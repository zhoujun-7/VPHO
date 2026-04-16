import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy import integrate
from timm.models import create_model
from manopth.manolayer import ManoLayer

from lib.model.sde import init_sde


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class DiffHand(nn.Module):
    _cfg = {
        "repeat_num": 20,
        'sde_mode': 've'
    }
    
    def __init__(
        self, 
        cfg=_cfg,
    ):
        super(DiffHand, self).__init__()
        self.cfg = cfg
        self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps, self.T = init_sde('ve')
        self.feature_extractor = create_model(
            "resnet50", 
            pretrained=False, 
            features_only=True,
            out_indices=(1, 2, 3, 4),
            in_chans=3,
        )
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(2048, 1024)


        self.act = nn.ReLU(True)
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            # self.act, # M4D26 update
            nn.Linear(128, 128),
            self.act,
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(58, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )

        self.fusion_tail = nn.Sequential(
            nn.Sequential(nn.Linear(128+256+1024, 512), self.act, zero_module(nn.Linear(512, 3))),
            nn.Sequential(nn.Linear(128+256+1024, 512), self.act, zero_module(nn.Linear(512, 45))),
            nn.Sequential(nn.Linear(128+256+1024, 512), self.act, zero_module(nn.Linear(512, 10))),
        )

        self.mano_layer_r = ManoLayer(flat_hand_mean=True, side="right", mano_root="asset/mano_v1_2/models", use_pca=False)

    def get_feature(self, sample):
        img_feat = self.feature_extractor(sample['rgb'])
        img_feat = self.max_pool(img_feat[-1])
        img_feat = img_feat.flatten(1)
        img_feat = self.act(img_feat)
        img_feat = self.fc(img_feat)
        return img_feat

    def get_score(self, sample):
        # img_feat = self.feature_extractor(sample['rgb'])
        # img_feat = self.max_pool(img_feat[-1])
        # img_feat = img_feat.flatten(1)
        img_feat = sample['feature']

        mano_feat = self.pose_encoder(sample['sampled_pose'])
        t_feat = self.t_encoder(sample['t'].squeeze(1))

        total_feat = torch.cat([img_feat, t_feat, mano_feat], dim=1)

        mano_rot = self.fusion_tail[0](total_feat)
        mano_pose = self.fusion_tail[1](total_feat)
        mano_shape = self.fusion_tail[2](total_feat)

        _, std = self.marginal_prob_fn(total_feat, sample['t'])
        out_score = torch.cat([mano_rot, mano_pose, mano_shape], dim=-1) / (std + 1e-7)

        return out_score

    def forward_train(self, sample):
        bs = sample['gt_mano'].shape[0]

        loss = 0
        for _ in range(self.cfg['repeat_num']):
            random_t = torch.rand(bs, device=sample['gt_mano'].device) * (1. - self.sampling_eps) + self.sampling_eps
            random_t = random_t.unsqueeze(-1)  
            mu, std = self.marginal_prob_fn(sample['gt_mano'], random_t)
            std = std.view(-1, 1)

            z = torch.randn_like(sample['gt_mano'])
            perturbed_x = mu + z * std
            sample['sampled_pose'] = perturbed_x
            sample['t'] = random_t
            estimated_score = self.get_score(sample)
            
            target_score = - z * std / (std**2)
            loss_ = (estimated_score - target_score)**2
            loss_ = std**2 * loss_
            loss_ = torch.mean(torch.sum(loss_, dim=-1))
            loss += loss_
        loss = loss / self.cfg['repeat_num']

        loss = {
            'loss_diffhand': loss
        }
        return loss
    
    def forward_infer(
            self, 
            sample, 
            init_x=None,
            T=1.0,
            eps=1e-5,
            num_steps=500,
            atol=1e-5, 
            rtol=1e-5, 
            denoise=True,
        ):
        bs = sample['rgb'].shape[0]
        device = sample['rgb'].device
        init_x = self.prior_fn((bs, 58), T=T).to(device) if init_x is None else init_x + self.prior_fn((bs, 58), T=T).to(device)

        def score_eval_wrapper(data):
            """A wrapper of the score-based model for use by the ODE solver."""
            with torch.no_grad():
                score = self.get_score(data)
            return score.cpu().numpy().reshape((-1,))
    
        def ode_func(t, x):      
            """The ODE function for use by the ODE solver."""
            x = torch.tensor(x.reshape(-1, 58), dtype=torch.float32, device=device)
            time_steps = torch.ones(bs, device=device).unsqueeze(-1) * t
            drift, diffusion = self.sde_fn(torch.tensor(t))
            drift = drift.cpu().numpy()
            diffusion = diffusion.cpu().numpy()
            sample['sampled_pose'] = x
            sample['t'] = time_steps
            return drift - 0.5 * (diffusion**2) * score_eval_wrapper(sample)
        
        # Run the black-box ODE solver, note the 
        t_eval = None
        if num_steps is not None:
            # num_steps, from T -> eps
            t_eval = np.linspace(T, eps, num_steps)

        res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
        xs = torch.tensor(res.y, device=device).T.view(-1, bs, 58) # [num_steps, bs, pose_dim]
        x = torch.tensor(res.y[:, -1], device=device).reshape(-1, 58) # [bs, pose_dim]
        # denoise, using the predictor step in P-C sampler

        # denoise, using the predictor step in P-C sampler
        if denoise:
            # Reverse diffusion predictor for denoising
            vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
            drift, diffusion = self.sde_fn(vec_eps)
            sample['sampled_pose'] = x.float()
            sample['t'] = vec_eps
            grad = self.get_score(sample)
            drift = drift - diffusion**2*grad       # R-SDE
            mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
            x = mean_x


        num_steps = xs.shape[0]
        xs = xs.reshape(bs*num_steps, -1).float()
        x = x.float()
        transl = sample['gt_hand_transl'][:, None].repeat(1, num_steps, 1).reshape(bs*num_steps, -1)
        process_verts, process_joints = self.mano_layer_r(xs[:, :48], xs[:, 48:], transl)
        process_verts = process_verts.reshape(bs, num_steps, -1, 3)
        process_joints = process_joints.reshape(bs, num_steps, -1, 3)
        verts, joints = self.mano_layer_r(x[:, :48], x[:, 48:], sample['gt_hand_transl'])

        return {
            'verts': verts / 1000,
            'joints': joints / 1000,
            'process_verts': process_verts / 1000,
            'process_joints': process_joints / 1000,
        }
    
    def forward(self, sample, is_train=True):
        sample['feature'] = self.get_feature(sample)
        # sample['feature'] = torch.randn([sample['gt_mano'].shape[0], 1024], device=sample['gt_mano'].device)
        if is_train:
            return self.forward_train(sample)
        else:
            y = self.forward_infer(sample)
            return y