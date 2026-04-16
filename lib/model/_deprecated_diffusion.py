import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy import integrate

from lib.model.sde import init_sde


class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.leakyrelu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual


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
        self.feature_extractor = []
        for i in range(4):
            self.feature_extractor.append(Residual(256, 256))
            self.feature_extractor.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_extractor = nn.Sequential(*self.feature_extractor)


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
            nn.Sequential(nn.Linear(128+256+1024, 512), self.act, zero_module(nn.Linear(512, 45))),
            nn.Sequential(nn.Linear(128+256+1024, 512), self.act, zero_module(nn.Linear(512, 3))),
            nn.Sequential(nn.Linear(128+256+1024, 512), self.act, zero_module(nn.Linear(512, 10))),
        )

    def get_score(self, sample, img_feat):
        img_feat = self.feature_extractor(img_feat)
        img_feat = img_feat.flatten(1)
        mano_feat = self.pose_encoder(sample['gt_mano'])
        t_feat = self.t_encoder(sample['t'].squeeze(1))
        total_feat = torch.cat([img_feat, t_feat, mano_feat], dim=1)

        mano_pose = self.fusion_tail[0](total_feat)
        mano_rot = self.fusion_tail[1](total_feat)
        mano_shape = self.fusion_tail[2](total_feat)

        _, std = self.marginal_prob_fn(total_feat, sample['t'])
        out_score = torch.cat([mano_pose, mano_rot, mano_shape], dim=-1) / (std + 1e-7)

        return out_score

    def forward_train(self, sample, feat):
        bs = sample['image'].shape[0]

        loss = 0
        for _ in range(self.cfg['repeat_num']):
            random_t = torch.rand(bs, device=feat.device) * (1. - self.sampling_eps) + self.sampling_eps
            random_t = random_t.unsqueeze(-1)  
            mu, std = self.marginal_prob_fn(sample['gt_mano'], random_t)
            std = std.view(-1, 1)

            z = torch.randn_like(sample['gt_mano'])
            perturbed_x = mu + z * std
            sample['sampled_pose'] = perturbed_x
            sample['t'] = random_t
            estimated_score = self.get_score(sample, feat)
            
            target_score = - z * std / (std ** 2)
            loss_weighting = std ** 2
            loss_ = torch.mean(torch.sum((loss_weighting * (estimated_score - target_score)**2).view(bs, -1), dim=-1))
            loss += loss_
        loss = loss / self.cfg['repeat_num']

        loss = {
            'loss_diffhand': loss
        }
        return loss
    
    def forward_infer(
            self, 
            sample, 
            feat, 
            init_x=None,
            T=1.0,
            eps=1e-5,
            num_steps=None,
            atol=1e-5, 
            rtol=1e-5, 
            denoise=True,
        ):
        bs = sample['image'].shape[0]
        init_x = self.prior_fn((bs, 58), T=T).to(feat.device) if init_x is None else init_x + self.prior_fn((bs, 58), T=T).to(feat.device)

        def score_eval_wrapper(data, feat):
            """A wrapper of the score-based model for use by the ODE solver."""
            with torch.no_grad():
                score = self.get_score(data, feat)
            return score.cpu().numpy().reshape((-1,))
    
        def ode_func(t, x):      
            """The ODE function for use by the ODE solver."""
            x = torch.tensor(x.reshape(-1, 58), dtype=torch.float32, device=feat.device)
            time_steps = torch.ones(bs, device=feat.device).unsqueeze(-1) * t
            drift, diffusion = self.sde_fn(torch.tensor(t))
            drift = drift.cpu().numpy()
            diffusion = diffusion.cpu().numpy()
            sample['sampled_pose'] = x
            sample['t'] = time_steps
            return drift - 0.5 * (diffusion**2) * score_eval_wrapper(sample, feat)
        
        # Run the black-box ODE solver, note the 
        t_eval = None
        if num_steps is not None:
            # num_steps, from T -> eps
            t_eval = np.linspace(T, eps, num_steps)

        res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
        xs = torch.tensor(res.y, device=feat.device).T.view(-1, bs, 58) # [num_steps, bs, pose_dim]
        x = torch.tensor(res.y[:, -1], device=feat.device).reshape(-1, 58) # [bs, pose_dim]
        # denoise, using the predictor step in P-C sampler

        # denoise, using the predictor step in P-C sampler
        if denoise:
            # Reverse diffusion predictor for denoising
            vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
            drift, diffusion = self.sde_fn(vec_eps)
            sample['sampled_pose'] = x.float()
            sample['t'] = vec_eps
            grad = self.get_score(sample, feat)
            drift = drift - diffusion**2*grad       # R-SDE
            mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
            x = mean_x


        num_steps = xs.shape[0]
        xs = xs.reshape(bs*num_steps, -1)

        return xs.permute(1, 0, 2), x