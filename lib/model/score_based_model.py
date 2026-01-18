import torch
import torch.nn as nn
import numpy as np
import time
from scipy import integrate

from lib.model.sde import init_sde
from lib.configs.args import cfg
from warnings import warn

def loss_fn(
        model, 
        data,
        marginal_prob_func, 
        sde_fn, 
        eps=1e-5, 
        mask_out=None,
    ):
    feat = data['feat']
    gt_pose = data['gt_pose']
    
    ''' get std '''
    bs = feat.shape[0]
    random_t = torch.rand(bs, device=feat.device) * (1. - eps) + eps         # [bs, ]
    random_t = random_t.unsqueeze(-1)                                   # [bs, 1]
    mu, std = marginal_prob_func(gt_pose, random_t)                     # [bs, pose_dim], [bs]
    std = std.view(-1, 1)                                               # [bs, 1]

    ''' perturb data and get estimated score '''
    z = torch.randn_like(gt_pose)                                       # [bs, pose_dim]
    perturbed_x = mu + z * std                                          # [bs, pose_dim]
    data['sampled_pose'] = perturbed_x
    data['t'] = random_t
    estimated_score = model(data)                                 # [bs, pose_dim]

    target_score = - z * std / (std ** 2)

    ''' loss weighting '''
    loss_weighting = std ** 2

    loss_ = torch.mean(torch.sum((loss_weighting * (estimated_score - target_score)**2).view(bs, -1), dim=-1))
    return loss_


def cond_ode_sampler(
        pose_dim,
        score_model,
        data,
        prior,
        sde_coeff,
        atol=3e-4, 
        rtol=3e-3, 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        denoise=True,
        init_x=None,
    ):
    device = data['feat'].device
    batch_size=data['feat'].shape[0]

    init_x = prior((batch_size, pose_dim), T=T).to(device) if init_x is None else init_x + prior((batch_size, pose_dim), T=T).to(device)
    shape = init_x.shape

    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
            if torch.any(torch.isnan(score)):
                print("\033[31mWarning: NaN detected in score evaluation. \033[0m")
                score = torch.nan_to_num_(score, nan=0.0, posinf=0.0, neginf=0.0)
        return score.cpu().numpy().reshape((-1,))
    
    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, pose_dim), device=device).float()
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        return drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
  
    # Run the black-box ODE solver, note the 
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)

    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval, max_step=10)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['sampled_pose'] = x.float()
        data['t'] = vec_eps
        grad = score_model(data)
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    return xs.permute(1, 0, 2), x


# This module dose not comprise the trainable part. 
class ScoreBasedModelAgent(nn.Module):
    def __init__(self):
        super(ScoreBasedModelAgent, self).__init__()
        self.cfg = cfg

        self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps, self.T = init_sde(cfg.sde_mode)


    def get_score_loss(self, denoiser, **kwargs):
        total_loss = 0
        for i in range(self.cfg.repeat_num):
            total_loss += loss_fn(
                model=denoiser,
                data=kwargs,
                marginal_prob_func=self.marginal_prob_fn,
                sde_fn=self.sde_fn,
                eps=self.sampling_eps
            )
        total_loss = total_loss / self.cfg.repeat_num
        return total_loss
    
    @torch.no_grad()
    def sample(self, data, denoiser, T0, init_x=None):
        if self.cfg.sampler == 'ode':
            in_process_sample, res =  cond_ode_sampler(
                denoiser.out_dim,
                score_model=denoiser,
                data=data,
                prior=self.prior_fn,
                sde_coeff=self.sde_fn,
                eps=self.sampling_eps,
                T=T0,
                num_steps=self.cfg.sampling_steps,
                init_x=init_x
            )
            return in_process_sample, res
        else:
            raise NotImplementedError("Only ode sampler is supported for now.")
        
    def get_score(self, data, denoiser):
        return denoiser(data)