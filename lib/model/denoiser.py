import torch 
import torch.nn as nn
import numpy as np
import math
from torch.nn import init
import warnings

from lib.model.parallel_linear import ParallelLinear

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.

        # self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        self.register_buffer('W', torch.randn(embed_dim // 2) * scale) # for DDP compatibility

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

class BaseDenoiser(nn.Module):
    def __init__(self, marginal_prob_fn, head='mano'):
        super(BaseDenoiser, self).__init__()
        self.act = nn.ReLU(True)
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            nn.Linear(128, 128),
            self.act,
        )

        self.marginal_prob_fn = marginal_prob_fn

        if head == 'mano':
            self.out_dim = 58
            self.head = ManoHead()
        elif head == 'obj':
            self.out_dim = 9
            self.head = ObjHead2()
        elif head == 'mano6d':
            self.out_dim = 16*6+10
            self.head = Mano6DHead()
        elif head == 'mano_pose':
            self.out_dim = 16*6
            self.head = ManoPoseHead2()
        else:
            raise ValueError('head should be either mano or obj')
        
        self.pose_encoder = nn.Sequential(
            nn.Linear(self.out_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )

    def forward(self, data):
        feat = data['feat']

        t = data['t']
        t_feat = self.t_encoder(t.squeeze(1))

        sampled_pose = data['sampled_pose']
        pose_feat = self.pose_encoder(sampled_pose)
        total_feat = torch.cat([t_feat, pose_feat, feat], dim=-1)

        _, std = self.marginal_prob_fn(sampled_pose, t)

        out = self.head(total_feat)
        out = out / (std + 1e-7)
        return out


class ManoHead(nn.Module):
    def __init__(self, ):
        super(ManoHead, self).__init__()
        self.act = nn.ReLU(True)
        self.head1 = nn.Sequential(
            nn.Linear(128+256+1024, 256),
            self.act,
            zero_module(nn.Linear(256, 3)),
        )

        self.head2 = nn.Sequential(
            nn.Linear(128+256+1024, 256),
            self.act,
            zero_module(nn.Linear(256, 45)),
        )

        self.head3 = nn.Sequential(
            nn.Linear(128+256+1024, 256),
            self.act,
            zero_module(nn.Linear(256, 10)),
        )

    def forward(self, total_feat):
        mano_pose = self.head1(total_feat)
        mano_rot = self.head2(total_feat)
        mano_shape = self.head3(total_feat)
        out = torch.cat([mano_pose, mano_rot, mano_shape], dim=-1)
        return out
    
class Mano6DHead(nn.Module):
    def __init__(self, ):
        super(Mano6DHead, self).__init__()
        self.act = nn.ReLU(True)
        self.head1 = nn.Sequential(
            nn.Linear(128+256+1024, 256),
            self.act,
            zero_module(nn.Linear(256, 6)),
        )

        self.head2 = nn.Sequential(
            nn.Linear(128+256+1024, 256),
            self.act,
            zero_module(nn.Linear(256, 15*6)),
        )

        self.head3 = nn.Sequential(
            nn.Linear(128+256+1024, 256),
            self.act,
            zero_module(nn.Linear(256, 10)),
        )

    def forward(self, total_feat):
        mano_pose = self.head1(total_feat)
        mano_rot = self.head2(total_feat)
        mano_shape = self.head3(total_feat)
        out = torch.cat([mano_pose, mano_rot, mano_shape], dim=-1)
        return out
    

class ManoPoseHead(nn.Module):
    def __init__(self, ):
        super(ManoPoseHead, self).__init__()
        self.act = nn.ReLU(True)
        self.head = []
        for i in range(16*2):
            self.head.append(nn.Sequential(
                nn.Linear(128+256+1024, 256),
                self.act,
                zero_module(nn.Linear(256, 3)),
            ))
        self.head = nn.ModuleList(self.head)
        
        warnings.warn("ManoPoseHead has been deprecated, which inefficiently processes multiple linear layers step-by-step. Please use ManoPoseHead2 instead.")

    def forward(self, total_feat):
        mano_pose = []
        for i in range(16*2):
            mano_pose.append(self.head[i](total_feat))
        mano_pose = torch.cat(mano_pose, dim=-1)
        return mano_pose
    
class ManoPoseHead2(nn.Module):
    def __init__(self, ):
        super(ManoPoseHead2, self).__init__()
        self.act = nn.ReLU(True)
        self.head = nn.Sequential(
            ParallelLinear(128+256+1024, 256, 16*2),
            self.act,
            zero_module(ParallelLinear(256, 3, 16*2)),
        )

    def forward(self, total_feat):
        mano_pose = self.head(total_feat)
        mano_pose = mano_pose.reshape(-1, 16*6)
        return mano_pose
    

class ManoPoseHead3(nn.Module):
    '''Deprecated: Add a another FC layer to ManoPoseHead2, but the performance is scarcely improved. '''
    def __init__(self, ):
        super(ManoPoseHead3, self).__init__()
        self.act = nn.ReLU(True)
        self.head = nn.Sequential(
            ParallelLinear(128+256+1024, 1024, 16*2),
            self.act,
            ParallelLinear(1024, 256, 16*2),
            self.act,
            zero_module(ParallelLinear(256, 3, 16*2)),
        )

    def forward(self, total_feat):
        mano_pose = self.head(total_feat)
        mano_pose = mano_pose.reshape(-1, 16*6)
        return mano_pose


class ObjHead(nn.Module):
    def __init__(self, ):
        super(ObjHead, self).__init__()
        self.act = nn.ReLU(True)
        self.head1 = nn.Sequential(
            nn.Linear(128+256+1024, 256),
            self.act,
            zero_module(nn.Linear(256, 3)),
        )

        self.head2 = nn.Sequential(
            nn.Linear(128+256+1024, 256),
            self.act,
            zero_module(nn.Linear(256, 3)),
        )

        self.head3 = nn.Sequential(
            nn.Linear(128+256+1024, 256),
            self.act,
            zero_module(nn.Linear(256, 3)),
        )

        warnings.warn("ObjHead has been deprecated, which inefficiently processes multiple linear layers step-by-step. Please use ObjHead2 instead.")


    def forward(self, total_feat):
        obj_pose = self.head1(total_feat)
        obj_rot = self.head2(total_feat)
        obj_shape = self.head3(total_feat)
        out = torch.cat([obj_pose, obj_rot, obj_shape], dim=-1)
        return out
    

class ObjHead2(nn.Module):
    def __init__(self, ):
        super(ObjHead2, self).__init__()
        self.act = nn.ReLU(True)
        self.head = nn.Sequential(
            ParallelLinear(128+256+1024, 256, 3),
            self.act,
            zero_module(ParallelLinear(256, 3, 3)),
        )

    def forward(self, total_feat):
        obj_pose = self.head(total_feat)
        obj_pose = obj_pose.reshape(-1, 3*3)
        return obj_pose