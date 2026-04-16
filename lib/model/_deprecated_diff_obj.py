import sys
import os
import torch
import numpy as np
import time
import torch.nn as nn
from easydict import EasyDict as edict
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix

from ipdb import set_trace
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .pts_encoder.pointnets import PointNetfeat
from .gf_algorithms.energynet import PoseEnergyNet
from .gf_algorithms.scorenet import PoseScoreNet, PoseDecoderNet
from .gf_algorithms.samplers import cond_ode_likelihood, cond_ode_sampler, cond_pc_sampler
from .sde import init_sde
from .gf_algorithms.score_utils import ExponentialMovingAverage


class GFObjectPose(nn.Module):
    _cfg = edict()
    _cfg.device = 'cuda'
    _cfg.posenet_mode = 'score'
    _cfg.pts_encoder = 'pointnet'
    _cfg.sampling_steps = 500
    _cfg.pose_mode = 'rot_matrix'
    _cfg.regression_head = 'Rx_Ry_and_T'
    _cfg.energy_mode = 'IP'
    _cfg.s_theta_mode = 'score'
    _cfg.norm_energy = 'identical'

    def __init__(self, prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T):
        super(GFObjectPose, self).__init__()
        cfg = self._cfg
        self.cfg = cfg
        self.device = cfg.device
        self.is_testing = False
        
        ''' Load model, define SDE '''
        # init SDE config
        self.prior_fn = prior_fn
        self.marginal_prob_fn = marginal_prob_fn
        self.sde_fn = sde_fn
        self.sampling_eps = sampling_eps
        self.T = T
        # self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps = init_sde(cfg.sde_mode)
        
        ''' encode pts '''
        # self.pts_encoder = PointNetfeat(778, True)
        # self.img_encoder = ImgEncoder()
        
        ''' score network'''
        # if self.cfg.sde_mode == 'edm':
        #     self.pose_score_net = PoseDecoderNet(
        #         self.marginal_prob_fn,
        #         sigma_data=1.4148, 
        #         pose_mode=self.cfg.pose_mode, 
        #         regression_head=self.cfg.regression_head
        #     )
        # else:
        per_point_feat = False
        
        if self.cfg.posenet_mode == 'score':
            self.pose_score_net = PoseScoreNet(self.marginal_prob_fn, self.cfg.pose_mode, self.cfg.regression_head, per_point_feat)
        elif self.cfg.posenet_mode == 'energy':
            self.pose_score_net = PoseEnergyNet(
                marginal_prob_func=self.marginal_prob_fn, 
                pose_mode=self.cfg.pose_mode,
                regression_head=self.cfg.regression_head,
                energy_mode=self.cfg.energy_mode,
                s_theta_mode=self.cfg.s_theta_mode,
                norm_energy=self.cfg.norm_energy)
        ''' ToDo: ranking network '''


    def extract_pts_feature(self, data):
        """extract the input pointcloud feature

        Args:
            data (dict): batch example without pointcloud feature. {'pts': [bs, num_pts, 3], 'sampled_pose': [bs, pose_dim], 't': [bs, 1]}
        Returns:
            data (dict): batch example with pointcloud feature. {'pts': [bs, num_pts, 3], 'pts_feat': [bs, c], 'sampled_pose': [bs, pose_dim], 't': [bs, 1]}
        """
        pts = data['pts']
        if self.cfg.pts_encoder == 'pointnet':
            pts_feat = self.pts_encoder(pts.permute(0, 2, 1))    # -> (bs, 3, 1024)
        elif self.cfg.pts_encoder in ['pointnet2']:
            pts_feat = self.pts_encoder(pts)
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            pts_pointnet_feat = self.pts_pointnet_encoder(pts.permute(0, 2, 1))
            pts_pointnet2_feat = self.pts_pointnet2_encoder(pts)
            pts_feat = self.fusion_layer(torch.cat((pts_pointnet_feat, pts_pointnet2_feat), dim=-1))
            pts_feat = self.act(pts_feat)
        else:
            raise NotImplementedError
        return pts_feat
    
    def extract_img_feature(self, data):
        return self.img_encoder(data['input_feat'])
    
   
    def sample(self, data, sampler, atol=1e-5, rtol=1e-5, snr=0.16, denoise=True, init_x=None, T0=None):
        if sampler == 'pc':
            in_process_sample, res = cond_pc_sampler(
                score_model=self,
                data=data,
                prior=self.prior_fn,
                sde_coeff=self.sde_fn,
                num_steps=self.cfg.sampling_steps,
                snr=snr,
                device=self.device,
                eps=self.sampling_eps,
                pose_mode=self.cfg.pose_mode,
                init_x=init_x
            )
            
        elif sampler == 'ode':
            T0 = self.T if T0 is None else T0

            
            in_process_sample, res =  cond_ode_sampler(
                score_model=self,
                data=data,
                prior=self.prior_fn,
                sde_coeff=self.sde_fn,
                atol=atol,
                rtol=rtol,
                device=self.device,
                eps=self.sampling_eps,
                T=T0,
                num_steps=self.cfg.sampling_steps,
                pose_mode=self.cfg.pose_mode,
                denoise=denoise,
                init_x=init_x
            )
        
        else:
            raise NotImplementedError
        
        return in_process_sample, res
    
   
    def calc_likelihood(self, data, atol=1e-5, rtol=1e-5):    
        latent_code, log_likelihoods = cond_ode_likelihood(
            score_model=self,
            data=data,
            prior=self.prior_fn,
            sde_coeff=self.sde_fn,
            marginal_prob_fn=self.marginal_prob_fn,
            atol=atol,
            rtol=rtol,
            device=self.device,
            eps=self.sampling_eps,
            num_steps=self.cfg.sampling_steps,
            pose_mode=self.cfg.pose_mode,
        )
        return log_likelihoods

    
    def forward(self, data, mode='score', init_x=None, T0=None):
        '''
        Args:
            data, dict {
                'pts': [bs, num_pts, 3]
                'pts_feat': [bs, c]
                'sampled_pose': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        if mode == 'score':
            out_score = self.pose_score_net(data) # normalisation
            return out_score
        elif mode == 'energy':
            out_energy = self.pose_score_net(data, return_item='energy')
            return out_energy
        elif mode == 'likelihood':
            likelihoods = self.calc_likelihood(data)
            return likelihoods
        elif mode == 'pts_feature':
            pts_feature = self.extract_pts_feature(data)
            return pts_feature
        elif mode == 'img_feature':
            img_feature = self.extract_img_feature(data)
            return img_feature
        elif mode == 'pc_sample':
            in_process_sample, res = self.sample(data, 'pc', init_x=init_x)
            return in_process_sample, res
        elif mode == 'ode_sample':
            in_process_sample, res = self.sample(data, 'ode', init_x=init_x, T0=T0)
            return in_process_sample, res
        else:
            raise NotImplementedError
        

def loss_fn(
        model, 
        data,
        marginal_prob_func, 
        sde_fn, 
        eps=1e-5, 
        likelihood_weighting=False,
        teacher_model=None,
        pts_feat_teacher=None
    ):
    device = model.device
    # pts = data['pts']
    gt_pose = data['obj_pose']
    
    ''' get std '''
    bs = gt_pose.shape[0]
    random_t = torch.rand(bs, device=device) * (1. - eps) + eps         # [bs, ]
    random_t = random_t.unsqueeze(-1)                                   # [bs, 1]
    mu, std = marginal_prob_func(gt_pose, random_t)                     # [bs, pose_dim], [bs]
    std = std.view(-1, 1)                                               # [bs, 1]

    ''' perturb data and get estimated score '''
    z = torch.randn_like(gt_pose)                                       # [bs, pose_dim]
    perturbed_x = mu + z * std                                          # [bs, pose_dim]
    data['sampled_pose'] = perturbed_x
    data['t'] = random_t
    estimated_score = model(data)                                 # [bs, pose_dim]

    ''' get target score '''
    if teacher_model is None:
        # theoretic estimation
        target_score = - z * std / (std ** 2)
        
    ''' loss weighting '''
    loss_weighting = std ** 2
    loss_ = torch.mean(torch.sum((loss_weighting * (estimated_score - target_score)**2).view(bs, -1), dim=-1))
    
    return loss_


class DiffusionObj(nn.Module):
    _cfg = edict()
    _cfg.repeat_num = 40
    _cfg.likelihood_weighting = False
    _cfg.ema_rate = 0.999
    _cfg.sampler_mode = ['ode']
    def __init__(self, ):
        super(DiffusionObj, self).__init__()
        self.cfg = self._cfg
        self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps, self.T = init_sde('ve')
        self.net = GFObjectPose(self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps, self.T)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=0.999)
        self.loss_fn = loss_fn

    def collect_score_loss(self, data, teacher_model=None, pts_feat_teacher=None):
        '''
        Args:
            data, dict {
                'pts': [bs, c]
                'gt_pose': [bs, pose_dim]
            }
        '''
        gf_loss = 0
        for _ in range(self.cfg.repeat_num):
            gf_loss += self.loss_fn(
                model=self.net,
                data=data,
                marginal_prob_func=self.marginal_prob_fn,
                sde_fn=self.sde_fn,
                likelihood_weighting=self.cfg.likelihood_weighting, 
                teacher_model=teacher_model,
                pts_feat_teacher=pts_feat_teacher
            )
        gf_loss /= self.cfg.repeat_num
        return gf_loss


    def collect_ema_loss(self, data):
        '''
        Args:
            data, dict {
                'pts': [bs, c]
                'gt_pose': [bs, pose_dim]
            }
        '''
        self.ema.store(self.net.parameters())
        self.ema.copy_to(self.net.parameters())
        with torch.no_grad():
            ema_loss = 0
            for _ in range(self.cfg.repeat_num):
                # calc score-matching loss
                ema_loss += self.loss_fn(
                    model=self.net,
                    data=data,
                    marginal_prob_func=self.marginal_prob_fn,
                    sde_fn=self.sde_fn,
                    likelihood_weighting=self.cfg.likelihood_weighting
                )
            ema_loss /= self.cfg.repeat_num
        self.ema.restore(self.net.parameters())
        ema_losses = {'ema': ema_loss}        
        return ema_losses     
    

    def train_score_func(self, data, teacher_model=None):
        """ One step of training """
        self.net.train()
        self.is_testing = False
        
        # data['pts_feat'] = self.net(data, mode='pts_feature')
        # data['img_feat'] = self.net(data, mode='img_feature')
        with torch.no_grad():
            if teacher_model is not None:
                teacher_model.eval()
            pts_feat_teacher = None if teacher_model is None else teacher_model(data, mode='pts_feature')
        self.pts_feature = True
        gf_losses = self.collect_score_loss(data, teacher_model, pts_feat_teacher)
        return {
            "loss_diff_obj": gf_losses
        }
    
        # self.update_network(gf_losses)
        # self.record_losses(gf_losses, 'train')
        # self.record_lr()
        
        # self.ema.update(self.net.parameters())
        # if self.cfg.ema_rate > 0 and self.clock.step % 5 == 0:
        #     ema_losses = self.collect_ema_loss(data)
        #     self.record_losses(ema_losses, 'train')
        # self.pts_feature = False
        # return gf_losses
    

    def eval_score_func(self, data):
        self.is_testing = True
        self.net.eval()
        # self.ema.store(self.net.parameters())
        # self.ema.copy_to(self.net.parameters())
        with torch.no_grad():
            # data['pts_feat'] = self.net(data, mode='pts_feature')
            # data['img_feat'] = self.net(data, mode='img_feature')
            self.pts_feature = True
            in_process_sample_list = []
            res_list = []
            sampler_mode_list = self.cfg.sampler_mode
            for sampler in sampler_mode_list:
                in_process_sample, res = self.net(data, mode=f'{sampler}_sample')
                in_process_sample_list.append(in_process_sample)
                res_list.append(res)
            
            self.pts_feature = False
        # self.ema.restore(self.net.parameters())

        inprocess_rt, res_rt = self.pose_process_rt(in_process_sample, res)

        return inprocess_rt, res_rt
    
    def pred_func(self, data, repeat_num=50, T0=0.55):
        self.is_testing = True
        self.net.eval()
        with torch.no_grad():
            data['pts_feat'] = self.net(data, mode='pts_feature')
            data['img_feat'] = self.net(data, mode='img_feature')
            self.pts_feature = True
            data['pts_feat'] = data['pts_feat'][:, None].repeat(1, repeat_num, 1).view(-1, data['pts_feat'].shape[-1])
            data['img_feat'] = data['img_feat'][:, None].repeat(1, repeat_num, 1).view(-1, data['img_feat'].shape[-1])
            sampler_mode_list = self.cfg.sampler_mode
            for sampler in sampler_mode_list:
                in_process_sample, res = self.net(data, mode=f'{sampler}_sample', T0=T0)
            self.pts_feature = False

            inprocess_rt, res_rt = self.pose_process_rt(in_process_sample, res)
            inprocess_rt = inprocess_rt.view(-1, repeat_num, inprocess_rt.shape[1], 4, 4)
            res_rt = res_rt.view(-1, repeat_num, 4, 4)
        return inprocess_rt, res_rt
            
    def pose_process_rt(self, inprocess, res):
        res_rot = rotation_6d_to_matrix(res[:, :-3])
        res_trans = res[:, -3:]
        res_rt = torch.cat([res_rot, res_trans[:, :, None]], dim=-1)
        _fill = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=res_rt.device)[None, None].repeat(res_rt.shape[0], 1, 1)
        res_rt = torch.cat([res_rt, _fill], dim=1)

        inprocess_rot = rotation_6d_to_matrix(inprocess[..., :-3])
        inprocess_trans = inprocess[..., -3:]
        inprocess_rt = torch.cat([inprocess_rot, inprocess_trans[..., None]], dim=-1)
        _fill = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=inprocess_rt.device)[None, None, None].repeat(inprocess_rt.shape[0], inprocess_rt.shape[1], 1, 1)
        inprocess_rt = torch.cat([inprocess_rt, _fill], dim=-2)
        return inprocess_rt, res_rt
    

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
    
class ImgEncoder(nn.Module):
    def __init__(self, num_heatmap_chan=256, num_feat_chan=256, size_input_feature=(32, 32),
                 nRegBlock=4, nRegModules=2):
        super(ImgEncoder, self).__init__()

        self.num_heatmap_chan = num_heatmap_chan
        self.num_feat_chan = num_feat_chan
        self.size_input_feature = size_input_feature

        self.nRegBlock = nRegBlock
        self.nRegModules = nRegModules

        reg = []
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                reg.append(Residual(self.num_feat_chan, self.num_feat_chan))

        self.reg = nn.ModuleList(reg)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample_scale = 2 ** self.nRegBlock

        # fc layers
        self.num_feat_out = self.num_feat_chan * (size_input_feature[0] * size_input_feature[1] // (self.downsample_scale ** 2))

    def forward(self, x):
        # x: B x num_feat_chan x 32 x 32
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                x = self.reg[i * self.nRegModules + j](x)
            x = self.maxpool(x)

        # x: B x num_feat_chan x 2 x 2
        out = x.view(x.size(0), -1)

        return out