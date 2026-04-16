import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from timm.models import create_model
from timm.utils import ModelEmaV3
# ops.feature_pyramid_network.FeaturePyramidNetwork
from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix, matrix_to_rotation_6d, matrix_to_axis_angle, rotation_6d_to_matrix,
    quaternion_to_matrix, matrix_to_quaternion, axis_angle_to_quaternion, quaternion_to_axis_angle
)

from lib.configs.args import cfg
from lib.model.backbone_FPN_HFL import FPN
from lib.model.denoiser import BaseDenoiser
from lib.model.head_inplane import HeadHeatmap, HeadHeatmap2
from lib.model.score_based_model import ScoreBasedModelAgent
from lib.model.ema import ExponentialMovingAverage
from lib.model.head_mano import HeadMano, mano_aa_to_6D, mano_6D_to_aa
from lib.model.head_object import HeadObject
from lib.model.attention import Transformer
from lib.utils.transform_fn import average_quaternion
from lib.utils.hand_fn import MANO_PARAMS_LEVEL, MANO_JOINT_LEVEL


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
    

class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, size_input_feature=(32, 32), nRegBlock=4, nRegModules=2):
        super(Encoder, self).__init__()

        self.project = nn.Conv2d(in_dim, hid_dim, bias=True, kernel_size=1, stride=1)

        self.nRegBlock = nRegBlock
        self.nRegModules = nRegModules
        reg = []
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                reg.append(Residual(hid_dim, hid_dim))
        self.reg = nn.ModuleList(reg)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downsample_scale = 2 ** self.nRegBlock
        self.out_dim = hid_dim * (size_input_feature[0] * size_input_feature[1] // (self.downsample_scale ** 2))

    def forward(self, x):
        """
            x: (B, in_dim, 32, 32)
            out: (B, num_feat_chan * 2 * 2)
        
        """
        # x: (B, in_dim, 32, 32)
        x = self.project(x)
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                x = self.reg[i * self.nRegModules + j](x)
            x = self.maxpool(x)
        out = x.flatten(1)
        return out
    

class DiffHandObj(nn.Module):
    def __init__(self):
        super(DiffHandObj, self).__init__()
        self.cfg = cfg
        self.feature_extractor = FPN()

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.score_agent = ScoreBasedModelAgent()
        self.mano_mode = 'mano_pose' # in [mano, mano_pose, mano6d]
        self.denoiser_hand = BaseDenoiser(self.score_agent.marginal_prob_fn, head=self.mano_mode)
        self.denoiser_obj = BaseDenoiser(self.score_agent.marginal_prob_fn, head='obj')

        # self.head_hm_hand = HeadHeatmap2(256, 21, 256, 2, 1)
        # self.head_hm_obj = HeadHeatmap2(256, 27, 256, 2, 1)
        self.head_hm_hand = HeadHeatmap2(256, 21, 128, 1, 1)
        self.head_hm_obj = HeadHeatmap2(256, 27, 128, 1, 1)

        self.transformer_obj = Transformer(inp_res=32, dim=256, depth=1, num_heads=4, mlp_ratio=2.)# TODO: try self-attention
        self.transformer_hand = Transformer(inp_res=32, dim=256, depth=1, num_heads=4, mlp_ratio=2.)

        self.encoder_hand = Encoder(256+21, 256, size_input_feature=(self.cfg.roi_size, self.cfg.roi_size))
        self.encoder_obj = Encoder(256+27, 256, size_input_feature=(self.cfg.roi_size, self.cfg.roi_size))

        self.head_mano = HeadMano(in_dim=1024, is_output_contact=True)  # TODO: use parallel linear
        self.head_obj = HeadObject()


    def forward(self, data, mode='predict'):
        """ data: {
            'rgb': (bs, 3, 256, 256),
            'root_joint': (bs, 3),
            'bbox_hand': (bs, 4),
            'bbox_obj': (bs, 4),
            'is_right': (bs,),

            if mode == 'train':
                'gt_obj': (bs, 9),
                'gt_mano': (bs, 3+45+10),
                'sampled_obj_pose': (bs, sample_num, 9),
                'sampled_mano_pose': (bs, sample_num, 3+45+10),

                "hm_hand": (bs, 21, 64, 64),
                "hm_obj": (bs, 21, 64, 64),
                "gt_hand_contact": (bs, 1080),
                
                "gt_hand_jt3d_flip": (bs, 21, 3)
                "gt_hand_vert_flip": (bs, 778, 3)
        }
        """
        assert mode in ['train', 'score', 'sample', 'predict']
        bs, device = data['rgb'].shape[0], data['rgb'].device
        hand_feat, obj_feat = self.feature_extractor(data['rgb'])  # (bs, 256, 56, 56), (bs, 256, 56, 56)
        
        idx_tensor = torch.arange(bs, device=device).float()[:, None]
        roi_boxes_hand = torch.cat((idx_tensor, data['bbox_hand']), dim=1)
        roi_boxes_obj = torch.cat((idx_tensor, data['bbox_obj']), dim=1)
        hf_hr = ops.roi_align(hand_feat, roi_boxes_hand, output_size=(self.cfg.roi_size, self.cfg.roi_size), spatial_scale=1/4.) #* checked, hand feature hand roi
        hf_or = ops.roi_align(hand_feat, roi_boxes_obj, output_size=(self.cfg.roi_size, self.cfg.roi_size), spatial_scale=1/4.) # hand feature obj roi
        of_hr = ops.roi_align(obj_feat, roi_boxes_hand, output_size=(self.cfg.roi_size, self.cfg.roi_size), spatial_scale=1/4.) # obj feature hand roi
        of_or = ops.roi_align(obj_feat, roi_boxes_obj, output_size=(self.cfg.roi_size, self.cfg.roi_size), spatial_scale=1/4.) # obj feature obj roi

        # region [mask out-of-bbox region]
        mask_inter = data['mask_inter'][:, None]
        mask_inter = F.interpolate(mask_inter, size=(self.cfg.roi_size, self.cfg.roi_size), mode='nearest')
        hf_or = hf_or * mask_inter
        of_hr = of_hr * mask_inter
        # endregion

        # heatmap
        pd_hm_hand = self.head_hm_hand(hf_hr)
        pd_hm_obj = self.head_hm_obj(of_or)
        
        #! flip back to original for object feature
        hf_or_ls, of_or_ls, pd_hm_obj_ls = [], [], []
        for i, is_right_i in enumerate(data['is_right']):
            if not is_right_i:
                hf_or_ls.append(hf_or[i].flip(-1))
                of_or_ls.append(of_or[i].flip(-1))
                pd_hm_obj_ls.append(pd_hm_obj[i].flip(-1))
            else:
                hf_or_ls.append(hf_or[i])
                of_or_ls.append(of_or[i])
                pd_hm_obj_ls.append(pd_hm_obj[i])
        hf_or = torch.stack(hf_or_ls, dim=0)
        of_or = torch.stack(of_or_ls, dim=0)
        pd_hm_obj = torch.stack(pd_hm_obj_ls, dim=0)

        # cross-attention for hand and object feature
        hf_hr_att = self.transformer_hand(hf_hr, of_hr) # TODO: mask out out-of-bbox region
        of_or_att = self.transformer_obj(of_or, hf_or)

        # squeeze feature to 1D
        if hf_hr.shape != pd_hm_hand.shape:
            pd_hm_hand_rs = F.interpolate(pd_hm_hand, size=hf_hr.shape[-2:], mode='bilinear', align_corners=False)
            pd_hm_obj_rs = F.interpolate(pd_hm_obj, size=of_or.shape[-2:], mode='bilinear', align_corners=False)
        else:
            pd_hm_hand_rs, pd_hm_obj_rs = pd_hm_hand, pd_hm_obj
        encoding_hand = self.encoder_hand(torch.cat((hf_hr_att, pd_hm_hand_rs), dim=1)) # (bs, 1024)
        encoding_obj = self.encoder_obj(torch.cat((of_or_att, pd_hm_obj_rs), dim=1)) # (bs, 1024)
        
        pd_mano_pose, pd_mano_shape, pd_hand_contact = self.head_mano(encoding_hand)
        pd_hand_vert, pd_hand_joint = self.head_mano.get_hand_verts(pose=pd_mano_pose, shape=pd_mano_shape)

        # pd_hand_joint2d = self.project_hand_joint(
        #     joint3d_local=pd_hand_joint, 
        #     joint_root=data['root_joint_flip'], 
        #     cam_intrinsic=data['cam_intr_crop'], 
        # )

        if mode == 'train':
            # diffusion loss
            loss_dt, pd_dt = {}, {}
            
            if self.mano_mode == 'mano6d':
                gt_mano_pose = mano_aa_to_6D(data['gt_mano'])
            elif self.mano_mode == 'mano_pose':
                gt_mano_pose = mano_aa_to_6D(data['gt_mano'])[..., :-10]
            else:
                gt_mano_pose = data['gt_mano']
            if use_axsym:=False:
                gt_pose = self.head_obj.to_axsym_pose(data['gt_obj'], data['obj_name'])
            else:
                gt_pose = data['gt_obj']

            loss_dt['diff_hand_loss'] = self.score_agent.get_score_loss(denoiser=self.denoiser_hand, feat=encoding_hand, gt_pose=gt_mano_pose)
            loss_dt['diff_obj_loss'] = self.score_agent.get_score_loss(denoiser=self.denoiser_obj, feat=encoding_obj, gt_pose=gt_pose)
        
            # heatmap loss
            loss_dt['hm_hand_loss'] = self.head_hm_hand.get_loss(pd_hm=pd_hm_hand, gt_hm=data['hm_hand'], weight=2)
            loss_dt['hm_obj_loss'] = self.head_hm_obj.get_loss(pd_hm=pd_hm_obj, gt_hm=data['hm_obj'], weight=0)

            # mano_loss
            gt_mano_pose, gt_mano_shape = data['gt_mano'][:, :48], data['gt_mano'][:, 48:]
            # gt_hand_vert, gt_hand_joint = self.head_mano.get_hand_verts(pose=gt_mano_pose, shape=gt_mano_shape) #! deprecated, as left hand shape is incompatible with right hand mano, use data['gt_hand_vert_flip'], 'gt_joint': data['gt_hand_jt3d_flip'] instead
            gt_hand_vert, gt_hand_joint = data['gt_hand_vert_flip'], data['gt_hand_jt3d_flip'] # checked, from annotation, no error introduced
            mano_loss_dt = self.head_mano.get_loss(pd_pose=pd_mano_pose, pd_shape=pd_mano_shape, pd_vert=pd_hand_vert, pd_joint=pd_hand_joint,
                                                   gt_pose=gt_mano_pose, gt_shape=gt_mano_shape, gt_vert=gt_hand_vert, gt_joint=gt_hand_joint,
                                                   is_right=data['is_right'])
            loss_dt.update(mano_loss_dt)
            loss_dt['hand_contact_loss'] = self.head_mano.get_contact_loss(pd_contact=pd_hand_contact, gt_contact=data['gt_hand_contact'])

            # apply weight to losses
            total_loss = 0
            for k, v in loss_dt.items():
                weighted_loss = v * getattr(self.cfg, f'weight_{k}')
                total_loss = total_loss + weighted_loss
                loss_dt[k] = weighted_loss
            loss_dt['total_loss'] = total_loss

            pd_dt['hand_vert'] = pd_hand_vert
            pd_dt['hand_joint'] = pd_hand_joint
            pd_dt['hand_contact'] = pd_hand_contact
            pd_dt['hand_heatmap'] = pd_hm_hand
            pd_dt['obj_heatmap'] = pd_hm_obj
            # pd_dt['hand_joint2d'] = pd_hand_joint2d
            return loss_dt, pd_dt
                
        elif mode == 'predict':
            pd_dt = {}
            pd_dt['hand_vert'] = pd_hand_vert
            pd_dt['hand_joint'] = pd_hand_joint
            pd_dt['hand_contact'] = pd_hand_contact
            pd_dt['hand_heatmap'] = pd_hm_hand
            pd_dt['obj_heatmap'] = pd_hm_obj
            # pd_dt['hand_joint2d'] = pd_hand_joint2d

            # region [diffusion generates hand results]
            _, hand_feat_dim = encoding_hand.shape
            encoding_hand_repeat = encoding_hand[:, None].repeat(1, self.cfg.sample_num, 1).reshape(-1, hand_feat_dim)
            hand_data = {'feat': encoding_hand_repeat}

            if denoise_from_regression:=False:
                mano_pose_regression = mano_aa_to_6D(pd_mano_pose)
                init_x = mano_pose_regression[:, None].repeat(1, self.cfg.sample_num, 1).reshape(bs*self.cfg.sample_num, -1)
                hand_inprocess, hand_final = self.score_agent.sample(hand_data, self.denoiser_hand, T0=0.55, init_x=init_x)
            else:
                hand_inprocess, hand_final = self.score_agent.sample(hand_data, self.denoiser_hand, self.cfg.sample_T0)
            hand_inprocess, hand_final = hand_inprocess.float(), hand_final.float()

            hand_inprocess, hand_final = self.poseprocess_diffusion_hand(hand_inprocess, hand_final, pd_mano_shape, sample_num=self.cfg.sample_num)
            pd_dt['hand_inprocess'] = hand_inprocess.reshape(bs, self.cfg.sample_num, -1, 58)
            pd_dt['hand_final'] = hand_final.reshape(bs, self.cfg.sample_num, 58)

            #! this part is for inprocess visualization and can be removed during runtime evaluation
            diff_inprocess_hand_vert, diff_inprocess_hand_joint = self.head_mano.get_hand_verts(pose=hand_inprocess[0, ::10, :48], shape=hand_inprocess[0, ::10, 48:]) #* only take every 10th sample of the first batchsample
            pd_dt['inprocess_hand_vert'] = diff_inprocess_hand_vert.reshape(-1, 778, 3)
            pd_dt['inprocess_hand_joint'] = diff_inprocess_hand_joint.reshape(-1, 21, 3)

            diff_final_hand_vert, diff_final_hand_joint = self.head_mano.get_hand_verts(pose=hand_final[:, :48], shape=hand_final[:, 48:])
            pd_dt['final_hand_vert'] = diff_final_hand_vert.reshape(bs, self.cfg.sample_num, 778, 3)
            pd_dt['final_hand_joint'] = diff_final_hand_joint.reshape(bs, self.cfg.sample_num, 21, 3)

            # region [rank by heatmap, fuse topk]
            select_data = self.select_topk_hand(
                # mode='heatmap', #* deprecated, MJE: 13.17 -> 11.42 useful
                mode='heatmap_cascade', #* MJE: 11.42 -> 11.22 useful
                pose=hand_final[:, :48],
                pose_regression=pd_mano_pose,
                shape=hand_final[:, 48:],
                joint_root=data['root_joint_flip'], 
                cam_intrinsic=data['cam_intr_crop_flip'], 
                heatmap=pd_hm_hand, 
                bbox=data['bbox_hand'], 
                k=self.cfg.topk,
            )
            pd_dt['hand_mean'] = select_data['diff_topk_fused_mano']
            pd_dt['final_hand_vert'] = select_data['diff_vert']
            pd_dt['final_hand_joint'] = select_data['diff_joint']
            pd_dt['mean_hand_vert'], pd_dt['mean_hand_joint'] = select_data['diff_topk_fused_vert'], select_data['diff_topk_fused_joint']
            # endregion
            # endregion

            # region [diffusion generates object results]
            bs, obj_feat_dim = encoding_obj.shape
            encoding_obj_repeat = encoding_obj[:, None].repeat(1, self.cfg.sample_num, 1).reshape(-1, obj_feat_dim)
            diff_obj_data = {'feat': encoding_obj_repeat}
            obj_inprocess, obj_final = self.score_agent.sample(diff_obj_data, self.denoiser_obj, self.cfg.sample_T0)
            pd_dt['obj_inprocess'] = obj_inprocess.reshape(bs, self.cfg.sample_num, -1, 9)
            pd_dt['obj_final'] = obj_final.reshape(bs, self.cfg.sample_num, 9)

            # region [rank by heatmap, fuse topk]
            diff_obj_point2d = self.project_object_point(
                pose6d=pd_dt['obj_final'], 
                joint_root=data['root_joint'], 
                cam_intrinsic=data['cam_intr_crop_flip'], 
                obj_name=data['obj_name'],
                is_right=data['is_right'],
            ) # (bs, sample_num, 21, 2)
            obj_topk = self.select_topk_obj(
                mode='heatmap',
                pt2d=diff_obj_point2d, 
                heatmap=pd_hm_obj, 
                bbox=data['bbox_obj'], 
                k=self.cfg.topk,
            )
            idx_tensor = torch.arange(bs, device=device)[:, None].repeat(1, self.cfg.topk)
            obj_final_topk = pd_dt['obj_final'][idx_tensor, obj_topk]
            pd_dt['obj_mean'] = average_obj_pose(obj_final_topk, mode='quat')
            pd_dt['obj_final'] = obj_final_topk
            # endregion

            # endregion
            return pd_dt
        
    def poseprocess_diffusion_hand(self, hand_inprocess, hand_final, pd_mano_shape, sample_num):
        bs, _ = pd_mano_shape.shape
        _, inprocess_num, pose_dim = hand_inprocess.shape
        if self.mano_mode == 'mano6d':
            hand_inprocess = mano_6D_to_aa(hand_inprocess) 
            hand_inprocess = hand_inprocess.reshape(-1, inprocess_num, 58)
            hand_final = mano_6D_to_aa(hand_final)
            hand_final = hand_final.reshape(-1, 58)
        elif self.mano_mode == 'mano_pose':
            hand_inprocess = hand_inprocess.reshape(-1, sample_num, inprocess_num, 16, 6)
            hand_inprocess = matrix_to_axis_angle(rotation_6d_to_matrix(hand_inprocess)) 
            hand_inprocess = hand_inprocess.reshape(bs, sample_num, inprocess_num, 16*3)
            inprocess_shape = pd_mano_shape[:, None, None].repeat(1, sample_num, inprocess_num, 1)
            hand_inprocess = torch.cat((hand_inprocess, inprocess_shape), dim=-1)
            hand_inprocess = hand_inprocess.reshape(-1, inprocess_num, 58)

            hand_final = hand_final.reshape(bs, sample_num, 16, 6)
            hand_final = matrix_to_axis_angle(rotation_6d_to_matrix(hand_final))
            hand_final = hand_final.reshape(bs, sample_num, 16*3)
            final_shape = pd_mano_shape[:, None].repeat(1, sample_num, 1)
            hand_final = torch.cat((hand_final, final_shape), dim=-1)
            hand_final = hand_final.reshape(-1, 58)
        elif self.mano_mode == 'mano':
            hand_inprocess = hand_inprocess.reshape(-1, inprocess_num, 58)
            hand_final = hand_final.reshape(-1, 58)
        return hand_inprocess, hand_final

    def project_hand_point(self, **kwargs):
        ''' project hand joint to 2D image plane
            cam_intrinsic: (bs, 3, 3)
        '''
        shape = kwargs['joint3d_cam'].shape
        joint2d = torch.einsum('b...ij,blj->b...il', kwargs['joint3d_cam'], kwargs['cam_intrinsic'])
        joint2d = joint2d[..., :2] / joint2d[..., 2:]
        return joint2d
        
    def select_topk_hand(self, **kwargs):
        ''' mode: 'heatmap' or 'heatmap_cascade' or 'energy'
            k: int
            
            if mode == 'heatmap':
                pose: (bs*sample_num, 48)
                shape: (bs*sample_num, 10)
                joint_root: (bs, 3), 
                cam_intrinsic: (bs, 3, 3), 
                heatmap: (bs, 21, H, W)
                bbox: (bs, 4)
                k: (k,)
                fuse_index: (n,)
                observe_index: (m,)
                is_independent: bool
            
            if mode == 'heatmap_cascade':
                pose: (bs*sample_num, 48)
                pose_regression: (bs*sample_num, 48)
                shape: (bs*sample_num, 10)
                joint_root: (bs, 3), 
                cam_intrinsic: (bs, 3, 3), 
                heatmap: (bs, 21, H, W)
                bbox: (bs, 4)
                k: (k,)
                fuse_index: (n,)
                observe_index: (m,)
                is_independent: bool
        '''
        bs = kwargs['joint_root'].shape[0]
        if kwargs['mode'] == 'heatmap': # checked
            
            fused_data = self.select_topk_hand_by_observed_heatmap_and_fuse_by_index(
                pose=kwargs['pose'],
                shape=kwargs['shape'],
                joint_root=kwargs['joint_root'],
                cam_intrinsic=kwargs['cam_intrinsic'],
                heatmap=kwargs['heatmap'],
                bbox=kwargs['bbox'],
                k=kwargs['k'],
                fuse_index=list(range(48)),
                observe_index=list(range(21)),
                is_independent=False,
            )
            fused_pose = fused_data['fused_pose'][:, 0] # (bs, 58)
            shape = kwargs['shape'].reshape(bs, -1, 10)[:, 0]
            fused_mano = torch.cat((fused_pose, shape), dim=-1)
            fused_vert, fused_joint = self.head_mano.get_hand_verts(pose=fused_pose, shape=shape)
            fused_vert, fused_joint = fused_vert.reshape(bs, 778, 3), fused_joint.reshape(bs, 21, 3)

            return {
                'topk': fused_data['topk'],
                'diff_topk_vert': fused_data['topk_vert'],
                'diff_topk_joint': fused_data['topk_joint'],
                'diff_topk_fused_mano': fused_mano,
                'diff_topk_fused_vert': fused_vert,
                'diff_topk_fused_joint': fused_joint,
                'fused_data_ls': [fused_data]
            }

        if kwargs['mode'] == 'heatmap_cascade':

            pose=kwargs['pose']
            shape=kwargs['shape']

            extra_pose = torch.zeros_like(pose).reshape(bs, -1, 48)
            extra_pose = extra_pose + kwargs['pose_regression'][:, None]
            pose = pose.reshape(bs, -1, 48)
            num_candidate = pose.shape[1]
            pose = torch.cat((pose, extra_pose), dim=1).reshape(-1, 48) #* useful, MJE: 11.22 -> 11.15, regression result as candidate
            shape = shape.reshape(bs, -1, 10).repeat(1, 2, 1).reshape(-1, 10)

            fused_data_ls = []
            for level_i in range(4):
                fuse_idx = MANO_PARAMS_LEVEL[level_i]
                # observe_idx = MANO_JOINT_LEVEL[level_i+1] # deprecated, MJE: 11.48 -> 11.61, only the direct children which is not very useful. This may be caused by the occlusion of the direct children joints.
                observe_idx = []
                for j in range(level_i+1, 5):  #* useful MJE: 11.48 -> 11.22 
                    observe_idx.extend(MANO_JOINT_LEVEL[j])

                if level_i in [0]: # only useful for the wrist
                    pose = pose.view(bs, -1, 48)
                    pose[:, num_candidate:, fuse_idx] = pose[:, :num_candidate, fuse_idx]
                    pose = pose.view(-1, 48)

                fused_data_i = self.select_topk_hand_by_observed_heatmap_and_fuse_by_index(
                    pose=pose,
                    shape=shape,
                    joint_root=kwargs['joint_root'],
                    cam_intrinsic=kwargs['cam_intrinsic'],
                    heatmap=kwargs['heatmap'],
                    bbox=kwargs['bbox'],
                    k=kwargs['k'],
                    fuse_index=fuse_idx,
                    observe_index=observe_idx,
                    is_independent=False if level_i == 0 else True,
                )

                pose = fused_data_i['fused_pose'].reshape(-1, 48)
                fused_data_ls.append(fused_data_i)

            fused_pose = fused_data_ls[-1]['fused_pose'][:, 0] # (bs, 58)
            shape = kwargs['shape'].reshape(bs, -1, 10)[:, 0]
            fused_mano = torch.cat((fused_pose, shape), dim=-1)
            fused_vert, fused_joint = self.head_mano.get_hand_verts(pose=fused_pose, shape=shape)
            fused_vert, fused_joint = fused_vert.reshape(bs, 778, 3), fused_joint.reshape(bs, 21, 3)

            return {
                'topk': fused_data_ls[-1]['topk'],
                'diff_topk_vert': fused_data_ls[-1]['topk_vert'],
                'diff_topk_joint': fused_data_ls[-1]['topk_joint'],
                'diff_topk_fused_mano': fused_mano,
                'diff_topk_fused_vert': fused_vert,
                'diff_topk_fused_joint': fused_joint,
                'diff_vert': fused_data_ls[0]['vert'],
                'diff_joint': fused_data_ls[0]['joint'],
                # 'fused_data_ls': fused_data_ls
            }

        elif kwargs['mode'] == 'energy':
            raise NotImplementedError

    
    def select_topk_hand_by_observed_heatmap_and_fuse_by_index(self, **kwargs):
        """
            pose: (bs*sample_num, 48)
            shape: (bs*sample_num, 10)
            joint_root: (bs, 3)
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            fuse_index: (n, )
            observe_index: (m, )
            is_independent: bool
        """
        bs, J, H, W = kwargs['heatmap'].shape
        K = kwargs['k']
        vert, joint = self.head_mano.get_hand_verts(pose=kwargs['pose'], shape=kwargs['shape'])
        vert, joint = vert.reshape(bs, -1, 778, 3), joint.reshape(bs, -1, 21, 3)
        joint_cam = joint + kwargs['joint_root'][:, None, None]
        pose = kwargs['pose'].reshape(bs, -1, 16, 3)

        pt2d = self.project_hand_point(joint3d_cam=joint_cam, cam_intrinsic=kwargs['cam_intrinsic'])
        bbox = kwargs['bbox'][:, None, None, :]
        pt2d = pt2d - bbox[..., :2]
        pt2d = 2 * pt2d / (bbox[..., 2:] - bbox[..., :2]) - 1

        heat_val = []
        for i in kwargs['observe_index']: # TODO: to be checked
            grid_i = pt2d[:, :, [i]]
            heatmap_i = kwargs['heatmap'][:, [i]]
            heat_val_i = F.grid_sample(heatmap_i, grid_i, align_corners=False, mode='bicubic')
            heat_val_i = heat_val_i.squeeze(1)
            heat_val.append(heat_val_i)
        heat_val = torch.concat(heat_val, dim=-1) # (bs, sample_num, m)

        if not kwargs['is_independent']: #* checked
            heat_val = heat_val.sum(dim=-1)
            val, topk = heat_val.topk(K, dim=1)

            idx_tensor = torch.arange(bs, device=heat_val.device)[:, None].repeat(1, K)
            topk_pose = kwargs['pose'].reshape(bs, -1, 48)[idx_tensor, topk]
            topk_idx_pose = topk_pose[:, :, kwargs['fuse_index']]
            topk_idx_pose_aa = topk_idx_pose.reshape(bs, K, -1, 3) # (bs, k, n, 3)
            topk_idx_pose_quat = axis_angle_to_quaternion(topk_idx_pose_aa) # (bs, k, n, 4)
            topk_idx_pose_quat = topk_idx_pose_quat.permute(0, 2, 1, 3)
            fuse_idx_pose_quat = average_quaternion(topk_idx_pose_quat) # (bs, n, 4) # TODO weighted average
            fused_idx_pose_aa = quaternion_to_axis_angle(fuse_idx_pose_quat) # (bs, n, 3)
            fused_idx_pose_aa = fused_idx_pose_aa.reshape(bs, -1)

            fused_pose = kwargs['pose'].reshape(bs, -1, 48)
            fused_pose[:, :, kwargs['fuse_index']] = fused_pose[:, :, kwargs['fuse_index']] * 0 + fused_idx_pose_aa[:, None]

            topk_vert = vert[idx_tensor, topk]
            topk_joint = joint[idx_tensor, topk]

        else:
            M, N = len(kwargs['observe_index']), len(kwargs['fuse_index'])
            assert M % (N // 3) == 0
            n_observed = M // (N // 3)
            heat_val = heat_val.reshape(bs, -1, n_observed, N//3).mean(dim=-2) # (bs, K, n)
            val, topk = heat_val.topk(K, dim=1) # (bs, K, n), (bs, K, n)

            pose = kwargs['pose'].reshape(bs, -1, 16, 3) # (bs, sample_num, 48)
            idx_tensor1 = torch.arange(bs, device=heat_val.device)[:, None, None].repeat(1, K, N//3)
            idx_tensor2 = torch.tensor(kwargs['fuse_index'], dtype=torch.long, device=heat_val.device)[None, None].repeat(bs, K, 1).reshape(bs, K, -1, 3)
            idx_tensor2 = idx_tensor2[:, :, :, 0] // 3  # TODO: To be more elegant

            topk_pose = pose[idx_tensor1, topk, idx_tensor2] # (bs, K, M, 3)
            topk_idx_posed_quat = axis_angle_to_quaternion(topk_pose)
            topk_idx_posed_quat = topk_idx_posed_quat.permute(0, 2, 1, 3)
            fused_idx_pose_quat = average_quaternion(topk_idx_posed_quat)
            fused_idx_pose_aa = quaternion_to_axis_angle(fused_idx_pose_quat) # (bs, M, 3)
            fused_idx_pose_aa = fused_idx_pose_aa.reshape(bs, -1)

            fused_pose = kwargs['pose'].reshape(bs, -1, 48)
            fused_pose[:, :, kwargs['fuse_index']] = fused_pose[:, :, kwargs['fuse_index']] * 0 + fused_idx_pose_aa[:, None]
            
            topk_vert = vert[idx_tensor1, topk]
            topk_joint = joint[idx_tensor1, topk]

        return {
            'val': val,
            'topk': topk,
            'fused_idx_pose': fused_idx_pose_aa,
            'fused_pose': fused_pose,
            'topk_vert': topk_vert,
            'topk_joint': topk_joint,
            'vert': vert,
            'joint': joint,
        }


    def project_object_point(self, **kwargs):
        ''' project hand joint to 2D image plane
            pose6d: (bs, 50, 9)
            joint_root=(bs, 3)
            cam_intrinsic: (bs, 3, 3) 
            is_right: (bs,)
        '''
        pose6d = kwargs['pose6d'].clone().float()
        if pose6d.dim() == 3:
            joint_root = kwargs["joint_root"].unsqueeze(1)
        pose6d[..., 6:] =  pose6d[..., 6:] + joint_root
        obj_pt3d = self.head_obj(pose6d, kwargs['obj_name'])

        idx_tensor = torch.arange(obj_pt3d.shape[0], device=obj_pt3d.device)
        flipped_idx = idx_tensor[~kwargs['is_right']]
        obj_pt3d[flipped_idx, :, :, 0] = obj_pt3d[flipped_idx, :, :, 0] * -1

        obj_pt2d = torch.einsum('b...ij,blj->b...il', obj_pt3d, kwargs['cam_intrinsic'])
        obj_pt2d = obj_pt2d[..., :2] / obj_pt2d[..., 2:]
        obj_pt2d = obj_pt2d[..., :2]
        return obj_pt2d
    
    def select_topk_obj(self, **kwargs):
        ''' mode: 'heatmap' or 'energy'
            pt2d: (bs, sample_num, 21, 2)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            is_right: (bs,)
            k: int
        '''
        if kwargs['mode'] == 'heatmap':
            bs, J, H, W = kwargs['heatmap'].shape
            bbox = kwargs['bbox'][:, None, None, :]
            pt2d = kwargs['pt2d'] - bbox[..., :2]
            pt2d = 2 * pt2d / (bbox[..., 2:] - bbox[..., :2]) - 1
            heatval = []
            for i in range(J):
            # for i in range(10):
                grid_i = pt2d[:, :, [i]]
                heatmap_i = kwargs['heatmap'][:, [i]]
                heatval_i = F.grid_sample(heatmap_i, grid_i, align_corners=False, mode='bicubic')
                heatval_i = heatval_i.squeeze(1)
                heatval.append(heatval_i)
            heatval = torch.concat(heatval, dim=-1) # (bs, sample_num, 21)
            heatval = heatval.sum(dim=-1)

            val, topk = heatval.topk(kwargs['k'], dim=1)

        elif kwargs['mode'] == 'energy':
            raise NotImplementedError

        return topk

    
            


def average_mano_params(mano_params, weights=None, mode='quat', pca_mat=None):
    """calculate the average mano parameters of the multiple mano parameters
    Args:
        mano_params: [B, ..., num_mano_params, 58]
        weights: [B, ..., num_mano_params]. Defaults to None.

    Returns:
        oriented_mano_avg: average mano parameters, [B, ..., 58]
    """

    assert mode in ['quat', 'rot6d', 'pca']

    shape = mano_params.shape
    mano_pose = mano_params[..., :48]
    mano_shape = mano_params[..., 48:]

    if weights is None:
        weights = torch.ones_like(mano_pose[..., 0]) / shape[-2]
    
    mano_shape_mean = mano_shape * weights[..., None]
    mano_shape_mean = mano_shape_mean.sum(dim=-2)

    if mode == 'quat':
        mano_pose_aa = mano_pose.reshape(*shape[:-1], 16, 3)
        mano_pose_quat = axis_angle_to_quaternion(mano_pose_aa) # (..., n, 16, 4)
        mano_pose_quat = mano_pose_quat.transpose(-2, -3) # (..., 16, n, 4)
        weights = weights[..., None, :].repeat(1, 16, 1)
        mano_pose_mean_quat = average_quaternion(mano_pose_quat, weights) # (..., 16, 4)
        mano_pose_mean_aa = quaternion_to_axis_angle(mano_pose_mean_quat) # (..., 16, 3)
        mano_pose_mean = mano_pose_mean_aa.reshape(*shape[:-2], 48)
    elif mode == 'rot6d':
        mano_pose_aa = mano_pose.reshape(*shape[:-1], 16, 3)
        mano_pose_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(mano_pose_aa))
        mano_pose_rot6d = mano_pose_rot6d.reshape(*shape[:-1], 16, 6)
        mano_pose_mean_rot6d = mano_pose_rot6d * weights[..., None, None]
        mano_pose_mean_rot6d = mano_pose_rot6d.sum(dim=-3)
        mano_pose_mean_aa = matrix_to_axis_angle(rotation_6d_to_matrix(mano_pose_mean_rot6d))
        mano_pose_mean = mano_pose_mean_aa.reshape(*shape[:-2], 48)
    elif mode == 'pca':
        raise NotImplementedError # TODO: mean in pca space
    
    mano_params_mean = torch.cat((mano_pose_mean, mano_shape_mean), dim=-1)
    return mano_params_mean

def average_obj_pose(obj_pose, weights=None, mode='quat'):
    """calculate the average object pose of the multiple object poses
    Args:
        obj_pose: [B, ..., N, 9]
        weights: [B, ..., N, 9]. Defaults to None.

    Returns:
        oriented_obj_avg: average object pose, [B, ..., 9]
    """

    assert mode in ['quat', 'rot6d']

    shape = obj_pose.shape
    obj_rot6d = obj_pose[..., :6]
    obj_trans = obj_pose[..., 6:]

    if weights is None:
        weights = torch.ones_like(obj_rot6d[..., 0]) / shape[-2]

    obj_trans_mean = obj_trans * weights[..., None]
    obj_trans_mean = obj_trans_mean.sum(dim=-2)

    if mode == 'quat':
        obj_quat = matrix_to_quaternion(rotation_6d_to_matrix(obj_rot6d))
        obj_quat_mean = average_quaternion(obj_quat, weights)
        obj_rot6d_mean = matrix_to_rotation_6d(quaternion_to_matrix(obj_quat_mean))
    elif mode == 'rot6d':
        obj_rot6d_mean = obj_rot6d * weights[..., None]
        obj_rot6d_mean = obj_rot6d_mean.sum(dim=-2)

    obj_pose_mean = torch.cat((obj_rot6d_mean, obj_trans_mean), dim=-1)
    return obj_pose_mean



