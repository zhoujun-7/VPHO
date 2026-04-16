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
from lib.utils.transform_fn import average_quaternion
from lib.model.selection import (
    project_point_by_cam_intrinsic,
    HandSelector,
    ObjectSelector,
)

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

        self.head_hm_hand = HeadHeatmap2(256, 21, 128, 1, 1)
        self.head_hm_obj = HeadHeatmap2(256, 27, 128, 1, 1)

        self.encoder_hand = Encoder(256+21, 256, size_input_feature=(self.cfg.roi_size, self.cfg.roi_size))
        self.encoder_obj = Encoder(256+27, 256, size_input_feature=(self.cfg.roi_size, self.cfg.roi_size))

        self.head_mano = HeadMano(in_dim=1024, is_output_contact=True)  # TODO: use parallel linear
        self.head_obj = HeadObject()

        self.hand_selector = HandSelector(self.head_mano.get_hand_verts)
        self.obj_selector = ObjectSelector(self.head_obj)


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
        # hf_or = ops.roi_align(hand_feat, roi_boxes_obj, output_size=(self.cfg.roi_size, self.cfg.roi_size), spatial_scale=1/4.) # hand feature obj roi
        # of_hr = ops.roi_align(obj_feat, roi_boxes_hand, output_size=(self.cfg.roi_size, self.cfg.roi_size), spatial_scale=1/4.) # obj feature hand roi
        of_or = ops.roi_align(obj_feat, roi_boxes_obj, output_size=(self.cfg.roi_size, self.cfg.roi_size), spatial_scale=1/4.) # obj feature obj roi

        # heatmap
        pd_hm_hand = self.head_hm_hand(hf_hr)
        pd_hm_obj = self.head_hm_obj(of_or)
        
        #! flip back to original for object feature
        hf_or_ls, of_or_ls, pd_hm_obj_ls = [], [], []
        for i, is_right_i in enumerate(data['is_right']):
            if not is_right_i:
                # hf_or_ls.append(hf_or[i].flip(-1))
                of_or_ls.append(of_or[i].flip(-1))
                pd_hm_obj_ls.append(pd_hm_obj[i].flip(-1))
            else:
                # hf_or_ls.append(hf_or[i])
                of_or_ls.append(of_or[i])
                pd_hm_obj_ls.append(pd_hm_obj[i])
        # hf_or = torch.stack(hf_or_ls, dim=0)
        of_or = torch.stack(of_or_ls, dim=0)
        pd_hm_obj = torch.stack(pd_hm_obj_ls, dim=0)

        # squeeze feature to 1D
        if hf_hr.shape != pd_hm_hand.shape:
            pd_hm_hand_rs = F.interpolate(pd_hm_hand, size=hf_hr.shape[-2:], mode='bilinear', align_corners=False)
            pd_hm_obj_rs = F.interpolate(pd_hm_obj, size=of_or.shape[-2:], mode='bilinear', align_corners=False)
        else:
            pd_hm_hand_rs, pd_hm_obj_rs = pd_hm_hand, pd_hm_obj
        encoding_hand = self.encoder_hand(torch.cat((hf_hr, pd_hm_hand_rs), dim=1)) # (bs, 1024)
        encoding_obj = self.encoder_obj(torch.cat((of_or, pd_hm_obj_rs), dim=1)) # (bs, 1024)
        
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
            hand_select_data = self.hand_selector.select_topk_hand(
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
            pd_dt['hand_mean'] = hand_select_data['diff_topk_fused_mano']
            pd_dt['final_hand_vert'] = hand_select_data['diff_vert']
            pd_dt['final_hand_joint'] = hand_select_data['diff_joint']
            pd_dt['mean_hand_vert'], pd_dt['mean_hand_joint'] = hand_select_data['diff_topk_fused_vert'], hand_select_data['diff_topk_fused_joint']
            # endregion
            # endregion

            # region [diffusion generates object results]
            bs, obj_feat_dim = encoding_obj.shape
            encoding_obj_repeat = encoding_obj[:, None].repeat(1, self.cfg.sample_num, 1).reshape(-1, obj_feat_dim)
            diff_obj_data = {'feat': encoding_obj_repeat}
            obj_inprocess, obj_final = self.score_agent.sample(diff_obj_data, self.denoiser_obj, self.cfg.sample_T0)
            pd_dt['obj_inprocess'] = obj_inprocess.reshape(bs, self.cfg.sample_num, -1, 9)
            pd_dt['obj_final'] = obj_final.reshape(bs, self.cfg.sample_num, 9)

            obj_select_data = self.obj_selector.select_topk_object(
                mode='heatmap',
                pose6d=pd_dt['obj_final'], 
                joint_root=data['root_joint'], 
                cam_intrinsic=data['cam_intr_crop_flip'], 
                obj_name=data['obj_name'],
                is_right=data['is_right'],
                heatmap=pd_hm_obj, 
                bbox=data['bbox_obj'], 
                k=self.cfg.topk,
            )
            pd_dt['obj_mean'] = obj_select_data['pose6d_fused']
            pd_dt['obj_final'] = obj_select_data['pose6d_topk']
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

    def _select_and_fuse_topk_obj(self, **kwargs):
        ''' offset inplane translation using heatmap of center point, then select topk obj and fuse using segmentation

            pose6d: (bs, sample_num, 9)
            root_joint: (bs, 3)
            obj_name: str
            heatmap: (bs, 1, H, W)
            bbox: (bs, 4)
            is_right: (bs,)
            k: int

        '''
        pose6d = kwargs['pose6d'].clone()
        pose6d[..., 6:] = pose6d[..., 6:] + kwargs['root_joint']
        pt3d = self.head_obj(pose6d, kwargs['obj_name'], use_bbox3d=True)
        idx_tensor = torch.arange(pt3d.shape[0], device=pt3d.device)
        flipped_idx = idx_tensor[~kwargs['is_right']]
        pt3d[flipped_idx, :, :, 0] = pt3d[flipped_idx, :, :, 0] * -1

        # fix the inplane translation
        offset = improve_obj_inplane_translation_using_heatmap(
            heatmap=kwargs['heatmap'], 
            pt3d=pt3d[:, :, [0], :], 
            bbox=kwargs['bbox'], 
        )
        pose6d = kwargs['pose6d'].clone()
        trans = pose6d[..., 6:] + offset
        trans_fused = trans.mean(dim=-2)
        pose6d[..., 6:] = trans_fused

        topk = self.select_topk_obj_by_segm(
            segm=kwargs['segm'], 
            pose6d=pose6d, 
            root_joint=torch.zeros_like(kwargs['root_joint']), 
            cam_intr_crop_flip=kwargs['cam_intr_crop_flip'], 
            obj_name=kwargs['obj_name'], 
            is_right=kwargs['is_right'], 
            bbox=kwargs['bbox'], 
            k=kwargs['k'],
        )

        idx_tensor = torch.arange(topk.size(0), device=topk.device)[:, None].repeat(1, topk.size(1))
        pose6d_topk = pose6d[idx_tensor, topk]
        rot_topk = pose6d_topk[..., :6]
        rot_fused = average_rot(rot_topk) # TODO: try weighted sum

        pose6d_fused = torch.cat([rot_fused, trans_fused], dim=-1) # (bs, 9)
        return {
            'pose6d_fused': pose6d_fused,
            'topk': topk,
            'pose6d_topk': pose6d_topk,
        }
    
    def select_topk_obj_by_segm(self, **kwargs):
        """ segm: (bs, 1, H, W)
            pose6d: (bs, sample_num, 9)
            root_joint: (bs, 3)
            cam_intr_crop_flip: (bs, 3, 3)
            obj_name: str
            is_right: (bs,)
            bbox: (bs, 4)
            k: int
        """
        bs, J, H, W = kwargs['segm'].shape
        bbox = kwargs['bbox'][:, None, None, :]

        verts2d = self.project_object_point(
            pose6d=kwargs['pose6d'], 
            joint_root=kwargs['root_joint'], 
            cam_intrinsic=kwargs['cam_intr_crop_flip'], 
            obj_name=kwargs['obj_name'],
            is_right=kwargs['is_right'],
            use_bbox3d=False,
        )
        verts2d = verts2d - bbox[..., :2]
        verts2d = verts2d / (bbox[..., 2:] - bbox[..., :2]) * (H - 1) # (bs, sample_num, N, 2)
        in_mask = (verts2d[..., 0] >= 0) & (verts2d[..., 0] < W) & (verts2d[..., 1] >= 0) & (verts2d[..., 1] < H)
        verts2d[~in_mask] = (H-1, W)
        verts2d = verts2d.round().long() # (bs, sample_num, N, 2)

        score_map = kwargs['segm'].repeat(1, kwargs['pose6d'].size(1), 1, 1) # (bs, sample_num, H, W)
        score_map = torch.cat([score_map, torch.zeros_like(score_map[:, :, :, :1])], dim=-1) # (bs, sample_num, H, W+1) add a column for out of bound
        idx_tensor1 = torch.arange(bs, device=score_map.device)[:, None, None].repeat(1, verts2d.size(1), verts2d.size(2))
        idx_tensor2 = torch.arange(verts2d.size(1), device=score_map.device)[None, :, None].repeat(bs, 1, verts2d.size(2))
        score_map[idx_tensor1, idx_tensor2, verts2d[1], verts2d[0]] -= 1
        score_map = score_map[:, :, :, :-1] # (bs, sample_num, H, W)
        score_map = score_map.sum(dim=-1).sum(dim=-1) * -1 # (bs, sample_num)

        val, topk = score_map.topk(kwargs['k'], dim=1)
        return topk
        

# TODO: to be checked
def improve_obj_inplane_translation_using_heatmap(**kwargs):
    """ offset object inplane translation using the 2D center point from heatmap
    
    """
    if kwargs['heatmap'].size(1) == 27:
        ct_hm = kwargs['heatmap'][:, 14]
        ct3d = kwargs['pt3d'][:, :, 14] # (bs, n, 3)
    elif kwargs['heatmap'].size(1) == 1:
        ct_hm = kwargs['heatmap'][:, 0]
        ct3d = kwargs['pt3d'][:, :, 0]
    else:
        raise NotImplementedError
    
    ct_hm = ct_hm.view(ct_hm.size(0), -1)
    idx, maxvals = torch.max(ct_hm, 1)
    ct2d = idx[:, None].repeat(1, 2)
    ct2d[:, 0] = ct2d % kwargs['heatmap'].size(3)
    ct2d[:, 1] = ct2d // kwargs['heatmap'].size(3)
    ct2d = ct2d.float() # (bs, 2)

    bbox = kwargs['bbox'][:, :2]
    ct2d = ct2d / (kwargs['heatmap'].size(3) - 1) * (bbox[:, 2:] - bbox[:, :2]) + kwargs['bbox'][:, :2] # (bs, 2)
    ct_uvd_hm = ct2d[:, None].repeat(1, ct3d.size(1), 1)
    ct_uvd_hm = torch.cat([ct_uvd_hm, ct3d[:, :, 2:]], dim=-1) # (bs, n, 3)
    ct3d_hm = inverse_project_uvd_to_xyz(ct_uvd_hm, kwargs['cam_intr_crop_flip']) # (bs, n, 3)

    weight = torch.clip(maxvals, 0, 1.0)
    weight = weight[:, None]
    ct3d_imporve = weight * ct3d_hm + (1 - weight) * ct3d

    offset = ct3d_imporve - ct3d
    return offset
    


# TODO: to be checked
def inverse_project_uvd_to_xyz(uvd, cam_intr):
    """inverse project uvd to xyz
    Args:
        uvd: [B, ..., 3]
        cam_intr: [B, 3, 3]

    Returns:
        xyz: [B, ..., 3]
    """
    cam_intr_inv = torch.inverse(cam_intr)

    xyz = torch.ones_like(uvd)
    xyz[..., :2] = uvd[..., :2]

    xyz = torch.einsum('b...i,bki->b...k', xyz, cam_intr_inv)
    xyz[..., 2] = uvd[..., 2]
    return xyz