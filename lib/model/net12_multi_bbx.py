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
from lib.model.encoding import Encoder
from lib.model.denoiser import BaseDenoiser
from lib.model.head_inplane import HeadHeatmap, HeadHeatmap2, HeadHeatmapSegm, HeadSegm
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
from lib.utils.hand_fn import get_joint_aligned_with_HO3D


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)


class DiffHandObj_MultiBBox(nn.Module):
    def __init__(self):
        super(DiffHandObj_MultiBBox, self).__init__()
        self.cfg = cfg
        self.feature_extractor = FPN()

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.score_agent = ScoreBasedModelAgent()
        self.mano_mode = 'mano_pose' # in [mano, mano_pose, mano6d]
        self.denoiser_hand = BaseDenoiser(self.score_agent.marginal_prob_fn, head=self.mano_mode)
        self.denoiser_obj = BaseDenoiser(self.score_agent.marginal_prob_fn, head='obj')

        self.head_hm_hand = HeadHeatmap2(256, 21, 128)
        self.head_hm_obj = HeadHeatmap2(256, 27, 128)

        self.encoder_hand = Encoder(256+21, 256, size_input_feature=(self.cfg.roi_size, self.cfg.roi_size))
        self.encoder_obj = Encoder(256+27, 256, size_input_feature=(self.cfg.roi_size, self.cfg.roi_size))

        self.head_mano = HeadMano(in_dim=1024, is_output_contact=False)
        self.head_obj = HeadObject()

        self.hand_selector = HandSelector(self.head_mano.get_hand_verts)
        self.obj_selector = ObjectSelector(self.head_obj)

        self.denoiser_hand.apply(init_weights)
        self.denoiser_obj.apply(init_weights)
        self.head_hm_hand.apply(init_weights)
        self.head_hm_obj.apply(init_weights)
        self.encoder_hand.apply(init_weights)
        self.encoder_obj.apply(init_weights)
        self.head_mano.apply(init_weights)
        self.head_obj.apply(init_weights)

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
        roi_boxes_hand_rect = torch.cat((idx_tensor, data['bbox_hand_rect']), dim=1)
        roi_boxes_obj_rect = torch.cat((idx_tensor, data['bbox_obj_rect']), dim=1)

        #* tight bbox for heatmap; rectangluar bbox for encoding 
        #! important: DexYCB MJE from 10.87 -> 10.26
        hf_hr = ops.roi_align(hand_feat, roi_boxes_hand, output_size=(self.cfg.roi_size, self.cfg.roi_size), spatial_scale=1/4.) #* checked, hand feature hand roi
        of_or = ops.roi_align(obj_feat, roi_boxes_obj, output_size=(self.cfg.roi_size, self.cfg.roi_size), spatial_scale=1/4.) # obj feature obj roi
        hf_hr_rect = ops.roi_align(hand_feat, roi_boxes_hand_rect, output_size=(self.cfg.roi_size, self.cfg.roi_size), spatial_scale=1/4.) #* checked, hand feature hand roi
        of_or_rect = ops.roi_align(obj_feat, roi_boxes_obj_rect, output_size=(self.cfg.roi_size, self.cfg.roi_size), spatial_scale=1/4.) # obj feature obj roi
        
        # heatmap
        pd_hm_hand = self.head_hm_hand(hf_hr)
        pd_hm_obj = self.head_hm_obj(of_or_rect)

        pd_hm_hand_rect = self.align_hm_to_bbox_rectangle(pd_hm_hand, data['bbox_hand'], data['bbox_hand_rect'])
        pd_hm_obj_rect = self.align_hm_to_bbox_rectangle(pd_hm_obj, data['bbox_obj'], data['bbox_obj_rect'])

        #! flip back to original for object feature
        of_or_rect = flip_tensor_by_mask_index(of_or_rect, is_flip=~data['is_right'])
        pd_hm_obj_rect_ori = flip_tensor_by_mask_index(pd_hm_obj_rect, is_flip=~data['is_right'])

        # squeeze feature to 1D
        if hf_hr.shape != pd_hm_hand.shape:
            pd_hm_hand_rs = F.interpolate(pd_hm_hand_rect, size=hf_hr.shape[-2:], mode='bilinear', align_corners=False)
            pd_hm_obj_ori_rs = F.interpolate(pd_hm_obj_rect_ori, size=of_or.shape[-2:], mode='bilinear', align_corners=False)
        else:
            pd_hm_hand_rs, pd_hm_obj_ori_rs = pd_hm_hand_rect, pd_hm_obj_rect_ori

        encoding_hand, _ = self.encoder_hand(torch.cat((hf_hr_rect, pd_hm_hand_rs), dim=1)) # (bs, 1024)
        encoding_obj, _ = self.encoder_obj(torch.cat((of_or_rect, pd_hm_obj_ori_rs), dim=1)) # (bs, 1024)

        pd_mano_pose, pd_mano_shape = self.head_mano(encoding_hand)
        pd_hand_vert, pd_hand_joint = self.head_mano.get_hand_verts(pose=pd_mano_pose, shape=pd_mano_shape)

        ho3d_mask = data['is_ho3d']
        idx_tensor = torch.arange(bs, device=device)
        idx_tensor = idx_tensor[ho3d_mask]
        pd_hand_joint[idx_tensor] = get_joint_aligned_with_HO3D(pd_hand_vert[idx_tensor], pd_hand_joint[idx_tensor])
        # pd_hand_joint = get_joint_aligned_with_HO3D(pd_hand_vert, pd_hand_joint) if self.cfg.dataset_name == 'ho3d' else pd_hand_joint

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
            loss_dt['hm_hand_loss'] = self.head_hm_hand.get_loss(pd_hm=pd_hm_hand, gt_hm=data['hm_hand'])
            loss_dt['hm_obj_loss'] = self.head_hm_obj.get_loss(pd_hm=pd_hm_obj, gt_hm=data['hm_obj'])

            # mano_loss
            gt_mano_pose, gt_mano_shape = data['gt_mano'][:, :48], data['gt_mano'][:, 48:]
            # gt_hand_vert, gt_hand_joint = self.head_mano.get_hand_verts(pose=gt_mano_pose, shape=gt_mano_shape) #! deprecated, as left hand shape is incompatible with right hand mano, use data['gt_hand_vert_flip'], 'gt_joint': data['gt_hand_jt3d_flip'] instead
            gt_hand_vert, gt_hand_joint = data['gt_hand_vert_flip'], data['gt_hand_jt3d_flip'] # checked, from annotation, no error introduced
            mano_loss_dt = self.head_mano.get_loss(pd_pose=pd_mano_pose, pd_shape=pd_mano_shape, pd_vert=pd_hand_vert, pd_joint=pd_hand_joint,
                                                   gt_pose=gt_mano_pose, gt_shape=gt_mano_shape, gt_vert=gt_hand_vert, gt_joint=gt_hand_joint,
                                                   is_right=data['is_right'])
            loss_dt.update(mano_loss_dt)

            # apply weight to losses
            total_loss = 0
            for k, v in loss_dt.items():
                weighted_loss = v * getattr(self.cfg, f'weight_{k}')
                total_loss = total_loss + weighted_loss
                loss_dt[k] = weighted_loss
            loss_dt['total_loss'] = total_loss

            pd_dt['hand_vert'] = pd_hand_vert
            pd_dt['hand_joint'] = pd_hand_joint
            pd_dt['hand_heatmap'] = pd_hm_hand
            pd_dt['obj_heatmap'] = pd_hm_obj
            return loss_dt, pd_dt
                
        elif mode == 'predict':
            pd_dt = {}
            pd_dt['hand_vert'] = pd_hand_vert
            pd_dt['hand_joint'] = pd_hand_joint
            pd_dt['hand_heatmap'] = pd_hm_hand
            pd_dt['obj_heatmap'] = pd_hm_obj

            # region [diffusion generates hand results]
            _, hand_feat_dim = encoding_hand.shape
            encoding_hand_repeat = encoding_hand[:, None].repeat(1, self.cfg.sample_num, 1).reshape(-1, hand_feat_dim)
            hand_data = {'feat': encoding_hand_repeat}

            hand_inprocess, hand_final = self.score_agent.sample(hand_data, self.denoiser_hand, self.cfg.sample_T0)
            hand_inprocess, hand_final = hand_inprocess.float(), hand_final.float()

            hand_inprocess, hand_final = self.postprocess_diffusion_hand(hand_inprocess, hand_final, pd_mano_shape, sample_num=self.cfg.sample_num)
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
                root_joint=data['root_joint_flip'], 
                cam_intrinsic=data['cam_intr_crop_flip'], 
                heatmap=pd_hm_hand, 
                bbox=data['bbox_hand'], 
                k=self.cfg.topk_hand,
            )
            pd_dt['hand_mean'] = hand_select_data['diff_topk_fused_mano']
            pd_dt['final_hand_vert'], pd_dt['final_hand_joint'] = hand_select_data['diff_vert'], hand_select_data['diff_joint']
            pd_dt['mean_hand_vert'], pd_dt['mean_hand_joint'] = hand_select_data['diff_topk_fused_vert'], hand_select_data['diff_topk_fused_joint']
            
            # pd_dt['final_hand_joint'] = get_joint_aligned_with_HO3D(pd_dt['final_hand_vert'], pd_dt['final_hand_joint']) if self.cfg.dataset_name == 'ho3d' else pd_dt['final_hand_joint']
            # pd_dt['mean_hand_joint'] = get_joint_aligned_with_HO3D(pd_dt['mean_hand_vert'], pd_dt['mean_hand_joint']) if self.cfg.dataset_name == 'ho3d' else pd_dt['mean_hand_joint']
            
            ho3d_mask = data['is_ho3d']
            idx_tensor = torch.arange(bs, device=device)
            idx_tensor = idx_tensor[ho3d_mask]
            pd_dt['final_hand_joint'][idx_tensor] = get_joint_aligned_with_HO3D(pd_dt['final_hand_vert'][idx_tensor], pd_dt['final_hand_joint'][idx_tensor])
            pd_dt['mean_hand_joint'][idx_tensor] = get_joint_aligned_with_HO3D(pd_dt['mean_hand_vert'][idx_tensor], pd_dt['mean_hand_joint'][idx_tensor])
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
                # mode='segm',
                # segm=pd_dt['obj_segm'],
                pose6d=pd_dt['obj_final'], 
                root_joint=data['root_joint'], 
                cam_intrinsic=data['cam_intr_crop_flip'], 
                obj_name=data['obj_name'],
                is_right=data['is_right'],
                heatmap=pd_hm_obj, 
                # heatmap=pd_hm_obj_ori_rs, #! ?????????
                bbox=data['bbox_obj_rect'], 
                k=self.cfg.topk_obj,
            )

            pd_dt['obj_mean'] = obj_select_data['pose6d_fused']
            pd_dt['obj_final'] = obj_select_data['pose6d_topk']
            # endregion
            # endregion
            return pd_dt
        
    def postprocess_diffusion_hand(self, hand_inprocess, hand_final, pd_mano_shape, sample_num):
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
    
    def align_hm_to_bbox_rectangle(self, hm, bbox, bbox_rect):
        device = hm.device
        bs = hm.size(0)
        xx, yy = torch.meshgrid(torch.arange(self.cfg.heatmap_size, device=device), torch.arange(self.cfg.heatmap_size, device=device))
        xx = xx / (self.cfg.heatmap_size-1) * 2 - 1
        yy = yy / (self.cfg.heatmap_size-1) * 2 - 1
        bbox_wh = bbox[:, 2:] - bbox[:, :2]
        bbox_rect_wh = bbox_rect[:, 2:] - bbox_rect[:, :2]
        relative_wh = bbox_rect_wh / bbox_wh 
        xx = xx * relative_wh[:, 0][:, None, None]
        yy = yy * relative_wh[:, 1][:, None, None]
        grid = torch.stack((xx, yy), dim=-1)
        hm_rect = F.grid_sample(hm, grid, mode='bilinear', align_corners=False)
        return hm_rect
    

def flip_tensor_by_mask_index(tensor, is_flip, dim=-1):
    flipped_ls = []
    for i, is_flip_i in enumerate(is_flip):
        if is_flip_i:
            flipped_ls.append(tensor[i].flip(dim))
        else:
            flipped_ls.append(tensor[i])
    flipped = torch.stack(flipped_ls, dim=0)
    return flipped

