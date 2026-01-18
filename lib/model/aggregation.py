import time
import torch
import torch.nn.functional as F
import copy
from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    quaternion_to_matrix,
    matrix_to_quaternion,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)
from pytorch3d.ops.knn import knn_points

from lib.utils.hand_fn import MANO_PARAMS_LEVEL, MANO_JOINT_LEVEL
from lib.utils.transform_fn import average_quaternion
from lib.model.head_object import HeadObject
from lib.model.head_mano import HeadMano
from lib.configs.args import cfg
from lib.model.physics import from_local_to_global

def project_point_by_cam_intrinsic(**kwargs):
    ''' project hand joint to 2D image plane
        pt3d_cam: (bs, ..., 3)
        cam_intrinsic: (bs, 3, 3)
    '''
    shape = kwargs['pt3d_cam'].shape
    joint2d = torch.einsum('b...ij,blj->b...il', kwargs['pt3d_cam'], kwargs['cam_intrinsic'])
    joint2d = joint2d[..., :2] / joint2d[..., 2:]
    return joint2d

# TODO: to be checked
def inverse_project_point_by_cam_intrinsic(uvd, cam_intr):
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
    xyz = xyz * uvd[..., 2:]
    xyz = torch.einsum('b...i,bki->b...k', xyz, cam_intr_inv)
    return xyz

def average_rot6d(rot6d, weights=None):
    if weights is None:
        weights = torch.ones_like(rot6d[..., 0]) / rot6d.shape[-2]
    quat = matrix_to_quaternion(rotation_6d_to_matrix(rot6d))
    quat_mean = average_quaternion(quat, weights)
    rot6d_mean = matrix_to_rotation_6d(quaternion_to_matrix(quat_mean))
    return rot6d_mean

#* checked
class HandAggregator:
    def __init__(self, mano_fn:HeadMano.get_hand_verts):
        self.mano_fn = mano_fn

    def __call__(self, **kwargs):
        if kwargs['mode'] == 'heatmap':
            return self.select_by_heatmap(**kwargs)
        elif kwargs['mode'] == 'heatmap_cascade':
            return self.select_by_heatmap_cascade(**kwargs)
        elif '2D_pt' in kwargs['mode']: # '2D_pt_pose' or '2D_pt_joint'
            return self.select_by_2D_pt(**kwargs)
        elif kwargs['mode'] == 'average_all':
            return self.average_all(**kwargs)
        elif kwargs['mode'] == 'random':
            return self.random(**kwargs)
        elif kwargs['mode'] == 'heatmap_cascade_n_level':
            return self.select_by_heatmap_cascade_n_level(**kwargs)
        elif kwargs['mode'] == 'physics':
            return self.select_by_physics(**kwargs)
        else:
            raise NotImplementedError
        
    
    def select_by_heatmap(self, **kwargs):
        bs = kwargs['root_joint'].shape[0]
        fused_data = self.select_topk_hand_by_observed_heatmap_and_fuse_by_index(
            pose=kwargs['pose'],
            shape=kwargs['shape'],
            root_joint=kwargs['root_joint'],
            cam_intrinsic=kwargs['cam_intrinsic'],
            heatmap=kwargs['heatmap'],
            bbox=kwargs['bbox'],
            k=kwargs['k'],
            fuse_index=list(range(48)),
            observe_index=list(range(21)),
            is_independent=False,
            is_weight=kwargs['is_weight'],
        )
        fused_pose = fused_data['fused_pose'][:, 0] # (bs, 58)
        shape = kwargs['shape'].reshape(bs, -1, 10)[:, 0]
        fused_mano = torch.cat((fused_pose, shape), dim=-1)
        fused_vert, fused_joint = self.mano_fn(pose=fused_pose, shape=shape)
        fused_vert, fused_joint = fused_vert.reshape(bs, 778, 3), fused_joint.reshape(bs, 21, 3)

        return {
            'topk': fused_data['topk'],
            'diff_topk_vert': fused_data['topk_vert'],
            'diff_topk_joint': fused_data['topk_joint'],
            'agg_hand_mano': fused_mano,
            'agg_vert': fused_vert,
            'agg_joint': fused_joint,
            'fused_data_ls': [fused_data],
            'diff_vert': fused_data['topk_vert'],
            'diff_joint': fused_data['topk_joint'],
        }
        
    def select_by_heatmap_cascade(self, **kwargs):
        bs = kwargs['root_joint'].shape[0]
        pose=kwargs['pose'].clone()
        shape=kwargs['shape'].clone()

        if kwargs['use_regression_as_candidate']:
            extra_pose = torch.zeros_like(pose).reshape(bs, -1, 48)
            extra_pose = extra_pose + kwargs['pose_regression'][:, None].clone()
            pose = pose.reshape(bs, -1, 48)
            num_candidate = pose.shape[1]
            pose = torch.cat((pose, extra_pose), dim=1).reshape(-1, 48) #* useful, MJE: 11.22 -> 11.15, regression result as candidate
            shape = shape.reshape(bs, -1, 10).repeat(1, 2, 1).reshape(-1, 10)
        else:
            num_candidate = pose.shape[1]
            pose = pose.reshape(-1, 48)
            shape = shape.reshape(-1, 10)


        fused_data_ls = []
        for level_i in range(4):
            fuse_idx = MANO_PARAMS_LEVEL[level_i]
            observe_idx = []
            for j in range(level_i+1, 5):  #* useful MJE: 11.48 -> 11.22 
                observe_idx.extend(MANO_JOINT_LEVEL[j])

            if kwargs['use_regression_as_candidate'] and level_i in [0]: # only useful for the wrist
                pose = pose.view(bs, -1, 48)
                pose[:, num_candidate:, fuse_idx] = pose[:, :num_candidate, fuse_idx]
                pose = pose.reshape(-1, 48)

            fused_data_i = self.select_topk_hand_by_observed_heatmap_and_fuse_by_index(
                pose=pose,
                shape=shape,
                root_joint=kwargs['root_joint'],
                cam_intrinsic=kwargs['cam_intrinsic'],
                heatmap=kwargs['heatmap'],
                bbox=kwargs['bbox'],
                k=kwargs['k'],
                fuse_index=fuse_idx,
                observe_index=observe_idx,
                is_independent=False if level_i == 0 else True,
                is_weight=kwargs['is_weight'],
            )

            pose = fused_data_i['fused_pose'].reshape(-1, 48)
            fused_data_ls.append(fused_data_i)

        fused_pose = fused_data_ls[-1]['fused_pose'][:, 0] # (bs, 58)
        shape = kwargs['shape'].reshape(bs, -1, 10)[:, 0]
        fused_mano = torch.cat((fused_pose, shape), dim=-1)
        fused_vert, fused_joint = self.mano_fn(pose=fused_pose, shape=shape)
        fused_vert, fused_joint = fused_vert.reshape(bs, 778, 3), fused_joint.reshape(bs, 21, 3)

        return {
            'topk': fused_data_ls[-1]['topk'],
            'diff_topk_vert': fused_data_ls[-1]['topk_vert'],
            'diff_topk_joint': fused_data_ls[-1]['topk_joint'],
            'agg_hand_mano': fused_mano,
            'agg_vert': fused_vert,
            'agg_joint': fused_joint,
            'diff_vert': fused_data_ls[0]['vert'],
            'diff_joint': fused_data_ls[0]['joint'],
            'middle_data': fused_data_ls,
        }

    def select_topk_hand_by_observed_heatmap_and_fuse_by_index(self, **kwargs):
        """
            pose: (bs*sample_num, 48)
            shape: (bs*sample_num, 10)
            root_joint: (bs, 3)
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            fuse_index: (n, )
            observe_index: (m, )
            is_independent: bool
            is_weight: bool
        """
        bs, J, H, W = kwargs['heatmap'].shape
        K = kwargs['k']
        vert, joint = self.mano_fn(pose=kwargs['pose'], shape=kwargs['shape'])
        vert, joint = vert.reshape(bs, -1, 778, 3), joint.reshape(bs, -1, 21, 3)
        joint_cam = joint + kwargs['root_joint'][:, None, None]
        pose = kwargs['pose'].reshape(bs, -1, 16, 3)

        pt2d = project_point_by_cam_intrinsic(pt3d_cam=joint_cam, cam_intrinsic=kwargs['cam_intrinsic'])
        bbox = kwargs['bbox'][:, None, None, :]
        pt2d = pt2d - bbox[..., :2]
        pt2d = 2 * pt2d / (bbox[..., 2:] - bbox[..., :2]) - 1 # (-1, 1)

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
            val, topk = heat_val.topk(K, dim=1) # (bs, K), (bs, K)
            weight = (val+1e-8) / (val.sum(dim=1, keepdim=True)+1e-8) # (bs, K)

            idx_tensor = torch.arange(bs, device=heat_val.device)[:, None].repeat(1, K)
            topk_pose = kwargs['pose'].reshape(bs, -1, 48)[idx_tensor, topk]
            topk_idx_pose = topk_pose[:, :, kwargs['fuse_index']]
            topk_idx_pose_aa = topk_idx_pose.reshape(bs, K, -1, 3) # (bs, k, n, 3)
            topk_idx_pose_quat = axis_angle_to_quaternion(topk_idx_pose_aa) # (bs, k, n, 4)
            topk_idx_pose_quat = topk_idx_pose_quat.permute(0, 2, 1, 3)

            if kwargs['is_weight']:
                fuse_idx_pose_quat = average_quaternion(topk_idx_pose_quat, weight[:, None]) # (bs, n, 4)
            else:
                fuse_idx_pose_quat = average_quaternion(topk_idx_pose_quat)

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
            weight = (val+1e-8) / (val.sum(dim=1, keepdim=True)+1e-8) # (bs, K, n)
            weight = weight.permute(0, 2, 1) # (bs, n, K)

            pose = kwargs['pose'].reshape(bs, -1, 16, 3) # (bs, sample_num, 48)
            idx_tensor1 = torch.arange(bs, device=heat_val.device)[:, None, None].repeat(1, K, N//3)
            idx_tensor2 = torch.tensor(kwargs['fuse_index'], dtype=torch.long, device=heat_val.device)[None, None].repeat(bs, K, 1).reshape(bs, K, -1, 3)
            idx_tensor2 = idx_tensor2[:, :, :, 0] // 3  # TODO: To be more elegant

            topk_pose = pose[idx_tensor1, topk, idx_tensor2] # (bs, K, M, 3)
            topk_idx_pose_aa = topk_pose
            topk_idx_posed_quat = axis_angle_to_quaternion(topk_pose)
            topk_idx_posed_quat = topk_idx_posed_quat.permute(0, 2, 1, 3) # (bs, M, K, 4)

            if kwargs['is_weight']:
                fused_idx_pose_quat = average_quaternion(topk_idx_posed_quat, weight) # (bs, M, 4)
            else:
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
            'topk_idx_pose_aa': topk_idx_pose_aa,
            'fused_pose': fused_pose,
            'topk_vert': topk_vert,
            'topk_joint': topk_joint,
            'vert': vert,
            'joint': joint,
        }
    
    def select_by_2D_pt(self, **kwargs):
        """
            pose: (bs*sample_num, 48)
            shape: (bs*sample_num, 10)
            root_joint: (bs, 3)
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            fuse_index: (n, )
            observe_index: (m, )
            is_independent: bool
            is_weight: bool
            level: str in ['pose', 'joint']
        """
        bs, J, H, W = kwargs['heatmap'].shape
        K = kwargs['k']
        vert, joint = self.mano_fn(pose=kwargs['pose'], shape=kwargs['shape'])
        vert, joint = vert.reshape(bs, -1, 778, 3), joint.reshape(bs, -1, 21, 3)
        joint_cam = joint + kwargs['root_joint'][:, None, None]
        pose = kwargs['pose'].reshape(bs, -1, 16, 3)

        pt2d_proj = project_point_by_cam_intrinsic(pt3d_cam=joint_cam, cam_intrinsic=kwargs['cam_intrinsic'])
        bbox = kwargs['bbox'][:, None, None, :]
        pt2d_proj = pt2d_proj - bbox[..., :2]
        pt2d_proj = 2 * pt2d_proj / (bbox[..., 2:] - bbox[..., :2]) - 1 # (bs, sample_num, 21, 2)

        X, Y = torch.arange(W, device=pt2d_proj.device), torch.arange(H, device=pt2d_proj.device)
        X, Y = X / (W-1) * 2 - 1, Y / (H-1) * 2 - 1
        XX, YY = torch.meshgrid(X, Y)
        XX, YY = XX[None, None].repeat(bs, J, 1, 1).reshape(bs, J, -1), YY[None, None].repeat(bs, J, 1, 1).reshape(bs, J, -1)
        hm = kwargs['heatmap'].reshape(bs, J, -1)
        ind = torch.argmax(hm, dim=-1)
        ind_1 = torch.arange(bs, device=pt2d_proj.device)[:, None].repeat(1, J)
        ind_2 = torch.arange(J, device=pt2d_proj.device)[None].repeat(bs, 1)
        pt2d_hm_x = XX[ind_1, ind_2, ind]
        pt2d_hm_y = YY[ind_1, ind_2, ind]
        pt2d_hm = torch.stack([pt2d_hm_x, pt2d_hm_y], dim=-1) # (bs, J, 2)
        score = -torch.norm(pt2d_proj - pt2d_hm[:, None], dim=-1) # (bs, sample_num, J)

        if 'pose' in kwargs['mode']:
            score = score.sum(-1)
            val, topk = score.topk(K, dim=1)

            idx_tensor = torch.arange(bs, device=score.device)[:, None].repeat(1, K)
            topk_pose_aa = pose[idx_tensor, topk] # (bs, K, 16, 3)
            topk_idx_pose_quat = axis_angle_to_quaternion(topk_pose_aa) # (bs, k, n, 4)
            topk_idx_pose_quat = topk_idx_pose_quat.permute(0, 2, 1, 3)
            fuse_idx_pose_quat = average_quaternion(topk_idx_pose_quat)
            fused_idx_pose_aa = quaternion_to_axis_angle(fuse_idx_pose_quat) # (bs, n, 3)
            fused_pose = fused_idx_pose_aa.reshape(bs, -1)

            topk_vert = vert[idx_tensor, topk]
            topk_joint = joint[idx_tensor, topk]

            shape = kwargs['shape'].reshape(bs, -1, 10)[:, 0]
            fused_mano = torch.cat((fused_pose, shape), dim=-1)
            fused_vert, fused_joint = self.mano_fn(pose=fused_pose, shape=shape)
            fused_vert, fused_joint = fused_vert.reshape(bs, 778, 3), fused_joint.reshape(bs, 21, 3)

            return {
                'topk': topk,
                'diff_topk_vert': topk_vert,
                'diff_topk_joint': topk_joint,
                'agg_hand_mano': fused_mano,
                'agg_vert': fused_vert,
                'agg_joint': fused_joint,
                'diff_vert': topk_vert,
                'diff_joint': topk_joint,
            }
        elif 'joint' in kwargs['mode']:
            val, topk = score.topk(K, dim=1) # (bs, K, J)
            
            idx_tensor1 = torch.arange(bs, device=score.device)[:, None, None].repeat(1, K, J)
            idx_tensor2 = torch.arange(J, device=score.device)[None, None].repeat(bs, K, 1)
            topk_joint = joint[idx_tensor1, topk, idx_tensor2] # (bs, K, J, 3)
            fused_joint = topk_joint.mean(dim=1)

            topk_vert = torch.zeros(bs, K, 778, 3, device=score.device)
            fused_vert = torch.zeros(bs, 778, 3, device=score.device)
            fused_mano = torch.zeros(bs, 58, device=score.device)

            return {
                'topk': topk,
                'diff_topk_vert': topk_vert,
                'diff_topk_joint': topk_joint,
                'agg_hand_mano': fused_mano,
                'agg_vert': fused_vert,
                'agg_joint': fused_joint,
                'diff_vert': topk_vert,
                'diff_joint': topk_joint,
            }
        
    def average_all(self, **kwargs):
        """
            pose: (bs*sample_num, 48)
            shape: (bs*sample_num, 10)
            root_joint: (bs, 3)
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            fuse_index: (n, )
            observe_index: (m, )
            is_independent: bool
            is_weight: bool
            level: str in ['pose', 'joint']
        """
        bs, J, H, W = kwargs['heatmap'].shape
        K = kwargs['k']
        
        pose_aa = kwargs['pose'].reshape(bs, -1, 16, 3)
        vert, joint = self.mano_fn(pose=kwargs['pose'], shape=kwargs['shape'])
        vert, joint = vert.reshape(bs, -1, 778, 3), joint.reshape(bs, -1, 21, 3)

        pose_quat = axis_angle_to_quaternion(pose_aa) # (bs, k, n, 4)
        pose_quat = pose_quat.permute(0, 2, 1, 3)
        fused_pose_quat = average_quaternion(pose_quat)
        fused_idx_pose_aa = quaternion_to_axis_angle(fused_pose_quat) # (bs, n, 3)
        fused_pose = fused_idx_pose_aa.reshape(bs, -1)

        topk_vert = vert
        topk_joint = joint

        shape = kwargs['shape'].reshape(bs, -1, 10)[:, 0]
        fused_mano = torch.cat((fused_pose, shape), dim=-1)
        fused_vert, fused_joint = self.mano_fn(pose=fused_pose, shape=shape)
        fused_vert, fused_joint = fused_vert.reshape(bs, 778, 3), fused_joint.reshape(bs, 21, 3)

        return {
            'topk': None,
            'diff_topk_vert': topk_vert,
            'diff_topk_joint': topk_joint,
            'agg_hand_mano': fused_mano,
            'agg_vert': fused_vert,
            'agg_joint': fused_joint,
            'diff_vert': topk_vert,
            'diff_joint': topk_joint,
        }

    def random(self, **kwargs):
        """
            pose: (bs*sample_num, 48)
            shape: (bs*sample_num, 10)
            root_joint: (bs, 3)
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            fuse_index: (n, )
            observe_index: (m, )
            is_independent: bool
            is_weight: bool
            level: str in ['pose', 'joint']
        """
        bs, J, H, W = kwargs['heatmap'].shape
        K = kwargs['k']
        
        pose_aa = kwargs['pose'].reshape(bs, -1, 16, 3)
        vert, joint = self.mano_fn(pose=kwargs['pose'], shape=kwargs['shape'])
        vert, joint = vert.reshape(bs, -1, 778, 3), joint.reshape(bs, -1, 21, 3)

        fused_pose = pose_aa[:, 0].reshape(bs, -1)

        topk_vert = vert
        topk_joint = joint

        shape = kwargs['shape'].reshape(bs, -1, 10)[:, 0]
        fused_mano = torch.cat((fused_pose, shape), dim=-1)
        fused_vert, fused_joint = self.mano_fn(pose=fused_pose, shape=shape)
        fused_vert, fused_joint = fused_vert.reshape(bs, 778, 3), fused_joint.reshape(bs, 21, 3)

        return {
            'topk': None,
            'diff_topk_vert': topk_vert,
            'diff_topk_joint': topk_joint,
            'agg_hand_mano': fused_mano,
            'agg_vert': fused_vert,
            'agg_joint': fused_joint,
            'diff_vert': topk_vert,
            'diff_joint': topk_joint,
        }

    def select_by_heatmap_cascade_n_level(self, **kwargs):
        n_level = kwargs.get("n_level", 2)
        bs = kwargs['root_joint'].shape[0]
        pose=kwargs['pose'].clone()
        shape=kwargs['shape'].clone()

        if kwargs['use_regression_as_candidate']:
            extra_pose = torch.zeros_like(pose).reshape(bs, -1, 48)
            extra_pose = extra_pose + kwargs['pose_regression'][:, None].clone()
            pose = pose.reshape(bs, -1, 48)
            num_candidate = pose.shape[1]
            pose = torch.cat((pose, extra_pose), dim=1).reshape(-1, 48) #* useful, MJE: 11.22 -> 11.15, regression result as candidate
            shape = shape.reshape(bs, -1, 10).repeat(1, 2, 1).reshape(-1, 10)
        else:
            num_candidate = pose.shape[1]
            pose = pose.reshape(-1, 48)
            shape = shape.reshape(-1, 10)


        fused_data_ls = []
        for level_i in range(4):
            if level_i >= n_level:
                break
            
            fuse_idx = MANO_PARAMS_LEVEL[level_i]
            observe_idx = []
            for j in range(level_i+1, 5):  #* useful MJE: 11.48 -> 11.22 
                observe_idx.extend(MANO_JOINT_LEVEL[j])

            if kwargs['use_regression_as_candidate'] and level_i in [0]: # only useful for the wrist
                pose = pose.view(bs, -1, 48)
                pose[:, num_candidate:, fuse_idx] = pose[:, :num_candidate, fuse_idx]
                pose = pose.reshape(-1, 48)

            fused_data_i = self.select_topk_hand_by_observed_heatmap_and_fuse_by_index(
                pose=pose,
                shape=shape,
                root_joint=kwargs['root_joint'],
                cam_intrinsic=kwargs['cam_intrinsic'],
                heatmap=kwargs['heatmap'],
                bbox=kwargs['bbox'],
                k=kwargs['k'],
                fuse_index=fuse_idx,
                observe_index=observe_idx,
                is_independent=False if level_i == 0 else True,
                is_weight=kwargs['is_weight'],
            )

            pose = fused_data_i['fused_pose'].reshape(-1, 48)
            fused_data_ls.append(fused_data_i)

        fused_pose = fused_data_ls[-1]['fused_pose'][:, 0] # (bs, 58)
        shape = kwargs['shape'].reshape(bs, -1, 10)[:, 0]
        fused_mano = torch.cat((fused_pose, shape), dim=-1)
        fused_vert, fused_joint = self.mano_fn(pose=fused_pose, shape=shape)
        fused_vert, fused_joint = fused_vert.reshape(bs, 778, 3), fused_joint.reshape(bs, 21, 3)

        return {
            'topk': fused_data_ls[-1]['topk'],
            'diff_topk_vert': fused_data_ls[-1]['topk_vert'],
            'diff_topk_joint': fused_data_ls[-1]['topk_joint'],
            'agg_hand_mano': fused_mano,
            'agg_vert': fused_vert,
            'agg_joint': fused_joint,
            'diff_vert': fused_data_ls[0]['vert'],
            'diff_joint': fused_data_ls[0]['joint'],
        }

    def select_by_physics(self, **kwargs):
        """
            pose: (bs, sample_num, 58)
            root_joint_flip: (bs, 3)
            cam_intrinsic: (bs, 3, 3)
            obj_vert: (bs, )
            obj_com: (bs, )
            is_right: (bs, )
            force_local: (bs, )
            physics_fn: physics_fn
            K: int
        """
        bs = kwargs['pose'].shape[0]
        pose = kwargs['pose'].reshape(-1, 58)
        vert, joint = self.mano_fn(pose=pose[:, :48], shape=pose[:, 48:])
        vert, joint = vert.reshape(bs, -1, 778, 3), joint.reshape(bs, -1, 21, 3)
        vert_cam = vert + kwargs['root_joint_flip'][:, None, None]
        _vert_cam = vert_cam.reshape(-1, 778, 3)

        _force_local = kwargs['force_local'][:, None].repeat(1, vert_cam.shape[1], 1, 1)
        _force_local = _force_local.reshape(-1, 32, 3)
        force_point, force_global = kwargs['physics_fn'].from_local_to_global(
            force_local=_force_local,
            hand_vert=_vert_cam,
        )
        force_point = force_point.reshape(bs, -1, 32, 3)
        force_global = force_global.reshape(bs, -1, 32, 3)
        force_global_norm = force_global.norm(dim=-1) # (bs, -1, 32)
        force_weight = force_global_norm / force_global_norm.sum(dim=-1, keepdim=True) # (bs, -1, 32)

        cdist = cdist_memory_save(force_point, kwargs['obj_vert'])
        score = force_weight * cdist

        force_global = force_global / force_global_norm[:, :, :, None]
        r = nn_for_r_memory_save2(force_global, kwargs['obj_vert'][:, None])

        r = r - kwargs['obj_com'][:, None]
        L = torch.cross(force_global, r, dim=-1)
        L = L.sum(-2)
        L = torch.norm(L, dim=-1)

        I = force_global.sum(-2)
        I = torch.norm(I, dim=-1)

        score = score * I[:, :, None]
        score = -score

        THUMB_FORCE_LEVEL = [1] + [2, 3, 4]
        INDEX_FORCE_LEVEL = [8] + [9, 10, 11]
        MIDDLE_FORCE_LEVEL = [14] + [15, 16, 17]
        RING_FORCE_LEVEL = [21] + [22, 23, 24]
        PINKY_FORCE_LEVEL = [28] + [29, 30, 31]

        fingle_force_level = [THUMB_FORCE_LEVEL, INDEX_FORCE_LEVEL, MIDDLE_FORCE_LEVEL, RING_FORCE_LEVEL, PINKY_FORCE_LEVEL]
        finger_topk_ls = []
        finger_score_ls = []
        finger_val_ls = []
        for finger_i in range(5):
            finger_i_score = score[:, :, fingle_force_level[finger_i]].sum(dim=-1)
            finger_i_val, finger_i_topk = finger_i_score.topk(kwargs['K'], dim=1)
            finger_topk_ls.append(finger_i_topk)
            finger_score_ls.append(finger_i_score)
            finger_val_ls.append(finger_i_val)
        finger_topk_ls = torch.stack(finger_topk_ls, dim=1)
        finger_score_ls = torch.stack(finger_score_ls, dim=1)
        finger_val_ls = torch.stack(finger_val_ls, dim=1)

        idx_tensor = torch.arange(bs, device=score.device)[:, None].repeat(1, kwargs['K'])
        fuse_pose = kwargs['pose'][:, 0].clone()
        for finger_i in range(5):
            fuse_idx = MANO_PARAMS_LEVEL[2][3*finger_i:3*finger_i+3] + MANO_PARAMS_LEVEL[3][3*finger_i:3*finger_i+3]
            finger_i_topk = finger_topk_ls[:, finger_i]
            finger_i_topk_pose = kwargs['pose'][:, :, :48][idx_tensor, finger_i_topk][:, :, fuse_idx]
            finger_i_topk_pose_aa = finger_i_topk_pose.reshape(-1, kwargs['K'], 2, 3)

            finger_i_topk_pose_quat = axis_angle_to_quaternion(finger_i_topk_pose_aa)
            finger_i_topk_pose_quat = finger_i_topk_pose_quat.permute(0, 2, 1, 3)
            finger_i_topk_pose_quat = average_quaternion(finger_i_topk_pose_quat)
            finger_i_topk_pose_aa = quaternion_to_axis_angle(finger_i_topk_pose_quat)
            
            finger_i_topk_pose_aa = finger_i_topk_pose_aa.reshape(-1, 6)
            fuse_pose[:, fuse_idx] = finger_i_topk_pose_aa

        fuse_vert, fuse_joint = self.mano_fn(pose=fuse_pose[:, :48], shape=fuse_pose[:, 48:])

        return {
            'agg_pose': fuse_pose,
            'agg_vert': fuse_vert,
            'agg_joint': fuse_joint,
        }
    
class ObjectAggregator:
    def __init__(self, obj_fn:HeadObject):
        self.obj_layer = obj_fn

    def __call__(self, **kwargs):
        if kwargs['mode'] == 'heatmap':
            return self.select_by_heatmap(**kwargs)
        elif kwargs['mode'] == 'heatmap_cascade':
            return self.select_by_heatmap_cascade(**kwargs)
        elif '2D_pt' in kwargs['mode']: # '2D_pt_pose'
            return self.select_by_2D_pt(**kwargs)
        elif kwargs['mode'] == 'average_all':
            return self.average_all(**kwargs)
        elif kwargs['mode'] == 'random':
            return self.random(**kwargs)
        else:
            raise NotImplementedError
        
    def select_by_heatmap(self, **kwargs):
        topk, weight = self.select_topk_object_by_heatmap(**kwargs)
        pose6d_fused = self.fuse_topk(topk=topk, **kwargs)
        pose6d_fused = pose6d_fused.float()
        pose6d = pose6d_fused.clone()
        pose6d[..., 6:] = pose6d[..., 6:] + kwargs['root_joint']
        obj_vert_fused = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts')
        obj_vert_fused = self.obj_layer.flip_pt3d(obj_vert_fused, kwargs['is_right']) 

        return {
            'agg_6d': pose6d_fused,
            'candidate_6d': kwargs['pose6d'],
            'agg_obj_vert': obj_vert_fused,
        }

    def select_by_heatmap_cascade(self, **kwargs):
        ori_pose6d = kwargs['pose6d'].clone()
        
        topk, weight = self.select_topk_object_by_heatmap(level='trans1', **kwargs)
        if kwargs['is_weight']:
            pose6d_fused = self.fuse_topk(topk=topk, weight=weight, **kwargs) # (bs, 9)
        else:
            pose6d_fused = self.fuse_topk(topk=topk, **kwargs)
        fused_trans1 = pose6d_fused[:, 6:]

        kwargs['pose6d'][..., 6:] = ori_pose6d[..., 6:] * 0 + fused_trans1[:, None]
        topk, weight = self.select_topk_object_by_heatmap(level='rot1', **kwargs)
        if kwargs['is_weight']:
            pose6d_fused = self.fuse_topk(topk=topk, weight=weight, **kwargs) # (bs, 9)
        else:
            pose6d_fused = self.fuse_topk(topk=topk, **kwargs)
        fused_rot1 = pose6d_fused[:, :6]

        kwargs['pose6d'] = ori_pose6d.clone()
        kwargs['pose6d'][..., :6] = ori_pose6d[..., :6] * 0 + fused_rot1[:, None]
        topk_trans2, weight_trans2 = self.select_topk_object_by_heatmap(level='trans2', **kwargs)
        trans2_candidate = self.topk_select(topk_trans2, kwargs['pose6d']) # (bs, k, 9)
        trans2_candidate[:, :, :6] = trans2_candidate[:, :, :6] * 0

        kwargs['pose6d'] = ori_pose6d.clone()
        kwargs['pose6d'][..., 6:] = ori_pose6d[..., 6:] * 0 + fused_trans1[:, None]
        topk_rot2, weight_rot2 = self.select_topk_object_by_heatmap(level='rot2', **kwargs)
        rot2_candidate = self.topk_select(topk_rot2, kwargs['pose6d'])
        rot2_candidate[:, :, 6:] = rot2_candidate[:, :, 6:] * 0

        if kwargs['is_force_selection']: #* useful, HO3Dv2-ADDS: 15.35 -> 14.15
            new_candidate = trans2_candidate[:, None] + rot2_candidate[:, :, None] # (bs, k, k, 9)
            new_candidate = new_candidate.reshape(ori_pose6d.size(0), -1, 9) # (bs, k*k, 9)
            kwargs['pose6d'] = new_candidate

            topk_physics, weight_physics = self.select_topk_object_by_physics3(level='final-physics', **kwargs)    
            topk_heatmap, weight_heatmap = self.select_topk_object_by_heatmap(level='final-heatmap', **kwargs)

            ungrasp_idx = torch.arange(ori_pose6d.size(0), device=kwargs['pose6d'].device)
            ungrasp_idx = ungrasp_idx[~kwargs['is_grasped']]
            new_topk = topk_physics.clone()
            new_topk[ungrasp_idx] = topk_heatmap[ungrasp_idx]
            new_weight = weight_physics.clone()
            new_weight[ungrasp_idx] = weight_heatmap[ungrasp_idx]
            pose6d_fused = self.fuse_topk(topk=new_topk, **kwargs)
        else:
            trans2 = self.fuse_topk(topk=topk_trans2, **kwargs)[:, 6:]
            rot2 = self.fuse_topk(topk=topk_rot2, **kwargs)[:, :6]
            pose6d_fused = torch.cat([rot2, trans2], dim=-1)

        pose6d_fused = pose6d_fused.float()
        pose6d = pose6d_fused.clone()
        pose6d[..., 6:] = pose6d[..., 6:] + kwargs['root_joint']
        obj_vert_fused = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts')
        obj_vert_fused = self.obj_layer.flip_pt3d(obj_vert_fused, kwargs['is_right']) 

        return {
            # 'topk': new_topk,
            'agg_6d': pose6d_fused,
            'pose6d_candidate': kwargs['pose6d'],
            'agg_obj_vert': obj_vert_fused,
        }
    
    def topk_select(self, topk, pose6d):
        bs = pose6d.size(0)
        idx_tensor = torch.arange(bs, device=pose6d.device)[:, None].repeat(1, topk.size(1))
        return pose6d[idx_tensor, topk]
    
    def fuse_topk(self, **kwargs):
        pose6d_topk = self.topk_select(kwargs['topk'], kwargs['pose6d'])
        if kwargs.get('weight', None) is None:
            trans_fused = pose6d_topk[:, :, 6:].mean(dim=1)
        else:
            weight = kwargs['weight']
            trans_fused = (pose6d_topk[:, :, 6:] * weight[:, :, None]).sum(dim=1)

        rot6d_topk = pose6d_topk[..., :6]
        rot6d_fused = average_rot6d(rot6d_topk, weights=kwargs.get('weight', None))
        pose6d_fused = torch.cat([rot6d_fused, trans_fused], dim=-1)
        return pose6d_fused # (bs, 9)

    def select_topk_object_by_heatmap(self, **kwargs):
        """ pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            is_right: (bs, )
            observe_index: (m,)
        """
        pose6d = kwargs['pose6d'].clone().float()
        root_joint = kwargs["root_joint"].unsqueeze(1)
        pose6d[..., 6:] =  pose6d[..., 6:] + root_joint
        obj_pt3d = self.obj_layer(pose6d, kwargs['obj_name'])
        obj_pt3d = self.obj_layer.flip_pt3d(obj_pt3d, kwargs['is_right'])
        pt2d = project_point_by_cam_intrinsic(pt3d_cam=obj_pt3d, cam_intrinsic=kwargs['cam_intrinsic'])

        bbox = kwargs['bbox'][:, None, None, :]
        pt2d = pt2d - bbox[..., :2]
        pt2d = 2 * pt2d / (bbox[..., 2:] - bbox[..., :2]) - 1 # to (-1, 1)

        bs, J, H, W = kwargs['heatmap'].shape
        observe_index = kwargs.get('observe_index', list(range(J)))

        heatval = []
        for i in observe_index:
            grid_i = pt2d[:, :, [i]]
            heatmap_i = kwargs['heatmap'][:, [i]]
            heatval_i = F.grid_sample(heatmap_i, grid_i, align_corners=False, mode='bicubic')
            heatval_i = heatval_i.squeeze(1)
            heatval.append(heatval_i)
        heatval = torch.concat(heatval, dim=-1) # (bs, sample_num, 21)
        heatval = heatval.sum(dim=-1) # (bs, sample_num)

        val, topk = heatval.topk(kwargs['k'], dim=1)
        weight = (val+1e-8) / (val.sum(dim=1, keepdim=True)+1e-8) # (bs, k)

        return topk, weight

    # deprecated, not stronger than the heatmap method
    def select_topk_object_by_segm(self, **kwargs):
        pose6d = kwargs['pose6d'].clone().float()
        pose6d[..., 6:] = pose6d[..., 6:] + kwargs['root_joint'][:, None]

        vert3d = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts')
        vert3d = self.obj_layer.flip_pt3d(vert3d, kwargs['is_right'])
        vert2d = project_point_by_cam_intrinsic(pt3d_cam=vert3d, cam_intrinsic=kwargs['cam_intrinsic'])

        bbox = kwargs['bbox'][:, None, None, :]
        bs, J, H, W = kwargs['segm'].shape
        vert2d = vert2d - bbox[..., :2]
        vert2d = vert2d / (bbox[..., 2:] - bbox[..., :2])
        vert2d[..., 0] = vert2d[..., 0] * (W - 1)
        vert2d[..., 1] = vert2d[..., 1] * (H - 1)

        vert2d = vert2d.round().long() # (bs, sample_num, N, 2)
        in_mask = (vert2d[..., 0] >= 0) & (vert2d[..., 0] <= W-1) & (vert2d[..., 1] >= 0) & (vert2d[..., 1] <= H-1)
        vert2d[~in_mask] = vert2d[~in_mask] *0 + torch.tensor([W, H-1], device=vert2d.device, dtype=vert2d.dtype)

        score_map = kwargs['segm'].repeat(1, kwargs['pose6d'].size(1), 1, 1) # (bs, sample_num, H, W)
        tmp_score = score_map.sum(dim=-1).sum(dim=-1) * -1 # (bs, sample_num)

        score_map = torch.cat([score_map, torch.zeros_like(score_map[:, :, :, :1])], dim=-1) # (bs, sample_num, H, W+1) add a column for out of bound
        
        idx_tensor1 = torch.arange(bs, device=score_map.device)[:, None, None].repeat(1, vert2d.size(1), vert2d.size(2))
        idx_tensor2 = torch.arange(vert2d.size(1), device=score_map.device)[None, :, None].repeat(bs, 1, vert2d.size(2))


        tmp_proj_mask = torch.zeros_like(score_map)

        score_map[idx_tensor1, idx_tensor2, vert2d[..., 1], vert2d[..., 0]] = 0
        score_map = score_map[:, :, :, :-1] # (bs, sample_num, H, W)
        score = score_map.sum(dim=-1).sum(dim=-1) * -1 # (bs, sample_num)

        val, topk = score.topk(kwargs['k'], dim=1)
        return topk


    # deprecated, useless
    def __improve_obj_inplane_translation_using_heatmap(self, **kwargs):
        """ offset object inplane translation using the 2D center point from heatmap
            little improvement for the object translation, deprecated.
        """
        if kwargs['heatmap'].size(1) == 27:
            ct_hm = kwargs['heatmap'][:, 13]
            ct3d = kwargs['pt3d'][:, :, 13] # (bs, n, 3)
        elif kwargs['heatmap'].size(1) == 1:
            ct_hm = kwargs['heatmap'][:, 0]
            ct3d = kwargs['pt3d'][:, :, 0]
        else:
            raise NotImplementedError
        
        ct_hm = ct_hm.view(ct_hm.size(0), -1)
        maxvals, idx = torch.max(ct_hm, 1)
        ct2d = idx[:, None].repeat(1, 2)
        ct2d[:, 0] = ct2d[:, 0] % kwargs['heatmap'].size(3)
        ct2d[:, 1] = ct2d[:, 1] // kwargs['heatmap'].size(2)
        ct2d = ct2d.float() # (bs, 2)
        bbox = kwargs['bbox']
        ct2d = ct2d / (kwargs['heatmap'].size(3) - 1) * (bbox[:, 2:] - bbox[:, :2]) + kwargs['bbox'][:, :2] # (bs, 2)

        ct_uvd_hm = ct2d[:, None].repeat(1, ct3d.size(1), 1)
        ct_uvd_hm = torch.cat([ct_uvd_hm, ct3d[:, :, 2:]], dim=-1) # (bs, n, 3)
        ct3d_hm = inverse_project_point_by_cam_intrinsic(ct_uvd_hm, kwargs['cam_intrinsic']) # (bs, n, 3)

        weight = maxvals
        weight = weight[:, None, None]
        weight = weight
        weight = torch.clip(weight, 0, 1.0)

        ct3d_imporve = weight * ct3d_hm + (1 - weight) * ct3d

        offset = ct3d_imporve - ct3d    
        return offset
    
    def search_along_z_axis(self, search_range=(-0.04, 0.04), **kwargs):
        """ search along the z-axis to find the best object pose
            pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            is_right: (bs, )
            observe_index: (m,)
        """
        pose6d = kwargs['pose6d'].clone().float()
        root_joint = kwargs["root_joint"].unsqueeze(1)
        pose6d[..., 6:] =  pose6d[..., 6:] + root_joint
        obj_pt3d = self.obj_layer(pose6d, kwargs['obj_name'])
        obj_pt3d = self.obj_layer.flip_pt3d(obj_pt3d, kwargs['is_right'])

        obj_ct = obj_pt3d[..., 13, :] # (bs, 1, 3)
        ray = obj_ct / (obj_ct.norm(dim=-1, keepdim=True)+1e-8)

        search_points = torch.arange(search_range[0], search_range[1]+1e-8, step=(search_range[1]-search_range[0])/100, device=pose6d.device)
        search_points = search_points[:, None] * ray

        pose6d_search = kwargs['pose6d'].clone().float()
        pose6d_search = pose6d_search.repeat(1, search_points.size(1), 1)
        pose6d_search[..., 6:] = pose6d_search[..., 6:] + search_points
        kwargs['pose6d'] = pose6d_search
        kwargs['k'] = 1
        topk = self.select_topk_object_by_heatmap(**kwargs)
        return topk, pose6d_search
    
    def select_topk_object_by_physics(self, **kwargs):
        """ Rank the object candidates by the torque

            pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            is_right: (bs, )
            force_point: (bs, 32, 3)
            force_global: (bs, 32, 3)
        """
        pose6d = kwargs['pose6d'].clone().float()
        root_joint = kwargs["root_joint"].unsqueeze(1)
        pose6d[..., 6:] =  pose6d[..., 6:] + root_joint
        obj_CoM = self.obj_layer(pose6d, kwargs['obj_name'], data_name='CoM') # (bs, n_candidate, 1, 3)
        obj_CoM = self.obj_layer.flip_pt3d(obj_CoM, kwargs['is_right'])
        
        force_point = kwargs['force_point'] + root_joint # (bs, 32, 3)
        force_global = kwargs['force_global'] # (bs, 32, 3)

        arm = force_point[:, None] - obj_CoM # (bs, n_candidate, 32, 3)
        torque = torch.cross(arm, force_global[:, None], dim=-1).sum(-2)
        torque = torch.norm(torque, dim=-1) # (bs, n_candidate)
        val, topk = (-torque).topk(kwargs['k'], dim=1)
        return topk

    def select_topk_object_by_physics2(self, **kwargs):
        """ Rank the object candidates by the weighed distance of force point
            weights = |force_global| / sum(|force_global|)

            pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            is_right: (bs, )
            force_point: (bs, 32, 3)
            force_global: (bs, 32, 3)
        """
        pose6d = kwargs['pose6d'].clone().float()
        root_joint = kwargs["root_joint"].unsqueeze(1)
        pose6d[..., 6:] =  pose6d[..., 6:] + root_joint
        obj_verts = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts') # (bs, n_candidate, 2048, 3)
        obj_verts = self.obj_layer.flip_pt3d(obj_verts, kwargs['is_right']) 
        
        force_point = kwargs['force_point'] # (bs, 32, 3)
        force_global = kwargs['force_global'] # (bs, 32, 3)
        force_global_norm = force_global.norm(dim=-1) # (bs, 32)
        force_weight = force_global_norm / force_global_norm.sum(dim=-1, keepdim=True) # (bs, 32)

        force_point = force_point[:, None] # (bs, 1, 32, 3)
        cdist = cdist_memory_save(force_point, obj_verts) # (bs, n_candidate, 32)

        score = -(cdist * force_weight[:, None]).sum(-1)
        val, topk = score.topk(kwargs['k'], dim=1)
        weight = torch.ones_like(val)
        weight = weight / weight.sum(dim=1, keepdim=True)
        return topk, weight


    def select_topk_object_by_physics3(self, **kwargs):
        """ Rank the object candidates by the weighed distance of force point
            weights = |force_global| / sum(|force_global|)

            pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            is_right: (bs, )
            force_point: (bs, 32, 3)
            force_global: (bs, 32, 3)
        """
        pose6d = kwargs['pose6d'].clone().float()
        root_joint = kwargs["root_joint"].unsqueeze(1)
        pose6d[..., 6:] =  pose6d[..., 6:] + root_joint
        obj_verts = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts') # (bs, n_candidate, 2048, 3)
        obj_verts = self.obj_layer.flip_pt3d(obj_verts, kwargs['is_right']) 
        obj_CoM = self.obj_layer(pose6d, kwargs['obj_name'], data_name='CoM') # (bs, n_candidate, 1, 3)
        obj_CoM = self.obj_layer.flip_pt3d(obj_CoM, kwargs['is_right']) # (bs, n_candidate, 1, 3)
        
        force_point = kwargs['force_point'] # (bs, 32, 3)
        force_global = kwargs['force_global'] # (bs, 32, 3)
        force_global_norm = force_global.norm(dim=-1) # (bs, 32)
        force_weight = force_global_norm / force_global_norm.sum(dim=-1, keepdim=True) # (bs, 32)

        force_point = force_point[:, None] # (bs, 1, 32, 3)
        cdist = cdist_memory_save(force_point, obj_verts) # (bs, n_candidate, 32)
        score = (cdist * force_weight[:, None])
        score = score.sum(-1) # (bs, n_candidate)

        force_global = force_global / force_global_norm[:, :, None] # (bs, 32, 3)
        r = nn_for_r_memory_save(force_point, obj_verts) # (bs, 100, 32, 3)
        r = r - obj_CoM # (bs, 100, 32, 3)
        L = torch.cross(force_global[:, None], r, dim=-1) # (bs, 100, 32, 3)
        L = L.sum(-2) # (bs, 100, 3)
        L = torch.norm(L, dim=-1) # (bs, 100)

        score = score * L

        score = -score

        val, topk = score.topk(kwargs['k'], dim=1)

        weight = val / val.sum(dim=1, keepdim=True)
        weight = torch.flip(weight, dims=[-1])

        # weight = weight ** 2
        # weight = weight / weight.sum(dim=1, keepdim=True)

        weight = torch.ones_like(val)
        weight = weight / weight.sum(dim=1, keepdim=True)
        return topk, weight

    

    def select_by_2D_pt(self, **kwargs):
        """ pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            is_right: (bs, )
            observe_index: (m,)
        """
        bs, J, H, W = kwargs['heatmap'].shape
        K = kwargs['k']

        pose6d = kwargs['pose6d'].clone().float()
        root_joint = kwargs["root_joint"].unsqueeze(1)
        pose6d[..., 6:] =  pose6d[..., 6:] + root_joint
        obj_pt3d = self.obj_layer(pose6d, kwargs['obj_name'])
        obj_pt3d = self.obj_layer.flip_pt3d(obj_pt3d, kwargs['is_right'])
        pt2d_proj = project_point_by_cam_intrinsic(pt3d_cam=obj_pt3d, cam_intrinsic=kwargs['cam_intrinsic'])
        bbox = kwargs['bbox'][:, None, None, :]
        pt2d_proj = pt2d_proj - bbox[..., :2]
        pt2d_proj = 2 * pt2d_proj / (bbox[..., 2:] - bbox[..., :2]) - 1 # (bs, sample_num, 21, 2)

        X, Y = torch.arange(W, device=pt2d_proj.device), torch.arange(H, device=pt2d_proj.device)
        X, Y = X / (W-1) * 2 - 1, Y / (H-1) * 2 - 1
        XX, YY = torch.meshgrid(X, Y)
        XX, YY = XX[None, None].repeat(bs, J, 1, 1).reshape(bs, J, -1), YY[None, None].repeat(bs, J, 1, 1).reshape(bs, J, -1)
        hm = kwargs['heatmap'].reshape(bs, J, -1)
        ind = torch.argmax(hm, dim=-1)
        ind_1 = torch.arange(bs, device=pt2d_proj.device)[:, None].repeat(1, J)
        ind_2 = torch.arange(J, device=pt2d_proj.device)[None].repeat(bs, 1)
        pt2d_hm_x = XX[ind_1, ind_2, ind]
        pt2d_hm_y = YY[ind_1, ind_2, ind]
        pt2d_hm = torch.stack([pt2d_hm_x, pt2d_hm_y], dim=-1) # (bs, J, 2)
        score = -torch.norm(pt2d_proj - pt2d_hm[:, None], dim=-1) # (bs, sample_num, J)

        if 'pose' in kwargs['mode']:
            score = score.sum(-1)
            val, topk = score.topk(K, dim=1)
            pose6d_fused = self.fuse_topk(topk=topk, **kwargs) # (bs, 9)
            pose6d_fused = pose6d_fused.float()
            pose6d = pose6d_fused.clone()
            pose6d[..., 6:] = pose6d[..., 6:] + kwargs['root_joint']
            obj_vert_fused = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts')
            obj_vert_fused = self.obj_layer.flip_pt3d(obj_vert_fused, kwargs['is_right']) 

            return {
                'agg_6d': pose6d_fused,
                'candidate_6d': kwargs['pose6d'],
                'agg_obj_vert': obj_vert_fused,
            }
        
    def average_all(self, **kwargs):
        """ pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            is_right: (bs, )
            observe_index: (m,)
        """
        bs, J, H, W = kwargs['heatmap'].shape
        K = kwargs['k']

        pose6d = kwargs['pose6d'].clone().float()
        topk = torch.arange(K, device=pose6d.device)[None].repeat(bs, 1)

        pose6d_fused = self.fuse_topk(topk=topk, **kwargs) # (bs, 9)
        pose6d_fused = pose6d_fused.float()
        pose6d = pose6d_fused.clone()
        pose6d[..., 6:] = pose6d[..., 6:] + kwargs['root_joint']
        obj_vert_fused = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts')
        obj_vert_fused = self.obj_layer.flip_pt3d(obj_vert_fused, kwargs['is_right']) 

        return {
            'agg_6d': pose6d_fused,
            'candidate_6d': kwargs['pose6d'],
            'agg_obj_vert': obj_vert_fused,
        }
        
    def random(self, **kwargs):
        """ pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            is_right: (bs, )
            observe_index: (m,)
        """
        bs, J, H, W = kwargs['heatmap'].shape
        K = kwargs['k']

        pose6d = kwargs['pose6d'].clone().float()
        topk = torch.zeros(bs, 1, device=pose6d.device, dtype=torch.long)

        pose6d_fused = self.fuse_topk(topk=topk, **kwargs) # (bs, 9)
        pose6d_fused = pose6d_fused.float()
        pose6d = pose6d_fused.clone()
        pose6d[..., 6:] = pose6d[..., 6:] + kwargs['root_joint']
        obj_vert_fused = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts')
        obj_vert_fused = self.obj_layer.flip_pt3d(obj_vert_fused, kwargs['is_right']) 

        return {
            'agg_6d': pose6d_fused,
            'candidate_6d': kwargs['pose6d'],
            'agg_obj_vert': obj_vert_fused,
        }


def cdist_memory_save(x, y):
    """ x: (bs, n1, v, d)
        y: (bs, n2, v, d)
    """
    cdist_ls = []
    for i in range(x.size(0)):
        cdist_i = torch.cdist(x[i], y[i], p=2)
        cdist_i = cdist_i.min(dim=-1)[0]
        cdist_ls.append(cdist_i)
    cdist = torch.stack(cdist_ls, dim=0)
    return cdist


def nn_for_r_memory_save(x, y):
    """ x: (bs, n1, v1, d)
        y: (bs, n2, v2, d)
    return: (bs, n1, n2, v1, d)
    """
    collector = []
    for i in range(x.size(0)):
        cdist_i = torch.cdist(x[i], y[i], p=2)
        cdist_i = cdist_i.min(dim=-1)[1]
        ind_tensor = torch.arange(cdist_i.shape[0])[:, None].repeat(1, cdist_i.shape[1])
        selected_y = y[i][ind_tensor, cdist_i]
        r = x[i] - selected_y
        collector.append(r)
    collector = torch.stack(collector, dim=0)
    return collector


def nn_for_r_memory_save2(x, y):
    """ x: (bs, n1, v1, d)
        y: (bs, n2, v2, d)
    return: (bs, n1, n2, v1, d)
    """
    collector = []
    for i in range(x.size(0)):
        cdist_i = torch.cdist(x[i], y[i], p=2)
        cdist_i = cdist_i.min(dim=-1)[1]
        selected_y = y[i][0][cdist_i]
        r = x[i] - selected_y
        collector.append(r)
    collector = torch.stack(collector, dim=0)
    return collector

class HOI_Aggregator:
    def __init__(self, mano_fn:HeadMano.get_hand_verts, obj_fn:HeadObject, physics_fn):
        self.mano_fn = mano_fn
        self.hand_aggregator = HandAggregator(mano_fn)
        self.obj_aggregator = ObjectAggregator(obj_fn)
        self.physics_fn = physics_fn

    def __call__(self, **kwargs):
        """
        1. aggregate hand by heatmap
        2. aggregate object translation by heatmap
        3. select topk object rotation by heatmap
        4. select topk object translation-z & rotation by physics
        5. aggregate hand 3,4 levels by physics
        """

        t_hv_1 = time.time()
        #* 1. aggregate hand by heatmap
        hand_select_data = self.hand_aggregator(
            mode='heatmap_cascade',
            pose=kwargs['hand_pose_diff'],
            pose_regression=kwargs['hand_pose_regression'],
            shape=kwargs['hand_shape'],
            root_joint=kwargs['root_joint_flip'], 
            cam_intrinsic=kwargs['cam_intrinsic'], 
            heatmap=kwargs['hand_heatmap'],  
            bbox=kwargs['hand_bbox'], 
            k=kwargs['hand_topk'],
            is_weight=True,
            use_regression_as_candidate=True,
        )
        t_hv_2 = time.time()

        agg_hand_mano = hand_select_data['agg_hand_mano']
        agg_hand_vert, agg_hand_joint = hand_select_data['agg_vert'], hand_select_data['agg_joint']
        hand_vert = agg_hand_vert + kwargs['root_joint_flip'][:, None]
        force_point, force_global = self.physics_fn.from_local_to_global(force_local=kwargs['force_local'], hand_vert=hand_vert)

        t_ov_1 = time.time()
        #* 2. aggregate object translation by heatmap
        obj_transl_topk, obj_transl_weight = self.obj_aggregator.select_topk_object_by_heatmap(
            pose6d=kwargs['obj_pose6d'],
            root_joint=kwargs['root_joint'], 
            cam_intrinsic=kwargs['cam_intrinsic'], 
            obj_name=kwargs['obj_name'],
            is_right=kwargs['is_right'],
            heatmap=kwargs['obj_heatmap'],  
            bbox=kwargs['obj_bbox'], 
            k=kwargs['obj_topk'],
            # observe_index=[14],
        )
        pose6d_fused = self.obj_aggregator.fuse_topk(
            topk=obj_transl_topk, 
            weight=obj_transl_weight, 
            pose6d=kwargs['obj_pose6d'],
        )
        obj_transl_fused = pose6d_fused[:, 6:]

        #* 3. select topk object rotation by heatmap
        updated_pose6d = kwargs['obj_pose6d'].clone()
        updated_pose6d[..., 6:] = obj_transl_fused[:, None]

        obj_rot_z_topk, obj_rot_z_weight = self.obj_aggregator.select_topk_object_by_heatmap(
            pose6d=updated_pose6d,
            root_joint=kwargs['root_joint'], 
            cam_intrinsic=kwargs['cam_intrinsic'], 
            obj_name=kwargs['obj_name'],
            is_right=kwargs['is_right'],
            heatmap=kwargs['obj_heatmap'],  
            bbox=kwargs['obj_bbox'], 
            k=kwargs['obj_topk'],
        )
        t_ov_2 = time.time()
        
        #* 4. select topk object translation-z & rotation by physics
        idx_tensor = torch.arange(kwargs['obj_pose6d'].size(0), device=kwargs['obj_pose6d'].device)
        idx_tensor = idx_tensor[:, None].repeat(1, kwargs['obj_topk'])
        candidate_transl = kwargs['obj_pose6d'][idx_tensor, obj_transl_topk][:, :, 6:]
        candidate_rot = kwargs['obj_pose6d'][idx_tensor, obj_rot_z_topk][:, :, :6]
        candidate_transl = candidate_transl[:, :, None].repeat(1, 1, kwargs['obj_topk'], 1)
        candidate_rot = candidate_rot[:, None, :].repeat(1, kwargs['obj_topk'], 1, 1)
        candidate_pose6d = torch.cat([candidate_rot, candidate_transl], dim=-1)
        candidate_pose6d = candidate_pose6d.reshape(kwargs['obj_pose6d'].size(0), -1, 9)
        # candidate_pose6d[:, :, 6:8] = obj_transl_fused[:, None, :2]

        t_op_1 = time.time()
        phy_topk = 5
        topk_physics, weight_physics = self.obj_aggregator.select_topk_object_by_physics3(
            level='final-physics',
            pose6d=candidate_pose6d,
            root_joint=kwargs['root_joint'],
            obj_name=kwargs['obj_name'],
            is_right=kwargs['is_right'],
            force_point=force_point,
            force_global=force_global,
            k=phy_topk,
        )    
        topk_heatmap, weight_heatmap = self.obj_aggregator.select_topk_object_by_heatmap(
            level='final-heatmap',
            pose6d=candidate_pose6d,
            root_joint=kwargs['root_joint'],
            obj_name=kwargs['obj_name'],
            cam_intrinsic=kwargs['cam_intrinsic'],
            heatmap=kwargs['obj_heatmap'],
            bbox=kwargs['obj_bbox'], 
            k=phy_topk,
            is_right=kwargs['is_right'],
        )
        t_op_2 = time.time()

        ungrasp_idx = torch.arange(kwargs['obj_pose6d'].size(0), device=kwargs['obj_pose6d'].device)
        ungrasp_idx = ungrasp_idx[~kwargs['is_grasped']]
        new_topk = topk_physics.clone()
        new_topk[ungrasp_idx] = topk_heatmap[ungrasp_idx]
        new_weight = weight_physics.clone()
        new_weight[ungrasp_idx] = weight_heatmap[ungrasp_idx]
        pose6d_fused = self.obj_aggregator.fuse_topk(
            topk=new_topk, 
            weight=new_weight, 
            pose6d=candidate_pose6d,
        )

        pose6d = pose6d_fused.clone().float()
        pose6d[..., 6:] = pose6d[..., 6:] + kwargs['root_joint']
        obj_vert_fused = self.obj_aggregator.obj_layer(pose6d, kwargs['obj_name'], data_name='verts')
        obj_vert_fused = self.obj_aggregator.obj_layer.flip_pt3d(obj_vert_fused, kwargs['is_right']) 
        obj_com_fused = self.obj_aggregator.obj_layer(pose6d, kwargs['obj_name'], data_name='CoM')
        obj_com_fused = self.obj_aggregator.obj_layer.flip_pt3d(obj_com_fused, kwargs['is_right'])


        #* 5. aggregate hand 3,4 levels by physics
        # level_3_topk_pose = hand_select_data['middle_data'][2]['topk_idx_pose_aa'][:, :10]
        # level_3_topk_pose = torch.cat([level_3_topk_pose, agg_hand_mano[:, MANO_PARAMS_LEVEL[2]].reshape(-1, 1, 5, 3)], dim=1)
        # level_4_topk_pose = hand_select_data['middle_data'][3]['topk_idx_pose_aa'][:, :10]
        # level_4_topk_pose = torch.cat([level_4_topk_pose, agg_hand_mano[:, MANO_PARAMS_LEVEL[3]].reshape(-1, 1, 5, 3)], dim=1)

        # new_candidate_param = torch.cat([
        #     level_3_topk_pose[:, :, None].repeat(1, 1, 11, 1, 1), # (B, 10, 10, 5, 3)
        #     level_4_topk_pose[:, None].repeat(1, 11, 1, 1, 1), # (B, 10, 10, 5, 3)
        # ], dim=-2) # (B, 10, 10, 10, 3)
        # new_candidate_param = new_candidate_param.reshape(new_candidate_param.shape[0], 121, -1, 3)
        # new_candidate_pose = agg_hand_mano.clone()[:, None, :48].repeat(1, 121, 1)
        # new_candidate_pose[:, :, MANO_PARAMS_LEVEL[2]] = new_candidate_param[:, :, :5].reshape(-1, 121, 15)
        # new_candidate_pose[:, :, MANO_PARAMS_LEVEL[3]] = new_candidate_param[:, :, 5:].reshape(-1, 121, 15)


        level_3_topk_pose = hand_select_data['middle_data'][2]['topk_idx_pose_aa'][:, :10]
        level_3_topk_pose = torch.cat([agg_hand_mano[:, MANO_PARAMS_LEVEL[2]].reshape(-1, 1, 5, 3)], dim=1)
        
        
        level_4_topk_pose = hand_select_data['middle_data'][3]['topk_idx_pose_aa'][:, :kwargs['hand_topk']]
        level_4_topk_pose = torch.cat([level_4_topk_pose, agg_hand_mano[:, MANO_PARAMS_LEVEL[3]].reshape(-1, 1, 5, 3)], dim=1)

        num_candidate = kwargs['hand_topk'] + 1
        new_candidate_param = torch.cat([
            level_3_topk_pose[:, :, None].repeat(1, 1, num_candidate, 1, 1), # (B, 10, 10, 5, 3)
            level_4_topk_pose[:, None].repeat(1, 1, 1, 1, 1), # (B, 10, 10, 5, 3)
        ], dim=-2) # (B, 10, 10, 10, 3)
        new_candidate_param = new_candidate_param.reshape(new_candidate_param.shape[0], num_candidate, -1, 3)
        new_candidate_pose = agg_hand_mano.clone()[:, None, :48].repeat(1, num_candidate, 1)
        new_candidate_pose[:, :, MANO_PARAMS_LEVEL[2]] = new_candidate_param[:, :, :5].reshape(-1, num_candidate, 15)
        new_candidate_pose[:, :, MANO_PARAMS_LEVEL[3]] = new_candidate_param[:, :, 5:].reshape(-1, num_candidate, 15)

        shape = agg_hand_mano[:, None, 48:].repeat(1, num_candidate, 1)   
        new_candidate_pose = torch.cat([new_candidate_pose, shape], dim=-1)

        t_hp_1 = time.time()
        hand_select_data = self.hand_aggregator(
            mode='physics',
            pose=new_candidate_pose,
            root_joint_flip=kwargs['root_joint_flip'],
            obj_vert=obj_vert_fused,
            obj_com=obj_com_fused,
            K=phy_topk,
            physics_fn=self.physics_fn,
            force_local=kwargs['force_local'],
            is_right=kwargs['is_right'],
        )
        t_hp_2 = time.time()

        data = {
            # 'topk': new_topk,
            'obj_agg_6d': pose6d_fused,
            'pose6d_candidate': candidate_pose6d,
            'agg_obj_vert': obj_vert_fused,

            'hand_agg_mano': hand_select_data['agg_pose'],
            'hand_agg_vert': hand_select_data['agg_vert'],
            'hand_agg_joint': hand_select_data['agg_joint'],
        }

        t6 = time.time()

        return data
    