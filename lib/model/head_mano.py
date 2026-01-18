import torch
from torch import nn
from torch.nn import functional as F
from manopth.manolayer import ManoLayer
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d, matrix_to_axis_angle, rotation_6d_to_matrix

from lib.model.parallel_linear import ParallelLinear
from lib.utils.transform_fn import project_pt3d_to_pt2d

def mano_aa_to_6D(mano_params):
    """mano_params: (bs, ..., 16*3+10)"""
    s = mano_params.shape[:-1]
    mano_6d = mano_params[..., :48].reshape(*s, 16, 3)
    mano_6d = matrix_to_rotation_6d(axis_angle_to_matrix(mano_6d))
    mano_6d = mano_6d.reshape(*s, -1)
    mano_6d = torch.cat((mano_6d, mano_params[..., 48:]), dim=-1)
    return mano_6d

def mano_6D_to_aa(mano_6d):
    """ mano_6d: (bs, ..., 16*6+10)"""
    s = mano_6d.shape[:-1]
    mano_pose = mano_6d[..., :16*6].reshape(*s, 16, 6)
    mano_pose = matrix_to_axis_angle(rotation_6d_to_matrix(mano_pose))
    mano_pose = mano_pose.reshape(*s, -1)
    mano_params = torch.cat((mano_pose, mano_6d[..., 16*6:]), dim=-1)
    return mano_params


# modified from 2023_CVPR_HFL
class HeadMano(nn.Module):
    def __init__(self, in_dim=1024, layer_dims=[1024, 512], is_output_contact=False, use_parallel_fc=False):
        super(HeadMano, self).__init__()
        
        all_dims = [in_dim] + layer_dims
        base_layers = []
        for i, (inp_neurons, out_neurons) in enumerate(
                zip(all_dims[:-1], all_dims[1:])):
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.LeakyReLU(inplace=True))
        self.base_layer = nn.Sequential(*base_layers)

        if use_parallel_fc:
            self.fc_pose = ParallelLinear(all_dims[-1], 3, 16*2)
        else:
            self.fc_pose = nn.Linear(all_dims[-1], 16*6)
        self.fc_shape = nn.Linear(all_dims[-1], 10)

        self.mano_layer = ManoLayer(
            ncomps=45, 
            center_idx=0, 
            flat_hand_mean=True,
            side="right", 
            mano_root="asset/mano_v1_2/models", 
            use_pca=False
        )

        self.is_output_contact = is_output_contact
        if self.is_output_contact:
            self.fc_contact = nn.Linear(all_dims[-1], 1080)

    def forward(self, x):
        bs = x.shape[0]
        base_feat = self.base_layer(x)

        pd_pose_rot6d = self.fc_pose(base_feat)
        pd_pose_rot6d = pd_pose_rot6d.reshape(bs, -1, 6)
        pd_pose_rotmat = rotation_6d_to_matrix(pd_pose_rot6d)
        pd_pose_rotaa = matrix_to_axis_angle(pd_pose_rotmat) # (bs, 16, 3)
        pd_pose_rotaa = pd_pose_rotaa.reshape(bs, -1)
        pd_shape = self.fc_shape(base_feat)

        if self.is_output_contact:
            pd_contact = self.fc_contact(base_feat)
            return pd_pose_rotaa, pd_shape, pd_contact
        else:
            return pd_pose_rotaa, pd_shape
    
    def get_hand_verts(self, **kwargs):
        """ data: {
            pose: (bs, 16*3),
            shape: (bs, 10),
        }
        """
        verts, joints = self.mano_layer(th_pose_coeffs=kwargs['pose'], th_betas=kwargs['shape'])
        verts = verts / 1000  # to mm
        joints = joints / 1000
        return verts, joints
    
    def get_loss(self, **kwargs):
        """ data: {
            pd_pose: (bs, 16*3),
            pd_shape: (bs, 10),
            pd_joint: (bs, 21, 3),
            pd_vert: (bs, 778, 3),

            gt_pose: (bs, 16*3),
            gt_shape: (bs, 10),
            gt_joint: (bs, 21, 3),
            gt_vert: (bs, 778, 3),

            is_right: (bs,)
        }
        """
        bs = kwargs["gt_pose"].shape[0]
        vert_loss = F.mse_loss(kwargs["pd_vert"], kwargs["gt_vert"])
        joint_loss = F.mse_loss(kwargs["pd_joint"], kwargs["gt_joint"])

        pd_mano6d = mano_aa_to_6D(kwargs["pd_pose"])
        gt_mano6d = mano_aa_to_6D(kwargs["gt_pose"])
        mano_pose_loss = F.mse_loss(pd_mano6d, gt_mano6d) # TODO: use axis-angle loss or rot6D loss

        #! only use right hand shape, since the shape from left hand is incompatible with the right hand mano layer
        #TODO: use left hand mano for left hand
        if only_right_shape:=True: 
            pd_shape = kwargs["pd_shape"]
            gt_shape = kwargs["gt_shape"]
            pd_shape = pd_shape[kwargs["is_right"]]
            gt_shape = gt_shape[kwargs["is_right"]]
            mano_shape_loss = F.mse_loss(pd_shape, gt_shape)
            
            right_num = gt_shape.shape[0]
            mano_shape_loss = mano_shape_loss / bs * right_num
        else:
            pd_shape = kwargs["pd_shape"]
            gt_shape = kwargs["gt_shape"]
            mano_shape_loss = F.mse_loss(pd_shape, gt_shape)

        return {
            "vert_loss": vert_loss,
            "joint_loss": joint_loss,
            "mano_pose_loss": mano_pose_loss,
            "mano_shape_loss": mano_shape_loss,
        }
    
    def get_contact_loss(self, **kwargs):
        """ data: {
            pd_contact: (bs, 1080),
            gt_contact: (bs, 1080),
        }
        """
        # contact_loss = F.mse_loss(data["pd_contact"], data["gt_contact"])
        contact_loss = F.binary_cross_entropy_with_logits(kwargs["pd_contact"], kwargs["gt_contact"])
        return contact_loss

    # region [dev]
    """ Use 2D joint loss to enchance generalization """
    def get_joint2d_loss(self, **kwargs):
        """ data: {
            pd_joint: (bs, 21, 3),
            gt_joint: (bs, 21, 3),
            root_joint: (bs, 3),
            cam_intr: (bs, 3, 3),
        }
        """
        pd_joint3d = kwargs["pd_joint"] + kwargs["root_joint"].unsqueeze(1)
        gt_joint3d = kwargs["gt_joint"] + kwargs["root_joint"].unsqueeze(1)
        pd_joint2d = project_pt3d_to_pt2d(pd_joint3d, kwargs["cam_intr"])
        gt_joint2d = project_pt3d_to_pt2d(gt_joint3d, kwargs["cam_intr"])
        joint2d_loss = F.mse_loss(pd_joint2d, gt_joint2d)
        return joint2d_loss
    
    
    def get_joint2hm_loss(self, **kwargs):
        """ data: {
            pd_joint: (bs, 21, 3),
            gt_joint: (bs, 21, 3),
            root_joint: (bs, 3),
            cam_intr: (bs, 3, 3),
            bbox: (bs, 4),
            hm_size: 64,
            sigma: 2
        }
        """
        device = kwargs["pd_joint"].device
        pd_joint3d = kwargs["pd_joint"] + kwargs["root_joint"].unsqueeze(1)
        gt_joint3d = kwargs["gt_joint"] + kwargs["root_joint"].unsqueeze(1)
        pd_joint2d = project_pt3d_to_pt2d(pd_joint3d, kwargs["cam_intr"])
        gt_joint2d = project_pt3d_to_pt2d(gt_joint3d, kwargs["cam_intr"])
        
        pd_joint2d_in_hm = (pd_joint2d - kwargs["bbox"][:, :2].unsqueeze(1)) / (kwargs["bbox"][:, 2:].unsqueeze(1) - kwargs["bbox"][:, :2].unsqueeze(1)) * (kwargs["hm_size"] - 1) # (bs, 21, 2)
        gt_joint2d_in_hm = (gt_joint2d - kwargs["bbox"][:, :2].unsqueeze(1)) / (kwargs["bbox"][:, 2:].unsqueeze(1) - kwargs["bbox"][:, :2].unsqueeze(1)) * (kwargs["hm_size"] - 1)

        xx, yy = torch.meshgrid(torch.arange(0, kwargs['hm_size']), torch.arange(0, kwargs['hm_size']))
        xx, yy = xx.to(device), yy.to(device)
        xx, yy = xx[None, None, ...], yy[None, None, ...] # (1, 1, 64, 64)
        pd_hm = torch.exp(-((xx - pd_joint2d_in_hm[:, :, 0:1, None]) ** 2 + (yy - pd_joint2d_in_hm[:, :, 1:2, None]) ** 2) / (2 * kwargs['sigma'] ** 2))
        gt_hm = torch.exp(-((xx - gt_joint2d_in_hm[:, :, 0:1, None]) ** 2 + (yy - gt_joint2d_in_hm[:, :, 1:2, None]) ** 2) / (2 * kwargs['sigma'] ** 2))

        # print(pd_hm.shape)
        # print(gt_hm.shape)
        # from lib.utils.viz_fn import depth_to_rgb
        # pd_viz = pd_hm[0].permute(1, 0, 2).reshape(64, -1).clone().detach().cpu().numpy()
        # gt_viz = gt_hm[0].permute(1, 0, 2).reshape(64, -1).clone().detach().cpu().numpy()
        # pd_viz = depth_to_rgb(pd_viz)
        # gt_viz = depth_to_rgb(gt_viz)
        # import cv2
        # import numpy as np 
        # print(pd_viz.shape)
        # viz = np.concatenate([pd_viz, gt_viz], axis=0)
        # print(viz.shape)
        # cv2.imwrite("pd_gt_hm.png", viz)
        # exit()

        joint2dhm_loss = F.mse_loss(pd_hm, gt_hm)
        return joint2dhm_loss
        

    # endregion [dev]

