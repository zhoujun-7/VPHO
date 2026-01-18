import torch
from torch import nn
from torch.nn import functional as F
from manopth.manolayer import ManoLayer
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d, matrix_to_axis_angle, rotation_6d_to_matrix


class HeadObjectRegress(nn.Module):
    def __init__(self, in_dim=1024, layer_dims=[1024, 512], is_output_contact=False, use_parallel_fc=False):
        super(HeadObjectRegress, self).__init__()
        
        all_dims = [in_dim] + layer_dims
        base_layers = []
        for i, (inp_neurons, out_neurons) in enumerate(
                zip(all_dims[:-1], all_dims[1:])):
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.LeakyReLU(inplace=True))
        self.base_layer = nn.Sequential(*base_layers)

        self.fc_r = nn.Linear(all_dims[-1], 6)
        self.fc_t = nn.Linear(all_dims[-1], 3)

    
    def forward(self, x):
        x = self.base_layer(x)

        pd_rot6d = self.fc_r(x)
        pd_trans = self.fc_t(x)

        pd_obj_pose = torch.cat([pd_rot6d, pd_trans], dim=-1)
        return pd_obj_pose
    

    def get_loss(self, **kwargs):
        """ data: {
            'pd_pose': (bs, 9),
            'pd_vert': (bs, 2048, 3)
            'pd_kpt': (bs, 27, 3)
            
            'gt_pose': (bs, 9),
            'gt_vert': (bs, 2048, 3)
            'gt_kpt': (bs, 27, 3)
        }
        """
        vert_loss = F.mse_loss(kwargs['pd_vert'], kwargs['gt_vert'])
        kpt_loss = F.mse_loss(kwargs['pd_kpt'], kwargs['gt_kpt'])

        rot6d_loss = F.mse_loss(kwargs['pd_pose'][:, :6], kwargs['gt_pose'][:, :6])
        trans_loss = F.mse_loss(kwargs['pd_pose'][:, 6:], kwargs['gt_pose'][:, 6:])

        return {
            "obj_reg_vert_loss": vert_loss,
            "obj_reg_kpt_loss": kpt_loss,
            "obj_reg_rot6d_loss": rot6d_loss,
            "obj_reg_trans_loss": trans_loss
        }