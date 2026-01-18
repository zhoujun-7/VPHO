import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

from lib.utils.transform_fn import obj_9D_to_mat
from lib.model.encoding import Residual
from lib.utils.hand_fn import FINGER_JOINT_IDX
from lib.utils.physics_fn import VERT2ANCHOR


def norm_vec(vec):
    return vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)

#* from nerf
class PosEmbedder:
    def __init__(self, input_dims=3, multires=10):
        self.kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
#* from transformer
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ForceEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_force, nRegBlock=3, nRegModules=2):
        super(ForceEncoder, self).__init__()
        self.project = nn.Conv2d(in_dim, hid_dim, bias=True, kernel_size=1, stride=1)

        self.nRegBlock = nRegBlock
        self.nRegModules = nRegModules
        reg = []
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                reg.append(Residual(hid_dim, hid_dim))
        self.reg = nn.ModuleList(reg)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        channel = out_dim // 4**2 * num_force
        self.final_conv = nn.Conv2d(hid_dim, channel, kernel_size=1, stride=1)
        self.num_force = num_force

    def forward(self, x):
        """ x: (B, in_dim, 32, 32)
            out: (B, num_feat_chan * 2 * 2)
        """
        # x: (B, in_dim, 32, 32)
        x = self.project(x)
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                x = self.reg[i * self.nRegModules + j](x)
            x = self.maxpool(x)
        # x: (B, hid_dim, 4, 4)
        out = self.final_conv(x)
        out = out.reshape(out.size(0), self.num_force, -1) # (bs, J, out_dim)
        return out
    

class HeadForce(nn.Module):
    def __init__(self, in_dim):
        super(HeadForce, self).__init__()
        self.num_force = 32
        self.hand_encoder = ForceEncoder(in_dim=in_dim, hid_dim=256, out_dim=128, num_force=self.num_force)

        self.gravity_embedder = PosEmbedder(input_dims=3, multires=10)
        self.gravity_proj = nn.Linear(self.gravity_embedder.out_dim, 128)

        self.pose_embedder = PositionalEncoding(128)

        self.hand_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=2),
            num_layers=4,
        )

        self.fc_scale = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 8),
            nn.Softmax(dim=-1),
        )

        num_anchor = 8
        anchor = torch.arange(0, 2*torch.pi, 2*torch.pi/num_anchor)[:num_anchor]
        anchor_x = torch.cos(anchor)
        anchor_y = torch.sin(anchor)
        anchor_z = torch.ones_like(anchor_x)
        anchor = torch.stack([anchor_x, anchor_y, anchor_z], dim=-1) / num_anchor
        self.register_buffer('anchor', anchor)

    def forward(self, **kwargs):
        """ mode: train or test
            feat: (B, in_dim, 32, 32)
            gravity: (B, 1, 3)
            hand_vert: (B, 778, 3)
        """
        
        enc_hand = self.hand_encoder(kwargs['feat']) # (bs, J, 128)
        enc_gravity = self.gravity_embedder.embed(kwargs['gravity'])
        enc_gravity = self.gravity_proj(enc_gravity)

        token = torch.cat([enc_hand, enc_gravity], dim=-2)
        token = self.pose_embedder(token)
        
        enc_force = self.hand_transformer(token)
        enc_force = enc_force[..., :self.num_force, :]

        scale = self.fc_scale(enc_force)
        scale = scale.squeeze(-1)
        weight = self.fc_weight(enc_force)

        force_local = self.get_local_force(scale, weight)
        return force_local, scale
        
    def from_local_to_global(self, force_local, hand_vert):
        """ force_local: (bs, J, 2)
            hand_pose: (bs, 778, 3)
        """
        return from_local_to_global(force_local, hand_vert)

    def get_loss(self, **kwargs):
        """ mode: supervised or unsupervised
            pd_force_global: (bs, J, 2)
            force_point: (bs, 32, 6) # not flipped
            gt_gravity: (bs, 3)
            obj_CoM: (bs, 1, 3)
            is_grasped: (bs,)
        """
        loss_force = self.get_force_loss(
            pd_force_global=kwargs['pd_force_global'], 
            gt_gravity=kwargs['gt_gravity'],
            is_grasped=kwargs['is_grasped'],
        )
        loss_gravity = self.get_gravity_loss(
            pd_force_global=kwargs['pd_force_global'], 
            gt_gravity=kwargs['gt_gravity'],
            is_grasped=kwargs['is_grasped'],
        )
        loss_torque = self.get_torque_loss(
            pd_force_global=kwargs['pd_force_global'], 
            force_point=kwargs['force_point'], 
            obj_CoM=kwargs['obj_CoM'],
            is_grasped=kwargs['is_grasped'],
        )
        # loss_distribution = self.get_distribution_loss(
        #     scale=kwargs['scale'],
        #     force_contact=kwargs['force_contact'],
        #     is_grasped=kwargs['is_grasped'],
        # )
        loss_supervised = self.get_supervised_loss(
            pd_force_local=kwargs['pd_force_local'],
            gt_force_local=kwargs['gt_force_local'],
        )
        return {
            'force_loss': loss_force,
            'gravity_loss': loss_gravity,
            'torque_loss': loss_torque,
            # 'distrib_loss': loss_distribution,
            # 'supervised_loss': loss_supervised,
        }
    
    def get_physics_metric(self, **kwargs):
        """ mode: supervised or unsupervised
            pd_force_global: (bs, J, 2)
            force_point: (bs, 32, 6) # not flipped
            gt_gravity: (bs, 3)
            obj_CoM: (bs, 1, 3)
        """
        force_blance = self.get_metric_force(
            pd_force_global=kwargs['pd_force_global'], 
            gt_gravity=kwargs['gt_gravity'],
            is_grasped=kwargs['is_grasped'],
        )
        # force_blance = force_blance.mean()
        
        gravity_blance = self.get_metric_gravity(
            pd_force_global=kwargs['pd_force_global'], 
            gt_gravity=kwargs['gt_gravity'],
            is_grasped=kwargs['is_grasped'],
        )
        # gravity_blance = gravity_blance.mean()

        torque_blance = self.get_metric_moment(
            pd_force_global=kwargs['pd_force_global'], 
            force_point=kwargs['force_point'], 
            obj_CoM=kwargs['obj_CoM'],
            is_grasped=kwargs['is_grasped'],
        )
        # torque_blance = torque_blance.mean()
        return {
            'force_blance': force_blance,
            'gravity_blance': gravity_blance,
            'torque_blance': torque_blance,
        }

    
    def get_local_force(self, scale, weight, friction_coeff=0.8):
        scale = torch.abs(scale)
        weight = torch.softmax(weight, dim=-1)
        anchor = self.anchor.clone()
        anchor[:, :2] *= friction_coeff
        anchor = anchor[None, :].repeat_interleave(scale.size(-1), dim=0)
        
        force_local = torch.einsum('...ij,ijk->...ik', weight, anchor) 
        force_direction = torch.einsum('...ij,ijk->...ik', weight, anchor)
        force_direction = norm_vec(force_direction)
        force_local = force_direction * scale[..., None]
        return force_local
    
    def get_metric_force(self, **kwargs):
        resultant_force = kwargs['pd_force_global'].sum(1, keepdim=True) + kwargs['gt_gravity'] # (bs, 1, 3)
        force_blance = torch.norm(resultant_force, dim=-1).squeeze(-1)
        force_blance = force_blance * kwargs['is_grasped']
        return force_blance

    def get_force_loss(self, **kwargs):
        """ pd_force_global: (bs, 32, 3)
            gt_gravity: (bs, 1, 3)
            is_grasped: (bs,)
        """
        force_blance = self.get_metric_force(**kwargs)
        force_loss = (force_blance**2).mean()
        return force_loss
    
    def get_metric_gravity(self, **kwargs):
        resultant_force = kwargs['pd_force_global'].sum(1, keepdim=True)
        cos_proj = torch.einsum('...i,...i->...', resultant_force, kwargs['gt_gravity'])
        gravity_blance = cos_proj.squeeze(-1)
        gravity_blance = gravity_blance * kwargs['is_grasped']
        return gravity_blance
    
    def get_gravity_loss(self, **kwargs):
        """ pd_force_global: (bs, 3)
            gt_gravity: (bs, 3)
            is_grasped: (bs,)
        """
        force_proj = self.get_metric_gravity(**kwargs)
        gt = torch.ones_like(force_proj) * kwargs['is_grasped'] * -1
        gravity_loss = F.mse_loss(force_proj, gt)
        return gravity_loss

    def get_metric_moment(self, **kwargs):
        arm = kwargs['force_point'] - kwargs['obj_CoM']
        torque_blance = torch.cross(arm, kwargs['pd_force_global'], dim=-1)
        torque_blance = torque_blance.sum(1)
        torque_blance = torch.norm(torque_blance, dim=-1)
        torque_blance = torque_blance * kwargs['is_grasped']
        return torque_blance
    
    def get_torque_loss(self, **kwargs):
        """ pd_force_global: (bs, 32, 3)
            force_point: (bs, 32, 3)
            obj_CoM: (bs, 1, 3)
            is_grasped: (bs,)
        """
        torque = self.get_metric_moment(**kwargs)
        torque_loss = (torque**2).mean()
        return torque_loss

    # deprecated, mice dropping
    def get_distribution_loss(self, **kwargs):
        """ scale: (bs, 32)
            force_contact: (bs, 32)
            is_grasped: (bs,)
        """
        force_mask = kwargs['force_contact'] > 0.05
        scale_norm = kwargs['scale'] / (kwargs['scale'].norm(dim=-1, keepdim=True).detach() + 1e-8)
        force_contact_norm = kwargs['force_contact'] / (kwargs['force_contact'].norm(dim=-1, keepdim=True).detach() + 1e-8)
        dist = torch.log(torch.abs(scale_norm / (force_contact_norm+1e-8))) * force_mask
        dist = (dist ** 2).mean(-1)
        dist = dist * kwargs['is_grasped']
        dist_loss = dist.mean()
        return dist_loss
    
    def get_supervised_loss(self, **kwargs):
        """ pd_force_local: (bs, J, 3)
            gt_force_local: (bs, J, 3)
        """
        force_loss = F.mse_loss(kwargs['pd_force_local'], kwargs['gt_force_local'])
        return force_loss

def from_local_to_global(force_local, vert):
    """ force_local: (bs, J, 3)
        hand_pose: (bs, 778, 3)
    """
    force_point, force_frame = VERT2ANCHOR(vert)
    if isinstance(vert, torch.Tensor):
        force_global = torch.einsum('...bi,...bji->...bj', force_local, force_frame)
    elif isinstance(vert, np.ndarray):
        force_global = np.einsum('...bi,...bji->...bj', force_local, force_frame)
    return force_point, force_global


def from_global_to_local(force_global, vert):
    # force_point, force_frame = get_force_point_and_direction_from_mano_vert(vert)
    force_point, force_frame = VERT2ANCHOR(vert)
    force_local = torch.einsum('...bi,...bji->...bj', force_global, force_frame.transpose(-2, -1))
    return force_local


class HeadForce2(nn.Module):
    def __init__(self):
        super(HeadForce2, self).__init__()
        self.num_force = 32
        self.hid_dim = 512

        self.hand_proj = Residual(256, 256)
        self.obj_proj = Residual(256, self.hid_dim)

        self.gravity_embedder = PosEmbedder(input_dims=3, multires=10)
        self.gravity_proj = nn.Linear(self.gravity_embedder.out_dim, self.hid_dim)

        self.pose_embedder = PositionalEncoding(self.hid_dim)

        self.hand_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=2),
            num_layers=1,
        )

        self.fc_scale = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, 1),
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, 8),
            nn.Softmax(dim=-1),
        )
        # self.fc_pt = nn.Sequential(
        #     nn.Linear(self.hid_dim, self.hid_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hid_dim, 3),
        # )

        num_anchor = 8
        anchor = torch.arange(0, 2*torch.pi, 2*torch.pi/num_anchor)[:num_anchor]
        anchor_x = torch.cos(anchor)
        anchor_y = torch.sin(anchor)
        anchor_z = torch.ones_like(anchor_x)
        anchor = torch.stack([anchor_x, anchor_y, anchor_z], dim=-1) / num_anchor
        self.register_buffer('anchor', anchor)

    def forward(self, enc_hand_ls, enc_obj_ls, gravity):
        enc_hand = enc_hand_ls[1]
        enc_obj = enc_obj_ls[1]

        enc_hand = self.hand_proj(enc_hand)
        enc_hand = enc_hand.reshape(enc_hand.size(0), self.num_force, self.hid_dim)
        enc_obj = self.obj_proj(enc_obj)
        enc_obj = F.adaptive_avg_pool2d(enc_obj, 1).squeeze(-1).squeeze(-1) # (bs, hid_dim, 1)

        enc_gravity = self.gravity_embedder.embed(gravity)

        enc_hand = enc_hand.reshape(enc_hand.size(0), self.num_force, self.hid_dim)
        enc_obj = enc_obj.unsqueeze(1)
        enc_gravity = self.gravity_proj(enc_gravity)

        token = torch.cat([enc_hand, enc_obj, enc_gravity], dim=-2)
        token = self.pose_embedder(token)

        token = self.hand_transformer(token)
        enc_force = token[..., :self.num_force, :]
        enc_point = token[..., :(self.num_force+1), :]

        scale = self.fc_scale(enc_force)
        scale = scale.squeeze(-1)
        weight = self.fc_weight(enc_force)
        # point = self.fc_pt(enc_point)

        force_local = self.get_local_force(scale, weight)

        return force_local, scale #, point
    
    def get_loss(self, **kwargs):
        """ mode: supervised or unsupervised
            gt_force_point: (bs, 32, 3) 
            pd_force_global: (bs, J, 2)
            gt_CoM: (bs, 1, 3)
            pd_CoM: (bs, 1, 3)
            gt_point: (bs, n, 3)
            pd_point: (bs, n, 3)
            gt_gravity: (bs, 3)
            is_grasped: (bs,)
        """
        loss_force = self.get_force_loss(
            pd_force_global=kwargs['pd_force_global'], 
            gt_gravity=kwargs['gt_gravity'],
            is_grasped=kwargs['is_grasped'],
        )
        loss_gravity = self.get_gravity_loss(
            pd_force_global=kwargs['pd_force_global'], 
            gt_gravity=kwargs['gt_gravity'],
            is_grasped=kwargs['is_grasped'],
        )
        loss_torque = self.get_torque_loss(
            pd_force_global=kwargs['pd_force_global'], 
            force_point=kwargs['gt_force_point'], 
            obj_CoM=kwargs['gt_CoM'],
            is_grasped=kwargs['is_grasped'],
        )
        loss_supervised = self.get_supervised_loss(
            pd_force_local=kwargs['pd_force_local'],
            gt_force_local=kwargs['gt_force_local'],
        )
        loss_CoM = self.get_point_loss(
            pd_point=kwargs['pd_CoM'],
            gt_point=torch.repeat_interleave(kwargs['gt_CoM'], kwargs['pd_CoM'].size(1), dim=1),
        )
        return {
            'force_loss': loss_force,
            'gravity_loss': loss_gravity,
            'torque_loss': loss_torque,
            # 'point_loss': loss_point,
            'supervised_loss': loss_supervised,
            'CoM_loss': loss_CoM,
        }
    
    def from_local_to_global(self, force_local, hand_vert):
        """ force_local: (bs, J, 2)
            hand_pose: (bs, 778, 3)
        """
        return from_local_to_global(force_local, hand_vert)

    def get_physics_metric(self, **kwargs):
        """ mode: supervised or unsupervised
            pd_force_global: (bs, J, 2)
            force_point: (bs, 32, 6) # not flipped
            gt_gravity: (bs, 3)
            obj_CoM: (bs, 1, 3)
        """
        force_blance = self.get_metric_force(
            pd_force_global=kwargs['pd_force_global'], 
            gt_gravity=kwargs['gt_gravity'],
            is_grasped=kwargs['is_grasped'],
        )
        # force_blance = force_blance.mean()
        
        gravity_blance = self.get_metric_gravity(
            pd_force_global=kwargs['pd_force_global'], 
            gt_gravity=kwargs['gt_gravity'],
            is_grasped=kwargs['is_grasped'],
        )
        # gravity_blance = gravity_blance.mean()

        torque_blance = self.get_metric_moment(
            pd_force_global=kwargs['pd_force_global'], 
            force_point=kwargs['force_point'], 
            obj_CoM=kwargs['obj_CoM'],
            is_grasped=kwargs['is_grasped'],
        )
        # torque_blance = torque_blance.mean()

        # point_distance = self.get_point_metric(
        #     pd_point=kwargs['pd_point'],
        #     gt_point=kwargs['gt_point'],
        # )
        return {
            'force_blance': force_blance,
            'gravity_blance': gravity_blance,
            'torque_blance': torque_blance,
            # 'point': point_distance,
        }

    def get_local_force(self, scale, weight, friction_coeff=0.8):
        scale = torch.abs(scale)
        weight = torch.softmax(weight, dim=-1)
        anchor = self.anchor.clone()
        anchor[:, :2] *= friction_coeff
        anchor = anchor[None, :].repeat_interleave(scale.size(-1), dim=0)
        
        force_local = torch.einsum('...ij,ijk->...ik', weight, anchor) 
        force_direction = torch.einsum('...ij,ijk->...ik', weight, anchor)
        force_direction = norm_vec(force_direction)
        force_local = force_direction * scale[..., None]
        return force_local
    
    def get_metric_force(self, **kwargs):
        resultant_force = kwargs['pd_force_global'].sum(1, keepdim=True) + kwargs['gt_gravity'] # (bs, 1, 3)
        force_blance = torch.norm(resultant_force, dim=-1).squeeze(-1)
        force_blance = force_blance * kwargs['is_grasped']
        return force_blance

    def get_force_loss(self, **kwargs):
        """ pd_force_global: (bs, 32, 3)
            gt_gravity: (bs, 1, 3)
            is_grasped: (bs,)
        """
        force_blance = self.get_metric_force(**kwargs)
        force_loss = (force_blance**2).mean()
        return force_loss
    
    def get_metric_gravity(self, **kwargs):
        resultant_force = kwargs['pd_force_global'].sum(1, keepdim=True)
        cos_proj = torch.einsum('...i,...i->...', resultant_force, kwargs['gt_gravity'])
        gravity_blance = cos_proj.squeeze(-1) + 1
        gravity_blance = gravity_blance * kwargs['is_grasped']
        return gravity_blance
    
    def get_gravity_loss(self, **kwargs):
        """ pd_force_global: (bs, 3)
            gt_gravity: (bs, 3)
            is_grasped: (bs,)
        """
        force_proj = self.get_metric_gravity(**kwargs)
        gravity_loss = (force_proj**2).mean()
        return gravity_loss

    def get_metric_moment(self, **kwargs):
        arm = kwargs['force_point'] - kwargs['obj_CoM']
        torque_blance = torch.cross(arm, kwargs['pd_force_global'], dim=-1)
        torque_blance = torque_blance.sum(1)
        torque_blance = torch.norm(torque_blance, dim=-1)
        torque_blance = torque_blance * kwargs['is_grasped']
        return torque_blance
    
    def get_torque_loss(self, **kwargs):
        """ pd_force_global: (bs, 32, 3)
            force_point: (bs, 32, 3)
            obj_CoM: (bs, 1, 3)
            is_grasped: (bs,)
        """
        torque = self.get_metric_moment(**kwargs)
        torque_loss = (torque**2).mean()
        return torque_loss

    # deprecated, mice dropping
    def get_distribution_loss(self, **kwargs):
        """ scale: (bs, 32)
            force_contact: (bs, 32)
            is_grasped: (bs,)
        """
        force_mask = kwargs['force_contact'] > 0.05
        scale_norm = kwargs['scale'] / (kwargs['scale'].norm(dim=-1, keepdim=True).detach() + 1e-8)
        force_contact_norm = kwargs['force_contact'] / (kwargs['force_contact'].norm(dim=-1, keepdim=True).detach() + 1e-8)
        dist = torch.log(torch.abs(scale_norm / (force_contact_norm+1e-8))) * force_mask
        dist = (dist ** 2).mean(-1)
        dist = dist * kwargs['is_grasped']
        dist_loss = dist.mean()
        return dist_loss
    
    def get_supervised_loss(self, **kwargs):
        """ pd_force_local: (bs, J, 3)
            gt_force_local: (bs, J, 3)
        """
        force_loss = F.mse_loss(kwargs['pd_force_local'], kwargs['gt_force_local'])
        return force_loss
    
    def get_point_metric(self, **kwargs):
        """ pd_point: (bs, J+1, 3)
            gt_point: (bs, J+1, 3)
        """
        diff = kwargs['pd_point'] - kwargs['gt_point']
        diff = torch.norm(diff, dim=-1)
        mean_distance = diff.mean(-1)
        return mean_distance
    
    def get_point_loss(self, **kwargs):
        """ supervise the estimation of force point and obj center of mass
            pd_point: (bs, J, 3)
            gt_point: (bs, J, 3)
        """
        point_loss = F.mse_loss(kwargs['pd_point'], kwargs['gt_point'])
        return point_loss


class HeadPhysics(HeadForce2):
    def __init__(self, hid_dim=256):
        super(HeadForce2, self).__init__()
        self.num_force = 32
        self.hid_dim = hid_dim

        self.fc_scale = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, 1),
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, 8),
            nn.Softmax(dim=-1),
        )
        # self.fc_pt = nn.Sequential(
        #     nn.Linear(self.hid_dim, self.hid_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hid_dim, 3),
        # )

        self.fc_CoM = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, 3),
        )
        # self.fc_rotx = nn.Sequential(
        #     nn.Linear(self.hid_dim, self.hid_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hid_dim, 3),
        # )
        # self.fc_roty = nn.Sequential(
        #     nn.Linear(self.hid_dim, self.hid_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hid_dim, 3),
        # )
        # self.fc_trans = nn.Sequential(
        #     nn.Linear(self.hid_dim, self.hid_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hid_dim, 3),
        # )

        num_anchor = 8
        anchor = torch.arange(0, 2*torch.pi, 2*torch.pi/num_anchor)[:num_anchor]
        anchor_x = torch.cos(anchor)
        anchor_y = torch.sin(anchor)
        anchor_z = torch.ones_like(anchor_x)
        anchor = torch.stack([anchor_x, anchor_y, anchor_z], dim=-1) / num_anchor
        self.register_buffer('anchor', anchor)

    def forward(self, x_hand, x_obj):
        """ x_hand: (bs, 32, 256)
            x_obj: (bs, 32, 256)
        """
        scale = self.fc_scale(x_hand)
        scale = scale.squeeze(-1)
        weight = self.fc_weight(x_obj)
        # point = self.fc_pt(enc_point)
        force_local = self.get_local_force(scale, weight)

        CoM = self.fc_CoM(x_obj)
        # rotx = self.fc_rotx(x_obj)
        # roty = self.fc_roty(x_obj)
        # trans = self.fc_trans(x_obj)

        return {
            'force_local': force_local,
            'scale': scale,
            'weight': weight,
            # 'point': point,
            'CoM': CoM,
        }