import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from timm.models import create_model
from timm.utils import ModelEmaV3
# ops.feature_pyramid_network.FeaturePyramidNetwork

from lib.configs.args import cfg
from lib.model.backbone_FPN_HFL import FPN
from lib.model.denoiser import BaseDenoiser
from lib.model.head_inplane import HeadHeatmap
from lib.model.score_based_model import ScoreBasedModelAgent
from lib.model.ema import ExponentialMovingAverage
from lib.model.head_mano import HeadMano, mano_aa_to_6D, mano_6D_to_aa


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
        if use_mano6d := False:
            self.mano_mode = 'mano6d'
        else:
            self.mano_mode = 'mano'
        self.denoiser_hand = BaseDenoiser(self.score_agent.marginal_prob_fn, head=self.mano_mode)
        self.denoiser_obj = BaseDenoiser(self.score_agent.marginal_prob_fn, head='obj')

        self.head_hm_hand = HeadHeatmap(256, 21, 128, 3)
        self.head_hm_obj = HeadHeatmap(256, 27, 128, 3)

        self.encoder_hand = Encoder(256+21, 256, size_input_feature=(self.cfg.roi_size, self.cfg.roi_size))
        self.encoder_obj = Encoder(256+27, 256, size_input_feature=(self.cfg.roi_size, self.cfg.roi_size))

        self.head_mano = HeadMano(in_dim=1024, is_output_contact=True)


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
        encoding_hand = self.encoder_hand(torch.cat((hf_hr, pd_hm_hand), dim=1)) # (bs, 1024)
        encoding_obj = self.encoder_obj(torch.cat((of_or, pd_hm_obj), dim=1)) # (bs, 1024)
        
        pd_mano_pose, pd_mano_shape, pd_hand_contact = self.head_mano(encoding_hand)
        pd_hand_vert, pd_hand_joint = self.head_mano.get_hand_verts({'pose': pd_mano_pose, 'shape': pd_mano_shape})

        if mode == 'train':
            # diffusion loss
            loss_dt, pd_dt = {}, {}
            
            gt_mano_6d = mano_aa_to_6D(data['gt_mano']) if self.mano_mode == 'mano6d' else data['gt_mano'] # convert mano(axis-angle) to mano(6d)
            diff_hand_data = {'feat': encoding_hand, 'gt_pose': gt_mano_6d}
            loss_dt['diff_hand_loss'] = self.score_agent.get_score_loss(diff_hand_data, self.denoiser_hand)
            diff_obj_data = {'feat': encoding_obj, 'gt_pose': data['gt_obj']}
            loss_dt['diff_obj_loss'] = self.score_agent.get_score_loss(diff_obj_data, self.denoiser_obj)
        
            # heatmap loss
            hm_hand_hr = F.interpolate(data['hm_hand'], size=(self.cfg.roi_size, self.cfg.roi_size), mode='bilinear', align_corners=False)
            hm_obj_hr = F.interpolate(data['hm_obj'], size=(self.cfg.roi_size, self.cfg.roi_size), mode='bilinear', align_corners=False)
            hm_hand_data = {'pd_hm': pd_hm_hand, 'gt_hm': hm_hand_hr}
            hm_obj_data = {'pd_hm': pd_hm_obj, 'gt_hm': hm_obj_hr}

            loss_dt['hm_hand_loss'] = self.head_hm_hand.get_loss(hm_hand_data)
            loss_dt['hm_obj_loss'] = self.head_hm_obj.get_loss(hm_obj_data)
    
            # mano_loss
            gt_mano_pose, gt_mano_shape = data['gt_mano'][:, :48], data['gt_mano'][:, 48:]
            # gt_hand_vert, gt_hand_joint = self.head_mano.get_hand_verts({'pose': gt_mano_pose, 'shape': gt_mano_shape}) #! deprecated, as left hand shape is incompatible with right hand mano, use data['gt_hand_vert_flip'], 'gt_joint': data['gt_hand_jt3d_flip'] instead
            gt_hand_vert, gt_hand_joint = data['gt_hand_vert_flip'], data['gt_hand_jt3d_flip'] # checked, from annotation, no error introduced
            mano_data = {'pd_pose': pd_mano_pose, 'pd_shape': pd_mano_shape, 'pd_vert': pd_hand_vert, 'pd_joint': pd_hand_joint,
                         'gt_pose': gt_mano_pose, 'gt_shape': gt_mano_shape, 'gt_vert': gt_hand_vert, 'gt_joint': gt_hand_joint, 
                         'is_right': data['is_right']}
            mano_loss_dt = self.head_mano.get_loss(mano_data)
            loss_dt.update(mano_loss_dt)
            loss_dt['hand_contact_loss'] = self.head_mano.get_contact_loss({'pd_contact': pd_hand_contact, 'gt_contact': data['gt_hand_contact']})

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
            return loss_dt, pd_dt
        
        elif mode == 'score':
            pd_dt = {}
            hand_data = {'feat': encoding_hand, 'sampled_pose': data['sampled_mano_pose']}
            hand_score = self.score_agent.get_score(hand_data, self.denoiser_hand)
            pd_dt['hand_score'] = hand_score

            diff_obj_data = {'feat': encoding_obj, 'sampled_pose': data['sampled_obj_pose']}
            obj_score = self.score_agent.get_score(diff_obj_data, self.denoiser_obj)
            pd_dt['obj_score'] = obj_score
            return pd_dt

        elif mode == 'sample':
            pd_dt = {}
            _, hand_feat_dim = encoding_hand.shape
            encoding_hand = encoding_hand[:, None].repeat(1, self.cfg.sample_num, 1).reshape(-1, hand_feat_dim)
            hand_data = {'feat': encoding_hand}
            hand_inprocess, hand_final = self.score_agent.sample(hand_data, self.denoiser_hand, self.cfg.sample_T0)
            pd_dt['hand_inprocess'] = hand_inprocess.reshape(bs, self.cfg.sample_num, -1, 58)
            pd_dt['hand_final'] = hand_final.reshape(bs, self.cfg.sample_num, 58)

            bs, obj_feat_dim = encoding_obj.shape
            encoding_obj = encoding_obj[:, None].repeat(1, self.cfg.sample_num, 1).reshape(-1, obj_feat_dim)
            diff_obj_data = {'feat': encoding_obj}
            obj_inprocess, obj_final = self.score_agent.sample(diff_obj_data, self.denoiser_obj, self.cfg.sample_T0)
            pd_dt['obj_inprocess'] = obj_inprocess.reshape(bs, self.cfg.sample_num, -1, 9)
            pd_dt['obj_final'] = obj_final.reshape(bs, self.cfg.sample_num, 9)
            return pd_dt
        
        elif mode == 'predict':
            pd_dt = {}
            pd_dt['hand_vert'] = pd_hand_vert
            pd_dt['hand_joint'] = pd_hand_joint
            pd_dt['hand_contact'] = pd_hand_contact
            pd_dt['hand_heatmap'] = pd_hm_hand
            pd_dt['obj_heatmap'] = pd_hm_obj

            _, hand_feat_dim = encoding_hand.shape
            encoding_hand = encoding_hand[:, None].repeat(1, self.cfg.sample_num, 1).reshape(-1, hand_feat_dim)
            hand_data = {'feat': encoding_hand}
            hand_inprocess, hand_final = self.score_agent.sample(hand_data, self.denoiser_hand, self.cfg.sample_T0)
            hand_inprocess, hand_final = hand_inprocess.float(), hand_final.float()
            hand_inprocess = mano_6D_to_aa(hand_inprocess) if self.mano_mode == 'mano6d' else hand_inprocess
            hand_final = mano_6D_to_aa(hand_final) if self.mano_mode == 'mano6d' else hand_final
            pd_dt['hand_inprocess'] = hand_inprocess.reshape(bs, self.cfg.sample_num, -1, 58)
            pd_dt['hand_final'] = hand_final.reshape(bs, self.cfg.sample_num, 58)

            #! this part is for hand visualization and can be removed during runtime evaluation
            diff_final_mano_pose, diff_final_mano_shape = hand_final[:, :48], hand_final[:, 48:]
            diff_final_hand_vert, diff_final_hand_joint = self.head_mano.get_hand_verts({'pose': diff_final_mano_pose, 'shape': diff_final_mano_shape})
            pd_dt['final_hand_vert'] = diff_final_hand_vert.reshape(bs, self.cfg.sample_num, 778, 3)
            pd_dt['final_hand_joint'] = diff_final_hand_joint.reshape(bs, self.cfg.sample_num, 21, 3)
            diff_inprocess_mano_pose, diff_inprocess_mano_shape = hand_inprocess[0, ::10, :48], hand_inprocess[0, ::10, 48:] #* only take every 10th sample of the first batchsample
            diff_inprocess_hand_vert, diff_inprocess_hand_joint = self.head_mano.get_hand_verts({'pose': diff_inprocess_mano_pose, 'shape': diff_inprocess_mano_shape})
            diff_inprocess_hand_vert = diff_inprocess_hand_vert.reshape(-1, 778, 3)
            diff_inprocess_hand_joint = diff_inprocess_hand_joint.reshape(-1, 21, 3)
            pd_dt['inprocess_hand_vert'] = diff_inprocess_hand_vert
            pd_dt['inprocess_hand_joint'] = diff_inprocess_hand_joint

            bs, obj_feat_dim = encoding_obj.shape
            encoding_obj = encoding_obj[:, None].repeat(1, self.cfg.sample_num, 1).reshape(-1, obj_feat_dim)
            diff_obj_data = {'feat': encoding_obj}
            obj_inprocess, obj_final = self.score_agent.sample(diff_obj_data, self.denoiser_obj, self.cfg.sample_T0)
            pd_dt['obj_inprocess'] = obj_inprocess.reshape(bs, self.cfg.sample_num, -1, 9)
            pd_dt['obj_final'] = obj_final.reshape(bs, self.cfg.sample_num, 9)
            return pd_dt
        


