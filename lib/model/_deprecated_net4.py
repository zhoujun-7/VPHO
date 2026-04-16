import torch
import torch.nn as nn
from torchvision import ops
from timm.models import create_model
from timm.utils import ModelEmaV3

from lib.configs.args import cfg
from lib.model.backbone_FPN_HFL import FPN
from lib.model.denoiser import BaseDenoiser
from lib.model.score_based_model import ScoreBasedModelAgent
from lib.model.ema import ExponentialMovingAverage

class SimpleDiffObj(nn.Module):
    def __init__(self):
        super(SimpleDiffObj, self).__init__()
        
        self.feature_extractor = create_model(
            "resnet50", 
            pretrained=False, 
            features_only=True,
            out_indices=(1, 2, 3, 4),
            in_chans=3,
        )
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(2048, 1024)
        self.act = nn.ReLU(True)
        # self.fc_hand = nn.Linear(2048, 1024)

        self.score_agent = ScoreBasedModelAgent()
        # self.denoiser_hand = BaseDenoiser(self.score_agent.marginal_prob_fn, head='mano')
        self.denoiser_obj = BaseDenoiser(self.score_agent.marginal_prob_fn, head='obj')
        self.cfg = cfg


    def forward(self, data, mode='train'):
        assert mode in ['train', 'score', 'sample']

        img_feat = self.feature_extractor(data['rgb'])
        img_feat = self.max_pool(img_feat[-1])
        img_feat = img_feat.flatten(1)
        img_feat = self.act(img_feat)
        obj_feat = self.fc(img_feat)
        # hand_feat = self.fc_hand(data['rgb'])
        
        if mode == 'train':
            # hand_data = {'feat': hand_feat, 'gt_pose': data['gt_mano']}
            obj_data = {'feat': obj_feat, 'gt_pose': data['gt_obj']}
            loss_dt, pd_dt = {}, {}

            # loss_dt['diff_hand_loss'] = self.score_agent.get_score_loss(hand_data, self.denoiser_hand)
            loss_dt['diff_obj_loss'] = self.score_agent.get_score_loss(obj_data, self.denoiser_obj)

            total_loss = 0
            for k, v in loss_dt.items():
                total_loss = total_loss + v * getattr(self.cfg, k + '_weight')
            loss_dt['total_loss'] = total_loss
            return loss_dt, pd_dt
        
        elif mode == 'score':
            pd_dt = {}
            # hand_data = {'feat': hand_feat, 'sampled_pose': data['sampled_mano_pose']}
            # hand_score = self.score_agent.get_score(hand_data, self.denoiser_hand)
            # pd_dt['hand_score'] = hand_score

            obj_data = {'feat': obj_feat, 'sampled_pose': data['sampled_obj_pose']}
            obj_score = self.score_agent.get_score(obj_data, self.denoiser_obj)
            pd_dt['obj_score'] = obj_score
            return pd_dt

        elif mode == 'sample':
            bs, obj_feat_dim = obj_feat.shape
            # hand_data = {'feat': hand_feat}
            obj_feat = obj_feat[:, None].repeat(1, self.cfg.sample_num, 1).reshape(-1, obj_feat_dim)
            obj_data = {'feat': obj_feat}
            pd_dt = {}
            # hand_inprocess, hand_final = self.score_agent.sample(hand_data, self.denoiser_hand, self.cfg.sample_T0)
            obj_inprocess, obj_final = self.score_agent.sample(obj_data, self.denoiser_obj, self.cfg.sample_T0)
            # pd_dt['hand_inprocess'] = hand_inprocess
            # pd_dt['hand_final'] = hand_final
            pd_dt['obj_inprocess'] = obj_inprocess.reshape(bs, self.cfg.sample_num, -1, 9)
            pd_dt['obj_final'] = obj_final.reshape(bs, self.cfg.sample_num, 9)
            return pd_dt