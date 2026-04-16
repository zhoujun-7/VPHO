import torch
import torch.nn as nn
from timm.models import create_model

from ._deprecated_diff_obj import DiffusionObj

class SimpleDiffusion(nn.Module):
    def __init__(
        self,
        cfg=None,
    ):
        super(SimpleDiffusion, self).__init__()
        self.cfg = cfg

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

        self.obj_diff = DiffusionObj()
    
    def get_feature(self, sample):
        img_feat = self.feature_extractor(sample['rgb'])
        img_feat = self.max_pool(img_feat[-1])
        img_feat = img_feat.flatten(1)
        img_feat = self.act(img_feat)
        img_feat = self.fc(img_feat)
        return img_feat

    def forward(self, sample, mode='train'):
        sample['img_feat'] = self.get_feature(sample)
        
        if mode == 'train':
            diff_loss = self.obj_diff.train_score_func(sample)
            return diff_loss
        elif mode == 'test':
            inprocess_rt, res_rt = self.honet.module.obj_diff.pred_func(sample)
            return inprocess_rt, res_rt
        elif mode == 'score':
            score = self.obj_diff(sample, mode='score')
            return score
        elif mode == 'energy':
            bs, repeat_num, c = sample['sampled_pose'].shape
            sample['sampled_pose'] = sample['sampled_pose'].reshape(bs*repeat_num, c)
            sample['img_feat'] = sample['img_feat'][:, None].repeat(1, repeat_num, 1).reshape(bs*repeat_num, -1)
            sample['t'] = torch.ones(bs*repeat_num, 1).type_as(sample['img_feat']) * 1e-5
            score = self.obj_diff.net(sample)

            # _, std = self.obj_diff.marginal_prob_fn(sample['sampled_pose'], sample['t'])
            # score = score / std
            score = score.reshape(bs, repeat_num, -1)
            return score

