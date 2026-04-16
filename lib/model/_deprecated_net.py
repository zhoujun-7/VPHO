import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model, resnet
from ._deprecated_mask import HeadMask, HeadDepth, HeadContact, HeadJoint2D


class MyNet(nn.Module):
    def __init__(self, ):
        super(MyNet, self).__init__()
        self.backbone = create_model(
            "resnet50", 
            pretrained=False, 
            features_only=True,
            out_indices=(1, 2, 3, 4),
            in_chans=3,
        )
        self.input_projection = nn.Conv2d(2048, 256, kernel_size=1)

        self.head_mask = HeadMask(
            dim=256 + 8, 
            fpn_dims=[1024, 512, 256], 
            context_dim=256,
        )
        self.head_depth = HeadDepth(
            dim=256 + 8, 
            fpn_dims=[1024, 512, 256], 
            context_dim=256,
        )

        self.head_contact = HeadContact(
            dim=256 + 8, 
            fpn_dims=[1024, 512, 256], 
            context_dim=256,
        )

        self.head_joint2d = HeadJoint2D(
            dim=256 + 8, 
            fpn_dims=[1024, 512, 256], 
            context_dim=256,
        )


    def forward(self, sample, is_train=True):
        for k, v in sample.items():
            print(k, v.shape)
        b, _, h, w = sample['rgb'].shape
        feat_ls = self.backbone(sample['rgb'])
        feat_map = self.input_projection(feat_ls[-1])

        bbox_att = torch.ones([b, 2, 8, feat_map.shape[-2], feat_map.shape[-1]], device=feat_map.device)
        mask_ho, fpn_feat = self.head_mask(feat_map, bbox_att, [feat_ls[2], feat_ls[1], feat_ls[0]])  # (B, 2, H//4, W//4)

        print(fpn_feat.shape)
        exit()

        # depth = self.head_depth(mask_ho, fpn_feat) # (b, 2, 56, 56)
        # contact = self.head_contact(mask_ho, fpn_feat) # (b, 2, 7, 56, 56)
        # joint2d = self.head_joint2d(fpn_feat)

        return contact
    

