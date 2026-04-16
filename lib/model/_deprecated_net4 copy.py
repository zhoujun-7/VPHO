import torch
import torch.nn as nn
from torchvision import ops

from lib.configs.args import cfg
from lib.model.backbone_FPN_HFL import FPN

class MyNet(nn.Module):
    def __init__(self, ):
        super(MyNet, self).__init__()
        
        self.backbone = FPN()
        self.out_res = 32


    def forward(self, data):
        bs, device = data['rgb'].shape[0], data['rgb'].device

        feat_h, feat_o = self.backbone(data['rgb'])
    
        idx_tensor = torch.arange(bs, device=device.device).float().view(-1, 1)
        roi_boxes_hand = torch.cat((idx_tensor, data['bbox_hand']), dim=1)
        roi_boxes_obj = torch.cat((idx_tensor, data['bbox_obj']), dim=1)  #! use hand in 2023-CVPR-HFL???

        x_hand = ops.roi_align(feat_h, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0, sampling_ratio=-1)
        x_obj = ops.roi_align(feat_o, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0, sampling_ratio=-1)
