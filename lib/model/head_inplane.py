import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
from lib.model.encoding import Residual


class HeadHeatmap(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_layers=4, act=nn.LeakyReLU(True)):
        super(HeadHeatmap, self).__init__()
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1)
                layer = nn.Sequential(conv, nn.BatchNorm2d(hidden_dim), act)
                layers.append(layer)
            elif i == num_layers - 1:
                conv = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
                layers.append(conv)
            else:
                conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
                layer = nn.Sequential(conv, nn.BatchNorm2d(hidden_dim), act)
                layers.append(layer)
        self.layers = nn.Sequential(*layers)
        self.loss_fn = JointsMSELoss()

    def forward(self, x):
        y_ls = [x]
        y = x
        for i in range(self.num_layers):
            y = self.layers[i](y)
            y_ls.append(y)
            if i > 0 and i < self.num_layers - 1:
                y = y + y_ls[i]
        return y

    def get_loss(self, **kwargs):
        return self.loss_fn(kwargs['pd_hm'], kwargs['gt_hm'])

class HeadHeatmap2(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_conv_layers=2, num_deconv_layer=1, act=nn.LeakyReLU(True)):
        super(HeadHeatmap2, self).__init__()
        self.num_layers = num_conv_layers
        layers = []
        for i in range(num_conv_layers):
            if i == 0:
                conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1)
                layers.append(conv)
            else:
                conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
                layer = [conv, nn.BatchNorm2d(hidden_dim, momentum=0.1), act]
                layers.extend(layer)
        self.conv_layers = nn.Sequential(*layers)
        self.inplanes = hidden_dim
        hidden_dim_ls = [hidden_dim//(2**(i+1)) for i in range(num_deconv_layer)]
        self.deconv_layers = self._make_deconv_layer(num_deconv_layer, hidden_dim_ls, [4]*num_deconv_layer)
        self.final_layer = nn.Conv2d(hidden_dim//2**num_deconv_layer, out_dim, kernel_size=1)
        self.loss_fn = JointsMSELoss()

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding
    
    """ from 2018_ECCV_SampleBaseline """
    def _make_deconv_layer(self, num_layers=3, num_filters=[256, 256, 256], num_kernels=[4, 4, 4]):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.deconv_layers(x)
        # x = self.conv_layers(x)
        x = self.final_layer(x)
        return x

    def get_loss(self, **kwargs):
        if kwargs['gt_hm'].shape != kwargs['pd_hm'].shape:
            kwargs['gt_hm'] = F.interpolate(kwargs['gt_hm'], size=kwargs['pd_hm'].shape[-2:], mode='bilinear', align_corners=False)
            warnings.warn(f"Ground truth heatmap ({kwargs['gt_hm'].shape}) does not match the prediction heatmap ({kwargs['pd_hm'].shape}).")
        return self.loss_fn(kwargs['pd_hm'], kwargs['gt_hm'])
    

# modified from 2022_CVPR_KeypointTransformer
class HeadHeatmap_KYPT(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_conv_layers=2, num_deconv_layer=1, act=nn.LeakyReLU(True)):
        super(HeadHeatmap_KYPT, self).__init__()
        self.num_layers = num_conv_layers
        layers = []
        for i in range(num_conv_layers):
            if i == 0:
                conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1)
                layer = nn.Sequential(conv, nn.BatchNorm2d(hidden_dim, momentum=0.1), act)
                layers.append(layer)
            else:
                conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
                layer = nn.Sequential(conv, nn.BatchNorm2d(hidden_dim, momentum=0.1), act)
                layers.append(layer)
        self.conv_layers = nn.Sequential(*layers)
        self.inplanes = hidden_dim
        hidden_dim_ls = [hidden_dim//(2**(i+1)) for i in range(num_deconv_layer)]
        self.deconv_layers = self._make_deconv_layer(num_deconv_layer, hidden_dim_ls, [4]*num_deconv_layer)
        self.final_layer = nn.Conv2d(hidden_dim//2**num_deconv_layer, 1, kernel_size=1)
        self.loss_fn = JointsMSELoss()

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding
    
    """ from 2018_ECCV_SampleBaseline """
    def _make_deconv_layer(self, num_layers=3, num_filters=[256, 256, 256], num_kernels=[4, 4, 4]):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.deconv_layers(x)
        # x = self.conv_layers(x)
        x = self.final_layer(x)
        return x

    def get_loss(self, **kwargs):
        if kwargs['gt_hm'].shape != kwargs['pd_hm'].shape:
            kwargs['gt_hm'] = F.interpolate(kwargs['gt_hm'], size=kwargs['pd_hm'].shape[-2:], mode='bilinear', align_corners=False)
            warnings.warn(f"Ground truth heatmap ({kwargs['gt_hm'].shape}) does not match the prediction heatmap ({kwargs['pd_hm'].shape}).")
        return self.loss_fn(kwargs['pd_hm'], kwargs['gt_hm'])

class JointsMSELoss(nn.Module):
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        loss = self.criterion(heatmaps_pred, heatmaps_gt)
        return loss

# from 2022_CVPR_KeypointTransformer
class JointHeatmapLoss_KYPT(nn.Module):
    def __ini__(self):
        super(JointHeatmapLoss_KYPT, self).__init__()

    def forward(self, joint_out, joint_gt, hm_valid=None):
        if hm_valid is not None:
            hm_valid = hm_valid[:, :, None] # N x 1 x 1
            loss = (joint_out - joint_gt)**2 * hm_valid
        else:
            loss = (joint_out - joint_gt)**2
        return loss    


class HeadSegm(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_conv_layers=2, num_deconv_layer=1, act=nn.LeakyReLU(True)):
        super(HeadSegm, self).__init__()
        self.num_layers = num_conv_layers
        layers = []
        for i in range(num_conv_layers):
            if i == 0:
                conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1)
                layer = nn.Sequential(conv, nn.BatchNorm2d(hidden_dim, momentum=0.1), act)
                layers.append(layer)
            else:
                conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
                layer = nn.Sequential(conv, nn.BatchNorm2d(hidden_dim, momentum=0.1), act)
                layers.append(layer)
        self.conv_layers = nn.Sequential(*layers)
        self.inplanes = hidden_dim
        hidden_dim_ls = [hidden_dim//(2**(i+1)) for i in range(num_deconv_layer)]
        self.deconv_layers = self._make_deconv_layer(num_deconv_layer, hidden_dim_ls, [4]*num_deconv_layer)
        self.final_layer = nn.Conv2d(hidden_dim//2**num_deconv_layer, out_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.loss_segm_fn = nn.BCEWithLogitsLoss()
        self.loss_hm_fn = JointsMSELoss()

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding
    
    """ from 2018_ECCV_SampleBaseline """
    def _make_deconv_layer(self, num_layers=3, num_filters=[256, 256, 256], num_kernels=[4, 4, 4]):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        # x_segm = self.sigmoid(x)
        return x

    def get_loss(self, **kwargs):
        if kwargs['pd_segm'].shape != kwargs['pd_segm'].shape:
            kwargs['gt_segm'] = F.interpolate(kwargs['gt_segm'], size=kwargs['pd_segm'].shape[-2:], mode='bilinear', align_corners=False)
            warnings.warn(f"Ground truth segmentation ({kwargs['gt_segm'].shape}) does not match the prediction segmentation ({kwargs['pd_segm'].shape}).")

        pd = kwargs['pd_segm']
        gt = kwargs['gt_segm']
        
        pd = pd.view(pd.size(0), -1)
        gt = gt.view(gt.size(0), -1)
        loss = self.loss_segm_fn(pd, gt)
        return loss




class HeadHeatmapSegm(nn.Module):
    def __init__(self, in_dim,out_dim, hidden_dim=256, num_conv_layers=2, num_deconv_layer=1, act=nn.LeakyReLU(True)):
        super(HeadHeatmapSegm, self).__init__()
        self.num_layers = num_conv_layers
        layers = []
        for i in range(num_conv_layers):
            if i == 0:
                conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1)
                layer = nn.Sequential(conv, nn.BatchNorm2d(hidden_dim, momentum=0.1), act)
                layers.append(layer)
            else:
                conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
                layer = nn.Sequential(conv, nn.BatchNorm2d(hidden_dim, momentum=0.1), act)
                layers.append(layer)
        self.conv_layers = nn.Sequential(*layers)
        self.inplanes = hidden_dim
        hidden_dim_ls = [hidden_dim//(2**(i+1)) for i in range(num_deconv_layer)]
        self.deconv_layers = self._make_deconv_layer(num_deconv_layer, hidden_dim_ls, [4]*num_deconv_layer)
        self.final_layer = nn.Conv2d(hidden_dim//2**num_deconv_layer, out_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.loss_segm_fn = nn.BCELoss()
        self.loss_hm_fn = JointsMSELoss()

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding
    
    """ from 2018_ECCV_SampleBaseline """
    def _make_deconv_layer(self, num_layers=3, num_filters=[256, 256, 256], num_kernels=[4, 4, 4]):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        x_hm = x[:, :-1]
        x_segm = self.sigmoid(x[:, [-1]])
        return x_hm, x_segm

    def get_segm_loss(self, **kwargs):
        if kwargs['pd_segm'].shape != kwargs['pd_segm'].shape:
            kwargs['gt_segm'] = F.interpolate(kwargs['gt_segm'], size=kwargs['pd_segm'].shape[-2:], mode='bilinear', align_corners=False)
            warnings.warn(f"Ground truth segmentation ({kwargs['gt_segm'].shape}) does not match the prediction segmentation ({kwargs['pd_segm'].shape}).")

        pd = kwargs['pd_segm']
        gt = kwargs['gt_segm']
        
        pd = pd.view(pd.size(0), -1)
        gt = gt.view(gt.size(0), -1)
        loss = self.loss_segm_fn(pd, gt)
        return loss
    
    def get_hm_loss(self, **kwargs):
        if kwargs['pd_hm'].shape != kwargs['gt_hm'].shape:
            kwargs['gt_hm'] = F.interpolate(kwargs['gt_hm'], size=kwargs['pd_hm'].shape[-2:], mode='bilinear', align_corners=False)
            warnings.warn(f"Ground truth heatmap ({kwargs['gt_hm'].shape}) does not match the prediction heatmap ({kwargs['pd_hm'].shape}).")
        return self.loss_hm_fn(kwargs['pd_hm'], kwargs['gt_hm'])
    
    def get_loss(self, **kwargs):
        return {
            'hm_obj_loss': self.get_hm_loss(pd_hm=kwargs['pd_hm'], gt_hm=kwargs['gt_hm']),
            'segm_obj_loss': self.get_segm_loss(pd_segm=kwargs['pd_segm'], gt_segm=kwargs['gt_segm'])
        }
