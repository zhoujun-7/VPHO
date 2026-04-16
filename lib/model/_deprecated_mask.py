import torch
from torch import Tensor, nn
from typing import Dict, List, Optional, Tuple, Union



def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum()


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class HeadMask(nn.Module):
    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        assert dim % 8 == 0
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]

        self.lay1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, dim)
        self.lay2 = nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, inter_dims[1]), inter_dims[1])
        self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = nn.GroupNorm(min(8, inter_dims[2]), inter_dims[2])
        self.lay4 = nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = nn.GroupNorm(min(8, inter_dims[3]), inter_dims[3])
        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(min(8, inter_dims[4]), inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = nn.functional.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = nn.functional.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = nn.functional.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = nn.functional.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")

        cur_fpn = x
        x = self.lay5(x)
        x = self.gn5(x)
        x = nn.functional.relu(x)

        x = self.out_lay(x)
        x = x.reshape(-1, 2, x.shape[-2], x.shape[-1])
        cur_fpn = cur_fpn.reshape(-1, 2, cur_fpn.shape[-3], cur_fpn.shape[-2], cur_fpn.shape[-1])
        return x, cur_fpn
    
    #* checked
    def get_loss(self, pd, gt):
        pd = nn.functional.interpolate(pd, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        pd = pd.flatten(1)
        gt = gt.flatten(1)
        loss = {
            "dice": dice_loss(pd, gt),
            "mask": sigmoid_focal_loss(pd, gt),
        }
        return loss
    

class HeadDepth(nn.Module):
    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        assert dim % 8 == 0
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        
        self.lay5 = nn.Conv2d(inter_dims[3]+1, inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(min(8, inter_dims[4]), inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 2, 3, padding=1)

    def forward(self, mask, cur_fpn):
        x = torch.cat([mask[:, [1]], cur_fpn[:, 1]], 1)
        x = self.lay5(x)
        x = self.gn5(x)
        x = nn.functional.relu(x)
        x = self.out_lay(x)
        return x
    
    #* checked
    def get_loss(self, pd, gt, gt_mask):
        pd = nn.functional.interpolate(pd, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        l2_loss = (pd[gt_mask] - gt[gt_mask])**2
        num = gt_mask.sum(dim=[-1, -2, -3])
        weight = num[:, None, None, None].repeat(1, gt.shape[-3], gt.shape[-2], gt.shape[-1])
        weight = weight[gt_mask]
        l2_loss[weight == 0] *= 0
        l2_loss = l2_loss / (weight + 1e-8)

        loss = {
            'depth_l1': l2_loss.mean(),
        }
        return loss
    

class HeadContact(nn.Module):
    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        assert dim % 8 == 0
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        
        self.lay5 = nn.Conv2d(inter_dims[3]*2+2, inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(min(8, inter_dims[4]), inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 2*7, 3, padding=1)

    def forward(self, mask, cur_fpn):
        x = torch.cat([mask, cur_fpn[:, 1], cur_fpn[:, 1]], 1)
        x = self.lay5(x)
        x = self.gn5(x)
        x = nn.functional.relu(x)
        x = self.out_lay(x)
        x = x.reshape(-1, 2, 7, x.shape[-2], x.shape[-1])
        return x
    
    #* checked
    def get_loss(self, pd, gt, mask):
        """ mask is the intersection of obj mask and hand mask
            pd: (B, 2, 7, H, W)
            gt: (B, 2, 7, H, W)
            mask: (B, H, W)
        """
        b, n, c, h, w = pd.shape
        H, W = gt.shape[-2:]
        pd = nn.functional.interpolate(pd.view(b*n, c, h, w), size=gt.shape[-2:], mode="bilinear", align_corners=False)
        pd = pd.view(b*n, c, H, W) # (B*2, 7, H, W)
        gt = gt.view(b*n, c, H, W) # (B*2, 7, H, W)
        mask = mask.unsqueeze(1).repeat(1, 2, 1, 1).reshape(b*n, H, W) # (B*2, H, W)
        
        weight = mask.clone().float()
        weight_sum = weight.sum(-1).sum(-1)
        zero_mask = weight_sum == 0
        weight[zero_mask] = 0
        weight = weight / (weight_sum[:, None, None] + 1e-8) # (B*2, H, W)
        weight = weight[mask]

        # classification loss
        pd_cls = pd[:, :-1].permute(0, 2, 3, 1).view(b*n, H, W, 6) # (B, 2, H, W, 6)
        pd_cls = pd_cls[mask]
        gt_cls = gt[:, :-1].argmax(1) #(B*2, H, W)
        gt_cls = gt_cls[mask]
        loss_ce = nn.functional.cross_entropy(pd_cls, gt_cls, reduction="none")
        loss_ce = loss_ce * weight
        loss_ce = loss_ce.sum(-1)
        
        # regression loss
        gt_reg = gt[:, -1]
        gt_reg = gt_reg[mask]
        pd_reg = pd[:, -1]
        pd_reg = pd_reg[mask]
        loss_l2 = (pd_reg - gt_reg)**2
        loss_l2 = loss_l2 * weight
        loss_l2 = loss_l2.sum(-1)

        loss = {
            'contact_cls': loss_ce.mean(),
            'contact_reg': loss_l2.mean(),
        }
        return loss


class HeadJoint2D(nn.Module):
    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        assert dim % 8 == 0
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        
        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(min(8, inter_dims[4]), inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 21, 3, padding=1)
        self.criterion = nn.MSELoss()

    def forward(self, cur_fpn):
        x = self.lay5(cur_fpn[:, 0])
        x = self.gn5(x)
        x = nn.functional.relu(x)
        x = self.out_lay(x)
        return x
    
    def get_loss(self, pd, gt):
        b, n, h, w = pd.shape
        H, W = gt.shape[-2:]
        pd = nn.functional.interpolate(pd, size=gt.shape[-2:], mode="bilinear", align_corners=False) 
        pd = pd.view(b*n, H, W)
        gt = gt.view(b*n, H, W)

        loss_mse = self.criterion(pd, gt)
        loss = {
            'joint2d': loss_mse,
        }
        return loss