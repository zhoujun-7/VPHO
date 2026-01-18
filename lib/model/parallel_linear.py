import torch 
import torch.nn as nn
import numpy as np
import math
from torch.nn import init


class ParallelLinear(nn.Module):
    """ do multi torch.nn.Linear parallelly in one step for speed up"""
    def __init__(self, in_features, out_features, num):
        super(ParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num = num
        self.weight = nn.Parameter(torch.Tensor(num, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(num, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # from torch.nn.linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # using n fc is time-consuming, so using einsum to replace n fc for speed up.
        # x: (B, c) or (B, n, c)
        # return: (B, n, c)
        if x.dim() == 2:
            y = torch.einsum('bc,ncd->bnd', x, self.weight) + self.bias
        elif x.dim() == 3:
            y = torch.einsum('bnc,ncd->bnd', x, self.weight) + self.bias
        return y