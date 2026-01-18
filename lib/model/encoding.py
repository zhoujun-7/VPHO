import torch
import torch.nn as nn


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
        x_ls = []
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                x = self.reg[i * self.nRegModules + j](x)
            x = self.maxpool(x)
            x_ls.append(x)
        out = x.flatten(1)
        return out, x_ls


class Encoder2(Encoder):
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

    # def forward(self, x):
    #     """
    #         x: (B, in_dim, 32, 32)
    #         out: (B, num_feat_chan * 2 * 2)
        
    #     """
    #     # x: (B, in_dim, 32, 32)
    #     x = self.project(x)
    #     x_ls = []
    #     for i in range(self.nRegBlock):
    #         for j in range(self.nRegModules):
    #             x = self.reg[i * self.nRegModules + j](x)
    #         x = self.maxpool(x)
    #         x_ls.append(x)
    #     out = x.flatten(1)
    #     return out, x_ls
    
    def stage1(self, x):
        x = self.project(x)
        x_ls = []
        for i in range(self.nRegBlock//2):
            for j in range(self.nRegModules):
                x = self.reg[i * self.nRegModules + j](x)
            x = self.maxpool(x)
            x_ls.append(x)
        return x, x_ls
    
    def stage2(self, x, x_ls):
        for i in range(self.nRegBlock//2, self.nRegBlock):
            for j in range(self.nRegModules):
                x = self.reg[i * self.nRegModules + j](x)
            x = self.maxpool(x)
            x_ls.append(x)
        out = x.flatten(1)
        return out, x_ls