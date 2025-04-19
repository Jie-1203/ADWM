import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat

class addconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(addconv, self).__init__()
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size

        self.point_wise = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)

        self.weight = nn.Parameter(torch.normal(mean=0,std=0.0001,size=(out_channels,1,kernel_size,kernel_size)))
        self.weight_2 = nn.Parameter(torch.normal(mean=0,std=0.0001,size=(out_channels,out_channels,1,1)))
        self.depth_wise = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=(kernel_size - 1) // 2,
                                    groups=out_channels,
                                    bias=True)

    def forward(self, x): 
        Blue_tmp = self.point_wise(x) 
        Blue_tmp = self.depth_wise(Blue_tmp) 

        bias = F.conv2d(Blue_tmp, self.weight, padding=(self.kernel_size - 1) // 2, groups=self.out_planes)
        bias = F.conv2d(bias,self.weight_2)

        out = Blue_tmp + bias

        return out

# --------------------------------Res Block -----------------------------------#
class Res_Block(nn.Module):
    def __init__(self,in_planes):
        super(Res_Block, self).__init__()
        self.conv1=addconv(in_planes,in_planes,3)
        self.relu1=nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2=addconv(in_planes,in_planes,3)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(in_planes)

    def forward(self,x):
        res=self.conv1(x)
        res=self.bn1(res)
        res=self.relu1(res)
        res=self.conv2(res)
        res=self.bn2(res)
        res=self.relu2(res)
        res = torch.add(x, res)
        return res

class BWNet(nn.Module):
    def __init__(self, pan_dim=1, ms_dim=8, channel1=20, channel2=32, channel3=16, num_blocks=4):
        super().__init__()

        self.ms_dim = ms_dim
        self.raise_dim = nn.Sequential(
            addconv(pan_dim + ms_dim, channel1, 3),
            addconv(channel1, channel2, 3),
            nn.LeakyReLU(inplace=True)
        )
        self.layers = nn.ModuleList([])
        for _ in range(num_blocks):
            self.layers.append(Res_Block(channel2))

        self.to_output = nn.Sequential(
            addconv(channel2, channel3, 3),
            addconv(channel3, ms_dim, 3),
            addconv(ms_dim, ms_dim, 3)
        )

    def forward(self, ms, pan):
        ms = F.interpolate(ms, scale_factor=4, mode='bicubic')
        input = torch.concatenate([pan, ms], dim=1)
        x = self.raise_dim(input)
        feature_list = []

        for layer in self.layers:
            x = layer(x)
            feature_list.append(x)

        return self.to_output(x) + ms


def summaries(model, input_size, grad=False):
    if grad:
        from torchinfo import summary
        summary(model, input_size=input_size)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

if __name__ == '__main__':
    model = BWNet().cuda()
    summaries(model, [(1, 8, 16, 16), (1, 1, 64, 64)], grad=True)