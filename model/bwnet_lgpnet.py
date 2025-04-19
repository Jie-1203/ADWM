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

    def forward(self,x):
        res=self.conv1(x)
        res=self.relu1(res)
        res=self.conv2(res)
        res=self.relu2(res)
        res = torch.add(x, res)
        return res
    
class CovBlock(nn.Module):
    def __init__(self, feature_dimension, features_num, hidden_dim, dropout=0.05):
        super().__init__()

        self.cov_mlp = nn.Sequential(
            nn.Linear(feature_dimension, feature_dimension),
            nn.Dropout(dropout, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feature_dimension, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, features_num),
        )

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)

        cov = x.transpose(-2, -1) @ x
        cov_norm = torch.norm(x, p=2, dim=-2, keepdim=True)
        cov_norm = cov_norm.transpose(-2, -1) @ cov_norm
        cov /= cov_norm

        weight = self.cov_mlp(cov)
        return weight


class BandSelectBlock(nn.Module):
    def __init__(self, feature_dimension, features_num):
        super().__init__()

        self.CovBlockList = nn.ModuleList([])
        for _ in range(features_num):
            self.CovBlockList.append(CovBlock(feature_dimension, 1, round(feature_dimension * 0.6), 0))

        self.global_covblock = CovBlock(features_num, 1, features_num, 0)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature_maps):
        H = feature_maps[0].shape[2]
        W = feature_maps[0].shape[3]
        C_weights = []

        for feature_map, block in zip(feature_maps, self.CovBlockList):
            input = rearrange(feature_map, 'B C H W -> B (H W) C', H=H) / (H * W - 1)
            C_weights.append(block(input).squeeze_(-1))

        weight_matrix = torch.stack(C_weights, dim=1)  # B x features_num x C
        feature_maps = torch.stack(feature_maps, dim=1)  # B x features_num x C x H x W
        output = weight_matrix.unsqueeze_(-1).unsqueeze_(-1) * feature_maps # B x features_num x C x H x W

        global_weight = self.global_pool(feature_maps).squeeze_(-1).squeeze_(-1)  # B x features_num x C
        global_weight = F.softmax(self.global_covblock(global_weight.transpose_(-1, -2)), dim=-2) # B x features_num x 1

        output = torch.sum(output * global_weight.unsqueeze(-1).unsqueeze(-1), dim=1) # B x C x H x W
        return output


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
        self.bw_output = BandSelectBlock(channel2, num_blocks)
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
        output = self.bw_output(feature_list)

        return self.to_output(output) + ms


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