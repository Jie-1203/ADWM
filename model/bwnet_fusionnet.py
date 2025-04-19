import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat

def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class Resblock(nn.Module):
    def __init__(self, channel):
        super(Resblock, self).__init__()

        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv20(x))
        rs1 = self.conv21(rs1)
        rs = torch.add(x, rs1)
        return rs

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
            self.CovBlockList.append(CovBlock(feature_dimension, 1, feature_dimension, 0))

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

        global_weight = self.global_pool(feature_maps).squeeze_(-1).squeeze_(-1) # B x features_num x C
        global_weight = F.softmax(self.global_covblock(global_weight.transpose_(-1, -2)), dim=-2) # B x features_num x 1

        output = torch.sum(output * global_weight.unsqueeze(-1).unsqueeze(-1), dim=1) # B x C x H x W
        return output

class BWNet(nn.Module):
    def __init__(self, pan_dim=1, ms_dim=8, channel=32, num_blocks=4):
        super().__init__()

        self.ms_dim = ms_dim
        self.raise_dim = nn.Sequential(
            nn.Conv2d(ms_dim, channel, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.layers = nn.ModuleList([])
        for _ in range(num_blocks):
            self.layers.append(Resblock(channel))
        self.bw_output = BandSelectBlock(channel, num_blocks)
        self.to_output = nn.Sequential(
            nn.Conv2d(channel, ms_dim, 3, 1, 1)
        )

    def forward(self, ms, pan):
        ms = F.interpolate(ms, scale_factor=4, mode='bicubic')
        pan = pan.repeat(1, self.ms_dim, 1, 1)
        input = pan - ms

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






