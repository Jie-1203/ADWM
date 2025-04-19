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






