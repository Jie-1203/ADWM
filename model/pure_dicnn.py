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

class BWNet(nn.Module):
    def __init__(self, pan_dim=1, ms_dim=8, channel=32, num_lacrbs=3):
        super().__init__()

        self.ms_dim = ms_dim
        self.raise_dim = nn.Sequential(
            nn.Conv2d(pan_dim + ms_dim, channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(inplace=True)
        )
        self.layers = nn.ModuleList([])
        for _ in range(num_lacrbs):
            self.layers.append(nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.LeakyReLU(inplace=True)))
        self.to_output = nn.Sequential(
            nn.Conv2d(channel, ms_dim, 3, 1, 1)
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






