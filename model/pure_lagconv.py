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

class LAConv2D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(LAConv2D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias
        
        # Generating local adaptive weights
        self.attention1=nn.Sequential(
            nn.Conv2d(in_planes, kernel_size**2, kernel_size, stride, padding),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(kernel_size**2,kernel_size**2,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),
            nn.Sigmoid()
        )
        if use_bias: # Global local adaptive weights
            self.attention3=nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_planes,out_planes,1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_planes,out_planes,1)
            )

        conv1=nn.Conv2d(in_planes,out_planes,kernel_size,stride,padding,dilation,groups)
        self.weight=conv1.weight # m, n, k, k


    def forward(self,x):
        (b, n, H, W) = x.shape
        m=self.out_planes
        k=self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1=self.attention1(x) #b,k*k,n_H,n_W
        #atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        atw1=atw1.permute([0,2,3,1]) #b,n_H,n_W,k*k
        atw1=atw1.unsqueeze(3).repeat([1,1,1,n,1]) #b,n_H,n_W,n,k*k
        atw1=atw1.view(b,n_H,n_W,n*k*k) #b,n_H,n_W,n*k*k

        #atw2=atw2.permute([0,2,3,1]) #b,n_H,n_W,n*k*k

        atw=atw1#*atw2 #b,n_H,n_W,n*k*k
        atw=atw.view(b,n_H*n_W,n*k*k) #b,n_H*n_W,n*k*k
        atw=atw.permute([0,2,1]) #b,n*k*k,n_H*n_W

        kx=F.unfold(x,kernel_size=k,stride=self.stride,padding=self.padding) #b,n*k*k,n_H*n_W
        atx=atw*kx #b,n*k*k,n_H*n_W

        atx=atx.permute([0,2,1]) #b,n_H*n_W,n*k*k
        atx=atx.view(1,b*n_H*n_W,n*k*k) #1,b*n_H*n_W,n*k*k

        w=self.weight.view(m,n*k*k) #m,n*k*k
        w=w.permute([1,0]) #n*k*k,m
        y=torch.matmul(atx,w) #1,b*n_H*n_W,m
        y=y.view(b,n_H*n_W,m) #b,n_H*n_W,m
        if self.bias==True:
            bias=self.attention3(x) #b,m,1,1
            bias=bias.view(b,m).unsqueeze(1) #b,1,m
            bias=bias.repeat([1,n_H*n_W,1]) #b,n_H*n_W,m
            y=y+bias #b,n_H*n_W,m

        y=y.permute([0,2,1]) #b,m,n_H*n_W
        y=F.fold(y,output_size=(n_H,n_W),kernel_size=1) #b,m,n_H,n_W
        return y


# LAC_ResBlocks
class LACRB(nn.Module):
    def __init__(self, in_planes, ms_dim):
        super(LACRB, self).__init__()
        self.conv1=LAConv2D(in_planes,in_planes,3,1,1,use_bias=True)
        self.relu1=nn.LeakyReLU(inplace=True)
        self.conv2=LAConv2D(in_planes,in_planes,3,1,1,use_bias=True)

    def forward(self,x):
        res=self.conv1(x)
        res=self.relu1(res)
        res=self.conv2(res)
        x=x+res
        return x
    
class BWNet(nn.Module):
    def __init__(self, pan_dim=1, ms_dim=8, channel=32, num_lacrbs=5):
        super().__init__()

        self.ms_dim = ms_dim
        self.raise_dim = nn.Sequential(
            LAConv2D(pan_dim + ms_dim, channel, 3, 1, 1, use_bias=True),
            nn.LeakyReLU(inplace=True)
        )
        self.layers = nn.ModuleList([])
        for _ in range(num_lacrbs):
            self.layers.append(LACRB(channel, ms_dim))
        self.to_output = nn.Sequential(
            LAConv2D(channel, ms_dim, 3, 1, 1, use_bias=True)
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






