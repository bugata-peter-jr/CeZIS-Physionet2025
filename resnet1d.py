# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:17:19 2020

@author: bugata
"""

import torch
from torch.nn import Conv1d, ReLU, MaxPool1d, AdaptiveAvgPool1d, Flatten
import  torch.nn.functional as F
from pytorch_model_summary import summary

# priestorovy dropout
class SpatialDropout(torch.nn.Module):
    
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.drop = torch.nn.Dropout2d(dropout_rate) 

    def forward(self, x):
        if self.dropout_rate <= 0:
            return x
        
        x = x.unsqueeze(-1)  # convert to [batch, channels, time, 1]
        x = self.drop(x)
        x = x.squeeze(-1)    # convert to [batch, channels, time]
        
        return x

class LayerNorm(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.eps = eps

    def forward(self, x):    # N, C, L
        mean = torch.mean(x, (1, 2), keepdim=True)
        var  = torch.var(x, (1, 2), unbiased=False, keepdim=True)        
        x = (x - mean)/torch.sqrt(var + self.eps)
        return x * self.gamma + self.beta

# vseobecne normalizacie pre konvolucnu 1D siet
# predpoklada, ze kanaly su na dim=1, t.j. N,C,L        
class Normalization(torch.nn.Module):
    
    def __init__(self, channels, norm_type='bn', num_groups=16):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == 'bn':
            self.norm = torch.nn.BatchNorm1d(channels, affine=True, track_running_stats=True)             
        elif norm_type == 'in':
            self.norm = torch.nn.InstanceNorm1d(channels, affine=True,track_running_stats=False) 
        elif norm_type == 'ln':    
            self.norm = LayerNorm() 
        elif norm_type == 'gn':       
            self.norm = torch.nn.GroupNorm(num_groups, channels, affine=True) 

    def forward(self, x):
        #print('before norm:', x.shape)   # (N,C,S)        
        return self.norm(x)

class ResBlock(torch.nn.Module):
    # ResBlock places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.    
    def __init__(self, in_channels, channels, out_channels, kernel=3, stride=1, norm_type='bn', num_groups=16, dropout_rate=0.0):
        super().__init__()

        # ci pouzit bias
        use_bias = norm_type not in ['bn', 'in']

        self.conv1 = Conv1d(in_channels, channels, kernel_size=1, bias=use_bias, stride=1)
        self.bn1   = Normalization(channels=channels, norm_type=norm_type, num_groups=num_groups)
        self.relu1 = ReLU(inplace=True)
        self.drop1 = SpatialDropout(dropout_rate)
        self.conv2 = Conv1d(channels, channels, kernel_size=kernel, bias=use_bias, stride=stride, padding=int((kernel-1)/2))  # strided conv - downsample
        self.bn2   = Normalization(channels=channels, norm_type=norm_type, num_groups=num_groups)
        self.relu2 = ReLU(inplace=True)
        self.drop2 = SpatialDropout(dropout_rate)
        self.conv3 = Conv1d(channels, out_channels, kernel_size=1, bias=use_bias, stride=1)
        self.bn3   = Normalization(channels=out_channels, norm_type=norm_type, num_groups=num_groups)
        
        if in_channels != out_channels:
            self.downsample = Conv1d(in_channels, out_channels, kernel_size=1, bias=use_bias, stride=stride)
            self.bn_down    = Normalization(channels=out_channels, norm_type=norm_type, num_groups=num_groups)    
        else:
            self.downsample = None
            self.bn_down    = None

        self.relu3 = ReLU(inplace=True)
        self.drop3 = SpatialDropout(dropout_rate)

        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn_down(identity)

        out += identity
        out = self.relu3(out)
        out = self.drop3(out)

        return out        


class ResLayer(torch.nn.Module):
    def __init__(self, n_blocks, in_channels, channels, kernel=3, stride=2, norm_type='bn', num_groups=16, dropout_rate=0.0):
        super().__init__()
        expansion = 4
        out_channels = channels * expansion

        resblocks = []
        b1 = ResBlock(in_channels, channels, out_channels, kernel=kernel, stride=stride, norm_type=norm_type, num_groups=num_groups, dropout_rate=dropout_rate)
        resblocks.append(b1)
        for i in range(n_blocks-1):
            b = ResBlock(out_channels, channels, out_channels, kernel=kernel, stride=1, norm_type=norm_type, num_groups=num_groups, dropout_rate=dropout_rate)
            resblocks.append(b)
        self.module_list = torch.nn.ModuleList(resblocks)
        
    def forward(self, x):
        o = x
        for module in self.module_list:
            o = module(o)
        return o
    

class ResNet(torch.nn.Module):
    def __init__(self, in_channels, kernel=9, norm_type='bn', num_groups=16, dropout_rate=0.0, width=1):
        super().__init__()
    
        # ci pouzit bias
        use_bias = norm_type not in ['bn', 'in']
        ch = int(64*width)
        
        self.conv1 = Conv1d(in_channels, ch, kernel_size=2*kernel+1, bias=use_bias, stride=2, padding=kernel)
        self.bn1   = Normalization(channels=ch, norm_type=norm_type, num_groups=num_groups)
        self.relu1 = ReLU(inplace=True)
        self.drop1 = SpatialDropout(dropout_rate)
        self.pool1 = MaxPool1d(kernel_size=kernel, stride=2, padding=int((kernel-1)/2))
        
        self.layer1 = ResLayer(3, in_channels= ch*1, channels=ch*1, kernel=kernel, norm_type=norm_type, num_groups=num_groups, dropout_rate=dropout_rate, stride=1)
        self.layer2 = ResLayer(4, in_channels= ch*4, channels=ch*2, kernel=kernel, norm_type=norm_type, num_groups=num_groups, dropout_rate=dropout_rate, stride=4)
        self.layer3 = ResLayer(6, in_channels= ch*8, channels=ch*4, kernel=kernel, norm_type=norm_type, num_groups=num_groups, dropout_rate=dropout_rate, stride=4)
        self.layer4 = ResLayer(3, in_channels=ch*16, channels=ch*8, kernel=kernel, norm_type=norm_type, num_groups=num_groups, dropout_rate=dropout_rate, stride=4)
        
        self.avgpool = AdaptiveAvgPool1d(1)
        self.flatten = Flatten()

        for m in self.modules():    
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)        
        
        return x
    
if __name__ == '__main__':
    inplen = 4096    
    channels = 8
        
    network = ResNet(in_channels=channels, kernel=9, norm_type='bn', num_groups=None, dropout_rate=0.0, width=1/4)

    summary(network, torch.zeros((1, channels, inplen)), show_hierarchical=True, 
            print_summary=True, show_parent_layers=True, max_depth=None)
    
    

    