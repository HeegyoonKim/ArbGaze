'''
https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class SA_Conv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
        super(SA_Conv, self).__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.num_experts = num_experts

        self.routing = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )

        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(out_c, in_c, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, dim=0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(8, out_c))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1/ math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)
    
    def forward(self, x, scale):
        # Generate routing weights
        scale = torch.ones(1,1).to(x.device) / scale
        routing_weights = self.routing(scale).view(self.num_experts, 1, 1)

        # Fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)    # size: (out_cXin_cXkXk) X 1
        fused_weight = fused_weight.view(-1, self.in_c, self.kernel_size, self.kernel_size)  # size: out_c X in_c X k X k

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None
        
        # Convolution
        out = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)

        return out


class SA_Adapt(nn.Module):
    def __init__(self, in_c, num_experts):
        super(SA_Adapt, self).__init__()

        self.in_c = in_c
        self.num_experts = num_experts

        self.mask = nn.Sequential(
            nn.Conv2d(in_c, in_c//4, 3, 1, 1),
            nn.BatchNorm2d(in_c//4),
            nn.ReLU(True),
            nn.Conv2d(in_c//4, in_c//16, 3, 1, 1),
            nn.BatchNorm2d(in_c//16),
            nn.ReLU(True),
            nn.Conv2d(in_c//16, in_c//16, 3, 1, 1),
            nn.BatchNorm2d(in_c//16),
            nn.ReLU(True),
            nn.Conv2d(in_c//16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.adapt = SA_Conv(in_c, in_c, 3, 1, 1, False, num_experts)
    
    def forward(self, x, scale):
        mask = self.mask(x)
        adapted = self.adapt(x, scale)

        return x + adapted * mask