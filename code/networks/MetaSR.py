'''
https://github.com/XuecaiHu/Meta-SR-Pytorch/blob/0.4.0/model/metardn.py
'''

import torch
import torch.nn as nn

import math


class RDB_Conv(nn.Module):
    def __init__(self, in_c, growth_rate, kernel_size=3):
        super(RDB_Conv, self).__init__()
        c_in = in_c
        G = growth_rate
        self.conv = nn.Sequential(*[
            nn.Conv2d(c_in, G, kernel_size, stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True)
        ])
    
    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growth_rate0, growth_rate, n_conv_layers, kernel_size=3):
        super(RDB, self).__init__()
        G0 = growth_rate0
        G = growth_rate
        C = n_conv_layers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        self.LFF = nn.Conv2d(G0 + C*G, G0, kernel_size=1, padding=0, stride=1)
    
    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class Pos2Weight(nn.Module):
    def __init__(self, in_c, kernel_size=3, out_c=1):
        super(Pos2Weight, self).__init__()
        self.in_c = in_c
        self.kernel_size = kernel_size
        self.out_c = out_c
        self.meta_block = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.kernel_size*self.kernel_size*self.in_c*self.out_c)
        )
    
    def forward(self, x):
        output = self.meta_block(x)
        return output


class MetaRDN(nn.Module):
    def __init__(self, D=16, C=8, G=64):
        super(MetaRDN, self).__init__()
        self.scale = 1.0
        self.D = D
        self.C = C
        self.G = G
 
        self.SFENet1 = nn.Conv2d(1, self.G, kernel_size=3, stride=1, padding=1)
        self.SFENet2 = nn.Conv2d(self.G, self.G, kernel_size=3, stride=1, padding=1)

        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growth_rate0=self.G, growth_rate=self.G, n_conv_layers=self.C)
            )
        
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G, self.G, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(self.G, self.G, kernel_size=3, padding=1, stride=1)
        ])

        self.P2W = Pos2Weight(in_c=self.G, out_c=1)
    
    def repeat_x(self, x):
        scale_int = math.ceil(self.scale)
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)

        x = torch.cat([x]*scale_int, 3)     # (N, C, H, r, W, 1)
        x = torch.cat([x]*scale_int, 5).permute(0, 3, 5, 1, 2, 4)   # (N, r, r, C, H, W)
        return x.contiguous().view(-1, C, H, W)
    
    def set_scale(self, scale):
        self.scale = scale
        
    def forward(self, x, pose_mat, scale):
        self.set_scale(scale)
        
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        local_weight = self.P2W(pose_mat.view(pose_mat.size(1), -1))
        # (out_H*out_W, in_C*out_C*3*3)

        up_x = self.repeat_x(x) # (N*r*r, in_c, in_H, in_W)

        cols = nn.functional.unfold(up_x, 3, padding=1) # (N*r*r, in_c*3*3, in_H*in_W   3=kernel_size)
        scale_int = math.ceil(self.scale)

        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()
        ###     (N, r*r, in_c*3*3, in_H*in_W, 1)                                                               # (N, r*r in_H*in_w, 1, in_c*3*3)

        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,1).permute(1,3,0,2,4,5).contiguous()
        ###     (in_H, r, in_W, r, in_c*3*3, out_c)                                                 # (r, r, in_H, in_W, in_c*3*3, out_c)
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, 1)
        # (r*r, in_H*in_W, in_c*3*3, out_c)

        #print(cols.size(), local_weight.size())
        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        # (N, r*r, in_H*in_W, 1, out_c)  # (N, r*r, out_c, in_H*in_W, 1)
        out = out.contiguous().view(x.size(0),scale_int,scale_int,1,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        # (N, r, r, out_c, in_H, in_W)                                                  # (N, out_c, in_H, r, in_W, r)
        out = out.contiguous().view(x.size(0),1, scale_int*x.size(2),scale_int*x.size(3))

        return out
    

if __name__ == '__main__':
    net = MetaRDN()
    print(net)