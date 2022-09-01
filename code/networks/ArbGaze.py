import torch
import torch.nn as nn

from .FA_module import SA_Adapt
from .backbones import resnet10, resnet18, resnet34, vgg13_bn, vgg16_bn, vgg19_bn


class ArbGaze(nn.Module):
    def __init__(self, backbone='resnet18', FA_module=False, num_experts=4, pretrained_path=None, p_dropout=0.0):
        super(ArbGaze, self).__init__()

        self.backbone = backbone
        self.FA_module = FA_module
        self.num_experts = num_experts
        self.pretrained_path = pretrained_path
        self.p_dropout = p_dropout

        if self.backbone == 'resnet18':
            backbone = resnet18()
        elif self.backbone == 'resnet34':
            backbone = resnet34()
        elif self.backbone == 'resnet10':
            backbone = resnet10()
        elif self.backbone == 'vgg13_bn':
            backbone = vgg13_bn()
        elif self.backbone == 'vgg16_bn':
            backbone = vgg16_bn()
        elif self.backbone == 'vgg19_bn':
            backbone = vgg19_bn()
        else:
            raise Exception('Invalid backbone model')
        
        # backbone model
        self.make_modules(backbone)
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )
        if p_dropout > 0:
            self.dropout = nn.Dropout(p_dropout)

        if pretrained_path is not None:
            self.load_weights_from_teacher()
        
        # Add feature adaptation module
        if self.FA_module:
            self.FA_modules = nn.ModuleList()
            self.FA_modules.append(SA_Adapt(64, self.num_experts))
            self.FA_modules.append(SA_Adapt(128, self.num_experts))
            self.FA_modules.append(SA_Adapt(256, self.num_experts))
            self.FA_modules.append(SA_Adapt(512, self.num_experts))
    
    def make_modules(self, backbone):
        module_list = list(backbone.children())
        self.init_module = nn.Sequential()
        self.conv_modules = nn.ModuleList()

        if 'resnet' in self.backbone:
            self.init_module = nn.Sequential(*module_list[0:4])
            for m in range(4, 8):
                self.conv_modules.append(module_list[m])
        elif 'vgg' in self.backbone:
            self.init_module = nn.Sequential(*list(module_list[0][0]))
            for m in range(1, 5):
                self.conv_modules.append(nn.Sequential(*list(module_list[0][m])))
    
    def load_weights_from_teacher(self):
        pretrained = torch.load(self.pretrained_path).copy()
        self.load_state_dict(pretrained['state_dict'])
    
    def forward(self, x, scale):
        intermediate_feats = []

        x = self.init_module(x)

        for n in range(len(self.conv_modules)):
            x = self.conv_modules[n](x)
            if self.p_dropout > 0:
                x = self.dropout(x)
            if self.FA_module:
                x = self.FA_modules[n](x, scale)
            intermediate_feats.append(x.clone())
        
        x = self.GAP(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        
        return x, intermediate_feats

if __name__ == '__main__':
    net = ArbGaze('vgg19_bn', False, 4, None, 0.0)
    input = torch.randn(7, 1, 36, 60)
    scale = torch.tensor([3.0]).view(1,1).float()
    output, feats = net(input, scale)