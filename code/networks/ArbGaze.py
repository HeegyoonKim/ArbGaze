from turtle import back
import torch
import torch.nn as nn

from .FA_module import SA_Adapt
from .baselines import resnet10, resnet18, resnet34, vgg19_bn


class ArbGaze(nn.Module):
    def __init__(
        self, baseline='resnet18', SA_adapt=False, num_experts=4,
        fine_tuning=False, teacher_path=None
    ):
        super(ArbGaze, self).__init__()

        self.baseline = baseline
        self.SA_adapt = SA_adapt
        self.num_experts = num_experts
        self.fine_tuning = fine_tuning
        self.teacher_path = teacher_path

        if self.baseline == 'resnet18':
            baseline = resnet18()
        elif self.baseline == 'resnet34':
            baseline = resnet34()
        elif self.baseline == 'resnet10':
            baseline = resnet10()
        elif self.baseline == 'vgg19':
            baseline = vgg19_bn()
        else:
            raise Exception('Invalid baseline model')
        
        self.make_modules(baseline)
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )
        if self.SA_adapt:
            self.FA_modules = nn.ModuleList()
            self.FA_modules.append(SA_Adapt(64, self.num_experts))
            self.FA_modules.append(SA_Adapt(128, self.num_experts))
            self.FA_modules.append(SA_Adapt(256, self.num_experts))
            self.FA_modules.append(SA_Adapt(512, self.num_experts))
    
    def make_modules(self, baseline):
        module_list = list(baseline.children())
        self.init_module = nn.Sequential()
        self.conv_modules = nn.ModuleList()

        if 'resnet' in self.baseline:
            self.init_module.add_module('0', nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))
            self.init_module.add_module('1', nn.BatchNorm2d(64))
            self.init_module.add_module('2', nn.ReLU(inplace=True))
            self.init_module.add_module('3', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.conv_modules.append(module_list[4])
            self.conv_modules.append(module_list[5])
            self.conv_modules.append(module_list[6])
            self.conv_modules.append(module_list[7])
        elif 'vgg' in self.baseline:
            self.init_module.add_module('0', nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1))
            self.init_module.add_module('1', nn.BatchNorm2d(64))
            self.init_module.add_module('2', nn.ReLU(inplace=True))
            self.conv_modules.append(nn.Sequential(*list(module_list[0][3:7])))
            self.conv_modules.append(nn.Sequential(*list(module_list[0][7:14])))
            self.conv_modules.append(nn.Sequential(*list(module_list[0][14:27])))
            self.conv_modules.append(nn.Sequential(*list(module_list[0][27:53])))
    
    def forward(self, x, scale):
        intermediate_features = []

        x = self.init_module(x)

        for n in range(len(self.conv_modules)):
            x = self.conv_modules[n](x)
            if self.SA_adapt:
                x = self.FA_modules[n](x, scale)
        intermediate_features.append(x.clone())
        
        x = self.GAP(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        
        return x, intermediate_features

if __name__ == '__main__':
    net = ArbGaze('resnet10', True, 4, False, None)
    input = torch.randn(7, 1, 36, 60)
    scale = torch.tensor([3.0]).view(1,1).float()
    print(net)
    # output, _ = net(input, scale)
    # print(output.size())
    # net = models.vgg19_bn()
    # print(net)