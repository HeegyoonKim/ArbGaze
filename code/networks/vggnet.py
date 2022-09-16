'''
https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
'''


import torch
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    layers.append([])
    layers.append([])
    layers.append([])
    layers.append([])
    layers.append([])
    module_idx = 0
    in_channels = 1
    for v in range(len(cfg)):
        if cfg[v] == 'M':
            layers[module_idx] += [nn.MaxPool2d(kernel_size=2, stride=2)]
            if in_channels != 512:
                layers[module_idx] = nn.Sequential(*layers[module_idx])
                module_idx += 1
        else:
            conv2d = nn.Conv2d(in_channels, cfg[v], kernel_size=3, padding=1)
            if batch_norm:
                layers[module_idx] += [conv2d, nn.BatchNorm2d(cfg[v]), nn.ReLU(inplace=True)]
            else:
                layers[module_idx] += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg[v]
        if v == 0:
            layers[module_idx] = nn.Sequential(*layers[module_idx])
            module_idx += 1
    layers[-1] = nn.Sequential(*layers[-1])
    return nn.Sequential(*[layers[i] for i in range(len(layers))])


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg, batch_norm):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm))
    return model

def vgg13_bn():
    return _vgg('B', True)

def vgg16_bn():
    return _vgg('D', True)

def vgg19_bn():
    return _vgg('E', True)


if __name__ == '__main__':
    net = vgg13_bn()
    print(net)