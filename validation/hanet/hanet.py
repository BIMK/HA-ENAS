import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import create_classifier, DropPath
from timm.models.registry import register_model
from timm.models.resnet import drop_blocks
from timm.models.vision_transformer import _cfg
# from torchsummary import summary
import torchvision
from torchinfo import summary

from hanet.ResConv import Conv
from hanet.ResAtt import Att
from hanet.MLP import Mixer
import os


def make_blocks(dim, encoding=None, heads=[1, 2, 4, 8], drop_path=0.):
    stages = []
    channels = [64, 128, 256, 512]
    patch_dims = [3136, 784, 196, 49]
    expansion_factor_tokens = [0.25, 0.5, 4, 8]
    expansion_factors = [8, 4, 4, 2]

    for stage_id, layer in enumerate(encoding):
        blocks = []
        stride = 1 if stage_id == 0 else 2

        num_blocks = len(layer)
        stage_name = f'layer{stage_id + 1}'
        for block_id in range(num_blocks):
            stride = stride if block_id == 0 else 1
            # 0 ：-> 卷积模块
            if layer[block_id] == 0:
                downsample = None
                if stride != 1 or dim != channels[stage_id] * Conv.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(dim, channels[stage_id] * Conv.expansion, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(channels[stage_id] * Conv.expansion),
                    )
                blocks.append(Conv(dim, planes=channels[stage_id], stride=stride, downsample=downsample))


            # 1 ： -> 注意力模块
            elif layer[block_id] == 1:
                # print(dim, channels[stage_id], heads[stage_id], stride)
                blocks.append(Att(dim=dim, attn_dim=channels[stage_id], heads=heads[stage_id], stride=stride))


            elif layer[block_id] == 2:
                if stride == 2 and stage_id != 0:
                    patch_dim = patch_dims[stage_id - 1]
                    expansion_factor_token = expansion_factor_tokens[stage_id - 1]
                    expansion_factor = expansion_factors[stage_id - 1]
                else:
                    patch_dim = patch_dims[stage_id]
                    expansion_factor_token = expansion_factor_tokens[stage_id]
                    expansion_factor = expansion_factors[stage_id]

                blocks.append(Mixer(dim=dim, attn_dim=channels[stage_id], patch_dim=patch_dim,
                                    expansion_factor_token=expansion_factor_token, expansion_factor=expansion_factor,
                                    stride=stride))

            dim = channels[stage_id] * Conv.expansion

        stages.append((stage_name, nn.Sequential(*blocks)))
    return stages


class HaNet(nn.Module):
    def __init__(self, num_classes=10, encoding=None):
        super(HaNet, self).__init__()
        if encoding == None:
            encoding = [[0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 0]]

        self.inplanes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 卷积和注意力层，共四层
        stage_modules = make_blocks(64, encoding=encoding)
        for stage in stage_modules:
            self.add_module(*stage)

        self.num_features = 2048
        self.global_pool, self.fc = create_classifier(self.num_features, num_classes, pool_type='avg')
        # 初始化网络权重
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x

    # 初始化网络权重
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

def HA_Net_A1(pretrained=False, **kwargs):
    encodings = [[2, 1, 2], [0, 1, 2, 2], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
    model = HaNet(encoding=encodings)
    if pretrained:
        checkpoint_path = 'checkpoints/hanet_a1.pth'
        checkpoint = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(checkpoint)
    print(model)
    return model

if __name__ == '__main__':
    model = HA_Net_A1(pretrained=False)
    print(model)
    x = torch.randn((1, 3, 224, 224))
    y = model(x)
    print(y.shape)

