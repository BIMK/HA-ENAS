import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import create_classifier, DropPath
from timm.models.registry import register_model
from timm.models.resnet import drop_blocks
from timm.models.vision_transformer import _cfg

import torchvision
from torchinfo import summary
from hanet.ResConv import Conv
from hanet.ResAtt import Att
from hanet.MLP import Mixer
from core.config import cfg

import os


# config.load_configs()


def random_op_encoding(num_of_ops, layers):
    encodings = []
    for l in layers:
        encodings.append(np.random.randint(0, num_of_ops, l).tolist())
    return encodings


def op_encoding(ops, layers):
    encodings = []
    for l in layers:
        encodings.append((np.ones([l], dtype=int) * ops).tolist())
    return encodings


class BasicalBlock(nn.Module):
    def __init__(self, stage_id, dim, stride):
        super(BasicalBlock, self).__init__()

        # 卷积模块
        # conv里面的下采样
        downsample = None
        if stride != 1 or dim != cfg.CONV.CHANNELS[stage_id] * cfg.CONV.EXPANSION:
            downsample = nn.Sequential(
                nn.Conv2d(dim, cfg.CONV.CHANNELS[stage_id] * cfg.CONV.EXPANSION, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(cfg.CONV.CHANNELS[stage_id] * cfg.CONV.EXPANSION),
            )

        self.conv = Conv(inplanes=dim, planes=cfg.CONV.CHANNELS[stage_id], stride=stride, downsample=downsample)

        # 注意力模型
        self.att = Att(dim=dim, attn_dim=cfg.CONV.CHANNELS[stage_id], heads=cfg.ATT.HEADS[stage_id],
                       stride=stride)
        # mlp模块
        mid = stage_id
        if stride == 2 and stage_id != 0:
            mid = stage_id - 1

        self.mlp = Mixer(dim=dim, attn_dim=cfg.CONV.CHANNELS[stage_id], patch_dim=cfg.MLP.PATCH_DIMS[mid],
                         expansion_factor_token=cfg.MLP.EXP_TOKENS[mid], expansion_factor=cfg.MLP.EXP[mid],
                         stride=stride)

    def forward(self, x, code):  # forward(self,x code=0)
        if code == 0:
            x = self.conv(x)
        elif code == 1:
            x = self.att(x)
        elif code == 2:
            x = self.mlp(x)
        return x


def make_blocks(encoding):
    stages = []
    dim = cfg.CONV.CHANNELS[0]
    for stage_id, layer in enumerate(encoding):
        blocks = []
        stride = 1 if stage_id == 0 else 2

        num_blocks = len(layer)
        stage_name = f'layer{stage_id + 1}'
        for block_id in range(num_blocks):
            stride = stride if block_id == 0 else 1
            blocks.append(
                BasicalBlock(stage_id=stage_id, dim=dim, stride=stride))
            dim = cfg.CONV.CHANNELS[stage_id] * cfg.CONV.EXPANSION
        stages.append((stage_name, nn.Sequential(*blocks)))
    # print(stages)
    return stages


class SuperHanet(nn.Module):
    def __init__(self, n_classes=10):
        super(SuperHanet, self).__init__()
        self.inplanes = cfg.CONV.CHANNELS[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.encodings = [[0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
        self._make_blocks()

        self.num_features = 2048
        self.global_pool, self.fc = create_classifier(self.num_features, n_classes, pool_type='avg')
        # 初始化网络权重
        self.apply(self._init_weights)

    def forward(self, x):  # forward(self, x,encodings = [[0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0]])
        x = self.stem(x)
        for idx, arch in enumerate(self.layer1):
            x = arch(x, self.encodings[0][idx])
        for idx, arch in enumerate(self.layer2):
            x = arch(x, self.encodings[1][idx])
        for idx, arch in enumerate(self.layer3):
            x = arch(x, self.encodings[2][idx])
        for idx, arch in enumerate(self.layer4):
            x = arch(x, self.encodings[3][idx])
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

    def _make_blocks(self):
        stage_modules = make_blocks(self.encodings)
        for stage in stage_modules:
            self.add_module(*stage)

    # 输出当前网络的编码信息
    def get_codings(self):
        print(self.encodings)

    # 分别获取全conv,att,mlp的网络
    def set_conv_net(self):
        self._sample_active_subnet(conv_net=True)

    def set_att_net(self):
        self._sample_active_subnet(att_net=True)

    def set_mlp_net(self):
        self._sample_active_subnet(mlp_net=True)

    def set_random_net(self):
        self._sample_active_subnet()

    def set_net(self, codings):
        self.encodings = codings

    def _sample_active_subnet(self, conv_net=False, att_net=False, mlp_net=False):
        if conv_net:
            self.encodings = op_encoding(0, cfg.SUPERNET_CFG.RATIO)
        elif att_net:
            self.encodings = op_encoding(1, cfg.SUPERNET_CFG.RATIO)
        elif mlp_net:
            self.encodings = op_encoding(2, cfg.SUPERNET_CFG.RATIO)
        else:
            self.encodings = random_op_encoding(3, cfg.SUPERNET_CFG.RATIO)

        # self._make_blocks()


def model_info(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = model.to(device)
    summary(backbone, (1, 3, 224, 224))


def save_test(model):
    model_state_dict = model.state_dict()
    state = {
        'state_dict': model_state_dict,
    }
    torch.save(state, 'ckpt.pth')


def ckp_ckeck():
    checkpoint_path = 'ckpt.pth'
    cpt = torch.load(checkpoint_path)['state_dict']
    for key in cpt:
        print(key)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    snet = SuperHanet().to(device)
    x = torch.randn((1, 3, 224, 224)).to(device)
    y = snet(x)
    print(y.shape)
    snet.get_codings()

    snet.set_conv_net()
    snet.get_codings()
    model_info(snet)
    y = snet(x)
    print(y.shape)

    snet.set_att_net()
    snet.get_codings()
    model_info(snet)
    y = snet(x)
    print(y.shape)

    snet.set_mlp_net()
    snet.get_codings()
    model_info(snet)
    y = snet(x)
    print(y.shape)

    snet.set_random_net()
    snet.get_codings()
    model_info(snet)
    y = snet(x)
    print(y.shape)
