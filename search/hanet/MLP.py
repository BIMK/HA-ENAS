from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import torch

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


class MLP(nn.Module):
    def __init__(self, dim, num_patches, expansion_factor_token, expansion_factor, dropout=0.):
        super(MLP, self).__init__()
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.cross_location = PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first))
        self.pre_location = PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.view(b, c, h * w)
        tokens = self.cross_location(tokens)
        tokens = self.pre_location(tokens).reshape(b, -1, h, w)

        return tokens


class Mixer(nn.Module):
    def __init__(self, dim, attn_dim, patch_dim, expansion_factor_token, expansion_factor, stride=2, act_layer=nn.ReLU):
        super(Mixer, self).__init__()
        activation = act_layer(inplace=False)
        dim_out = attn_dim * 4

        # shortcut部分
        if stride == 2:
            # 需要升维和降分辨率
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        elif dim != dim_out:
            # 当att出现在第一阶段的第一个块时，才用到这个
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        # bottleneck的降维投影
        self.proj = nn.Sequential(nn.Conv2d(dim, attn_dim, 1, bias=False),
                                  nn.BatchNorm2d(attn_dim)
                                  )
        # MLP
        self.net = nn.Sequential(
            activation,
            MLP(patch_dim, attn_dim, expansion_factor_token, expansion_factor),
            nn.BatchNorm2d(attn_dim),
            activation,
        )

        # bottleneck 投影回来
        self.out = nn.Sequential(
            nn.Conv2d(attn_dim, dim_out, 1, stride=stride, bias=False),
            nn.BatchNorm2d(dim_out)
        )

        self.activation = activation

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.proj(x)

        x = self.net(x)
        x = self.out(x)
        x += shortcut
        return self.activation(x)


