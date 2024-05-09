from torch import nn
import torch
import os

# 注意力部分
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=128):
        super(Attention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).view(b, 3 * self.heads, self.dim_head, h * w).chunk(3, dim=1)

        attn = q.transpose(-2, -1) @ k * self.scale
        attn = attn.softmax(dim=-1)

        out = (v @ attn.transpose(-2, -1)).reshape(b, -1, h, w)

        return out


# 条件位置编码部分，更多详情信息，百度CPVT
class PEG(nn.Module):
    def __init__(self, dim, stride):
        super(PEG, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim)

    def forward(self, x):
        return self.conv(x)


# 源码中在Att中还添加了升维，降维和残差连接等操作
class Att(nn.Module):
    def __init__(self, dim, attn_dim, heads=8, stride=1, act_layer=nn.ReLU):
        super(Att, self).__init__()
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

        # 位置编码部分
        self.postion = nn.Sequential(PEG(dim=attn_dim, stride=1),
                                     nn.BatchNorm2d(attn_dim))

        # 注意力部分
        self.net = nn.Sequential(
            activation,
            Attention(attn_dim, heads, dim_head=attn_dim // heads),
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

        x = x + self.postion(x)
        x = self.net(x)
        x = self.out(x)
        x += shortcut
        return self.activation(x)


class Convp(nn.Module):
    def __init__(self):
        super(Convp, self).__init__()
        self.conv = nn.Conv2d(256, 1024, 3, 1, padding=1)

    def forward(self, x):
        return self.conv(x)

