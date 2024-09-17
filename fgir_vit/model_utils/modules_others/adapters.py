from einops import rearrange
from einops.layers.torch import Rearrange

import torch
from torch import nn

from .drop_path import DropPath


class Adapter(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, adapter, dim, adapter_dim, kernel_size):  # , seq_len=197, attn_mask=False):
        super().__init__()

        self.adapter_down = nn.Linear(dim, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, dim)
        self.act = nn.GELU()

        if adapter == 'convpass':
            padding = kernel_size // 2
            # conv with kernel size 3, stride 1, padding=1
            self.adapter_conv = nn.Conv2d(adapter_dim, adapter_dim, kernel_size, 1, padding)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if hasattr(self, 'adapter_down'):
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.xavier_uniform_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)
        if hasattr(self, 'adapter_conv'):
            nn.init.xavier_uniform_(self.adapter_conv.weight)

    def forward(self, x):
        x = self.act(self.adapter_down(x))

        if hasattr(self, 'adapter_conv'):
            _, s, _ = x.shape
            h = int(s ** 0.5)
            cls = False if (h ** 2) == s else True

            if cls:
                x_cls, x_patches = torch.split(x, [1, s - 1], dim=1)
                x_cls = rearrange(x_cls, 'b 1 d -> b d 1 1')
                x_cls = self.adapter_conv(x_cls)
                x_cls = rearrange(x_cls, 'b d 1 1 -> b 1 d')

                x_patches = rearrange(x_patches, 'b (h w) d -> b d h w', h=h)
                x_patches = self.adapter_conv(x_patches)
                x_patches = rearrange(x_patches, 'b d h w -> b (h w) d')

                x = torch.cat([x_cls, x_patches], dim=1)
                x = self.act(x)
            else:
                x = rearrange(x, 'b (h w) d -> b d h w', h=h)
                x = self.adapter_conv(x)
                x = rearrange(x, 'b d h w -> b (h w) d')                
                x = self.act(x)

        x = self.adapter_up(x)
        return x


class ILAdapter(nn.Module):
    def __init__(
        self, dim=768, adapter_dim=8, kernel_size=3,
        cls=False, cls_conv=False, ds_conv=None, ds_conv_type='dws_near_ones_init',
        norm1=None, norm2=None, norm3=None,
        padding=False, dilation=1, groups=0,
        sd=0, hidden_dropout_prob=0.1,
    ):
        super().__init__()
        self.adapter_down = nn.Linear(dim, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, dim)
        self.act = nn.GELU()

        if norm1 == 'bn':
            self.norm1 = nn.Sequential(
                Rearrange('b s d -> b d s'),
                nn.BatchNorm1d(adapter_dim),
                Rearrange('b d s -> b s d'),
            )
        elif norm1 == 'ln':
            self.norm1 = nn.LayerNorm(adapter_dim)

        if norm2 == 'bn':
            norm2 = nn.Sequential(
                nn.BatchNorm2d(adapter_dim * groups),
                self.act,
            )
        elif norm2 == 'ln':
            norm2 = nn.Sequential(
                Rearrange('b d h w -> b h w d'),
                nn.LayerNorm(adapter_dim * groups),
                self.act,
                Rearrange('b h w d -> b d h w')
            )
        else:
            norm2 = nn.Identity()

        if norm3 == 'bn':
            self.norm3 = nn.Sequential(
                Rearrange('b s d -> b d s'),
                nn.BatchNorm1d(adapter_dim),
                Rearrange('b d s -> b s d')
            )
        elif norm3 == 'ln':
            self.norm3 = nn.LayerNorm(adapter_dim)

        if cls and not cls_conv:
            padding = True
        elif cls and cls_conv:
            self.adapter_cls = nn.Conv2d(adapter_dim, adapter_dim, (1, 1))

        if padding:
            padding = kernel_size // 2
            padding = padding + (dilation - 1)
        else:
            padding = 0

        if ds_conv and 'dws' in ds_conv_type:
            if ds_conv_type in ('dws_near_ones_init', 'dws_ones_init'):
                self.dws_init = ds_conv_type
            self.ds_conv = nn.Conv2d(dim, dim, kernel_size, 1, padding, groups=dim, bias=False)
        elif ds_conv and ds_conv_type == 'conv':
            self.ds_conv = nn.Conv2d(dim, dim, kernel_size, 1, padding, bias=False)
        elif ds_conv and ds_conv_type == 'avg_pool':
            self.ds_conv = nn.AvgPool2d(kernel_size, 1, padding)

        if groups > 0:
            self.dws_conv = True
            self.adapter_conv = nn.Sequential(
                nn.Conv2d(
                    adapter_dim,
                    adapter_dim * groups,
                    kernel_size=(kernel_size, kernel_size),
                    stride=1,
                    padding=(padding, padding),
                    dilation=(dilation, dilation),
                    groups=adapter_dim,
                ),
                norm2,
                nn.Conv2d(adapter_dim * groups, adapter_dim, 1, 1, 0),
            )
        else:
            self.adapter_conv = nn.Conv2d(
                adapter_dim,
                adapter_dim,
                kernel_size=(kernel_size, kernel_size),
                stride=1,
                padding=(padding, padding),
                dilation=(dilation, dilation),
            )

        if padding or ds_conv:
            if sd > 0:
                self.drop = DropPath(sd)
            else:
                self.drop = nn.Dropout(hidden_dropout_prob)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.xavier_uniform_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        if hasattr(self, 'dws_conv'):
            nn.init.xavier_uniform_(self.adapter_conv[0].weight)
            nn.init.xavier_uniform_(self.adapter_conv[-1].weight)
        else:
            nn.init.xavier_uniform_(self.adapter_conv.weight)

        if hasattr(self, 'ds_conv') and hasattr(self, 'dws_init') and self.dws_init == 'dws_near_ones_init':
            print('Near ones init for ds_conv')
            nn.init.normal_(self.ds_conv.weight, mean=1.0, std=0.001)
        elif hasattr(self, 'ds_conv') and hasattr(self, 'dws_init') and self.dws_init == 'dws_ones_init':
            print('Ones init for ds_conv')
            nn.init.ones_(self.ds_conv.weight)
        elif hasattr(self, 'ds_conv') and hasattr(self.ds_conv, 'weight'):
            print('Random init for ds_conv')
            nn.init.xavier_uniform_(self.ds_conv.weight)

        if hasattr(self, 'adapter_cls'):
            nn.init.xavier_uniform_(self.adapter_cls.weight)

    def forward(self, x):
        # tensor with shape b, s, d
        res = x

        # downsampling block with optional norm
        x = self.adapter_down(x)
        if hasattr(self, 'norm1'):
            x = self.norm1(x)
        x = self.act(x)

        _, s, _ = x.shape
        h = int(s ** 0.5)
        cls = False if (h**2) == s else True

        if cls:
            x_cls, x_patches = torch.split(x, [1, s - 1], dim=1)

            x_cls = rearrange(x_cls, "b 1 d -> b d 1 1")
            if hasattr(self, 'adapter_cls'):
                x_cls = self.adapter_cls(x_cls)
            else:
                x_cls = self.adapter_conv(x_cls)
            x_cls = rearrange(x_cls, "b d 1 1 -> b 1 d")

            x_patches = rearrange(x_patches, "b (h w) d -> b d h w", h=h)
            x_patches = self.adapter_conv(x_patches)
            x_patches = rearrange(x_patches, "b d h w -> b (h w) d")

            x = torch.cat([x_cls, x_patches], dim=1)
        else:
            x = rearrange(x, "b (h w) d -> b d h w", h=h)
            x = self.adapter_conv(x)
            x = rearrange(x, "b d h w -> b (h w) d")

        # upsampling block with optional norm
        if hasattr(self, 'norm3'):
            x = self.norm3(x)
        x = self.act(x)
        x = self.adapter_up(x)

        if hasattr(self, 'ds_conv'):
            if cls:
                res_cls, res_patches = torch.split(res, [1, res.shape[1] - 1], dim=1)
                res_patches = rearrange(res_patches, 'b (h w) d -> b d h w', h=h)
                res_patches = self.ds_conv(res_patches)
                res_patches = rearrange(res_patches, 'b d h w -> b (h w) d')
                res = torch.cat([res_cls, res_patches], dim=1)
            else:
                res = rearrange(res, 'b (h w) d -> b d h w', h=h)
                res = self.ds_conv(res)
                res = rearrange(res, 'b d h w -> b (h w) d')

        if hasattr(self, 'drop'):
            x = res + self.drop(x)
        return x


if __name__ == '__main__':
    x = torch.rand(2, 196, 768)

    convpass = Adapter()
    params_cp = sum([p.numel() for p in convpass.parameters()])
    # 13648

    iladapter = ILAdapter()
    params_lcp = sum([p.numel() for p in iladapter.parameters()])
    # 14800

    dwiladapter = ILAdapter(groups=2)
    params_dwlcp = sum([p.numel() for p in dwiladapter.parameters()])
    # 13648

