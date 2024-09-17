import math
import torch
from torch import nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

from .matrix_sqrt import matrix_sqrt
from .mpncov import MPNCOV


class Heads(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.classifier = config.classifier

        if 'cls' in config.classifier:
            self.class_token = True

        if config.prompt == 'vqt':
            del self.class_token
            input_size = config.hidden_size + (config.num_hidden_layers * config.hidden_size * config.prompt_len)
            self.head = nn.Linear(input_size, config.num_classes)
        elif config.classifier == 'cls':
            self.head = nn.Linear(config.hidden_size, config.num_classes)
        elif config.classifier == 'pool':
            self.head = nn.Sequential(
                Reduce('b s d -> b d', 'mean'),
                nn.Linear(config.hidden_size, config.num_classes)
            )
        elif config.classifier == 'flatten':
            self.head = nn.Sequential(
                Rearrange('b s d -> b (s d)'),
                nn.Linear(config.hidden_size * config.seq_len, config.num_classes)
            )
        elif config.classifier == 'mpncov':
            # class proj size for vgg/resnet by def is 256 (512 for vgg / 2048 for rn -> 256)
            # it uses a classifier factor (5, 100, 1000) that increases lr by factor for classifier
            self.mpncov = MPNCOV(input_dim=config.hidden_size, dimension_reduction=config.class_proj_size)
            self.head = nn.Sequential(
                Rearrange('b d 1 -> b d'),
                nn.Linear((config.class_proj_size * (config.class_proj_size + 1)) // 2, config.num_classes)
            )
        elif config.classifier == 'lrblp':
            # https://www.ics.uci.edu/~skong2/img/LRBP_poster_v0.4.pdf
            self.lr_proj = nn.Linear(config.hidden_size, config.class_proj_size)
            self.blp_head = nn.Linear(config.class_proj_size * config.class_proj_size, config.num_classes)
        elif config.classifier == 'iblp':
            # https://arxiv.org/abs/1707.06772
            # https://github.com/DennisLeoUTS/improved-bilinear-pooling/
            # the improved norm can be sqrt matrix or log matrix (not element-wise)
            self.matrix_sqrt = matrix_sqrt.apply
            self.blp_head = nn.Linear(config.hidden_size ** 2, config.num_classes)
        elif config.classifier == 'blp':
            # https://github.com/HaoMood/blinear-cnn-faster/blob/master/src/model.py
            self.blp_head = nn.Linear(config.hidden_size ** 2, config.num_classes)
        elif config.classifier == 'ifacls':
            self.ifa_head = nn.Sequential(
                nn.Linear(config.num_hidden_layers, 1),
                Rearrange(' b c 1 -> b c'),
                nn.ReLU(inplace=True),
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                nn.Linear(config.hidden_size, config.num_classes)
            )

    def forward(self, x):
        if hasattr(self, 'mpncov'):
            # 2d input: b, c, h, w -> 1d output: b, dim_out*(dim_out+1)/2, 1
            x = rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.shape[1])))
            x = self.mpncov(x)
            x = self.head(x)

        elif hasattr(self, 'blp_head'):
            if hasattr(self, 'lr_proj'):
                x = self.lr_proj(x)

            x = torch.matmul(rearrange(x, 'b s d -> b d s'), x) / x.shape[1]

            if hasattr(self, 'matrix_sqrt'):
                # matrix square root
                x = self.matrix_sqrt(x)

            x = rearrange(x, 'b d1 d2 -> b (d1 d2)')
            # https://github.com/pascal-niklaus/pascal/blob/master/pascal/R/sgnsqrt.R
            x = torch.sign(x) * torch.sqrt(torch.abs(x)) # + 1e-5
            x = torch.nn.functional.normalize(x)

            x = self.blp_head(x)

        elif hasattr(self, 'head') and hasattr(self, 'class_token'):
            x = self.head(x[:, 0, :])

        elif hasattr(self, 'head'):
            x = self.head(x)

        elif hasattr(self, 'ifa_head'):
            x = rearrange(x, 'b l d -> b d l')
            x = self.ifa_head(x)

        return x
