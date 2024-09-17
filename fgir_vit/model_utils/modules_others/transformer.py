import math
import numpy as np

from einops import repeat, rearrange
import torch
from torch import nn
from torch.nn import functional as F

from .drop_path import DropPath
from .adapters import Adapter, ILAdapter


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class AllYouNeedAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout, proj_dim=None):
        super().__init__()
        if not proj_dim:
            proj_dim = dim

        self.proj_q = nn.Linear(dim, proj_dim)
        self.proj_k = nn.Linear(dim, proj_dim)
        self.proj_v = nn.Linear(dim, proj_dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads

    def forward(self, x, mask=None, context=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        if context is None:
            q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        else:
            q, k, v = self.proj_q(x), self.proj_k(context), self.proj_v(context)

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        # this is what's used to visualize attention
        scores_soft = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores_soft @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)

        return h, scores_soft, scores


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class BlockVanilla(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, hidden_dropout_prob,
                 attention_probs_dropout_prob, layer_norm_eps, sd=0,
                 adapter=None, adapter_dim=8, adapter_kernel_size=3,
                 adapter_path='pfeiffer', scale=1.0):
        super().__init__()
        self.attn = AllYouNeedAttention(dim, num_heads, attention_probs_dropout_prob)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)

        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)

        if sd > 0:
            self.drop = DropPath(sd)
        else:
            self.drop = nn.Dropout(hidden_dropout_prob)

        if adapter:
            # https://arxiv.org/abs/2005.00247
            self.adapter_path = adapter_path
            self.scale = scale
            self.adapter_pwffn = Adapter(adapter, dim, adapter_dim, adapter_kernel_size)
            if self.adapter_path in ('convpass'):
                self.adapter_attn = Adapter(adapter, dim, adapter_dim, adapter_kernel_size)
 
    def forward(self, x, prompt_len=0):
        if hasattr(self, 'adapter_path') and self.adapter_path in ('convpass', 'ours'):
            res = x
            x = self.norm1(x)

            if prompt_len > 0:
                h, scores_soft, scores = self.attn(x, context=x[:, prompt_len:])
            else:
                h, scores_soft, scores = self.attn(x)
            h = self.drop(self.proj(h))

            if hasattr(self, 'adapter_attn'):
                h2 = self.drop(self.adapter_attn(x))
                x = res + h + (self.scale * h2)
            else:
                x = res + h

            res = x
            x = self.norm2(x)

            h = self.drop(self.pwff(x))

            if hasattr(self, 'adapter_pwffn'):
                h2 = self.drop(self.adapter_pwffn(x))
                x = res + h + (self.scale * h2)
            else:
                x = res + h

        else:
            res = x

            if prompt_len > 0:
                h = self.norm1(x)
                h, scores_soft, scores = self.attn(h, context=h[:, prompt_len:])
            else:
                h, scores_soft, scores = self.attn(self.norm1(x))
            h = self.drop(self.proj(h))

            if hasattr(self, 'adapter_attn'):
                h2 = self.drop(self.adapter_attn(res + h, scores_soft))
                x = res + h + (self.scale * h2)
            else:
                x = res + h

            res = x

            h = self.drop(self.pwff(self.norm2(x)))

            if hasattr(self, 'adapter_pwffn'):
                h2 = self.drop(self.adapter_pwffn(res + h))
                x = res + h + (self.scale * h2)
            else:
                x = res + h

        return x, scores_soft, scores


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, num_layers=12, dim=768, num_heads=12, ff_dim=768*4,
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.0,
                 layer_norm_eps=1e-12, sd=0, ret_inter=False, cls=True,
                 adapter=None, adapter_dim=8,
                 adapter_kernel_size=3, adapter_path='pfeiffer',
                 ila=False, ila_locs=None, ila_ds_locs=None, ila_cls_conv=False,
                 ila_ds_conv=None, ila_ds_conv_type='dws_near_ones_init',
                 ila_ds_kernel_size=3,
                 ila_norm1=None, ila_norm2=None, ila_norm3=None,
                 ila_padding=False, ila_dilation=1, ila_dws_conv_groups=1,
                 ila_sd=0.0, prompt=None, prompt_len=None, patch_size=16):
        super().__init__()
        self.ret_inter = ret_inter
            
        self.blocks = nn.ModuleList([
            BlockVanilla(
                dim, num_heads, ff_dim, hidden_dropout_prob, attention_probs_dropout_prob,
                layer_norm_eps, sd, adapter, adapter_dim,adapter_kernel_size,
                adapter_path) for _ in range(num_layers)])

        if ila:
            self.ila_locs = ila_locs
            if self.ila_locs:
                self.ila = nn.ModuleList([
                    ILAdapter(
                        dim, adapter_dim, adapter_kernel_size, cls,
                        ila_cls_conv, False, ila_ds_conv_type,
                        ila_norm1, ila_norm2, ila_norm3, True,
                        ila_dilation, ila_dws_conv_groups, ila_sd, hidden_dropout_prob
                    )
                for _ in range(len(ila_locs))])

            self.ila_ds_locs = ila_ds_locs
            if self.ila_ds_locs:
                self.ila_ds = nn.ModuleList([
                    ILAdapter(
                        dim, adapter_dim, ila_ds_kernel_size, cls,
                        ila_cls_conv, ila_ds_conv, ila_ds_conv_type,
                        ila_norm1, ila_norm2, ila_norm3, ila_padding,
                        ila_dilation, ila_dws_conv_groups, ila_sd, hidden_dropout_prob
                    )
                for _ in range(len(ila_ds_locs))])

        if prompt == 'vpt_deep' or prompt == 'vqt':
            if prompt == 'vqt':
                self.prompt_len = prompt_len
            self.prompt = nn.Parameter(torch.zeros(num_layers, prompt_len, dim))
            self.prompt_dropout = nn.Dropout(hidden_dropout_prob)
            val = math.sqrt(6. / float(3 * (patch_size ** 2) + dim))
            nn.init.uniform_(self.prompt.data, -val, val)

    def forward(self, x, vis=False):
        scores_soft_list = []
        scores_list = []
        inter = []
        level_ila = 0
        level_ila_ds = 0

        b, seq_len, _ = x.shape

        for i, block in enumerate(self.blocks):

            if hasattr(self, 'prompt'):
                prompts = repeat(self.prompt[i, :, :], 's d -> b s d', b=b)
                prompts = self.prompt_dropout(prompts)
                if hasattr(self, 'prompt_len'):
                    x = torch.cat((prompts, x), dim=1)
                else:
                    x = torch.cat((x[:, :1, :], prompts, x[:, -(seq_len - 1):, :]), dim=1)

            x, scores_soft, scores = block(x, getattr(self, 'prompt_len', 0))
            # print(i, x.shape)

            if hasattr(self, 'prompt') and hasattr(self, 'prompt_len'):
                vqt = x[:, :self.prompt_len]
                vqt = F.normalize(rearrange(vqt, 'b s d -> b (s d)'), dim=-1)
                x = x[:, self.prompt_len:]
            elif hasattr(self, 'prompt'):
                x = torch.cat((
                    x[:, :1, :],
                    x[:, -(seq_len - 1):, :]
                ), dim=1)

            if hasattr(self, 'ila') and (i in self.ila_locs):
                x = self.ila[level_ila](x)
                level_ila += 1
            elif hasattr(self, 'ila_ds') and (i in self.ila_ds_locs):
                x = self.ila_ds[level_ila_ds](x)
                level_ila_ds += 1

            if self.ret_inter or vis:
                inter.append(x)
                scores_soft_list.append(scores_soft)
                scores_list.append(scores)
            elif hasattr(self, 'prompt_len'):
                inter.append(vqt)
                scores_soft_list.append(None)
                scores_list.append(None)
            else:
                inter.append(None)
                scores_soft_list.append(None)
                scores_list.append(None)

        if self.ret_inter:
            return inter, scores_soft_list, scores_list, _
        return x, inter, scores_soft_list, scores_list
