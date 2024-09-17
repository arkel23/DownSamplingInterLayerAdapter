import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat, rearrange

from .modules_others import Transformer, LearnedPositionalEmbedding1D
from .modules_fgir import Heads
from .misc import load_pretrained_weights


class ViT(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, out_channels=config.hidden_size,
            kernel_size=(config.fh, config.fw), stride=(config.patch_stride, config.patch_stride))

        # Class token
        if 'cls' in config.classifier:
            self.class_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Positional embedding
        if config.pos_embedding_type == 'learned':
            self.positional_embedding = LearnedPositionalEmbedding1D(
                config.seq_len, config.hidden_size, config.patch_size[0], config.prompt, config.prompt_len)

        # Transformer encoder
        self.ret_inter = 'ifa' in config.classifier
        self.encoder = Transformer(
            num_layers=config.num_hidden_layers,
            dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            ff_dim=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            sd=config.sd,
            ret_inter=('ifa' in config.classifier),
            cls=('cls' in config.classifier),
            adapter=config.adapter,
            adapter_dim=config.adapter_dim,
            adapter_kernel_size=config.adapter_kernel_size,
            adapter_path=config.adapter_path,
            ila=config.ila,
            ila_locs=config.ila_locs,
            ila_ds_locs=config.ila_ds_locs,
            ila_cls_conv=config.ila_cls_conv,
            ila_ds_conv=config.ila_ds_conv,
            ila_ds_conv_type=config.ila_ds_conv_type,
            ila_ds_kernel_size=config.ila_ds_kernel_size,
            ila_norm1=config.ila_norm1,
            ila_norm2=config.ila_norm2,
            ila_norm3=config.ila_norm3,
            ila_padding=config.ila_padding,
            ila_dilation=config.ila_dilation,
            ila_dws_conv_groups=config.ila_dws_conv_groups,
            ila_sd=config.ila_sd,
            prompt=config.prompt,
            prompt_len=config.prompt_len,
            patch_size=config.patch_size[0]
        )

        if config.encoder_norm:
            self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Classifier head
        if config.load_fc_layer:
            self.classifier = config.classifier
            self.head = Heads(config)

        if config.prompt == 'vqt':
            self.vqt = True

        # Initialize weights
        self.init_weights()

        if pretrained:
            load_pretrained_weights(self, config, config.model_name)

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        if hasattr(self, 'positional_embedding'):
            nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)
        if hasattr(self, 'class_token'):
            nn.init.constant_(self.class_token, 0)

    def encode_images(self, x, ret_dist=False):
        x = self.patch_embedding(x)
        x = rearrange(x, 'b d gh gw -> b (gh gw) d')
        b, _, _ = x.shape

        if hasattr(self, 'class_token'):
            cls_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)  # b s+1 d

        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)

        if self.ret_inter:
            inter, scores_soft, scores, _ = self.encoder(x, vis=ret_dist)
            x = inter[-1]
        else:
            x, inter, scores_soft, scores = self.encoder(x, vis=ret_dist)

        if hasattr(self, 'encoder_norm'):
            x = self.encoder_norm(x)
        x_norm = x if ret_dist else None 

        if hasattr(self, 'vqt'):
            inter.append(F.normalize(x[:, 0], dim=-1))
            x = torch.cat(inter, dim=1)
        elif self.ret_inter:
            feats = [feats[:, 0] for feats in inter]
            x = torch.stack(feats, dim=1)

        return x, x_norm, inter, scores_soft, scores


    def forward(self, images, ret_dist=False):
        """
        x (tensor): b k c fh fw -> b s d
        """
        x, x_norm, inter, scores_soft, scores = self.encode_images(images, ret_dist=ret_dist)

        if hasattr(self, 'head'):
            x = self.head(x)

        return x

