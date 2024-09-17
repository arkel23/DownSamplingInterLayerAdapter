import torch
from torch import nn
from einops import repeat, rearrange

from .modules_others import Transformer, LearnedPositionalEmbedding1D
from .misc import load_pretrained_weights
from .modules_fgir import PSM, MAWS


class ViTFG(nn.Module):
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
        self.encoder = Transformer(
            num_layers=config.num_hidden_layers,
            dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            ff_dim=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            sd=config.sd,
            ret_inter=True,
            adapter=config.adapter,
            adapter_dim=config.adapter_dim,
            prompt=config.prompt,
            prompt_len=config.prompt_len,
            patch_size=config.patch_size[0]
        )

        if config.selector == 'maws':
            self.selector = MAWS(num_token=config.selector_num_tokens)
        elif config.selector == 'psm':
            self.selector = PSM()

        self.aggregator = Transformer(
            num_layers=config.aggregator_num_hidden_layers,
            dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            ff_dim=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            sd=config.sd)

        if config.encoder_norm and config.selector == 'psm':
            self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        elif config.encoder_norm and config.selector == 'maws':
            self.aggregator_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Classifier head
        if config.load_fc_layer:
            self.head = nn.Linear(config.hidden_size, config.num_classes)

        if config.classifier_aux:
            self.head_aux = nn.Identity()

        # Initialize weights
        self.init_weights()

        if pretrained:
            load_pretrained_weights(self, config, 'vit_b16')

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
        if hasattr(self, 'head'):
            nn.init.constant_(self.head.weight, 0)
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        """
        x (tensor): b k c fh fw -> b s d
        """
        x = self.patch_embedding(x)
        x = rearrange(x, 'b d gh gw -> b (gh gw) d')
        b, _, _ = x.shape

        if hasattr(self, 'class_token'):
            cls_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)  # b s+1 d

        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)

        inter, attn_weights_soft, attn_weights, _ = self.encoder(x)

        x = self.selector(inter, attn_weights_soft, attn_weights)

        if hasattr(self, 'aggregator'):
            x, _, _, _ = self.aggregator(x)

        if hasattr(self, 'encoder_norm'):
            x = self.encoder_norm(x)
        elif hasattr(self, 'aggregator_norm'):
            x = self.aggregator_norm(x)

        if hasattr(self, 'head_aux'):
            x_aux = self.head_aux(x)[:, 0]

        if hasattr(self, 'head') and hasattr(self, 'class_token'):
            x = self.head(x[:, 0, :])

        if hasattr(self, 'head_aux'):
            return x, x_aux
        return x
