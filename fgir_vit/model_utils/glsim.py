import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

from .modules_others import Transformer, LearnedPositionalEmbedding1D
from .modules_fgir import GLSimCropDrop
from .misc import load_pretrained_weights


class ViTGLSim(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.debugging = config.debugging

        self.seq_len_post_reducer = config.seq_len_post_reducer
        self.selector = config.selector

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
                in_channels=config.num_channels, out_channels=config.hidden_size,
                kernel_size=(config.fh, config.fw), stride=(config.fh, config.fw))

        # Class token
        if 'cls' in config.classifier:
            self.class_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Positional embedding
        if config.pos_embedding_type == 'learned':
            self.positional_embedding = LearnedPositionalEmbedding1D(
                config.seq_len_ind, config.hidden_size)

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
            adapter=config.adapter,
            adapter_dim=config.adapter_dim,
        )

        if config.encoder_norm:
            self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # aux loss
        if config.classifier_aux:
            self.classifier_aux = config.classifier_aux
            if config.classifier_aux in ('cls', 'pool'):
                self.head_aux = nn.Sequential(
                    Rearrange('b k d -> (b k) d'),
                    nn.Linear(config.representation_size, config.num_classes)
                )
            elif config.classifier_aux in ('multi_cont'):
                self.head_aux = nn.Sequential(
                    Rearrange('b k d -> (b k) d'),
                    nn.Linear(config.representation_size, config.representation_size * 2),
                    Rearrange('(b k) d -> b k d', k=config.seq_len_post_reducer)
                )
            else:
                self.head_aux = Rearrange('b k d -> (b k) d')

        # reducer
        self.num_anchors_total = 2 if config.dynamic_anchor else 1

        if config.representation_size != config.hidden_size:
            reducer_resize = nn.Linear(config.hidden_size, config.representation_size)
        else:
            reducer_resize = nn.Identity()

        if config.reducer == 'cls':
            self.reducer = reducer_resize

        # aggregator
        if config.aggregator:
            self.aggregator = Transformer(
                    num_layers=config.aggregator_num_hidden_layers,
                    dim=config.representation_size,
                    num_heads=config.representation_size // 64,
                    ff_dim=config.representation_size * 4,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    layer_norm_eps=config.layer_norm_eps,
                    sd=config.sd,
            )
            self.aggregator_norm = nn.LayerNorm(config.representation_size, eps=config.layer_norm_eps)



        # Classifier head
        if config.load_fc_layer:
            if config.classifier in ('cls', 'avg_cls', 'max_cls'):
                self.head = nn.Linear(config.representation_size, config.num_classes)
                if config.classifier in ('avg_cls', 'max_cls'):
                    reduce = 'mean' if config.classifier == 'avg_cls' else 'max'
                    self.head_reducer = nn.Sequential(
                        self.head,
                        Reduce('b s d -> b d', reduction=reduce)
                    )

        if config.dynamic_anchor:
            # cropping module
            self.get_crops = GLSimCropDrop(config)
            self.selector_num_tokens = config.selector_num_tokens
            self.crop_mode = config.crop_mode

            if config.anchor_class_token:
                self.anchor_class_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

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
        if hasattr(self, 'anchor_class_token'):
            nn.init.constant_(self.anchor_class_token, 0)
        if hasattr(self, 'head'):
            nn.init.constant_(self.head.weight, 0)
            nn.init.constant_(self.head.bias, 0)

    def patchify_tokenize(self, x, crops=False):
        self.maybe_print('Before tokenizing: ', x.shape)

        x = self.patch_embedding(x)

        x = rearrange(x, 'b d gh gw -> b (gh gw) d')
        b, _, _ = x.shape

        if hasattr(self, 'class_token'):
            if crops and hasattr(self, 'anchor_class_token'):
                cls_tokens = repeat(self.anchor_class_token, '1 1 d -> b 1 d', b=b)
            else:
                cls_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)  # b s+1 d

        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)

        self.maybe_print('After tokenizing: ', x.shape)
        return x

    def forward_encoder(self, x, vis=False):
        self.maybe_print('Before encoder: ', x.shape)

        ret_attn = True if self.selector == 'rollout' else False
        x, inter, scores_soft, scores = self.encoder(x, vis=(vis or ret_attn))
        if hasattr(self, 'encoder_norm'):
            x = self.encoder_norm(x)

        self.maybe_print('After encoder: ', x.shape)
        return x, inter, scores_soft, scores

    def process_crops(self, x, x_crops):
        self.maybe_print('\nProcessing crops: ', x.shape, x_crops.shape)

        x_crops = self.patchify_tokenize(x_crops, crops=True)

        x_crops, _, _, _ = self.forward_encoder(x_crops)

        x = torch.cat([x, x_crops], dim=1)

        self.maybe_print('After concatenating: ', x.shape)
        return x

    def forward_reducer(self, x):
        if hasattr(self, 'reducer'):
            if hasattr(self, 'class_token'):
                x = rearrange(x, 'b (k s) d -> b k s d', k=self.num_anchors_total)
                x = self.reducer(x[:, :, 0, :])
            else:
                x = self.reducer(x)

        self.maybe_print('\nAfter reducer: ', x.shape)
        return x

    def forward_aggregator(self, x):
        x = self.forward_reducer(x)

        if hasattr(self, 'aggregator'):
            x, _, _, _= self.aggregator(x)
            if hasattr(self, 'aggregator_norm'):
                x = self.aggregator_norm(x)

        self.maybe_print('After aggregator: ', x.shape)

        if hasattr(self, 'head_aux'):
            x_aux = self.head_aux(x)
            self.maybe_print('After auxiliary head: ', x_aux.shape)
            return x, x_aux
        return x

    def classify(self, x):
        if hasattr(self, 'head') and hasattr(self, 'class_token'):
            if hasattr(self, 'head_aux') and self.classifier_aux == 'shared':
                x = rearrange(x, 'b s d -> (b s) d')
                x = self.head(x)
            elif hasattr(self, 'head_reducer'):
                x = self.head_reducer(x)
            else:
                x = x[:, 0, :]
                x = self.head(x)

        self.maybe_print('After classifier head: ', x.shape)
        return x

    def forward(self, x, ret_dist=False):
        """
        x (tensor): b c h w -> b s d
        """
        images = x

        x = self.patchify_tokenize(x)

        x, inter, scores_soft, scores = self.forward_encoder(x, vis=ret_dist)
        if ret_dist:
            x_norm = x

        if hasattr(self, 'get_crops'):
            images_crops = self.get_crops(x, scores_soft, images, self.selector_num_tokens, self.crop_mode)
            x = self.process_crops(x, images_crops)

        x = self.forward_aggregator(x)
        if hasattr(self, 'head_aux'):
            x, x_aux = x

        x = self.classify(x)

        if hasattr(self, 'head_aux') and self.classifier_aux == 'shared':
            x = rearrange(x, '(b s) k -> b s k', s=self.seq_len_post_reducer)
            x, x_aux = torch.split(x, [1, self.seq_len_post_reducer - 1], dim=1)
            x = rearrange(x, 'b 1 d -> b d')
            x_aux = rearrange(x_aux, 'b s d -> (b s) d')
            self.maybe_print('After classifier head with shared aux: ', x.shape, x_aux.shape)

        if ret_dist:
            if hasattr(self, 'get_crops'):
                return x, images_crops, x_norm, inter, scores_soft, scores
            else:
                return x, x_norm, inter, scores_soft, scores

        elif hasattr(self, 'head_aux') and hasattr(self, 'get_crops'):
            return x, x_aux, images_crops
        elif hasattr(self, 'head_aux'):
            return x, x_aux
        elif hasattr(self, 'get_crops'):
            return x, images_crops
        return x

    def maybe_print(self, *args):
        if self.debugging:
            print(*args)
