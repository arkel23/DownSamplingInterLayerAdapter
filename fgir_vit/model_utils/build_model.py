from types import SimpleNamespace

import timm
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

from .vit import ViT
from .glsim import ViTGLSim
from .vit_fg import ViTFG
from .cal import CAL, CONFIGS_CAL
from .misc import ViTConfig
from .timm_vit import *
from .modules_fgir.simtrans import SIMTrans


VITS = (
    'vit_t4', 'vit_t8', 'vit_t16', 'vit_t32', 'vit_s8', 'vit_s16', 'vit_s32',
    'vit_b8', 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32', 'vit_h14')


def build_model(args):
    # initiates model and loss
    if args.model_name in VITS and args.dynamic_anchor:
        model = VisionTransformerGLSim(args)
    elif args.model_name in VITS and args.selector in ('psm', 'maws'):
        model = VisionTransformerFG(args)
    elif args.model_name in VITS and args.selector == 'sil':
        model = SIMTrans(args.model_name, args.image_size, args.num_classes)
    elif args.selector == 'cal':
        model = CounterfactualAttentionLearning(args)
    elif args.model_name in VITS:
        model = VisionTransformer(args)
    elif args.model_name in timm.list_models(pretrained=True):
        model = TIMMViT(args)
    else:
        raise NotImplementedError

    args.seq_len_post_reducer = model.cfg.seq_len_post_reducer

    if args.ckpt_path:
        state_dict = torch.load(
            args.ckpt_path, map_location=torch.device('cpu'))['model']
        expected_missing_keys = []
        if args.transfer_learning:
            # modifications to load partial state dict
            if ('model.head.weight' in state_dict):
                expected_missing_keys += ['model.head.weight', 'model.head.bias']
            for key in expected_missing_keys:
                state_dict.pop(key)
        ret = model.load_state_dict(state_dict, strict=False)
        print('''Missing keys when loading pretrained weights: {}
              Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
        print('Unexpected keys when loading pretrained weights: {}'.format(
            ret.unexpected_keys))
        print('Loaded from custom checkpoint.')

    if args.freeze_backbone:
        freeze_backbone(args, model)

    if args.distributed:
        model.cuda()
    else:
        model.to(args.device)

    print(f'Initialized classifier: {args.model_name}')
    return model


def freeze_backbone(args, model):
    keywords = ['head', 'aggregator', 'adapter', 'dfsm', 'feature_center', 'prompt', 'ila']

    if args.unfreeze_cls:
        keywords.append('class_token')
    if args.unfreeze_positional_embedding:
        keywords.append('positional_embedding')
    if args.unfreeze_encoder_norm:
        keywords.append('encoder_norm')
    if args.unfreeze_mhsa_proj:
        keywords.append('proj')
    if args.unfreeze_encoder_last_block:
        keywords.append(str(args.num_hidden_layers - 1))

    for name, param in model.named_parameters():
        if any(kw in name for kw in keywords):
            param.requires_grad = True
            # print(name)
        else:
            param.requires_grad = False

    print('Total parameters (M): ', sum([p.numel() for p in model.parameters()]) / (1e6))
    print('Trainable parameters (M): ', sum([p.numel() for p in model.parameters() if p.requires_grad]) / (1e6))


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        # init default config
        cfg = ViTConfig(model_name=args.model_name)
        # modify config if given an arg otherwise keep defaults
        args_temp = vars(args)
        for k, v in args_temp.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v if v is not None else getattr(cfg, k))
        cfg.assertions_corrections()
        cfg.calc_dims()
        # update the args with the final model config
        for attribute in vars(cfg):
            if hasattr(args, attribute):
                setattr(args, attribute, getattr(cfg, attribute))
        # init model
        self.model = ViT(cfg, pretrained=args.pretrained)
        self.cfg = cfg

    def forward(self, images, targets=None, ret_dist=False):
        out = self.model(images, ret_dist=ret_dist)
        return out


class VisionTransformerGLSim(nn.Module):
    def __init__(self, args):
        super(VisionTransformerGLSim, self).__init__()
        # init default config
        cfg = ViTConfig(model_name=args.model_name)
        # modify config if given an arg otherwise keep defaults
        args_temp = vars(args)
        for k, v in args_temp.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v if v is not None else getattr(cfg, k))
        cfg.assertions_corrections()
        cfg.calc_dims()
        # update the args with the final model config
        for attribute in vars(cfg):
            if hasattr(args, attribute):
                setattr(args, attribute, getattr(cfg, attribute))
        # init model
        self.model = ViTGLSim(cfg, pretrained=args.pretrained)
        self.cfg = cfg

    def forward(self, images, targets=None, ret_dist=False):
        out = self.model(images, ret_dist=ret_dist)
        return out


class VisionTransformerFG(nn.Module):
    def __init__(self, args):
        super(VisionTransformerFG, self).__init__()
        # init default config
        cfg = ViTConfig(model_name=args.model_name)
        # modify config if given an arg otherwise keep defaults
        args_temp = vars(args)
        for k, v in args_temp.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v if v is not None else getattr(cfg, k))
        cfg.assertions_corrections()
        cfg.calc_dims()

        # update the args with the final model config
        for attribute in vars(cfg):
            if hasattr(args, attribute):
                setattr(args, attribute, getattr(cfg, attribute))
        # init model
        self.model = ViTFG(cfg, pretrained=args.pretrained)
        self.cfg = cfg

    def forward(self, images, targets=None, ret_dist=False):
        out = self.model(images)
        return out


class CounterfactualAttentionLearning(nn.Module):
    def __init__(self, args):
        super(CounterfactualAttentionLearning, self).__init__()
        # init default config
        cfg = CONFIGS_CAL['cal']
        cfg.backbone = args.model_name.replace('cal_', '')
        cfg.image_size = args.image_size
        cfg.num_classes = args.num_classes
        cfg.device = args.device
        cfg.single_crop = args.cal_single_crop
        cfg.adapter = args.adapter

        self.model = CAL(cfg)

        self.cfg = cfg
        if cfg.backbone == 'resnet101':
            self.cfg.seq_len_post_reducer = (args.image_size // 32) ** 2
        elif 'vit' in cfg.backbone:
            self.cfg.seq_len_post_reducer = self.model.cfg.seq_len_post_reducer

    def forward(self, images, targets=None, ret_dist=False):
        out = self.model(images, targets)
        return out


class TIMMViT(nn.Module):
    def __init__(self, args):
        super(TIMMViT, self).__init__()
        # init default config

        self.model = timm.create_model(
            args.model_name, pretrained=args.pretrained, num_classes=1000,
            img_size=args.image_size, drop_path_rate=args.sd, args=args)
        self.model.reset_classifier(args.num_classes)

        self.cfg = SimpleNamespace(**{'seq_len_post_reducer': args.image_size // 16})

    def forward(self, images, targets=None, ret_dist=False):
        x = self.model(images)
        return x
