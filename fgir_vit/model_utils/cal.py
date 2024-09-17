'''
https://github.com/raoyongming/CAL/tree/master/fgvc
https://github.com/raoyongming/CAL/blob/master/fgvc/train_distributed.py
https://github.com/raoyongming/CAL/blob/master/fgvc/models/cal.py
https://github.com/raoyongming/CAL/blob/master/fgvc/infer.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
from einops.layers.torch import Rearrange

from .modules_fgir import WSDAN_CAL, batch_augment
from .modules_others import resnet101
from .misc import ViTConfig
from .vit import ViT

EPSILON = 1e-6


def get_cal_config():
    """Returns the CAL configuration."""
    config = ml_collections.ConfigDict()
    config.backbone = 'resnet101'
    config.num_attention_maps = 32
    config.beta = 5e-2
    config.image_size = 224
    config.num_classes = 21843
    config.device = 'cuda'
    config.single_crop = False
    config.adapter = None
    return config


class CAL(nn.Module):
    def __init__(self, config):
        super(CAL, self).__init__()
        self.config = config

        # Network Initialization
        if config.backbone == 'resnet101':
            self.encoder = resnet101(pretrained=True).get_features()
            num_encoded_channels = 512 * self.encoder[-1][-1].expansion
        elif 'vit' in config.backbone:
            cfg = ViTConfig(model_name=config.backbone, image_size=config.image_size,
                            classifier='none', adapter=config.adapter, load_fc_layer=False)
            self.encoder = nn.Sequential(
                ViT(cfg, pretrained=True),
                Rearrange('b (h w) d -> b d h w', h=cfg.gh)
            )
            num_encoded_channels = cfg.hidden_size
            self.cfg = cfg
        else:
            raise ValueError('Unsupported net: %s' % config.backbone)

        # discriminative feature selection mechanism
        self.dfsm = WSDAN_CAL(config.num_classes, num_encoded_channels, config.num_attention_maps)

        self.feature_center = torch.zeros(
            config.num_classes, config.num_attention_maps * num_encoded_channels, device=config.device)

        self.single_crop = config.single_crop

        print('WSDAN: using {} as feature extractor, num_classes: {}, num_attention_maps: {}'.format(
            config.backbone, config.num_classes, config.num_attention_maps))

    def forward(self, x, y=None):
        if self.training and y is not None:
            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw, y_pred_aux, feature_matrix, attention_map = self.dfsm(feature_maps)

            # Update Feature Center
            feature_center_batch = F.normalize(self.feature_center[y], dim=-1)
            self.feature_center[y] += self.config.beta * (feature_matrix.detach() - feature_center_batch)

            # attention cropping
            with torch.no_grad():
                crop_images = batch_augment(
                    x, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
                drop_images = batch_augment(x, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
            aug_images = torch.cat([crop_images, drop_images], dim=0)

            # crop images forward
            feature_maps = self.encoder(aug_images)
            y_pred_aug, y_pred_aux_aug, _, _ = self.dfsm(feature_maps)

            y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)

            # final prediction
            y_pred_aug_crops, _ = torch.split(y_pred_aug, [x.shape[0], x.shape[0]], dim=0)
            y_pred = (y_pred_raw + y_pred_aug_crops) / 2.

            return (y_pred, y_pred_raw, y_pred_aux, feature_matrix, feature_center_batch,
                    y_pred_aug, crop_images)
        elif self.single_crop:
            feature_maps = self.encoder(x)
            y_pred_raw, y_pred_aux, _, attention_map = self.dfsm(feature_maps)

            crop_images3 = batch_augment(x, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images3)
            y_pred_crop3, _, _, _ = self.dfsm(feature_maps)

            # final prediction
            y_pred = (y_pred_raw + y_pred_crop3) / 2.

            return y_pred, crop_images3
        else:
            x_m = torch.flip(x, [3])

            # Raw Image
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map = self.dfsm(feature_maps)

            feature_maps = self.encoder(x_m)
            y_pred_raw_m, _, _, attention_map_m = self.dfsm(feature_maps)

            # Object Localization and Refinement
            crop_images = batch_augment(x, attention_map, mode='crop', theta=0.3, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images)
            y_pred_crop, _, _, _ = self.dfsm(feature_maps)

            crop_images2 = batch_augment(x, attention_map, mode='crop', theta=0.2, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images2)
            y_pred_crop2, _, _, _ = self.dfsm(feature_maps)

            crop_images3 = batch_augment(x, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images3)
            y_pred_crop3, _, _, _ = self.dfsm(feature_maps)

            crop_images_m = batch_augment(x_m, attention_map_m, mode='crop', theta=0.3, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m)
            y_pred_crop_m, _, _, _ = self.dfsm(feature_maps)

            crop_images_m2 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.2, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m2)
            y_pred_crop_m2, _, _, _ = self.dfsm(feature_maps)

            crop_images_m3 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.1, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images_m3)
            y_pred_crop_m3, _, _, _ = self.dfsm(feature_maps)

            y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.
            y_pred_m = (y_pred_raw_m + y_pred_crop_m + y_pred_crop_m2 + y_pred_crop_m3) / 4.
            y_pred = (y_pred + y_pred_m) / 2.

            # crop_images = torch.stack((crop_images, crop_images2, crop_images3,
            #                            crop_images_m, crop_images_m2, crop_images_m3), dim=-1)
            return y_pred, crop_images


CONFIGS_CAL = {
    'cal': get_cal_config()
}
