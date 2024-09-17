# https://github.com/raoyongming/CAL/blob/master/fgvc/utils.py
import random

import torch
import torch.nn.functional as F


# augment function
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1, percent_max=True):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                if percent_max:
                    theta_c = random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = random.uniform(*theta) * atten_map.mean()
            else:
                if percent_max:
                    theta_c = theta * atten_map.max()
                else:
                    theta_c = theta * atten_map.mean()

            # 0 / 1 mask based on if attention at x,y is higher than max value * threshold percentage
            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c

            # x, y indices for 1 values in mask
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])

            # select highest/min height/width
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(
                    images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                if percent_max:
                    theta_d = random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = random.uniform(*theta) * atten_map.mean()
            else:
                if percent_max:
                    theta_c = theta * atten_map.max()
                else:
                    theta_c = theta * atten_map.mean()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], \
            but received unsupported augmentation method %s' % mode)
