import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce

from .ramstrans_dcal_attention_rollout import attention_rollout


class GLSimCropDrop(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.debugging = config.debugging

        self.anchor_random = config.anchor_random

        self.selector = config.selector
        self.sim_metric = config.sim_metric

        self.patch_equivalent = config.patch_equivalent

        if 'cls' in config.classifier:
            self.class_token = True

    def get_glsim_indices(self, x, num_tokens):
        if hasattr(self, 'class_token'):
            g = x[:, :1, :]
            l = x[:, 1:, :]
        else:
            g = reduce(x, 'b s d -> b d', 'mean')
            l = x

        if self.sim_metric in ('l1'):
            distances = F.l1_loss(g, l, reduction='none')
        elif self.sim_metric in ('l2'):
            distances = F.mse_loss(g, l, reduction='none')
        elif self.sim_metric in ('cos'):
            distances = F.cosine_similarity(g, l, dim=-1)

        if self.sim_metric != 'cos':
            distances = reduce(distances, 'b a k -> b a', 'mean')
            distances = torch.abs(distances)

        largest = True if self.sim_metric == 'cos' else False

        dist, ind = distances.topk(num_tokens, dim=-1, largest=largest)
        self.maybe_print('Distances and 1-d indexes: ', distances.shape, dist.shape,
                         ind.shape, dist[0], ind[0])

        return ind

    @torch.no_grad()
    def get_attention_rollout_indices(self, scores, num_tokens):
        rollout = attention_rollout(scores)

        if hasattr(self, 'class_token'):
            # Attention from the output token to the input space.
            attn = rollout[:, 0, 1:]
        else:
            attn = reduce(rollout, 'b s1 s2 -> b s2', 'mean')

        dist, ind = attn.topk(num_tokens, dim=-1, largest=True)
        self.maybe_print('Distances and 1-d indexes: ', rollout.shape, attn.shape, dist.shape,
                         ind.shape, dist[0], ind[0])

        return ind

    @torch.no_grad()
    def get_indices(self, x, scores, num_tokens):

        if self.anchor_random:
            ind = torch.randint(0, x.shape[1] - 1, [x.shape[0], num_tokens], device=x.get_device())
        elif self.selector == 'glsim':
            ind = self.get_glsim_indices(x, num_tokens)
        elif self.selector == 'rollout':
            ind = self.get_attention_rollout_indices(scores, num_tokens)

        return ind

    @torch.no_grad()
    def crop_image(self, images, i, x_0, x_1, y_0, y_1):
        crop = images[i:i+1, :, x_0:x_1, y_0:y_1]
        crop = F.upsample_bilinear(crop, size=(images.shape[-1], images.shape[-1]))
        return crop

    @torch.no_grad()
    def mask_image(self, images, i, x_0, x_1, y_0, y_1):
        images[i:i+1, :, x_0:x_1, y_0:y_1] = 0
        masked_image = images[i:i+1]
        return masked_image

    @torch.no_grad()
    def forward(self, x, scores, images, num_tokens=8, crop_mode='crop'):
        self.maybe_print('Obtaining crops: ', x.shape, images.shape)
        if isinstance(images.shape[-1], torch.Tensor):
            img_size = images.shape[-1].item()
        else:
            img_size = images.shape[-1]
        ind = self.get_indices(x, scores, num_tokens)

        ind_x = torch.floor(torch.div(ind, int(math.sqrt(x.shape[1])))).int()
        ind_y = ind % int(math.sqrt(x.shape[1]))

        ind_x_i = reduce(ind_x, 'b a -> b', 'min')
        ind_x_f = reduce(ind_x, 'b a -> b', 'max')
        ind_y_i = reduce(ind_y, 'b a -> b', 'min')
        ind_y_f = reduce(ind_y, 'b a -> b', 'max')

        x_i = ind_x_i * self.patch_equivalent
        y_i = ind_y_i * self.patch_equivalent
        x_f = ind_x_f * self.patch_equivalent
        y_f = ind_y_f * self.patch_equivalent

        self.maybe_print('2d indexes: ', ind_x[0], ind_y[0], x_i, x_f, y_i, y_f)

        images_crops = []
        for i in range(ind.shape[0]):
            x_0 = max(x_i[i], 0)
            y_0 = max(y_i[i], 0)
            x_1 = min(max(x_f[i], x_i[i] + self.patch_equivalent), img_size)
            y_1 = min(max(y_f[i], y_i[i] + self.patch_equivalent), img_size)

            if crop_mode == 'crop_mask' and self.training:
                switching_prob = np.random.rand(1)
                if switching_prob > 0.5:
                    aug = self.crop_image(images, i, x_0, x_1, y_0, y_1)
                else:
                    aug = self.mask_image(images, i, x_0, x_1, y_0, y_1)
            elif crop_mode == 'crop_mask':
                aug = self.crop_image(images, i, x_0, x_1, y_0, y_1)

            elif crop_mode == 'crop':
                aug = self.crop_image(images, i, x_0, x_1, y_0, y_1)

            elif crop_mode == 'mask':
                aug = self.mask_image(images, i, x_0, x_1, y_0, y_1)

            else:
                raise NotImplementedError

            images_crops.append(aug)

        images_crops = torch.cat(images_crops, dim=0)

        return images_crops

    def maybe_print(self, *args):
        if self.debugging:
            print(*args)

