# https://github.com/raoyongming/CAL/blob/master/fgvc/models/cal.py
# https://github.com/raoyongming/CAL/blob/master/fgvc/infer.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules_others import BasicConv2D 


EPSILON = 1e-6


# Bilinear Attention Pooling
class BAP_Counterfactual(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP_Counterfactual, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + EPSILON)

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


# Center Loss for Attention Regularization
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, output, targets):
        return self.l2_loss(output, targets) / output.size(0)


# Overall CAL Loss
class CALLoss(nn.Module):
    def __init__(self):
        super(CALLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss()

    def forward(self, output, y):
        if len(output) == 7:
            (_, y_pred_raw, y_pred_aux, feature_matrix, feature_center_batch,
             y_pred_aug, _) = output

            y_aug = torch.cat([y, y], dim=0)
            y_aux = torch.cat([y, y_aug], dim=0)
 
            batch_loss = (self.cross_entropy_loss(y_pred_raw, y) / 3. +
                          self.cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. +
                          self.cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. +
                          self.center_loss(feature_matrix, feature_center_batch))

        elif len(output) == 2:
            y_pred, _ = output
            batch_loss = self.cross_entropy_loss(y_pred, y)

        return batch_loss            


class WSDAN_CAL(nn.Module):
    """
    WS-DAN models
    Hu et al.,
    "See Better Before Looking Closer: Weakly Supervised Data Augmentation Network
    for Fine-Grained Visual Classification",
    arXiv:1901.09891
    """
    def __init__(self, num_classes, num_features=2048, num_attention_maps=32):
        super(WSDAN_CAL, self).__init__()
        # Attention Maps
        self.num_attention_maps = num_attention_maps
        self.attentions = BasicConv2D(num_features, num_attention_maps, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP_Counterfactual(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(num_attention_maps * num_features, num_classes, bias=False)

    def visualize(self, feature_maps):
        # Feature Maps, Attention Maps and Feature Matrix
        attention_maps = self.attentions(feature_maps)

        feature_matrix, _ = self.bap(feature_maps, attention_maps)
        p = self.fc(feature_matrix * 100.)

        return p, attention_maps

    def forward(self, feature_maps):
        # Feature Maps, Attention Maps and Feature Matrix
        batch_size = feature_maps.size(0)
        attention_maps = self.attentions(feature_maps)

        feature_matrix, feature_matrix_hat = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.num_attention_maps, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        return p, p - self.fc(feature_matrix_hat * 100.), feature_matrix, attention_map
