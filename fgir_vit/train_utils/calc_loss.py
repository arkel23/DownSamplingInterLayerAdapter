import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from timm.loss import LabelSmoothingCrossEntropy

from .contrastive_loss import cont_loss, multi_cont_loss
from .focal_loss import FocalLoss
from .mix import mixup_criterion
from ..model_utils.modules_fgir.cal_wsdan_bap_loss import CALLoss

class OverallLoss(nn.Module):
    def __init__(self, args):
        super(OverallLoss, self).__init__()

        self.args = args

        if args.selector == 'cal':
            self.criterion = CALLoss()
        elif args.focal_gamma:
            self.criterion = FocalLoss(args.focal_gamma, smoothing=args.ls)
        elif args.ls:
            self.criterion = LabelSmoothingCrossEntropy(args.smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, output, targets, y_a=None, y_b=None, lam=None):
        if self.args.classifier_aux and self.args.dynamic_anchor:
            output, output_aux, images_crops = output
        elif self.args.classifier_aux:
            output, output_aux = output
        elif self.args.dynamic_anchor:
            output, images_crops = output

        if self.args.dynamic_anchor:
            r = self.args.seq_len_post_reducer
        elif self.args.classifier in ('ifacls', 'ifaconv'):
            r = self.args.num_hidden_layers

        if y_a is not None:
            loss = mixup_criterion(self.criterion, output, y_a, y_b, lam)
        else:
            loss = self.criterion(output, targets)

        if self.args.selector == 'cal':
            if len(output) == 7:
                output, _, _, _, _, _, _ = output
            elif len(output) == 2:
                output, _ = output

        if self.args.classifier_aux:
            if self.args.classifier_aux == 'cont':
                loss_aux = cont_loss(output_aux, targets)
            elif self.args.classifier_aux == 'multi_cont':
                if self.args.supcon:
                    loss_aux = multi_cont_loss(output_aux, targets, norm_ind=self.args.norm_ind)
                else:
                    loss_aux = multi_cont_loss(output_aux, norm_ind=self.args.norm_ind)
            elif self.args.classifier_aux == 'consistency':
                logits = torch.split(
                    output_aux, output.size(0), dim=0)
                logits = [F.softmax(lg, dim=1) for lg in logits]
                p_mixture = torch.clamp(
                    (torch.sum(torch.stack(logits, dim=1), dim=1)) /
                    r, 1e-7, 1)
                loss_aux = sum([F.kl_div(p_mixture, lg, reduction='batchmean')
                                for lg in logits]) / r
            else:
                if self.args.classifier_aux == 'shared':
                    r = r - 1
                targets_aux = repeat(targets, 'b -> (b n)', n=r)
                loss_aux = self.criterion(output_aux, targets_aux)
            loss = loss + (self.args.loss_aux_weight * loss_aux)

        assert math.isfinite(loss), f'Loss is not finite: {loss}, stopping training'

        return output, loss
