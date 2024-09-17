# https://github.com/Lyken17/pytorch-OpCounter
# https://github.com/sovrasov/flops-counter.pytorch
# https://github.com/zhijian-liu/torchprofile
import torch

from thop import profile
from ptflops import get_model_complexity_info
from torchprofile import profile_macs

from fgir_vit import ViTConfig
from fgir_vit.other_utils.build_args import parse_train_args
from fgir_vit.model_utils.build_model import build_model


def count_params(model, trainable=False):
    if trainable:
        return sum([p.numel() for p in model.parameters()])
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


class FGFLOPS(object):
    """Computes the inference flops for FGIR transformers."""

    def __init__(self, model_name, image_size=224, patch_size=16, hidden_size=768, 
                 num_classes=1000, channels_in=3, **kwargs):
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.channels_in = channels_in

        self.seq_len = self.calc_seq_len()

        self.num_attention_heads = self.hidden_size // 64

        if self.model_name in ('transfg', 'ffvt'):
            self.num_hidden_layers = 11
        else:
            self.num_hidden_layers = 12

    def calc_seq_len(self):
        if self.model_name in ('transfg', 'aftrans'):
            stride_size = 12
        else:
            stride_size = 16
        seq_len = (((self.image_size - self.patch_size) / stride_size) + 1) ** 2
        seq_len += 1
        return seq_len

    def vit_flops(self):
        patch_flops = 2 * (self.seq_len - 1) * (self.patch_size ** 2) * self.channels_in * self.hidden_size

        msa_flops = (4 * self.seq_len * (self.hidden_size ** 2)) + (2 * (self.seq_len ** 2) * self.hidden_size)
        pwffn_flops = 8 * self.seq_len * (self.hidden_size ** 2)
        layerwise_flops = msa_flops + pwffn_flops

        out_flops = self.hidden_size * self.num_classes

        flops = patch_flops + (self.num_hidden_layers * layerwise_flops) + out_flops

        return flops
    
    def transfg_flops(self):
        # recursive matrix-matrix multiplication of head-wise attention scores
        # nm * (2p - 1)
        headwise_flops = (self.seq_len ** 2) * ((2 * self.seq_len) - 1)
        # nm * (2p - 1) * num_heads
        layerwise_flops = headwise_flops * self.num_attention_heads
        # nm * (2p - 1) * num_heads * (num_layers - 1)
        flops = layerwise_flops * (self.num_hidden_layers - 1)
        return flops

    def ffvt_flops(self):
        # softmax normalized element-wise first-row and first-column of attention scores
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/profiler/internal/flops_registry.py
        # 2 * n (compute shifted logits) + n (exp of shifted logits) + 2 * n (softmax from exp of shifted logits)
        softmax_flops = 5 * self.seq_len * self.num_attention_heads
        # mean of each head: sum heads and then divide for both a and b vector
        avg_flops = ((self.num_attention_heads - 1) + 1) * 2
        element_wise_prod_flops = self.seq_len
        layerwise_flops = softmax_flops + avg_flops + element_wise_prod_flops
        flops = layerwise_flops * self.num_hidden_layers
        return flops

    def aftrans_flops(self):
        # head-wise element-wise multiplication then sequence-wise GAP
        # followed by excitation-like MLP (2 layers)
        headwise_flops = (self.seq_len ** 2)
        layerwise_flops = (self.num_attention_heads - 1) * headwise_flops

        # squeeze
        gap_flops = (self.seq_len ** 2) + 1

        # excitation: L to C/R dimension (since C/R is not given assume same as L) + L (activation)
        excitation_mlp_layerwise_flops = (self.num_hidden_layers * 2) + self.num_hidden_layers
        # assume select cls tokens only to multiply element-wise excitation of each layer (:, :, 0, 1:)
        excitation_flops = self.num_hidden_layers * (self.seq_len - 1)

        flops = (self.num_hidden_layers * layerwise_flops) + gap_flops + (excitation_mlp_layerwise_flops * 2) + excitation_flops
        return flops

    def ramstrans_flops(self):
        # recursive matrix-matrix multiplication of each layer renormalized attention weights
        # head-wise mean + add diagonal matrix (assume seq_len * seq_len size)
        numerator_flops = (self.num_attention_heads * (self.seq_len ** 2)) + (self.seq_len ** 2)
        # seq-wise (last dimension) mean of head-wise mean (+ diagonal matrix) (already computed in previous step)
        denominator_flops = self.seq_len * (self.seq_len + 1)

        layerwise_norm_flops = numerator_flops + denominator_flops + 1

        # recursive matrix-matrix multiplication
        flops_matrix_mult = (self.seq_len * self.seq_len) * ((2 * self.seq_len) - 1)
        flops = (self.num_hidden_layers * layerwise_norm_flops) + (self.num_hidden_layers - 1) * flops_matrix_mult
        return flops

    def dcal_flops(self):
        # attention rollout:
        # recursive matrix-matrix multiplication of attention weights + identity to account for residual
        # head-wise mean + add diagonal matrix (assume seq_len * seq_len size)
        numerator_flops = (self.num_attention_heads * (self.seq_len ** 2)) + (self.seq_len ** 2)
        # divide all elements by 0.5
        denominator_flops = self.seq_len

        layerwise_norm_flops = numerator_flops + denominator_flops + 1

        # recursive matrix-matrix multiplication
        flops_matrix_mult = (self.seq_len * self.seq_len) * ((2 * self.seq_len) - 1)
        flops = (self.num_hidden_layers * layerwise_norm_flops) + (self.num_hidden_layers - 1) * flops_matrix_mult
        return flops

    def pim_flops(self):
        # weakly supervised selector (linear layer to predict classes for each token) for 4 blocks
        tokenwise_linear_flops = self.hidden_size * self.num_classes
        layerwise_flops = self.seq_len * tokenwise_linear_flops
        flops = 4 * layerwise_flops        
        return flops

    def glsim_flops(self):
        # cosine similarity between CLS token (global) and each token in sequence (local)
        # element-wise multiplication
        numerator_flops = self.hidden_size
        # element-wise squared followed by squared root times 2 (A and B norm each)
        denominator_flops = (self.hidden_size + 1) * 2 
        # add numerator and denominator flops followed by division
        cos_flops = numerator_flops + denominator_flops + 1
        # cosine similarity for each element in sequence
        flops = cos_flops * (self.seq_len - 1)
        return flops

    def davit_flops(self):
        # recursive matrix-matrix multiplication of each layer renormalized attention weights
        # head-wise mean + add diagonal matrix (assume seq_len * seq_len size)
        numerator_flops = (self.num_attention_heads * (self.seq_len ** 2)) + (self.seq_len ** 2)
        # seq-wise (last dimension) mean of head-wise mean (+ diagonal matrix) (already computed in previous step)
        denominator_flops = self.seq_len * (self.seq_len + 1)

        layerwise_norm_flops = numerator_flops + denominator_flops + 1

        # single matrix-matrix multiplication
        flops_matrix_mult = (self.seq_len * self.seq_len) * ((2 * self.seq_len) - 1)
        flops = (2 * layerwise_norm_flops) + flops_matrix_mult
        return flops

    def get_discriminative_flops(self):
        if self.model_name == 'vit':
            flops = self.vit_flops()
        if self.model_name == 'transfg':
            flops = self.transfg_flops()
        elif self.model_name == 'ffvt':
            flops = self.ffvt_flops()
        elif self.model_name == 'aftrans':
            flops = self.aftrans_flops()
        elif self.model_name == 'ramstrans':
            flops = self.ramstrans_flops()
        elif self.model_name == 'dcal':
            flops = self.dcal_flops()
        elif self.model_name == 'pim':
            flops = self.pim_flops()
        elif self.model_name == 'glsim':
            flops = self.glsim_flops()
        elif self.model_name == 'davit':
            flops = self.davit_flops()
        return flops


def main():

    args = parse_train_args()

    args.num_classes = 1000

    x = torch.rand(1, 3, args.image_size, args.image_size).to(args.device)

    model = build_model(args)

    params = count_params(model, trainable=True) / 1e6
    trainable_params = count_params(model, trainable=False) / 1e6

    macs, _ = profile(model, inputs=(x, ))
    macs = macs / 1e9

    macs2, _ = get_model_complexity_info(
        model, (3, args.image_size, args.image_size), as_strings=False,
        print_per_layer_stat=args.debugging, verbose=args.debugging)
    macs2 = macs2 / 1e9

    macs3 = profile_macs(model, x)
    macs3 = macs3 / 1e9

    print('model_name,no_params_1k,no_trainable_params_1k,gmacs_thop,gmacs_ptflops,gmacs_torchprofile')
    line = f'{args.model_name},{params},{trainable_params},{macs},{macs2},{macs3}\n'
    print(line)

    for model_name in ('vit', 'transfg', 'ffvt', 'aftrans', 'ramstrans', 'dcal',
                       'pim', 'glsim', 'davit'):
        flops = FGFLOPS(model_name, image_size=args.image_size,
                        num_classes=args.num_classes).get_discriminative_flops()
        print('{}: {:.2f} MFLOPs'.format(model_name, (flops / (1e6))))

if __name__ == "__main__":
    main()
