import torch
from torch import nn


class PSM(nn.Module):
    def __init__(self):
        super(PSM, self).__init__()

    def forward(self, x, attn_weights, *args):
        length = len(attn_weights)

        last_map = attn_weights[0]
        for i in range(1, length):
            last_map = torch.matmul(attn_weights[i], last_map)
        last_map = last_map[:, :, 0, 1:]

        _, max_inx = last_map.max(2)

        part_inx = max_inx + 1

        parts = []

        B, num = part_inx.shape

        for i in range(B):
            parts.append(x[-1][i, part_inx[i, :]])

        parts = torch.stack(parts).squeeze(1)
        concat = torch.cat((x[-1][:, 0].unsqueeze(1), parts), dim=1)

        return concat

