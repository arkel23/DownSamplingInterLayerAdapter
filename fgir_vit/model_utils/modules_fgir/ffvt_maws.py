import torch
from torch import nn


class MAWS(nn.Module):
    # mutual attention weight selection
    def __init__(self, num_token):
        super(MAWS, self).__init__()
        self.num_token = num_token
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x, attn_weights_soft, attn_weights):
        B = x[0].shape[0]

        tokens = [[] for _ in range(x[0].shape[0])]

        for l in range(len(x)):

            contributions = self.softmax(attn_weights[l])[:, :, :, 0]
            contributions = contributions.mean(1)

            weights = attn_weights_soft[l][:, :, 0, :].mean(1)

            scores = contributions*weights

            max_inx = torch.argsort(scores, dim=1, descending=True)

            for i in range(B):
                tokens[i].extend(
                    x[l][i, max_inx[i, :self.num_token]])

        tokens = [torch.stack(token) for token in tokens]
        tokens = torch.stack(tokens).squeeze(1)
        concat = torch.cat((x[-1][:, 0].unsqueeze(1), tokens), dim=1)

        return concat
