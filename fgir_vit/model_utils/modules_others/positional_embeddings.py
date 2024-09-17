import math
import torch
from torch import nn

from einops import repeat

class LearnedPositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim, patch_size=16, prompt=False, prompt_len=None):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
        if prompt == 'vpt_shallow':
            self.prompt = nn.Parameter(torch.zeros(1, prompt_len, dim))
            val = math.sqrt(6. / float(3 * (patch_size ** 2) + dim))
            nn.init.uniform_(self.prompt.data, -val, val)

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        x = x + self.pos_embedding

        if hasattr(self, 'prompt'):
            x = torch.cat((
                x[:, :1, :],
                repeat(self.prompt, '1 s d -> b s d', b=x.shape[0]),
                x[:, 1:, :]
            ), dim=1)

        return x
