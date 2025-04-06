import torch
from torch import nn
from causal_attention import CausalAttention

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                      for _ in range(num_heads)]

    def forward(self, x):
        # These are processed sequentially
        return torch.cat([head(x) for head in self.heads], dim=-1)
