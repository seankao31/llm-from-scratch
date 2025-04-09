import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # Buffers are automatically moved to the appropriate device along with our model
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split from (b, num_tokens, d_out) into multiple heads
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Shape (b, num_heads, num_tokens, head_dim) for matrix multiplication on the last 2 dim
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values: torch.Tensor = values.transpose(1, 2)

        # Computes dot product for each head in each batch
        attn_scores = queries @ keys.transpose(2, 3)

        # Masks truncated to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights: torch.Tensor = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, head_dim)
        # -> (b, num_tokens, num_heads, head_dim)
        # -> (b, num_tokens, d_out)
        context_vec = (attn_weights @ values) \
            .transpose(1, 2) \
            .contiguous() \
            .view(b, num_tokens, self.d_out)
        # Output projection layer not strictly necessary, but commonly used
        context_vec = self.out_proj(context_vec)
        return context_vec
