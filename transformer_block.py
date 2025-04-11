from torch import nn
from feed_forward import FeedForward
from gpt_config import GPTConfig
from layer_norm import LayerNorm
from multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate_attn,
            qkv_bias=cfg.qkv_bias
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop = nn.Dropout(cfg.drop_rate_shortcut)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x + shortcut

        return x
