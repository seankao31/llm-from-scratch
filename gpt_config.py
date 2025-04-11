from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate_attn: float
    drop_rate_shortcut: float
    drop_rate_emb: float
    qkv_bias: bool
