from torch import nn
from gelu_activation import GELU
from gpt_model import GPTConfig

class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim)
        )

    def forward(self, x):
        return self.layers(x)
