import torch.nn as nn
from block import Block


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layers)),
                ln_f=nn.RMSNorm(config.n_embed),
            )
        )
        self.ln_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
