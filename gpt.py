import torch.nn as nn
from block import Block
import torch.nn.functional as F


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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size

        tok_emb = self.transformer.wte(idx)

        x = tok_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.ln_head(x)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1), targets.view(-1)))
        return logits, loss
