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

        # weight sharing scheme
        self.transformer.wte.weight = self.ln_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights of the model with a normal distribution.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.006)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.RMSNorm):
            nn.init.constant_(module.weight, 1.0)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size

        tok_emb = self.transformer.wte(idx)
        x = tok_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.ln_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits
