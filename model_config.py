from dataclasses import dataclass


@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 6
    n_head: int = 6
    n_embed: int = 384
