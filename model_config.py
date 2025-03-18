from dataclasses import dataclass


@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layers: int = 4
    n_head: int = 4
    n_embed: int = 384
