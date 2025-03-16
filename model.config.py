from dataclasses import dataclass


@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 65
    n_layers: int = 12
    n_head: int = 12
    n_embed: int = 768
