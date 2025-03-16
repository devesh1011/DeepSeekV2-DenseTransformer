import torch.nn as nn
from attention import MultiHeadLatentAttention
from feed_forward_net import FeedForwardNet

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rms_1 = nn.RMSNorm(config.n_embed)
        self.attn = MultiHeadLatentAttention(config)
        self.rms_2 = nn.RMSNorm(config.n_embed)
        self.ffn = FeedForwardNet(config)

    def forward(self, x):
        x = x + self.attn(self.rms_1(x))
        x = x + self.ffn(self.rms_2(x))
        return x
