import torch
import numpy as np

# total_tokens // (B * T)
class Dataloader:
    def __init__(self, B, T, split="train"):
        self.B = B
        self.T = T
        self.split = split.lower()

        # load pre tokenized data from binary files
        if self.split == "train":
            tokens_np = np.fromfile("train.bin", dtype=np.uint16)
            self.tokens = torch.tensor(tokens_np, dtype=torch.float16)
        elif self.split == "val":
            tokens_np = np.fromfile("val.bin", dtype=np.uint16)
            self.tokens = torch.tensor(tokens_np, dtype=torch.float16)
        else:
            raise ValueError("split must be 'train' or 'val'")
        self.curr_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.curr_pos : self.curr_pos + B * T + 1]

        # Handle case where remaining tokens are less than B * T + 1
        if len(buf) < B * T + 1:
            # Pad with zeros or cycle back (cycling is more common for language modeling)
            pad_len = (B * T + 1) - len(buf)
            buf = torch.cat([buf, self.tokens[:pad_len]])

        x = buf[:-1].view(B, T)  # Input: all tokens except the last
        y = buf[1:].view(B, T)  # Target: all tokens except the first

        self.curr_pos += B * T
        # Reset position if we've reached or exceeded the end
        if self.curr_pos + B * T >= len(self.tokens):
            self.curr_pos = 0

        return x, y
