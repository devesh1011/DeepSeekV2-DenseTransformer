import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("mahabharata.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text[:1000])
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // B * T} batches")

        self.curr_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.curr_pos : self.curr_pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.curr_pos += B * T
        if self.curr_pos + (B * T + 1) > len(self.tokens):
            self.curr_pos = 0
        return x, y


# with open("mahabharata.txt", "r") as f:
#     text = f.read()
# vocab_size = enc.n_vocab
# print(f"Vocab size: {vocab_size}")  # Outputs 50257

# # Tokenize your dataset to verify
# tokens = enc.encode(text)
# print(f"Number of tokens in dataset: {len(tokens)}")
# print(f"First few tokens: {tokens[:10]}")

# # Decode back to text (optional, for verification)
# decoded_text = enc.decode(tokens[:10])
# print(decoded_text)
