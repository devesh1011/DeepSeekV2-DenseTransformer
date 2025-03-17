import torch
from dataset import DataLoaderLite
from gpt import GPT
from model_config import Config
from torch.utils.data import Dataset, DataLoader


class SyntheticDataset(Dataset):
    def __init__(self, block_size, vocab_size, size=1000):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = torch.randint(0, vocab_size, (size, block_size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y


config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT(config).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# dataset = SyntheticDataset(config.block_size, config.vocab_size)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
# for idx, targets in dataloader:
#     idx, targets = idx.to(device), targets.to(device)
#     break

# print(f"Input shape: {idx.shape}")  # Should be [4, 1023]
# logits = model(idx)
# print(f"Logits shape: {logits.shape}")

# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# logits, loss = model(idx, targets)
# loss.backward()
# optimizer.step()
# optimizer.zero_grad()
# print(f"Loss: {loss.item()}")