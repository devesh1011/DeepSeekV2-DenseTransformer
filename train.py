import torch
from gpt import GPT
from model_config import Config
from dataloader import Dataloader
import time
import tiktoken
import torch.nn.functional as F

enc = tiktoken.get_encoding("gpt2")


def get_lr(
    it,
    total_tokens=338025,
    max_lr=2.4e-4,
    warmup_steps=100,
    decay_factors=[0.316, 0.316],
    decay_points=[0.6, 0.9],
):
    # Convert decay points to token counts
    decay_token_counts = [int(p * total_tokens) for p in decay_points]

    if it < warmup_steps:
        # Linear warmup from 0 to max_lr
        lr = max_lr * (it + 1 / warmup_steps)
    else:
        # Base LR after warmup
        lr = max_lr
        # Apply step decay based on tokens processed
        for decay_tokens, factor in zip(decay_token_counts, decay_factors):
            if tokens_processed >= decay_tokens:
                lr *= factor
            else:
                break

    return lr


total_batch_size = 1024
B = 16
T = 256
assert (
    total_batch_size % (B * T) == 0
), "maken sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size: {total_batch_size}")
print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = Dataloader(B=B, T=T, split="train")
val_loader = Dataloader(B=B, T=T, split="val")

torch.set_float32_matmul_precision("high")

config = Config(vocab_size=50304)

model = GPT(Config())
model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, betas=[0.9, 0.95], weight_decay=0.1
)

max_steps = 5000

for step in range(max_steps):
    t0 = time.time()

    # model evaluation
    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        print(f"Validation loss: {val_loss_accum.item():.4f}")

    if step > 0 and step % 100 == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 10
        tokens = enc.encode("We are accounted poor citizens")
        tokens = torch.tensor(tokens, dtype=torch.float16)
        xgen = tokens.to(device.type)
        sample_rng = torch.Generator(device=device.type)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(xgen)

                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(probs, 30, dim=-1)

                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=-1)
        for i in range(num_return_sequences):
            tokens - xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"sample {i}: {decoded}")

    # model training
    model.train()
    loss_accum = 0.0
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    # gradient clipping -> as per deepseek-v2 paper
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)  # learning rate scheduler
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(
        f"step {step:4d}, loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}"
    )
