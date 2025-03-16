import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadLatentAttention(nn.Module):
    """
    Multi Head Latent Attention (MLA) module for a transformer model.
    """

    def __init__(self, config):
        super().__init__()
        self.d_kv = 128
        self.d_c = 64
        self.d_r = 64
        self.d_h = config.n_embed // config.n_head

        # Shared projection matrices
        self.W_dkv = nn.Linear(config.n_embed, self.d_kv, bias=False)
        self.W_dq = nn.Linear(config.n_embed, self.d_kv, bias=False)
        self.W_kr = nn.Linear(config.n_embed, self.d_r, bias=False)

        # Per-head projection matrices
        self.W_qc = nn.ModuleList(
            [nn.Linear(self.d_kv, self.d_c, bias=False) for _ in range(self.n_head)]
        )
        self.W_qr = nn.ModuleList(
            [nn.Linear(self.d_kv, self.d_r, bias=False) for _ in range(self.n_head)]
        )
        self.W_uk = nn.ModuleList(
            [nn.Linear(self.d_kv, self.d_c, bias=False) for _ in range(self.n_head)]
        )
        self.W_uv = nn.ModuleList(
            [nn.Linear(self.d_kv, self.d_h, bias=False) for _ in range(self.n_head)]
        )
        # Output projection
        self.W_o = nn.Linear(config.n_head * self.d_h, config.n_embed, bias=False)

        freqs = torch.exp(
            torch.arange(0, self.d_r // 2, dtype=torch.float)
            * (-math.log(10000) / (self.d_r // 2))
        )

        # Precompute RoPE (Rotary Position Embedding) cosines and sines
        positions = torch.arange(0, config.block_size)
        angles = positions[:, None] * freqs[None, :]
        self.register_buffer(
            "cos_angles", torch.cos(angles)
        )  # Shape: [block_size, d_r//2]
        self.register_buffer("sin_angles", torch.sin(angles))

    def apply_rope(self, x, seq_len):
        batch_size, _, d_r = x.size()
        cos = self.cos_angles[:seq_len]
        sin = self.sin_angles[:seq_len]
        x_pairs = x.view(batch_size, seq_len, d_r // 2, 2)
        rotated_x0 = (
            x_pairs[..., 0] * cos[None, :, :] - x_pairs[..., 1] * sin[None, :, :]
        )
        rotated_x1 = (
            x_pairs[..., 0] * sin[None, :, :] + x_pairs[..., 1] * cos[None, :, :]
        )
        rotated_pairs = torch.stack([rotated_x0, rotated_x1], dim=-1)
        return rotated_pairs.view(batch_size, seq_len, d_r)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        c_kv = self.W_dkv(x)  # [batch_size, seq_len, 128]
        x_dq = self.W_dq(x)  # [batch_size, seq_len, 128]
        k_rope = self.W_kr(x)  # [batch_size, seq_len, 64]
        k_rope = self.apply_rope(k_rope, seq_len)

        outputs = []
        for h in range(self.n_head):
            # Query projections
            q_nope = self.W_qc[h](x_dq)  # [batch_size, seq_len, 64]
            q_rope = self.W_qr[h](x_dq)  # [batch_size, seq_len, 64]
            q_rope = self.apply_rope(q_rope, seq_len)

            # Key and value projections from latent space
            k_nope = self.W_uk[h](c_kv)  # [batch_size, seq_len, 64]
            v = self.W_uv[h](c_kv)  # [batch_size, seq_len, 128]

            # Compute attention scores
            score_nope = torch.bmm(
                q_nope, k_nope.transpose(1, 2)
            )  # [batch_size, seq_len, seq_len]
            score_rope = torch.bmm(
                q_rope, k_rope.transpose(1, 2)
            )  # [batch_size, seq_len, seq_len]
            score = score_nope + score_rope

            # Apply causal mask
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            score = score.masked_fill(mask, float("-inf"))

            # Compute attention weights
            attn_weights = F.softmax(
                score / math.sqrt(self.d_h), dim=-1
            )  # Scale by sqrt(d_h) = sqrt(128)

            # Compute output for this head
            out = torch.bmm(attn_weights, v)  # [batch_size, seq_len, 128]
            outputs.append(out)

        # Concatenate outputs from all heads
        out = torch.cat(outputs, dim=-1)  # [batch_size, seq_len, 768]

        # Final output projection
        out = self.W_o(out)  # [batch_size, seq_len, 768]
        return out
