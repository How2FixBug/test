#!/usr/bin/env python3
"""ASTRA 0.1 — LoopCore language model."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

def causal_prefix_mean(x: torch.Tensor) -> torch.Tensor:
    """For each position t, average only positions 0..t."""
    length = x.size(1)
    counts = torch.arange(1, length + 1, device=x.device, dtype=x.dtype)
    return x.cumsum(dim=1) / counts.view(1, length, 1)

class SharedLoopBlock(nn.Module):
    """One transformer-like block reused at every LoopCore pass."""
    def __init__(self, width: int, heads: int, ff_width: int, local_window: int):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width, self.heads = width, heads
        self.head_width, self.local_window = width // heads, local_window
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.attn_out = nn.Linear(width, width, bias=False)
        self.memory_candidate = nn.Linear(width, width)
        self.memory_gate = nn.Linear(width * 2, width)
        self.memory_out = nn.Linear(width, width, bias=False)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp_gate = nn.Linear(width, ff_width, bias=False)
        self.mlp_up = nn.Linear(width, ff_width, bias=False)
        self.mlp_down = nn.Linear(ff_width, width, bias=False)

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        batch, length, width = x.shape
        normalized = self.attn_norm(x)
        qkv = self.qkv(normalized).view(
            batch, length, 3, self.heads, self.head_width
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        positions = torch.arange(length, device=x.device)
        distance = positions[:, None] - positions[None, :]
        attention_mask = (distance >= 0) & (distance < self.local_window)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        x = x + self.attn_out(attended)

        # Position-wise recurrent memory receives only causal prefix information.
        prefix = causal_prefix_mean(x)
        candidate = torch.tanh(self.memory_candidate(prefix))
        gate = torch.sigmoid(self.memory_gate(torch.cat((prefix, memory), dim=-1)))
        memory = gate * memory + (1.0 - gate) * candidate
        x = x + self.memory_out(memory)

        normalized = self.mlp_norm(x)
        hidden = F.silu(self.mlp_gate(normalized)) * self.mlp_up(normalized)
        return x + self.mlp_down(hidden), memory

class LoopCore(nn.Module):
    """Autoregressive model with one shared block applied recurrently."""
    def __init__(
        self, vocab_size: int, width: int = 512, heads: int = 8,
        ff_width: int = 1792, max_seq_len: int = 512,
        local_window: int = 128, loop_passes: int = 4,
    ):
        super().__init__()
        self.vocab_size, self.width = vocab_size, width
        self.max_seq_len, self.loop_passes = max_seq_len, loop_passes
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(max_seq_len, width)
        self.pass_embedding = nn.Embedding(loop_passes, width)
        self.loop_controller = nn.Linear(width * 2, 1)
        self.shared_block = SharedLoopBlock(width, heads, ff_width, local_window)
        self.final_norm = nn.LayerNorm(width)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor):
        batch, length = input_ids.shape
        if length > self.max_seq_len:
            raise ValueError(f"Sequence length {length} exceeds max_seq_len={self.max_seq_len}")
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        memory = x.new_zeros(batch, length, self.width)

        for pass_index in range(self.loop_passes):
            pass_vector = self.pass_embedding.weight[pass_index].view(1, 1, -1)
            prefix = causal_prefix_mean(x)
            control_input = torch.cat(
                (prefix, pass_vector.expand(batch, length, -1)), dim=-1
            )
            strength = torch.sigmoid(self.loop_controller(control_input))
            updated, memory = self.shared_block(x + pass_vector, memory)
            x = x + strength * (updated - x)

        return F.linear(self.final_norm(x), self.token_embedding.weight)
