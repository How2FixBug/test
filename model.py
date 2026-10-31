"""A small GPT-style language model built from scratch in PyTorch."""
import math
import torch
from torch import nn
from torch.nn import functional as F

VOCAB_SIZE = 259  # PAD, BOS, EOS, then all 256 byte values
PAD, BOS, EOS = 0, 1, 2
BYTE_OFFSET = 3


def encode(text):
    return [BOS] + [b + BYTE_OFFSET for b in text.encode("utf-8")] + [EOS]


def decode(tokens):
    raw = bytes(t - BYTE_OFFSET for t in tokens if BYTE_OFFSET <= t < VOCAB_SIZE)
    return raw.decode("utf-8", errors="replace")


class CausalBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        n = x.size(1)
        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, width=384, layers=8, heads=6, context=512, dropout=0.05):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(VOCAB_SIZE, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([CausalBlock(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, VOCAB_SIZE, bias=False)
        self.output.weight = self.token.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        batch, length = ids.shape
        if length > self.context:
            ids = ids[:, -self.context:]
            if targets is not None:
                targets = targets[:, -self.context:]
            length = self.context
        positions = torch.arange(length, device=ids.device)
        x = self.token(ids) + self.position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.norm(x))
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), ignore_index=PAD) if targets is not None else None
        return logits, loss

    @torch.inference_mode()
    def generate(self, ids, max_new_tokens=240, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -self.context:])
            scores = logits[:, -1, :] / max(temperature, 0.05)
            if top_k:
                values, _ = torch.topk(scores, min(top_k, scores.size(-1)))
                scores[scores < values[:, [-1]]] = -float("inf")
            probs = F.softmax(scores, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)
            if next_id.item() == EOS:
                break
        return ids
