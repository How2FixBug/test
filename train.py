import json
import math
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Small character-level GPT: deliberately initialized from random weights.
SEED = 1337
BLOCK_SIZE = 96
BATCH_SIZE = 24
N_EMBD = 48
N_HEAD = 4
N_LAYER = 2
MAX_STEPS = int(os.environ.get("TRAIN_STEPS", "500"))
EVAL_INTERVAL = 100
LEARNING_RATE = 3e-3
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

torch.manual_seed(SEED)
torch.set_num_threads(2)
device = torch.device("cpu")

def get_text():
    path = Path("training_data.txt")
    if not path.exists():
        print("Downloading Tiny Shakespeare training text…", flush=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding="utf-8")

text = get_text()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

def get_batch(source):
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(N_EMBD, dim=2)
        q = q.view(b, t, N_HEAD, c // N_HEAD).transpose(1, 2)
        k = k.view(b, t, N_HEAD, c // N_HEAD).transpose(1, 2)
        v = v.view(b, t, N_HEAD, c // N_HEAD).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.mask[:t, :t] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.ModuleDict({
            "fc1": nn.Linear(N_EMBD, 4 * N_EMBD),
            "fc2": nn.Linear(4 * N_EMBD, N_EMBD),
        })

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp["fc2"](F.gelu(self.mlp["fc1"](self.ln2(x))))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        positions = torch.arange(t, device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        return logits, loss

model = TinyGPT().to(device)
parameter_count = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
print(f"Initialized {parameter_count:,} random parameters; vocab={vocab_size}; device=CPU", flush=True)

@torch.no_grad()
def estimate_loss():
    model.eval()
    values = {}
    for name, source in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(12)
        for k in range(12):
            xb, yb = get_batch(source)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        values[name] = round(losses.mean().item(), 4)
    model.train()
    return values

steps = MAX_STEPS
last_metrics = {"train": None, "val": None}
for step in range(steps):
    xb, yb = get_batch(train_data)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if (step + 1) % EVAL_INTERVAL == 0 or step == 0 or step + 1 == steps:
        last_metrics = estimate_loss()
        print(f"step {step+1}/{steps} train={last_metrics['train']:.4f} val={last_metrics['val']:.4f}", flush=True)

# Export plain JSON so the browser needs no ML runtime or model download.
def array(tensor):
    return tensor.detach().cpu().tolist()

state = {
    "wte": array(model.wte.weight),
    "wpe": array(model.wpe.weight),
    "ln_f": {"weight": array(model.ln_f.weight), "bias": array(model.ln_f.bias)},
    "lm_head": {"weight": array(model.lm_head.weight), "bias": array(model.lm_head.bias)},
    "blocks": []
}
for block in model.blocks:
    state["blocks"].append({
        "ln1": {"weight": array(block.ln1.weight), "bias": array(block.ln1.bias)},
        "attn": {
            "qkv": {"weight": array(block.attn.qkv.weight), "bias": array(block.attn.qkv.bias)},
            "proj": {"weight": array(block.attn.proj.weight), "bias": array(block.attn.proj.bias)}
        },
        "ln2": {"weight": array(block.ln2.weight), "bias": array(block.ln2.bias)},
        "mlp": {
            "fc1": {"weight": array(block.mlp["fc1"].weight), "bias": array(block.mlp["fc1"].bias)},
            "fc2": {"weight": array(block.mlp["fc2"].weight), "bias": array(block.mlp["fc2"].bias)}
        }
    })

payload = {
    "format": "littleGPT-v1",
    "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "config": {"block_size": BLOCK_SIZE, "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER},
    "vocab": chars,
    "parameters": parameter_count,
    "training": {"steps": steps, "loss": last_metrics["val"], "train_loss": last_metrics["train"], "dataset": "Tiny Shakespeare", "seed": SEED},
    "state": state
}
Path("model.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
print(f"Exported model.json ({Path('model.json').stat().st_size / 1024:.1f} KB)", flush=True)
