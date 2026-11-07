import json
import os
import re
import random
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# A small word-level GPT. The model is initialized randomly every run.
SEED = 1337
BLOCK_SIZE = 128
BATCH_SIZE = 16
N_EMBD = 112
N_HEAD = 4
N_LAYER = 3
VOCAB_LIMIT = 4096
DATASET_ROWS = int(os.environ.get("DATASET_ROWS", "6000"))
MAX_STEPS = int(os.environ.get("TRAIN_STEPS", "5000"))
CHUNK_STEPS = int(os.environ.get("CHUNK_STEPS", str(MAX_STEPS)))
CHECKPOINT_PATH = Path("checkpoint.pt")
EVAL_INTERVAL = max(200, min(400, MAX_STEPS // 8))
LEARNING_RATE = 2.0e-3
DATASET = "roneneldan/TinyStories"
ROWS_API = "https://datasets-server.huggingface.co/rows"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", re.IGNORECASE)

torch.manual_seed(SEED)
torch.set_num_threads(2)
torch.backends.mkldnn.enabled = True
device = torch.device("cpu")


def download_stories():
    cached_stories = Path("training_stories.json")
    if cached_stories.exists():
        stories = json.loads(cached_stories.read_text(encoding="utf-8"))
        print(f"Loaded {len(stories)} cached stories.", flush=True)
        random.Random(SEED).shuffle(stories)
        return stories
    stories = []
    page_size = 100
    for offset in range(0, DATASET_ROWS, page_size):
        length = min(page_size, DATASET_ROWS - offset)
        params = urllib.parse.urlencode({
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        })
        request = urllib.request.Request(
            ROWS_API + "?" + params,
            headers={"User-Agent": "littleGPT-training/2.0"},
        )
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                batch = [item.get("row", {}).get("text", "") for item in payload.get("rows", [])]
                batch = [story.strip() for story in batch if isinstance(story, str) and len(story.strip()) > 30]
                if not batch:
                    raise RuntimeError(f"Dataset returned no stories at offset {offset}")
                stories.extend(batch)
                print(f"Downloaded {len(stories)} stories…", flush=True)
                break
            except Exception as error:
                last_error = error
                if attempt == 2:
                    raise RuntimeError(f"Could not download TinyStories at offset {offset}: {last_error}") from error
                time.sleep(2 * (attempt + 1))
    if len(stories) < 100:
        raise RuntimeError("Not enough training stories were downloaded.")
    cached_stories.write_text(json.dumps(stories), encoding="utf-8")
    random.Random(SEED).shuffle(stories)
    return stories


stories = download_stories()
split_at = max(1, int(len(stories) * 0.95))
train_stories, val_stories = stories[:split_at], stories[split_at:]

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

counter = Counter()
for story in train_stories:
    counter.update(tokenize(story))
special = ["<unk>", "<bos>", "<eos>"]
vocab = special + [token for token, _ in counter.most_common(VOCAB_LIMIT - len(special))]
stoi = {token: index for index, token in enumerate(vocab)}
UNK, BOS, EOS = stoi["<unk>"], stoi["<bos>"], stoi["<eos>"]

def encode_story(story):
    return [BOS] + [stoi.get(token, UNK) for token in tokenize(story)] + [EOS]

train_ids = torch.tensor([token for story in train_stories for token in encode_story(story)], dtype=torch.long)
val_ids = torch.tensor([token for story in val_stories for token in encode_story(story)], dtype=torch.long)
if len(val_ids) < BLOCK_SIZE + 2:
    val_ids = train_ids[-max(BLOCK_SIZE * 4, 512):]

vocab_size = len(vocab)

def get_batch(source):
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(N_EMBD, dim=2)
        head_dim = c // N_HEAD
        q = q.view(b, t, N_HEAD, head_dim).transpose(1, 2)
        k = k.view(b, t, N_HEAD, head_dim).transpose(1, 2)
        v = v.view(b, t, N_HEAD, head_dim).transpose(1, 2)
        # PyTorch's optimized CPU attention kernel avoids the slow Python-level mask path.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        _, t = idx.shape
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
parameter_count = sum(parameter.numel() for parameter in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.08)
start_step = 0
last_metrics = {"train": None, "val": None}
if CHECKPOINT_PATH.exists():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if checkpoint.get("format") != "littleGPT-training-checkpoint-v1" or checkpoint.get("vocab") != vocab:
        raise RuntimeError("Checkpoint does not match this dataset/vocabulary; refusing unsafe resume.")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = int(checkpoint["step"])
    last_metrics = checkpoint.get("metrics", last_metrics)
    torch.set_rng_state(checkpoint["torch_rng"])
    print(f"RESUMED from checkpoint at step {start_step}/{MAX_STEPS}", flush=True)


def learning_rate(step):
    warmup = min(1.0, (step + 1) / 150)
    progress = min(1.0, step / max(1, MAX_STEPS - 1))
    cosine = 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
    return LEARNING_RATE * warmup * (0.15 + 0.85 * cosine)
print(
    f"Random init: {parameter_count:,} parameters | vocab={vocab_size} | "
    f"train tokens={len(train_ids):,} | CPU",
    flush=True,
)

@torch.no_grad()
def estimate_loss():
    model.eval()
    metrics = {}
    for name, source in (("train", train_ids), ("val", val_ids)):
        losses = []
        for _ in range(8):
            xb, yb = get_batch(source)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        metrics[name] = round(sum(losses) / len(losses), 4)
    model.train()
    return metrics


for step in range(start_step, min(start_step + CHUNK_STEPS, MAX_STEPS)):
    lr = learning_rate(step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    xb, yb = get_batch(train_ids)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if (step + 1) % EVAL_INTERVAL == 0 or step == 0 or step + 1 == MAX_STEPS:
        last_metrics = estimate_loss()
        print(
            f"step {step + 1}/{MAX_STEPS} "
            f"train={last_metrics['train']:.4f} val={last_metrics['val']:.4f}",
            flush=True,
        )

completed_steps = min(start_step + CHUNK_STEPS, MAX_STEPS)
torch.save({
    "format": "littleGPT-training-checkpoint-v1",
    "step": completed_steps,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "vocab": vocab,
    "metrics": last_metrics,
    "torch_rng": torch.get_rng_state(),
}, CHECKPOINT_PATH)
print(f"CHECKPOINT SAVED at step {completed_steps}/{MAX_STEPS}", flush=True)

if completed_steps < MAX_STEPS:
    print("Chunk complete; waiting for the next checkpointed chunk.", flush=True)
    raise SystemExit(0)


def array(tensor):
    return tensor.detach().cpu().tolist()


state = {
    "wte": array(model.wte.weight),
    "wpe": array(model.wpe.weight),
    "ln_f": {"weight": array(model.ln_f.weight), "bias": array(model.ln_f.bias)},
    "lm_head": {"weight": array(model.lm_head.weight), "bias": array(model.lm_head.bias)},
    "blocks": [],
}
for block in model.blocks:
    state["blocks"].append({
        "ln1": {"weight": array(block.ln1.weight), "bias": array(block.ln1.bias)},
        "attn": {
            "qkv": {"weight": array(block.attn.qkv.weight), "bias": array(block.attn.qkv.bias)},
            "proj": {"weight": array(block.attn.proj.weight), "bias": array(block.attn.proj.bias)},
        },
        "ln2": {"weight": array(block.ln2.weight), "bias": array(block.ln2.bias)},
        "mlp": {
            "fc1": {"weight": array(block.mlp["fc1"].weight), "bias": array(block.mlp["fc1"].bias)},
            "fc2": {"weight": array(block.mlp["fc2"].weight), "bias": array(block.mlp["fc2"].bias)},
        },
    })

payload = {
    "format": "littleGPT-v3",
    "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "config": {
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "unk_id": UNK,
        "bos_id": BOS,
        "eos_id": EOS,
    },
    "vocab": vocab,
    "parameters": parameter_count,
    "training": {
        "steps": MAX_STEPS,
        "loss": last_metrics["val"],
        "train_loss": last_metrics["train"],
        "dataset": f"TinyStories ({len(stories)} stories)",
        "seed": SEED,
        "train_tokens": len(train_ids),
        "validation_tokens": len(val_ids),
        "learning_rate": LEARNING_RATE,
    },
    "state": state,
}
Path("model.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
print(f"Exported model.json ({Path('model.json').stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
