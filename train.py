import json
import os
import re
import time
import urllib.parse
import urllib.request
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# littleGPT v4: story-isolated batches, regularization, and best-validation export.
SEED = 1337
BLOCK_SIZE = 96
BATCH_SIZE = 16
N_EMBD = 96
N_HEAD = 4
N_LAYER = 3
DROPOUT = 0.08
VOCAB_LIMIT = 4096
DATASET_ROWS = int(os.environ.get("DATASET_ROWS", "30000"))
MAX_STEPS = int(os.environ.get("TRAIN_STEPS", "3000"))
CHUNK_STEPS = int(os.environ.get("CHUNK_STEPS", "500"))
CHECKPOINT_PATH = Path("checkpoint.pt")
EVAL_INTERVAL = 500
EVAL_BATCHES = 8
LEARNING_RATE = 1.8e-3
DATASET = "roneneldan/TinyStories"
ROWS_API = "https://datasets-server.huggingface.co/rows"
TOKEN_RE = re.compile(r"""[a-z0-9]+(?:'[a-z0-9]+)?|[.,!?;:"()\-]""")

torch.manual_seed(SEED)
torch.set_num_threads(2)
torch.backends.mkldnn.enabled = True
device = torch.device("cpu")


def download_stories():
    cache = Path("training_stories.json")
    stories = []
    if cache.exists():
        try:
            stories = json.loads(cache.read_text(encoding="utf-8"))
            if not isinstance(stories, list):
                stories = []
        except Exception:
            stories = []
    if len(stories) >= DATASET_ROWS:
        print(f"Loaded {len(stories)} cached stories.", flush=True)
        return stories[:DATASET_ROWS]

    # Resume partial downloads instead of throwing away thousands of rows after a rate limit.
    page_size = 100
    offset = len(stories)
    print(f"Continuing dataset download from story {offset:,}.", flush=True)
    while offset < DATASET_ROWS:
        length = min(page_size, DATASET_ROWS - offset)
        params = urllib.parse.urlencode({
            "dataset": DATASET, "config": "default", "split": "train",
            "offset": offset, "length": length,
        })
        request = urllib.request.Request(
            ROWS_API + "?" + params,
            headers={"User-Agent": "littleGPT-training/4.1"},
        )
        last_error = None
        for attempt in range(7):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                batch = [row.get("row", {}).get("text", "") for row in payload.get("rows", [])]
                batch = [story.strip() for story in batch if isinstance(story, str) and len(story.strip()) > 30]
                if not batch:
                    raise RuntimeError(f"No stories returned at offset {offset}")
                stories.extend(batch)
                offset += len(batch)
                # Persist every page so a later run can resume after throttling/cancellation.
                cache.write_text(json.dumps(stories, ensure_ascii=False), encoding="utf-8")
                print(f"Downloaded {len(stories):,} / {DATASET_ROWS:,} stories…", flush=True)
                time.sleep(0.35)
                break
            except Exception as error:
                last_error = error
                if attempt == 6:
                    raise RuntimeError(f"TinyStories download failed at offset {offset}: {last_error}") from error
                retry_after = 60 * (attempt + 1)
                if hasattr(error, "headers") and error.headers:
                    try:
                        retry_after = max(retry_after, int(error.headers.get("Retry-After", "0")))
                    except (TypeError, ValueError):
                        pass
                print(f"Dataset request throttled/failed ({error}); retrying in {retry_after}s…", flush=True)
                time.sleep(retry_after)

    if len(stories) < 1000:
        raise RuntimeError(f"Only {len(stories)} stories downloaded; refusing to train.")
    return stories[:DATASET_ROWS]

stories = download_stories()
# Shuffle before splitting so validation stories are held out from training.
import random
random.Random(SEED).shuffle(stories)
split_at = max(1, int(len(stories) * 0.95))
train_stories, val_stories = stories[:split_at], stories[split_at:]


def normalize_text(text):
    # Repair common mojibake and normalize smart punctuation before vocabulary creation.
    replacements = {
        "â€œ": '"', "â€": '"', "â€˜": "'", "â€™": "'",
        "â€“": "-", "â€”": "-", "â€¦": "...",
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        # These artifacts appeared in generated text in place of quotes/apostrophes.
        "€": '"', "™": "'",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def tokenize(text):
    return TOKEN_RE.findall(normalize_text(text).lower())


counter = Counter()
for story in train_stories:
    counter.update(tokenize(story))
special = ["<unk>", "<bos>", "<eos>"]
vocab = special + [token for token, _ in counter.most_common(VOCAB_LIMIT - len(special))]
stoi = {token: index for index, token in enumerate(vocab)}
UNK, BOS, EOS = stoi["<unk>"], stoi["<bos>"], stoi["<eos>"]


def encode_stories(source):
    encoded = []
    for story in source:
        ids = [BOS] + [stoi.get(token, UNK) for token in tokenize(story)] + [EOS]
        if len(ids) >= BLOCK_SIZE + 2:
            encoded.append(torch.tensor(ids, dtype=torch.long))
    if not encoded:
        raise RuntimeError("No stories are long enough for the selected context length.")
    return encoded


train_sequences = encode_stories(train_stories)
val_sequences = encode_stories(val_stories)
vocab_size = len(vocab)


def get_batch(source):
    # Every row in this batch comes from a different story.
    indices = torch.randperm(len(source))[:BATCH_SIZE].tolist()
    xs, ys = [], []
    for index in indices:
        story = source[index]
        start = torch.randint(0, len(story) - BLOCK_SIZE, (1,)).item()
        xs.append(story[start:start + BLOCK_SIZE])
        ys.append(story[start + 1:start + BLOCK_SIZE + 1])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


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
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.fc1 = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.fc2 = nn.Linear(4 * N_EMBD, N_EMBD)
        self.mlp_dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = x + self.attn_dropout(self.attn(self.ln1(x)))
        x = x + self.mlp_dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        positions = torch.arange(t, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        return logits, loss


model = TinyGPT().to(device)
parameter_count = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.10)
start_step = 0
best_val = float("inf")
best_model_state = None
last_metrics = {"train": None, "val": None}
config_signature = {
    "block_size": BLOCK_SIZE, "n_embd": N_EMBD, "n_head": N_HEAD,
    "n_layer": N_LAYER, "dropout": DROPOUT, "dataset_rows": DATASET_ROWS,
    "tokenizer_version": 2,
}

if CHECKPOINT_PATH.exists():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if (checkpoint.get("format") != "littleGPT-training-checkpoint-v3"
            or checkpoint.get("vocab") != vocab
            or checkpoint.get("config") != config_signature):
        raise RuntimeError("Checkpoint belongs to another model version/configuration; refusing to resume it.")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = int(checkpoint["step"])
    best_val = float(checkpoint.get("best_val", float("inf")))
    best_model_state = checkpoint.get("best_model")
    last_metrics = checkpoint.get("metrics", last_metrics)
    torch.set_rng_state(checkpoint["torch_rng"])
    print(f"RESUMED v4 from checkpoint at step {start_step}/{MAX_STEPS}", flush=True)

print(
    f"littleGPT v5 | {parameter_count:,} parameters | vocab={vocab_size} | "
    f"stories={len(stories):,} | train={len(train_sequences):,} | "
    f"validation={len(val_sequences):,} | batch={BATCH_SIZE} distinct stories | CPU",
    flush=True,
)


@torch.no_grad()
def estimate_loss():
    model.eval()
    metrics = {}
    for name, source in (("train", train_sequences), ("val", val_sequences)):
        losses = []
        for _ in range(EVAL_BATCHES):
            xb, yb = get_batch(source)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        metrics[name] = round(sum(losses) / len(losses), 4)
    model.train()
    return metrics


def clone_state_dict(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def learning_rate(step):
    warmup = min(1.0, (step + 1) / 200)
    progress = min(1.0, step / max(1, MAX_STEPS - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * warmup * (0.10 + 0.90 * cosine)


model.train()
for step in range(start_step, min(start_step + CHUNK_STEPS, MAX_STEPS)):
    lr = learning_rate(step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    xb, yb = get_batch(train_sequences)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if (step + 1) % EVAL_INTERVAL == 0 or step == 0 or step + 1 == MAX_STEPS:
        last_metrics = estimate_loss()
        improved = last_metrics["val"] < best_val
        if improved:
            best_val = last_metrics["val"]
            best_model_state = clone_state_dict(model)
        marker = " BEST" if improved else ""
        print(
            f"step {step + 1}/{MAX_STEPS} "
            f"train={last_metrics['train']:.4f} val={last_metrics['val']:.4f} "
            f"best_val={best_val:.4f}{marker}",
            flush=True,
        )

completed_steps = min(start_step + CHUNK_STEPS, MAX_STEPS)
torch.save({
    "format": "littleGPT-training-checkpoint-v2",
    "step": completed_steps,
    "model": model.state_dict(),
    "best_model": best_model_state,
    "best_val": best_val,
    "optimizer": optimizer.state_dict(),
    "vocab": vocab,
    "config": config_signature,
    "metrics": last_metrics,
    "torch_rng": torch.get_rng_state(),
}, CHECKPOINT_PATH)
print(f"CHECKPOINT SAVED at step {completed_steps}/{MAX_STEPS} (best val={best_val:.4f})", flush=True)

if completed_steps < MAX_STEPS:
    print("Chunk complete; waiting for the next checkpointed chunk.", flush=True)
    raise SystemExit(0)

# Export the checkpoint with the lowest held-out validation loss.
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    model.eval()
    print(f"Exporting best-validation checkpoint (val={best_val:.4f})", flush=True)


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
            "fc1": {"weight": array(block.fc1.weight), "bias": array(block.fc1.bias)},
            "fc2": {"weight": array(block.fc2.weight), "bias": array(block.fc2.bias)},
        },
    })

payload = {
    "format": "littleGPT-v5",
    "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "config": {
        "block_size": BLOCK_SIZE, "n_embd": N_EMBD, "n_head": N_HEAD,
        "n_layer": N_LAYER, "unk_id": UNK, "bos_id": BOS, "eos_id": EOS,
    },
    "vocab": vocab,
    "parameters": parameter_count,
    "training": {
        "steps": MAX_STEPS, "loss": best_val,
        "train_loss": last_metrics["train"],
        "dataset": f"TinyStories ({len(stories)} stories)",
        "seed": SEED, "train_tokens": sum(len(s) for s in train_sequences),
        "validation_tokens": sum(len(s) for s in val_sequences),
        "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "dropout": DROPOUT, "best_validation_loss": best_val,
    },
    "state": state,
}
Path("model.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
print(f"Exported model.json ({Path('model.json').stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
