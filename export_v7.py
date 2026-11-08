import torch, json, base64, datetime
from pathlib import Path

CHECKPOINT = Path("checkpoint-v7-25k.pt")
OUTPUT = Path("model.json")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Missing {CHECKPOINT}")

print("Loading LittleGPT checkpoint...")
ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

weights = ck.get("best_model")
if weights is None:
    raise RuntimeError("No best_model found in checkpoint!")

vocab = ck["vocab"]
config = ck["config"]

def pack(tensor):
    raw = tensor.detach().cpu().to(torch.float16).contiguous().numpy().tobytes()
    return {"f16": base64.b64encode(raw).decode("ascii")}

def get(name):
    return pack(weights[name])

state = {
    "wte": get("wte.weight"),
    "wpe": get("wpe.weight"),
    "ln_f": {
        "weight": get("ln_f.weight"),
        "bias": get("ln_f.bias")
    },
    "lm_head": {
        "weight": get("lm_head.weight"),
        "bias": get("lm_head.bias")
    },
    "blocks": []
}

for i in range(config["n_layer"]):
    p = f"blocks.{i}."
    state["blocks"].append({
        "ln1": {
            "weight": get(p+"ln1.weight"),
            "bias": get(p+"ln1.bias")
        },
        "attn": {
            "qkv": {
                "weight": get(p+"attn.qkv.weight"),
                "bias": get(p+"attn.qkv.bias")
            },
            "proj": {
                "weight": get(p+"attn.proj.weight"),
                "bias": get(p+"attn.proj.bias")
            }
        },
        "ln2": {
            "weight": get(p+"ln2.weight"),
            "bias": get(p+"ln2.bias")
        },
        "mlp": {
            "fc1": {
                "weight": get(p+"fc1.weight"),
                "bias": get(p+"fc1.bias")
            },
            "fc2": {
                "weight": get(p+"fc2.weight"),
                "bias": get(p+"fc2.bias")
            }
        }
    })

max_width = max(len(token.encode("latin1")) for token in vocab[3:])

payload = {
    "format": "littleGPT-v7",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "config": {
        "block_size": config["block_size"],
        "n_embd": config["n_embd"],
        "n_head": config["n_head"],
        "n_layer": config["n_layer"],
        "vocab_size": config["vocab_size"],
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "tokenizer": "byte-subword-greedy-v1",
        "max_token_bytes": max_width
    },
    "vocab": vocab,
    "parameters": 9796448,
    "training": {
        "steps": 10500,
        "loss": ck["best"],
        "best_validation_loss": ck["best"],
        "train_loss": None,
        "dataset": "TinyStories official train split (25000 stories); official validation split (21990 stories); story-boundary-safe windows",
        "seed": 1337,
        "learning_rate": 8e-4,
        "batch_size": 16,
        "dropout": 0.10,
        "device": "mps"
    },
    "state": state
}

OUTPUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

print("Export complete!")
print("Best validation loss:", ck["best"])
print("Exported model size:", round(OUTPUT.stat().st_size / 1024 / 1024, 2), "MB")
print("Model steps: 10,500 (best checkpoint)")
