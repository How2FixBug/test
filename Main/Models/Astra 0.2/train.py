#!/usr/bin/env python3
"""Train ASTRA 0.2 locally on Apple Silicon (MPS) or CPU."""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import AstraGPT

HERE = Path(__file__).resolve().parent
ASTRA_TOKEN = 258
IGNORE_INDEX = -100

def read_records(dataset_path: Path, expected_fingerprint: str):
    records = []
    with dataset_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("tokenizer_fingerprint") != expected_fingerprint:
                raise ValueError(
                    f"Tokenizer fingerprint mismatch on line {line_number}. "
                    "Run python3 tokenizer.py to regenerate both artifacts."
                )
            ids = record.get("input_ids")
            if not isinstance(ids, list) or len(ids) < 4:
                raise ValueError(f"Invalid token sequence on line {line_number}")
            if ASTRA_TOKEN not in ids:
                raise ValueError(f"Missing <astra> marker on line {line_number}")
            records.append([int(token) for token in ids])
    if len(records) < 10:
        raise ValueError("Need at least 10 tokenized examples.")
    return records

def split_records(records, validation_fraction: float, seed: int):
    """Split by exact response so paraphrases cannot leak across splits."""
    groups = {}
    for record in records:
        key = tuple(record[record.index(ASTRA_TOKEN) + 1:])
        groups.setdefault(key, []).append(record)
    target = max(1, round(len(records) * validation_fraction))
    rng = random.Random(seed)
    buckets = {}
    for group in groups.values():
        buckets.setdefault(len(group), []).append(group)
    validation, count = [], 0
    for size in sorted(buckets):
        rng.shuffle(buckets[size])
        for group in buckets[size]:
            if count < target and count + len(group) <= target:
                validation.extend(group)
                count += len(group)
    if not validation:
        raise ValueError("Could not create a non-empty validation split.")
    held_out = {tuple(r[r.index(ASTRA_TOKEN) + 1:]) for r in validation}
    training = [
        r for r in records
        if tuple(r[r.index(ASTRA_TOKEN) + 1:]) not in held_out
    ]
    if not training:
        raise ValueError("Validation split left no training examples.")
    return training, validation, len(groups)

def build_windows(records, block_size: int):
    """Cover every assistant token using overlapping context windows."""
    target_size = max(1, block_size // 2)
    context_size = block_size - target_size
    windows = []

    for record in records:
        try:
            assistant_start = record.index(ASTRA_TOKEN)
        except ValueError:
            raise ValueError("A training example is missing its <astra> marker.")

        first_target = assistant_start + 1
        if first_target >= len(record):
            raise ValueError("A training example has no assistant response tokens.")

        # Each target token appears in exactly one window. Later windows retain
        # up to context_size preceding tokens, including earlier assistant text.
        for target_start in range(first_target, len(record), target_size):
            target_end = min(len(record), target_start + target_size)
            input_start = max(0, target_start - context_size - 1)

            inputs = record[input_start:target_end - 1]
            targets = record[input_start + 1:target_end]
            mask_count = target_start - (input_start + 1)
            if mask_count:
                targets[:mask_count] = [IGNORE_INDEX] * mask_count

            if not inputs or len(inputs) != len(targets) or len(inputs) > block_size:
                raise ValueError("Invalid training window generated from a tokenized example.")
            windows.append((inputs, targets))

    if not windows:
        raise ValueError("No assistant training windows were generated.")
    return windows


def get_batch(windows, batch_size: int, block_size: int, device: torch.device, rng=None):
    """Sample fixed-size batches from prebuilt assistant-response windows."""
    rng = rng or random
    x_batch = torch.zeros((batch_size, block_size), dtype=torch.long)
    y_batch = torch.full((batch_size, block_size), IGNORE_INDEX, dtype=torch.long)

    for row in range(batch_size):
        inputs, targets = windows[rng.randrange(len(windows))]
        length = len(inputs)
        x_batch[row, :length] = torch.tensor(inputs, dtype=torch.long)
        y_batch[row, :length] = torch.tensor(targets, dtype=torch.long)

    return x_batch.to(device), y_batch.to(device)

@torch.no_grad()
def estimate_loss(model, train_windows, val_windows, batch_size, block_size, device):
    model.eval()
    result = {}
    for name, windows, seed in (
        ("train", train_windows, 1234), ("val", val_windows, 5678)
    ):
        losses = []
        rng = random.Random(seed)
        for _ in range(20):
            x, y = get_batch(windows, batch_size, block_size, device, rng=rng)
            logits = model(x)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                ignore_index=IGNORE_INDEX,
            ).item())
        result[name] = sum(losses) / len(losses)
    model.train()
    return result

def main():
    parser = argparse.ArgumentParser(description="Train ASTRA's LoopCore neural network.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Continue from the best checkpoint")
    args = parser.parse_args()

    config_path = HERE / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "vocab_hex" not in config or "merges" not in config or "fingerprint" not in config:
        raise SystemExit("Tokenizer artifacts are outdated. Run python3 tokenizer.py first.")
    vocab_size = int(config["vocab_size"])
    records = read_records(HERE / "tokenized_dataset.jsonl", config["fingerprint"])
    largest_id = max(max(record) for record in records)
    if largest_id >= vocab_size:
        raise SystemExit(
            f"Dataset contains token ID {largest_id}, but vocab_size is {vocab_size}. "
            "Run python3 tokenizer.py again."
        )

    train_records, val_records, response_groups = split_records(
        records, args.validation_fraction, args.seed
    )
    block_size = min(args.block_size, 512)
    if block_size < 2:
        raise SystemExit("Block size must be at least 2 tokens.")
    train_windows = build_windows(train_records, block_size)
    val_windows = build_windows(val_records, block_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = AstraGPT(vocab_size=vocab_size, max_seq_len=512).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )
    output_dir = HERE / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "astra-0.2-best.pt"
    start_step, best_val = 0, float("inf")
    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit("Cannot resume: checkpoint not found.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("architecture_version") != 3:
            raise SystemExit("Cannot resume: checkpoint architecture version does not match.")
        if checkpoint.get("tokenizer_fingerprint") != config["fingerprint"]:
            raise SystemExit("Cannot resume: checkpoint tokenizer does not match current dataset.")
        if checkpoint.get("model_config", {}).get("vocab_size") != vocab_size:
            raise SystemExit("Cannot resume: checkpoint vocabulary size does not match.")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("best_val_loss", float("inf")))
        print("Resuming from step {} | previous best val {:.4f}".format(start_step, best_val))
    stale_evaluations = 0
    started = time.time()

    print("ASTRA 0.2 — AstraGPT (decoder-only Transformer)")
    print(
        f"Examples: {len(records):,} | Train: {len(train_records):,} | "
        f"Validation: {len(val_records):,} | Unique response groups: {response_groups:,}"
    )
    print(f"Vocabulary: {vocab_size:,} | Parameters: {parameter_count:,}")
    print(f"Device: {device} | Context: {block_size} | Transformer layers: {model.num_layers}")
    print(f"Response windows: {len(train_windows):,} train | {len(val_windows):,} validation")
    print("Validation groups are isolated by exact response.")
    print("Loss is calculated on assistant replies only; user prompts are context, not targets.")
    if args.resume:
        print("Continuing from the saved checkpoint; optimizer state is freshly initialized.")
    else:
        print("Training from random initialization. No pretrained weights are used.")

    final_step = start_step + args.steps
    for step in range(start_step + 1, final_step + 1):
        x, y = get_batch(train_windows, args.batch_size, block_size, device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == start_step + 1 or step % args.eval_interval == 0 or step == final_step:
            metrics = estimate_loss(
                model, train_windows, val_windows, args.batch_size, block_size, device
            )
            elapsed = time.time() - started
            print(
                f"step {step:5d}/{final_step} | train {metrics['train']:.4f} | "
                f"val {metrics['val']:.4f} | {elapsed:.1f}s"
            )
            if metrics["val"] < best_val:
                best_val, stale_evaluations = metrics["val"], 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "vocab_size": vocab_size, "width": 384, "heads": 6,
                        "ff_width": 1024, "max_seq_len": 512,
                        "layers": 4, "dropout": 0.05,
                    },
                    "architecture_version": 3,
                    "tokenizer_fingerprint": config["fingerprint"],
                    "step": step,
                    "best_val_loss": best_val,
                }, checkpoint_path)
                print("  saved new best checkpoint")
            else:
                stale_evaluations += 1
                if stale_evaluations >= args.patience:
                    print("Early stopping: validation loss stopped improving.")
                    break

    print(f"Finished. Best validation loss: {best_val:.4f}")
    print(f"Checkpoint: {output_dir / 'astra-0.2-best.pt'}")

if __name__ == "__main__":
    main()
