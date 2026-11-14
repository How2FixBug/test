#!/usr/bin/env python3
"""Train ASTRA 0.1 locally on Apple Silicon (MPS) or CPU."""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import LoopCore

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

def get_batch(records, batch_size: int, block_size: int, device: torch.device, rng=None):
    """Sample rows from individual conversations, never across boundaries."""
    rng = rng or random
    x_batch = torch.zeros((batch_size, block_size), dtype=torch.long)
    y_batch = torch.full((batch_size, block_size), IGNORE_INDEX, dtype=torch.long)
    for row in range(batch_size):
        record = records[rng.randrange(len(records))]
        max_start = max(0, len(record) - (block_size + 1))
        start = rng.randint(0, max_start)
        chunk = record[start:start + block_size + 1]
        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
        usable = len(chunk) - 1
        if usable < 1:
            raise ValueError("A training example must contain at least two tokens.")
        x_batch[row, :usable] = chunk_tensor[:-1]
        y_batch[row, :usable] = chunk_tensor[1:]
    return x_batch.to(device), y_batch.to(device)

@torch.no_grad()
def estimate_loss(model, train_records, val_records, batch_size, block_size, device):
    model.eval()
    result = {}
    for name, records, seed in (
        ("train", train_records, 1234), ("val", val_records, 5678)
    ):
        losses = []
        rng = random.Random(seed)
        for _ in range(20):
            x, y = get_batch(records, batch_size, block_size, device, rng=rng)
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
    parser.add_argument("--block-size", type=int, default=256)
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
    if min(min(map(len, train_records)), min(map(len, val_records))) < 2:
        raise SystemExit("A training or validation example is too short.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = LoopCore(vocab_size=vocab_size, max_seq_len=512).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )
    output_dir = HERE / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "astra-0.1-best.pt"
    start_step, best_val = 0, float("inf")
    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit("Cannot resume: checkpoint not found.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("architecture_version") != 2:
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

    print("ASTRA 0.1 — LoopCore (causal-memory revision)")
    print(
        f"Examples: {len(records):,} | Train: {len(train_records):,} | "
        f"Validation: {len(val_records):,} | Unique response groups: {response_groups:,}"
    )
    print(f"Vocabulary: {vocab_size:,} | Parameters: {parameter_count:,}")
    print(f"Device: {device} | Context: {block_size} | Shared passes: {model.loop_passes}")
    print("Validation groups are isolated by exact response.")
    if args.resume:
        print("Continuing from the saved checkpoint; optimizer state is freshly initialized.")
    else:
        print("Training from random initialization. No pretrained weights are used.")

    final_step = start_step + args.steps
    for step in range(start_step + 1, final_step + 1):
        x, y = get_batch(train_records, args.batch_size, block_size, device)
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
                model, train_records, val_records, args.batch_size, block_size, device
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
                        "vocab_size": vocab_size, "width": 512, "heads": 8,
                        "ff_width": 1792, "max_seq_len": 512,
                        "local_window": 128, "loop_passes": 4,
                    },
                    "architecture_version": 2,
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
    print(f"Checkpoint: {output_dir / 'astra-0.1-best.pt'}")

if __name__ == "__main__":
    main()
