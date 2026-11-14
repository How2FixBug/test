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


def read_streams(dataset_path: Path, validation_fraction: float, seed: int):
    records = []
    with dataset_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            ids = record.get("input_ids")
            if not isinstance(ids, list) or len(ids) < 3:
                raise ValueError(f"Invalid token sequence on line {line_number}")
            records.append(ids)
    if len(records) < 10:
        raise ValueError("Need at least 10 tokenized examples for a train/validation split.")
    random.Random(seed).shuffle(records)
    split = max(1, int(len(records) * (1.0 - validation_fraction)))
    train_records, validation_records = records[:split], records[split:]
    return (
        torch.tensor([token for row in train_records for token in row], dtype=torch.long),
        torch.tensor([token for row in validation_records for token in row], dtype=torch.long),
        len(records),
    )


def get_batch(stream: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    starts = torch.randint(0, stream.numel() - block_size - 1, (batch_size,))
    offsets = torch.arange(block_size + 1)
    batch = stream[starts[:, None] + offsets[None, :]].to(device)
    return batch[:, :-1], batch[:, 1:]


@torch.no_grad()
def estimate_loss(model, train_stream, val_stream, batch_size, block_size, device):
    model.eval()
    result = {}
    for name, stream in (("train", train_stream), ("val", val_stream)):
        losses = []
        for _ in range(20):
            x, y = get_batch(stream, batch_size, block_size, device)
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
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
    args = parser.parse_args()

    config_path = HERE / "tokenizer_config.json"
    data_path = HERE / "tokenized_dataset.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vocab_size = int(config["vocab_size"])
    if "vocab_hex" not in config or "merges" not in config:
        raise SystemExit(
            "Your tokenizer config is still the old 262-token version. "
            "Run: python3 tokenizer.py"
        )

    train_stream, val_stream, examples = read_streams(
        data_path, args.validation_fraction, args.seed
    )
    largest_id = max(int(train_stream.max()), int(val_stream.max()))
    if largest_id >= vocab_size:
        raise SystemExit(
            f"Dataset contains token ID {largest_id}, but tokenizer vocab_size is {vocab_size}. "
            "Regenerate it with: python3 tokenizer.py"
        )

    block_size = min(args.block_size, 512)
    if min(train_stream.numel(), val_stream.numel()) <= block_size + 1:
        raise SystemExit(
            "Not enough tokens in one split for this block size. Try --block-size 128."
        )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = LoopCore(vocab_size=vocab_size, max_seq_len=512).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    output_dir = HERE / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    best_val = float("inf")
    stale_evaluations = 0
    started = time.time()

    print(f"ASTRA 0.1 — LoopCore")
    print(f"Examples: {examples:,} | Vocabulary: {vocab_size:,} | Parameters: {parameter_count:,}")
    print(f"Device: {device} | Context: {block_size} | Shared passes: {model.loop_passes}")
    print("Training from random initialization. No pretrained weights are used.")

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_stream, args.batch_size, block_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            metrics = estimate_loss(
                model, train_stream, val_stream, args.batch_size, block_size, device
            )
            elapsed = time.time() - started
            print(
                f"step {step:5d}/{args.steps} | "
                f"train {metrics['train']:.4f} | val {metrics['val']:.4f} | "
                f"{elapsed:.1f}s"
            )
            if metrics["val"] < best_val:
                best_val = metrics["val"]
                stale_evaluations = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "vocab_size": vocab_size,
                            "width": 512,
                            "heads": 8,
                            "ff_width": 1792,
                            "max_seq_len": 512,
                            "local_window": 128,
                            "loop_passes": 4,
                        },
                        "step": step,
                        "best_val_loss": best_val,
                    },
                    output_dir / "astra-0.1-best.pt",
                )
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
