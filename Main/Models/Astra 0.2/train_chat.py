#!/usr/bin/env python3
"""Fine-tune ASTRA 0.2 on the isolated chat corpus; never writes the original checkpoint."""
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
CHAT_DATASET = HERE / "tokenized_chat_dataset.jsonl"
BASE_CHECKPOINT = HERE / "checkpoints" / "astra-0.2-best.pt"
CHAT_CHECKPOINT = HERE / "checkpoints" / "astra-0.2-chat-best.pt"
ASTRA_TOKEN = 258
ASTRA_END_TOKEN = 259
IGNORE_INDEX = -100


def read_records(path: Path, fingerprint: str, vocab_size: int):
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("tokenizer_fingerprint") != fingerprint:
                raise ValueError(f"Tokenizer fingerprint mismatch on line {line_number}.")
            ids = record.get("input_ids")
            if not isinstance(ids, list) or len(ids) < 4:
                raise ValueError(f"Invalid token sequence on line {line_number}.")
            ids = [int(token) for token in ids]
            if ASTRA_TOKEN not in ids or max(ids) >= vocab_size or min(ids) < 0:
                raise ValueError(f"Invalid token IDs or missing ASTRA marker on line {line_number}.")
            records.append(ids)
    if len(records) < 10:
        raise ValueError("Need at least 10 tokenized conversations.")
    return records


def split_records(records, fraction: float, seed: int):
    groups = {}
    for record in records:
        groups.setdefault(tuple(record), []).append(record)
    buckets = {}
    for group in groups.values():
        buckets.setdefault(len(group), []).append(group)
    rng = random.Random(seed)
    target = max(1, round(len(records) * fraction))
    validation, count = [], 0
    for size in sorted(buckets):
        rng.shuffle(buckets[size])
        for group in buckets[size]:
            if count < target and count + len(group) <= target:
                validation.extend(group)
                count += len(group)
    if not validation:
        raise ValueError("Could not create a non-empty validation split.")
    held_out = {tuple(record) for record in validation}
    training = [record for record in records if tuple(record) not in held_out]
    if not training:
        raise ValueError("Validation split left no training examples.")
    return training, validation


def build_windows(records, block_size: int):
    target_size = max(1, block_size // 2)
    context_size = block_size - target_size
    windows = []
    for record in records:
        assistant_targets = [False] * len(record)
        inside_astra = False
        for position, token in enumerate(record):
            if token == ASTRA_TOKEN:
                if inside_astra:
                    raise ValueError("Nested <astra> message found.")
                inside_astra = True
                continue
            if token == ASTRA_END_TOKEN:
                if not inside_astra:
                    raise ValueError("Unmatched </astra> marker found.")
                assistant_targets[position] = True
                inside_astra = False
                continue
            if inside_astra:
                assistant_targets[position] = True
        if inside_astra or not any(assistant_targets):
            raise ValueError("Conversation has an unclosed or empty ASTRA response.")

        for target_start in range(1, len(record), target_size):
            target_end = min(len(record), target_start + target_size)
            input_start = max(0, target_start - context_size - 1)
            inputs = record[input_start:target_end - 1]
            targets = record[input_start + 1:target_end]
            mask_count = target_start - (input_start + 1)
            if mask_count:
                targets[:mask_count] = [IGNORE_INDEX] * mask_count
            for offset, position in enumerate(range(target_start, target_end), start=mask_count):
                if not assistant_targets[position]:
                    targets[offset] = IGNORE_INDEX
            if inputs and len(inputs) == len(targets) and len(inputs) <= block_size:
                if any(token != IGNORE_INDEX for token in targets):
                    windows.append((inputs, targets))
    if not windows:
        raise ValueError("No assistant-response training windows were generated.")
    return windows


def get_batch(windows, batch_size, block_size, device, rng=None):
    rng = rng or random
    x = torch.zeros((batch_size, block_size), dtype=torch.long)
    y = torch.full((batch_size, block_size), IGNORE_INDEX, dtype=torch.long)
    for row in range(batch_size):
        inputs, targets = windows[rng.randrange(len(windows))]
        x[row, :len(inputs)] = torch.tensor(inputs, dtype=torch.long)
        y[row, :len(targets)] = torch.tensor(targets, dtype=torch.long)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, train_windows, val_windows, batch_size, block_size, device):
    model.eval()
    scores = {}
    for name, windows, seed in (
        ("train", train_windows, 1234), ("val", val_windows, 5678)
    ):
        rng = random.Random(seed)
        losses = []
        for _ in range(20):
            x, y = get_batch(windows, batch_size, block_size, device, rng)
            logits = model(x)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=IGNORE_INDEX,
            ).item())
        scores[name] = sum(losses) / len(losses)
    model.train()
    return scores


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ASTRA 0.2 for casual conversation.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.eval_interval < 1 or args.patience < 1:
        parser.error("steps, batch-size, eval-interval, and patience must be positive")
    if args.learning_rate <= 0 or not 0 < args.validation_fraction < 1:
        parser.error("learning-rate must be positive and validation-fraction between 0 and 1")
    if not 2 <= args.block_size <= 512:
        parser.error("block-size must be between 2 and 512")
    if CHAT_CHECKPOINT.exists():
        raise SystemExit(
            f"Refusing to overwrite {CHAT_CHECKPOINT}. Rename or move it manually "
            "if you intentionally want a fresh chat fine-tune."
        )
    if not CHAT_DATASET.exists() or not BASE_CHECKPOINT.exists():
        raise SystemExit("Missing chat dataset or original ASTRA checkpoint.")

    config = json.loads((HERE / "tokenizer_config.json").read_text(encoding="utf-8"))
    vocab_size = int(config["vocab_size"])
    records = read_records(CHAT_DATASET, config["fingerprint"], vocab_size)
    train_records, val_records = split_records(records, args.validation_fraction, args.seed)
    train_windows = build_windows(train_records, args.block_size)
    val_windows = build_windows(val_records, args.block_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    base = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    if base.get("architecture_version") != 3:
        raise SystemExit("Original checkpoint architecture version is incompatible.")
    if base.get("tokenizer_fingerprint") != config["fingerprint"]:
        raise SystemExit("Original checkpoint tokenizer does not match the chat dataset.")
    model_config = base.get("model_config", {})
    if model_config.get("vocab_size") != vocab_size:
        raise SystemExit("Original checkpoint vocabulary size does not match.")
    model = AstraGPT(**model_config).to(device)
    model.load_state_dict(base["model_state_dict"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    output_dir = CHAT_CHECKPOINT.parent
    output_dir.mkdir(exist_ok=True)
    best_val = float("inf")
    stale = 0
    started = time.time()

    print("ASTRA 0.2 — CHAT FINE-TUNING")
    print(f"Conversations: {len(records):,} | Train: {len(train_records):,} | Validation: {len(val_records):,}")
    print(f"Windows: {len(train_windows):,} train | {len(val_windows):,} validation")
    print(f"Parameters: {parameter_count:,} | Device: {device} | Block size: {args.block_size}")
    print(f"Starting weights: {BASE_CHECKPOINT.name}")
    print(f"New output only: {CHAT_CHECKPOINT.name}")
    print(f"Learning rate: {args.learning_rate:g} | Maximum steps: {args.steps}")
    print("Only ASTRA response tokens contribute to loss. Validation conversations are held out.")
    print("Original checkpoint will not be modified.")

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_windows, args.batch_size, args.block_size, device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            metrics = evaluate(
                model, train_windows, val_windows,
                args.batch_size, args.block_size, device
            )
            print(
                f"step {step:4d}/{args.steps} | train {metrics['train']:.4f} | "
                f"val {metrics['val']:.4f} | {time.time() - started:.1f}s",
                flush=True,
            )
            if metrics["val"] < best_val:
                best_val = metrics["val"]
                stale = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "architecture_version": 3,
                    "tokenizer_fingerprint": config["fingerprint"],
                    "step": step,
                    "best_val_loss": best_val,
                    "fine_tuned_from": "astra-0.2-best.pt",
                    "training_dataset": "tokenized_chat_dataset.jsonl",
                }, CHAT_CHECKPOINT)
                print("  saved new chat checkpoint", flush=True)
            else:
                stale += 1
                if stale >= args.patience:
                    print("Early stopping: validation loss stopped improving.")
                    break

    print(f"Finished. Best validation loss: {best_val:.4f}")
    print(f"Chat checkpoint: {CHAT_CHECKPOINT}")
    print(f"Original checkpoint untouched: {BASE_CHECKPOINT}")


if __name__ == "__main__":
    main()
