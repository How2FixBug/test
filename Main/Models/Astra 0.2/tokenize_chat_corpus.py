#!/usr/bin/env python3
"""Tokenize ASTRA's chat corpus with the existing saved tokenizer, without replacing artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from tokenizer import encode, decode, load_conversations, tokenizer_fingerprint

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset_chat_filtered.txt"
CONFIG_PATH = HERE / "tokenizer_config.json"
OUTPUT_PATH = HERE / "tokenized_chat_dataset.jsonl"


def main() -> None:
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"Refusing to overwrite existing file: {OUTPUT_PATH.name}\n"
            "Move or rename it manually if you intentionally want to regenerate it."
        )
    if not DATASET_PATH.exists():
        raise SystemExit(f"Missing chat corpus: {DATASET_PATH.name}")
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing saved tokenizer: {CONFIG_PATH.name}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    required = ("fingerprint", "vocab_size", "vocab_hex", "merges", "special_tokens")
    if any(key not in config for key in required):
        raise SystemExit("Saved tokenizer config is incomplete.")

    vocab = [
        bytes.fromhex(value) if value is not None else None
        for value in config["vocab_hex"]
    ]
    merges = [tuple(pair) for pair in config["merges"]]
    special_tokens = {
        str(token): int(token_id)
        for token, token_id in config["special_tokens"].items()
    }
    fingerprint = tokenizer_fingerprint(vocab, merges)
    if fingerprint != config["fingerprint"]:
        raise SystemExit("Saved tokenizer fingerprint does not match its vocabulary and merges.")
    if len(vocab) != int(config["vocab_size"]):
        raise SystemExit("Saved vocabulary length does not match vocab_size.")

    conversations = load_conversations(DATASET_PATH)
    merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
    records = []
    total_tokens = 0

    for index, conversation in enumerate(conversations):
        ids = encode(conversation, merge_ranks, special_tokens)
        if not ids or max(ids) >= len(vocab):
            raise SystemExit(f"Invalid token ID in conversation {index + 1}.")
        if decode(ids, vocab, special_tokens) != conversation:
            raise SystemExit(f"Exact round-trip failed for conversation {index + 1}.")
        records.append({
            "id": index,
            "tokenizer_fingerprint": fingerprint,
            "input_ids": ids,
        })
        total_tokens += len(ids)

    # Exclusive creation prevents accidental replacement of a previous dataset.
    with OUTPUT_PATH.open("x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
            output.write("\n")

    print("ASTRA CHAT DATASET TOKENIZED")
    print(f"Conversations: {len(records):,}")
    print(f"Encoded tokens: {total_tokens:,}")
    print(f"Vocabulary: {len(vocab):,}")
    print(f"Special tokens: {len(special_tokens)} (loaded from saved config)")
    print("Tokenizer fingerprint: verified")
    print("Encode/decode round-trip: exact for every conversation")
    print(f"New output: {OUTPUT_PATH}")
    print("Original tokenized_dataset.jsonl: untouched")
    print("tokenizer_config.json: untouched")
    print("All checkpoints: untouched")
    print("Model training: not started")


if __name__ == "__main__":
    main()
