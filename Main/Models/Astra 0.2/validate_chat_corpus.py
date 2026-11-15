#!/usr/bin/env python3
"""Read-only validation for ASTRA's chat-focused corpus and saved tokenizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizer import (
    BASE_VOCAB_SIZE,
    decode,
    encode,
    load_conversations,
    tokenizer_fingerprint,
)

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a chat corpus using the existing tokenizer; writes nothing."
    )
    parser.add_argument(
        "--dataset", type=Path, default=HERE / "dataset_chat_filtered.txt"
    )
    parser.add_argument(
        "--config", type=Path, default=HERE / "tokenizer_config.json"
    )
    args = parser.parse_args()

    conversations = load_conversations(args.dataset)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    special_tokens = config["special_tokens"]
    merges = [tuple(pair) for pair in config["merges"]]
    vocab = [
        bytes.fromhex(value) if value is not None else None
        for value in config["vocab_hex"]
    ]
    fingerprint = tokenizer_fingerprint(vocab, merges)

    if fingerprint != config["fingerprint"]:
        raise SystemExit("FAIL: saved tokenizer fingerprint does not match its vocab/merges.")
    if len(vocab) != int(config["vocab_size"]):
        raise SystemExit("FAIL: saved vocabulary length does not match vocab_size.")
    if len(vocab) != BASE_VOCAB_SIZE + len(special_tokens) + len(merges):
        raise SystemExit(
            "FAIL: vocab size does not equal byte tokens + special tokens + BPE merges."
        )

    total_tokens = 0
    max_token_id = -1
    round_trip_failures = []
    for index, conversation in enumerate(conversations, 1):
        token_ids = encode(conversation, dict((pair, rank) for rank, pair in enumerate(merges)), special_tokens)
        total_tokens += len(token_ids)
        if token_ids:
            max_token_id = max(max_token_id, max(token_ids))
        if max_token_id >= len(vocab):
            raise SystemExit(
                f"FAIL: token ID {max_token_id} exceeds vocabulary on conversation {index}."
            )
        if decode(token_ids, vocab, special_tokens) != conversation:
            round_trip_failures.append(index)
            if len(round_trip_failures) >= 5:
                break

    if round_trip_failures:
        raise SystemExit(
            "FAIL: tokenizer round-trip mismatch in conversation(s): "
            + ", ".join(map(str, round_trip_failures))
        )

    print("ASTRA CHAT CORPUS VALIDATION PASSED")
    print(f"Conversations parsed: {len(conversations):,}")
    print(f"Tokenizer fingerprint: verified")
    print(f"Special tokens loaded from saved config: {len(special_tokens)}")
    print(f"Saved vocabulary: {len(vocab):,}")
    print(f"Encoded tokens checked: {total_tokens:,}")
    print(f"Highest token ID: {max_token_id:,}")
    print("Encode/decode round-trip: exact for every conversation")
    print("Files changed: none")
    print("Tokenizer rebuilt: no")
    print("Tokenized dataset overwritten: no")
    print("Checkpoint changed: no")
    print("Training started: no")


if __name__ == "__main__":
    main()
