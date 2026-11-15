#!/usr/bin/env python3
"""Validate ASTRA's local curated conversation corpus; never downloads or imports data."""
from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

from tokenizer import load_conversations

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset_conversation_candidate.txt"
USER_RE = re.compile(r"<user>([\s\S]*?)</user>")
ASTRA_RE = re.compile(r"<astra>([\s\S]*?)</astra>")


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check ASTRA's local curated conversation corpus. "
            "This command is read-only and performs no downloads or imports."
        )
    )
    parser.add_argument(
        "--dataset", type=Path, default=DATASET_PATH,
        help="Local serialized corpus to inspect (defaults to dataset_conversation_candidate.txt).",
    )
    parser.add_argument(
        "--sample-count", type=int, default=0,
        help="Optionally print this many conversations for manual review.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Accepted for compatibility; this checker is always read-only.",
    )
    args = parser.parse_args()

    if args.sample_count < 0:
        parser.error("--sample-count cannot be negative.")

    conversations = load_conversations(args.dataset)
    users = [body.strip() for sample in conversations for body in USER_RE.findall(sample)]
    answers = [body.strip() for sample in conversations for body in ASTRA_RE.findall(sample)]
    user_counts = [len(USER_RE.findall(sample)) for sample in conversations]
    astra_counts = [len(ASTRA_RE.findall(sample)) for sample in conversations]

    bad_turns = [
        (index + 1, count)
        for index, count in enumerate(user_counts)
        if count != 3
    ]
    if bad_turns:
        raise SystemExit(
            "Every curated conversation must contain exactly 3 user messages; "
            f"mismatches: {bad_turns[:12]}"
        )

    duplicate_users = sum(count - 1 for count in Counter(map(normalize, users)).values() if count > 1)
    duplicate_answers = sum(count - 1 for count in Counter(map(normalize, answers)).values() if count > 1)

    print("ASTRA LOCAL CURATED CORPUS CHECK")
    print(f"Source: {args.dataset}")
    print(f"Serialized conversations: {len(conversations):,}")
    print(f"User messages: {len(users):,}")
    print(f"ASTRA messages: {len(answers):,}")
    print(f"Conversations with exactly 3 user messages: {len(conversations):,}/{len(conversations):,}")
    print(f"Additional ASTRA messages beyond 3 per conversation: {sum(astra_counts) - 3 * len(conversations):,}")
    print(f"Exact duplicate user messages: {duplicate_users:,}")
    print(f"Exact duplicate ASTRA messages: {duplicate_answers:,}")
    print("File changes: none")
    print("Network access: none")
    print("Tokenizer rebuild: none")
    print("Model training: none")

    if args.sample_count:
        count = min(args.sample_count, len(conversations))
        indices = sorted(random.Random(42).sample(range(len(conversations)), count))
        print(f"\nMANUAL REVIEW SAMPLE ({count} of {len(conversations)})")
        for number, index in enumerate(indices, 1):
            print(f"\n[{number}/{count}]")
            print(conversations[index])


if __name__ == "__main__":
    main()
