#!/usr/bin/env python3
"""Audit ASTRA's serialized multi-turn conversation corpus without changing it."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from tokenizer import load_conversations

HERE = Path(__file__).resolve().parent
MESSAGE_RE = {
    "user": re.compile(r"<user>([\s\S]*?)</user>"),
    "astra": re.compile(r"<astra>([\s\S]*?)</astra>"),
}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def repeated(values: list[str], limit: int = 8):
    counts = Counter(normalize(value) for value in values)
    return [(count, value[:130].replace("\n", " ")) for value, count in counts.most_common(limit) if count > 1]


def repeated_phrases(values: list[str], minimum: int = 5, limit: int = 12):
    counts = Counter()
    for value in values:
        clean = re.sub(r"<think>[\s\S]*?</think>", " ", value)
        words = normalize(clean).split()
        for size in (4, 5, 6):
            for index in range(len(words) - size + 1):
                counts[" ".join(words[index:index + size])] += 1
    return [
        (phrase_count, phrase)
        for phrase, phrase_count in counts.most_common()
        if phrase_count >= minimum
    ][:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect ASTRA conversation quality and tokenization; changes no files."
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=HERE / "dataset_conversation_candidate.txt",
    )
    parser.add_argument(
        "--tokenized", type=Path, default=HERE / "tokenized_dataset.jsonl"
    )
    parser.add_argument(
        "--config", type=Path, default=HERE / "tokenizer_config.json"
    )
    parser.add_argument("--expected-count", type=int, default=512, help="Expected conversation count; use 0 to disable.")
    parser.add_argument("--minimum-multi-turn", type=int, default=512, help="Minimum number of multi-turn conversations.")
    parser.add_argument("--exact-user-turns", type=int, default=3, help="Require exactly this many user messages in every conversation; use 0 to disable.")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    conversations = load_conversations(args.dataset)
    if args.expected_count and len(conversations) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count:,} conversations, found {len(conversations):,}.")
    user_turn_counts = [len(MESSAGE_RE["user"].findall(sample)) for sample in conversations]
    if args.exact_user_turns and any(count != args.exact_user_turns for count in user_turn_counts):
        bad = [(index + 1, count) for index, count in enumerate(user_turn_counts) if count != args.exact_user_turns]
        raise SystemExit(f"Expected exactly {args.exact_user_turns} user messages per conversation; mismatches: {bad[:12]}")
    users = [body.strip() for sample in conversations for body in MESSAGE_RE["user"].findall(sample)]
    answers = [body.strip() for sample in conversations for body in MESSAGE_RE["astra"].findall(sample)]
    if len({normalize(value) for value in users}) != len(users):
        raise SystemExit("Duplicate user messages found.")
    if len({normalize(value) for value in answers}) != len(answers):
        raise SystemExit("Duplicate ASTRA messages found.")
    multi_turn = sum(
        len(MESSAGE_RE["user"].findall(sample)) > 1 for sample in conversations
    )
    if multi_turn < args.minimum_multi_turn:
        raise SystemExit(f"Expected at least {args.minimum_multi_turn:,} multi-turn conversations, found {multi_turn:,}.")
    user_lengths = [len(value) for value in users]
    answer_lengths = [len(value) for value in answers]

    print("ASTRA CONVERSATION CORPUS AUDIT")
    print(f"Serialized conversations: {len(conversations):,}")
    print("Exact duplicate message check: PASS")
    print(f"User turns: {len(users):,} ({len({normalize(x) for x in users}):,} unique)")
    print(f"ASTRA turns: {len(answers):,} ({len({normalize(x) for x in answers}):,} unique)")
    print(f"Multi-turn conversations: {multi_turn:,}")
    print(f"Tool requests: {sum(sample.count('<getinfo>') for sample in conversations):,}")
    print(f"Tool results: {sum(sample.count('<getinforesult>') for sample in conversations):,}")
    print(f"Thinking notes: {sum(sample.count('<think>') for sample in conversations):,}")
    print(
        "User message lengths p10/p50/p90: "
        f"{percentile(user_lengths, .1)}/{percentile(user_lengths, .5)}/{percentile(user_lengths, .9)}"
    )
    print(
        "ASTRA message lengths p10/p50/p90: "
        f"{percentile(answer_lengths, .1)}/{percentile(answer_lengths, .5)}/{percentile(answer_lengths, .9)}"
    )

    print("Repeated user messages:")
    groups = repeated(users)
    if groups:
        for count, value in groups:
            print(f"  x{count}: {value}")
    else:
        print("  none")
    print("Repeated ASTRA messages:")
    groups = repeated(answers)
    if groups:
        for count, value in groups:
            print(f"  x{count}: {value}")
    else:
        print("  none")

    print("Repeated 4–6 word phrases (review for template patterns):")
    phrases = repeated_phrases(answers)
    if phrases:
        for count, phrase in phrases:
            print(f"  x{count}: {phrase}")
    else:
        print("  none")

    if args.sample_count < 0:
        raise SystemExit("--sample-count cannot be negative.")
    if args.sample_count:
        count = min(args.sample_count, len(conversations))
        rng = random.Random(args.sample_seed)
        indices = sorted(rng.sample(range(len(conversations)), count))
        print(f"\nQUALITY SAMPLE ({count} of {len(conversations)}; seed={args.sample_seed})")
        for number, index in enumerate(indices, 1):
            print(f"\n[{number}/{count}]")
            print(conversations[index])

    if not args.tokenized.exists():
        print("\nTokenized dataset not found; skipping token statistics.")
        return

    records = 0
    token_counts: Counter[int] = Counter()
    sequence_lengths: list[int] = []
    fingerprint = None
    with args.tokenized.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                ids = item["input_ids"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise SystemExit(f"Invalid tokenized record on line {line_number}: {exc}")
            if not isinstance(ids, list):
                raise SystemExit(f"input_ids is not a list on line {line_number}")
            item_fingerprint = item.get("tokenizer_fingerprint")
            if fingerprint is None:
                fingerprint = item_fingerprint
            elif item_fingerprint != fingerprint:
                raise SystemExit(f"Tokenizer fingerprint changes on line {line_number}.")
            records += 1
            sequence_lengths.append(len(ids))
            token_counts.update(int(token) for token in ids)

    print("\nTOKENIZED DATA")
    print(f"Records: {records:,}")
    print(f"Total tokens: {sum(token_counts.values()):,}")
    print(f"Distinct token IDs used: {len(token_counts):,}")
    print(
        "Sequence lengths p10/p50/p90: "
        f"{percentile(sequence_lengths, .1)}/"
        f"{percentile(sequence_lengths, .5)}/"
        f"{percentile(sequence_lengths, .9)}"
    )
    if records != len(conversations):
        print(f"WARNING: {records:,} tokenized records do not match {len(conversations):,} conversations.")
    if args.config.exists():
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if fingerprint and fingerprint != config.get("fingerprint"):
            print("WARNING: tokenized data fingerprint does not match tokenizer_config.json.")
        special = {
            int(value): key for key, value in config.get("special_tokens", {}).items()
        }
        print("Special-token counts:")
        for token_id, name in special.items():
            count = token_counts.get(token_id, 0)
            if count:
                print(f"  {name}: {count:,}")


if __name__ == "__main__":
    main()
