#!/usr/bin/env python3
"""Build a separate chat-focused ASTRA corpus without modifying source files.

Combines the known 531-conversation casual section at the start of dataset.txt
with the separately curated candidate corpus. Does not tokenize or train.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def split_base_dataset(path: Path, expected_count: int) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    marker = raw.find("<user>")
    if marker < 0:
        raise SystemExit(f"No <user> marker found in {path}")
    body = raw[marker:]
    records = [
        part.strip()
        for part in re.split(r"\n---\n", body)
        if part.strip()
    ]
    if len(records) < expected_count:
        raise SystemExit(
            f"{path} contains only {len(records)} separator-delimited records; "
            f"expected at least {expected_count}. No output was written."
        )
    selected = records[:expected_count]
    for index, record in enumerate(selected, 1):
        if not record.startswith("<user>") or not record.endswith("</astra>"):
            raise SystemExit(
                f"Base record {index} does not have the expected outer markers. "
                "No output was written."
            )
    return selected


def split_candidate(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").strip()
    records = [
        part.strip()
        for part in re.split(r"\n\s*\n(?=<user>)", raw)
        if part.strip()
    ]
    if not records:
        raise SystemExit(f"No conversations found in {path}")
    for index, record in enumerate(records, 1):
        if not record.startswith("<user>") or not record.endswith("</astra>"):
            raise SystemExit(
                f"Candidate conversation {index} has invalid outer markers. "
                "No output was written."
            )
    return records


def normalize(record: str) -> str:
    return re.sub(r"\s+", " ", record).strip().casefold()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a separate chat-focused corpus; source files are never changed."
    )
    parser.add_argument("--base", type=Path, default=HERE / "dataset.txt")
    parser.add_argument(
        "--candidate", type=Path,
        default=HERE / "dataset_conversation_candidate.txt",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "dataset_chat_filtered.txt"
    )
    parser.add_argument(
        "--base-count", type=int, default=531,
        help="Number of known casual-conversation records to take from the start of dataset.txt.",
    )
    args = parser.parse_args()

    if args.base_count < 1:
        parser.error("--base-count must be positive.")
    if args.output.exists():
        raise SystemExit(
            f"Refusing to overwrite {args.output}. Rename or move it first if replacement is intended."
        )
    if not args.base.exists() or not args.candidate.exists():
        raise SystemExit("Both source files must exist. No output was written.")

    base_records = split_base_dataset(args.base, args.base_count)
    candidate_records = split_candidate(args.candidate)

    combined: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for record in base_records + candidate_records:
        key = normalize(record)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        combined.append(record)

    args.output.write_text("\n\n".join(combined) + "\n", encoding="utf-8")

    text = "\n".join(combined)
    print("ASTRA CHAT CORPUS CREATED")
    print(f"Base casual records selected: {len(base_records):,}")
    print(f"Curated candidate records: {len(candidate_records):,}")
    print(f"Duplicate conversations skipped: {duplicates:,}")
    print(f"Final conversations: {len(combined):,}")
    print(f"<think> markers preserved: {text.count('<think>'):,}")
    print(f"<getinfo> markers preserved: {text.count('<getinfo>'):,}")
    print(f"<getinforesult> markers preserved: {text.count('<getinforesult>'):,}")
    print(f"Output: {args.output}")
    print("Source files changed: no")
    print("Tokenizer or checkpoint changed: no")
    print("Model training started: no")


if __name__ == "__main__":
    main()
