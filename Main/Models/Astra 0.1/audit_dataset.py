#!/usr/bin/env python3
"""Audit ASTRA's text corpus and tokenized training data without changing them."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIR_RE = re.compile(\n    r"(?m)^<user>\\s*\\n([\\s\\S]*?)\\n</user>\\s*\\n"\n    r"<astra>\\s*\\n(?:<think>[\\s\\S]*?</think>\\s*)?([\\s\\S]*?)\\n</astra>"\n)


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def top_groups(values: list[str], limit: int = 8):
    counts = Counter(normalize(value) for value in values)
    return [(count, value[:130].replace("\n", " ")) for value, count in counts.most_common(limit) if count > 1]


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def main():
    parser = argparse.ArgumentParser(description="Inspect ASTRA dataset quality; does not modify files.")
    parser.add_argument("--dataset", type=Path, default=HERE / "dataset.txt")
    parser.add_argument("--tokenized", type=Path, default=HERE / "tokenized_dataset.jsonl")
    parser.add_argument("--config", type=Path, default=HERE / "tokenizer_config.json")
    args = parser.parse_args()

    raw = args.dataset.read_text(encoding="utf-8")
    marker = re.search(r"(?m)^<user>\s*$", raw)
    if not marker:
        raise SystemExit("No training examples found.")
    matches = list(PAIR_RE.finditer(raw[marker.start():]))
    users = [m.group(1).strip() for m in matches]
    thinks = [m.group(2).strip() for m in matches]
    answers = [m.group(3).strip() for m in matches]
    pairs = [(normalize(u), normalize(a)) for u, a in zip(users, answers)]
    source_counts = Counter(re.findall(r"<!-- ASTRA_SOURCE: ([^>]+) -->", raw))
    lengths = [len(u) + len(a) for u, a in zip(users, answers)]

    print("ASTRA DATASET AUDIT")
    print(f"Examples parsed: {len(matches):,}")
    print(f"Unique prompts (normalized): {len({normalize(x) for x in users}):,}")
    print(f"Unique answers (normalized): {len({normalize(x) for x in answers}):,}")
    print(f"Unique prompt/answer pairs: {len(set(pairs)):,}")
    print(f"Repeated-answer examples: {len(answers) - len({normalize(x) for x in answers}):,}")
    print(f"Prompt+answer character lengths p10/p50/p90: {percentile(lengths,.1)}/{percentile(lengths,.5)}/{percentile(lengths,.9)}")
    print("Source markers:")
    if source_counts:
        for source, count in source_counts.most_common():
            print(f"  {source}: {count:,}")
    else:
        print("  none found")
    print("Most repeated answers:")
    for count, value in top_groups(answers):
        print(f"  x{count}: {value}")
    print("Most repeated prompts:")
    for count, value in top_groups(users):
        print(f"  x{count}: {value}")

    if not args.tokenized.exists():
        print("\nTokenized dataset not found; skipping token statistics.")
        return
    records, token_counts, sequence_lengths = 0, Counter(), []
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
            records += 1
            sequence_lengths.append(len(ids))
            token_counts.update(ids)

    print("\nTOKENIZED DATA")
    print(f"Records: {records:,}")
    print(f"Total tokens: {sum(token_counts.values()):,}")
    print(f"Distinct token IDs used: {len(token_counts):,}")
    print(f"Sequence lengths p10/p50/p90: {percentile(sequence_lengths,.1)}/{percentile(sequence_lengths,.5)}/{percentile(sequence_lengths,.9)}")
    if args.config.exists():
        config = json.loads(args.config.read_text(encoding="utf-8"))
        vocab = config.get("vocab_hex", [])
        special = {int(v): k for k, v in config.get("special_tokens", {}).items()}
        print("Most frequent token pieces:")
        for token_id, count in token_counts.most_common(15):
            if token_id in special:
                piece = special[token_id]
            elif 0 <= token_id < len(vocab) and vocab[token_id] is not None:
                piece = repr(bytes.fromhex(vocab[token_id]).decode("utf-8", errors="replace"))
            else:
                piece = "<unknown>"
            print(f"  id={token_id:>5} count={count:>9,} piece={piece[:70]}")


if __name__ == "__main__":
    main()
