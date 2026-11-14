#!/usr/bin/env python3
"""ASTRA 0.1 tokenizer and dataset encoder. No third-party packages required."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

SPECIAL_TOKENS = {
    "<user>": 256, "</user>": 257,
    "<astra>": 258, "</astra>": 259,
    "<think>": 260, "</think>": 261,
}
VOCAB_SIZE = 262
PAIR_RE = re.compile(
    r"<user>\s*([\s\S]*?)\s*</user>\s*"
    r"<astra>\s*([\s\S]*?)\s*</astra>"
)
MARKER_RE = re.compile("(" + "|".join(
    re.escape(token) for token in sorted(SPECIAL_TOKENS, key=len, reverse=True)
) + ")")

def encode(text: str) -> list[int]:
    ids: list[int] = []
    for part in MARKER_RE.split(text):
        if not part:
            continue
        if part in SPECIAL_TOKENS:
            ids.append(SPECIAL_TOKENS[part])
        else:
            ids.extend(part.encode("utf-8"))
    return ids

def decode(ids: list[int]) -> str:
    reverse = {value: key for key, value in SPECIAL_TOKENS.items()}
    output: list[str] = []
    byte_buffer = bytearray()
    def flush() -> None:
        if byte_buffer:
            output.append(byte_buffer.decode("utf-8"))
            byte_buffer.clear()
    for token_id in ids:
        if token_id in reverse:
            flush()
            output.append(reverse[token_id])
        elif 0 <= token_id <= 255:
            byte_buffer.append(token_id)
        else:
            raise ValueError(f"Unknown token ID: {token_id}")
    flush()
    return "".join(output)

def load_pairs(dataset_path: Path) -> list[tuple[str, str]]:
    raw = dataset_path.read_text(encoding="utf-8")
    # The token guide mentions marker strings inline. The real dataset begins
    # only when <user> appears alone on a line; everything before it is ignored.
    match = re.search(r"(?m)^<user>\s*$", raw)
    if not match:
        raise ValueError("No standalone <user> training marker found.")
    pairs = [(u.strip(), a.strip()) for u, a in PAIR_RE.findall(raw[match.start():])]
    if not pairs:
        raise ValueError("No complete user/ASTRA pairs found.")
    for index, (user, astra) in enumerate(pairs, 1):
        if not user or not astra:
            raise ValueError(f"Example {index} has an empty message.")
        if "<think>" not in astra or "</think>" not in astra:
            raise ValueError(f"Example {index} is missing a think section.")
    return pairs

def tokenize_dataset(dataset_path: Path, output_path: Path) -> int:
    pairs = load_pairs(dataset_path)
    with output_path.open("w", encoding="utf-8") as output:
        for index, (user, astra) in enumerate(pairs):
            sample = f"<user>\n{user}\n</user>\n<astra>\n{astra}\n</astra>"
            record = {"id": index, "input_ids": encode(sample)}
            output.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
            output.write("\n")
    return len(pairs)

def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize ASTRA 0.1; this does not train.")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("dataset.txt"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("tokenized_dataset.jsonl"))
    args = parser.parse_args()
    count = tokenize_dataset(args.dataset, args.output)
    print(f"Tokenized {count} examples.")
    print(f"Vocabulary size: {VOCAB_SIZE} (256 byte IDs + 6 special tokens)")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
