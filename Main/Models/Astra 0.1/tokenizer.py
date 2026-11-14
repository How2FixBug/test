#!/usr/bin/env python3
"""Train ASTRA's byte-level BPE tokenizer and encode its dataset.

No third-party packages required. This trains the tokenizer only; it does not
train the neural network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
import heapq
import time
from pathlib import Path

TARGET_VOCAB_SIZE = 8192
BASE_VOCAB_SIZE = 256
SPECIAL_TOKENS = {
    "<user>": 256, "</user>": 257,
    "<astra>": 258, "</astra>": 259,
}
SPECIAL_RE = re.compile("(" + "|".join(
    re.escape(token) for token in sorted(SPECIAL_TOKENS, key=len, reverse=True)
) + ")")
PAIR_RE = re.compile(
    r"<user>\s*([\s\S]*?)\s*</user>\s*"
    r"<astra>\s*([\s\S]*?)\s*</astra>"
)
# Keep whitespace, words, numbers, and punctuation in separate pieces.
PRETOKEN_RE = re.compile(r"\s+|[^\W_]+|_+|[^\w\s]+", re.UNICODE)


def load_pairs(dataset_path: Path) -> list[tuple[str, str]]:
    raw = dataset_path.read_text(encoding="utf-8")
    # Ignore the explanatory material before the first actual training sample.
    marker = re.search(r"(?m)^<user>\s*$", raw)
    if not marker:
        raise ValueError("No standalone <user> training marker found.")
    pairs = [(u.strip(), a.strip()) for u, a in PAIR_RE.findall(raw[marker.start():])]
    # Legacy datasets may contain think blocks; remove them completely.
    pairs = [
        (user, re.sub(r"<think>[\\s\\S]*?</think>\\s*", "", answer, count=1).strip())
        for user, answer in pairs
    ]\n    if not pairs:
        raise ValueError("No complete user/ASTRA pairs found.")
    for index, (user, astra) in enumerate(pairs, 1):
        if not user or not astra:
            raise ValueError(f"Example {index} has an empty message.")
    return pairs


def text_pieces(text: str) -> list[str]:
    return PRETOKEN_RE.findall(text)


def train_bpe(pairs: list[tuple[str, str]], target_size: int):
    """Train BPE using incremental pair counts instead of rescanning all text.

    A merge only changes sequences that currently contain the selected pair.
    Pair counts are maintained with a lazy priority queue, so unrelated
    sequences are not revisited on every vocabulary step.
    """
    piece_counts: Counter[tuple[int, ...]] = Counter()
    for user, astra in pairs:
        clean_user = SPECIAL_RE.sub("", user)
        clean_astra = SPECIAL_RE.sub("", astra)
        for piece in text_pieces(clean_user) + text_pieces(clean_astra):
            piece_counts[tuple(piece.encode("utf-8"))] += 1

    sequences = [(list(piece), count) for piece, count in piece_counts.items()]
    vocab: list[bytes | None] = (
        [bytes([i]) for i in range(BASE_VOCAB_SIZE)]
        + [None] * len(SPECIAL_TOKENS)
    )
    merges: list[tuple[int, int]] = []

    # pair -> {sequence index: occurrences in that sequence}
    pair_sequences: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    pair_totals: Counter[tuple[int, int]] = Counter()

    def sequence_pairs(sequence: list[int]) -> Counter:
        return Counter(zip(sequence, sequence[1:]))

    def add_sequence_pairs(index: int) -> None:
        sequence, frequency = sequences[index]
        for pair, occurrences in sequence_pairs(sequence).items():
            pair_sequences[pair][index] = occurrences
            pair_totals[pair] += frequency * occurrences
            heapq.heappush(heap, (-pair_totals[pair], pair))

    def remove_sequence_pairs(index: int) -> None:
        sequence, frequency = sequences[index]
        for pair, occurrences in sequence_pairs(sequence).items():
            pair_totals[pair] -= frequency * occurrences
            pair_sequences[pair].pop(index, None)
            if pair_totals[pair] <= 0:
                pair_totals.pop(pair, None)
                pair_sequences.pop(pair, None)
            else:
                heapq.heappush(heap, (-pair_totals[pair], pair))

    heap: list[tuple[int, tuple[int, int]]] = []
    for index in range(len(sequences)):
        add_sequence_pairs(index)

    started = time.time()
    while len(vocab) < target_size and heap:
        negative_count, pair = heapq.heappop(heap)
        frequency = -negative_count
        if pair_totals.get(pair, 0) != frequency:
            continue
        if frequency < 2:
            break

        left, right = pair
        new_id = len(vocab)
        merges.append(pair)
        vocab.append(vocab[left] + vocab[right])

        # Only sequences containing this pair need to be updated.
        affected = list(pair_sequences.get(pair, {}).keys())
        for index in affected:
            remove_sequence_pairs(index)
            sequence, _ = sequences[index]
            merged = []
            cursor = 0
            while cursor < len(sequence):
                if (
                    cursor + 1 < len(sequence)
                    and sequence[cursor] == left
                    and sequence[cursor + 1] == right
                ):
                    merged.append(new_id)
                    cursor += 2
                else:
                    merged.append(sequence[cursor])
                    cursor += 1
            sequences[index] = (merged, sequences[index][1])
            add_sequence_pairs(index)

        if len(merges) % 250 == 0:
            elapsed = time.time() - started
            print(
                f"  BPE merges: {len(merges):,}/{target_size - BASE_VOCAB_SIZE - len(SPECIAL_TOKENS):,} "
                f"| {elapsed:.1f}s",
                flush=True,
            )

    print(f"  BPE training finished: {len(merges):,} merges in {time.time() - started:.1f}s")
    return vocab, merges


def encode_plain(text: str, merge_ranks: dict[tuple[int, int], int]) -> list[int]:
    """Apply ranked BPE merges efficiently with a min-heap."""
    import heapq
    # Backward compatibility: older callers pass the serialized merge list.
    if not isinstance(merge_ranks, dict):
        merge_ranks = {tuple(pair): rank for rank, pair in enumerate(merge_ranks)}

    ids: list[int] = []
    for piece in text_pieces(text):
        values = list(piece.encode("utf-8"))
        size = len(values)
        if size < 2:
            ids.extend(values)
            continue

        previous = [i - 1 for i in range(size)]
        following = [i + 1 for i in range(size)]
        following[-1] = -1
        alive = [True] * size
        heap = []

        def add_pair(left: int):
            right = following[left]
            if right != -1:
                rank = merge_ranks.get((values[left], values[right]))
                if rank is not None:
                    heapq.heappush(heap, (rank, left, right))

        for i in range(size - 1):
            add_pair(i)

        while heap:
            rank, left, right = heapq.heappop(heap)
            if not alive[left] or not alive[right] or following[left] != right:
                continue
            if merge_ranks.get((values[left], values[right])) != rank:
                continue

            values[left] = BASE_VOCAB_SIZE + len(SPECIAL_TOKENS) + rank
            alive[right] = False
            after = following[right]
            following[left] = after
            if after != -1:
                previous[after] = left

            before = previous[left]
            if before != -1:
                add_pair(before)
            add_pair(left)

        index = 0
        while index != -1:
            if alive[index]:
                ids.append(values[index])
            index = following[index]
    return ids


def encode(text: str, merge_ranks: dict[tuple[int, int], int]) -> list[int]:
    ids: list[int] = []
    for part in SPECIAL_RE.split(text):
        if not part:
            continue
        if part in SPECIAL_TOKENS:
            ids.append(SPECIAL_TOKENS[part])
        else:
            ids.extend(encode_plain(part, merge_ranks))
    return ids

def decode(ids: list[int], vocab: list[bytes | None]) -> str:
    reverse = {value: key for key, value in SPECIAL_TOKENS.items()}
    output: list[str] = []
    byte_buffer = bytearray()

    def flush():
        if byte_buffer:
            output.append(byte_buffer.decode("utf-8"))
            byte_buffer.clear()

    for token_id in ids:
        if token_id in reverse:
            flush()
            output.append(reverse[token_id])
        elif 0 <= token_id < len(vocab) and vocab[token_id] is not None:
            byte_buffer.extend(vocab[token_id])
        else:
            raise ValueError(f"Unknown token ID: {token_id}")
    flush()
    return "".join(output)


def tokenizer_fingerprint(vocab: list[bytes | None], merges: list[tuple[int, int]]) -> str:
    payload = {
        "vocab_hex": [token.hex() if token is not None else None for token in vocab],
        "merges": [list(pair) for pair in merges],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_tokenizer(config_path: Path, vocab: list[bytes | None], merges: list[tuple[int, int]], target_size: int):
    fingerprint = tokenizer_fingerprint(vocab, merges)
    config = {
        "fingerprint": fingerprint,
        "name": "ASTRA 0.1 Byte-Level BPE",
        "type": "byte_level_bpe",
        "target_vocab_size": target_size,
        "vocab_size": len(vocab),
        "base_byte_vocab_size": BASE_VOCAB_SIZE,
        "special_tokens": SPECIAL_TOKENS,
        "special_token_count": len(SPECIAL_TOKENS),
        "vocab_hex": [token.hex() if token is not None else None for token in vocab],
        "merges": [list(pair) for pair in merges],
        "notes": (
            "UTF-8 byte fallback preserves every Unicode character. "
            "BPE merges are learned from the ASTRA dataset. "
            "The tokenizer is trained separately from the language model."
        ),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return fingerprint


def tokenize_dataset(pairs, output_path: Path, merge_ranks, fingerprint: str):
    total_tokens = 0
    with output_path.open("w", encoding="utf-8") as output:
        for index, (user, astra) in enumerate(pairs):
            sample = f"<user>\n{user}\n</user>\n<astra>\n{astra}\n</astra>"
            input_ids = encode(sample, merge_ranks)
            total_tokens += len(input_ids)
            record = {
                "id": index,
                "tokenizer_fingerprint": fingerprint,
                "input_ids": input_ids,
            }
            output.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
            output.write("\n")
    return total_tokens

def main():
    parser = argparse.ArgumentParser(
        description="Train ASTRA's BPE tokenizer and tokenize its dataset. Does not train ASTRA."
    )
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("dataset.txt"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("tokenized_dataset.jsonl"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("tokenizer_config.json"))
    parser.add_argument("--vocab-size", type=int, default=TARGET_VOCAB_SIZE)
    args = parser.parse_args()

    if args.vocab_size <= BASE_VOCAB_SIZE + len(SPECIAL_TOKENS):
        parser.error("--vocab-size must be greater than 262")

    pairs = load_pairs(args.dataset)
    vocab, merges = train_bpe(pairs, args.vocab_size)
    fingerprint = save_tokenizer(args.config, vocab, merges, args.vocab_size)
    merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
    total_tokens = tokenize_dataset(pairs, args.output, merge_ranks, fingerprint)
    actual_size = len(vocab)
    print(f"Tokenizer trained on {len(pairs)} conversation pairs.")
    print(f"Vocabulary: {actual_size:,} tokens (target: {args.vocab_size:,}).")
    print(f"Learned BPE merges: {len(merges):,}.")
    print(f"Encoded dataset: {total_tokens:,} tokens.")
    print(f"Saved tokenizer: {args.config}")
    print(f"Saved dataset: {args.output}")
    if actual_size < args.vocab_size:
        print("Stopped early: no repeated byte pairs remained worth adding.")


if __name__ == "__main__":
    main()
