#!/usr/bin/env python3
"""Correctness checks for ASTRA's tokenizer, conversation masks, and causal model."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from model import AstraGPT
from tokenizer import encode, decode, load_conversations
from train import build_windows, IGNORE_INDEX

HERE = Path(__file__).resolve().parent


def main():
    config = json.loads((HERE / "tokenizer_config.json").read_text(encoding="utf-8"))
    if "vocab_hex" not in config or "merges" not in config:
        raise SystemExit("Run python3 prepare_dataset.py before this test.")
    vocab = [bytes.fromhex(item) if item is not None else None for item in config["vocab_hex"]]
    merges = [tuple(pair) for pair in config["merges"]]

    sample = "ASTRA tokenizer check: café, snow ☃, punctuation! <think>"
    encoded = encode(sample, merges)
    decoded = decode(encoded, vocab)
    if decoded != sample:
        raise AssertionError("Tokenizer encode/decode round-trip failed.")
    print("PASS: tokenizer round-trip (including Unicode and special markers)")

    conversations = load_conversations(HERE / "dataset_conversation_candidate.txt")
    if not conversations:
        raise AssertionError("The candidate corpus is empty.")
    if not any(sample.count("<user>") > 1 for sample in conversations):
        raise AssertionError("The corpus has no genuine multi-turn conversations.")
    if not any("<getinfo>" in sample and "<getinforesult>" in sample for sample in conversations):
        raise AssertionError("The corpus has no complete information-tool examples.")
    if not any("<think>" in sample for sample in conversations):
        raise AssertionError("The corpus has no selected thinking-note examples.")
    for conversation in (conversations[0], next(s for s in conversations if "<getinfo>" in s)):
        if decode(encode(conversation, merges), vocab) != conversation:
            raise AssertionError("A serialized conversation failed its tokenizer round-trip.")
    print(
        f"PASS: {len(conversations):,} serialized conversations parse; "
        "multi-turn, thought, and tool markers survive round-trip"
    )

    # Only tokens inside ASTRA messages (including closing markers) may receive loss.
    synthetic = [
        256, 97, 257,                         # <user>a</user>
        258, 260, 98, 261, 262, 99, 263, 259, # <astra><think>b</think><getinfo>c</getinfo></astra>
        264, 100, 265,                        # <getinforesult>d</getinforesult>
        258, 101, 259,                        # <astra>e</astra>
    ]
    windows = build_windows([synthetic], block_size=12)
    predicted = [
        token
        for _, targets in windows
        for token in targets
        if token != IGNORE_INDEX
    ]
    expected = [260, 98, 261, 262, 99, 263, 259, 101, 259]
    if predicted != expected:
        raise AssertionError(
            f"Conversation loss mask is incorrect. Expected {expected}, got {predicted}"
        )
    print("PASS: loss is applied to ASTRA turns only; user and tool-result tokens are masked")

    torch.manual_seed(7)
    model = AstraGPT(
        vocab_size=config["vocab_size"], width=64, heads=4,
        ff_width=128, max_seq_len=32, layers=3, dropout=0.0,
    ).eval()
    prefix = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
    changed_future = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33]])
    with torch.no_grad():
        logits_a = model(prefix)
        logits_b = model(changed_future)
    torch.testing.assert_close(
        logits_a[:, :8], logits_b[:, :8], rtol=1e-5, atol=1e-5
    )
    print("PASS: changing future tokens does not change earlier-position logits")
    print("All ASTRA correctness checks passed.")


if __name__ == "__main__":
    main()
