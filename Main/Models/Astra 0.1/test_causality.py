#!/usr/bin/env python3
"""Small correctness checks for ASTRA's tokenizer and causal architecture."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from model import LoopCore
from tokenizer import encode, decode

HERE = Path(__file__).resolve().parent

def main():
    config = json.loads((HERE / "tokenizer_config.json").read_text(encoding="utf-8"))
    if "vocab_hex" not in config or "merges" not in config:
        raise SystemExit("Run python3 tokenizer.py before this test.")
    vocab = [bytes.fromhex(item) if item is not None else None for item in config["vocab_hex"]]
    merges = [tuple(pair) for pair in config["merges"]]

    sample = "ASTRA tokenizer check: café, snow ☃, punctuation! <think>"
    encoded = encode(sample, merges)
    decoded = decode(encoded, vocab)
    if decoded != sample:
        raise AssertionError("Tokenizer encode/decode round-trip failed.")
    print("PASS: tokenizer round-trip (including Unicode and special markers)")

    torch.manual_seed(7)
    model = LoopCore(
        vocab_size=config["vocab_size"], width=64, heads=4,
        ff_width=128, max_seq_len=32, local_window=8, loop_passes=3,
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
