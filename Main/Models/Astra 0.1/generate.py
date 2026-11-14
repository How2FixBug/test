#!/usr/bin/env python3
"""Generate a reply with a trained ASTRA 0.1 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import LoopCore
from tokenizer import SPECIAL_TOKENS, encode, decode


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Hello! What can you do?")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    config = json.loads((HERE / "tokenizer_config.json").read_text(encoding="utf-8"))
    vocab = [bytes.fromhex(item) if item is not None else None for item in config["vocab_hex"]]
    merges = [tuple(pair) for pair in config["merges"]]
    checkpoint = torch.load(
        HERE / "checkpoints" / "astra-0.1-best.pt",
        map_location="cpu",
        weights_only=False,
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LoopCore(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = f"<user>\n{args.prompt}\n</user>\n<astra>\n<think>\n"
    input_ids = encode(prompt, merges)
    generated = list(input_ids)
    stop_id = SPECIAL_TOKENS["</astra>"]

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = torch.tensor(
                [generated[-model.max_seq_len:]], dtype=torch.long, device=device
            )
            logits = model(context)[0, -1] / max(args.temperature, 1e-5)
            if args.top_k > 0:
                values, _ = torch.topk(logits, min(args.top_k, logits.numel()))
                logits[logits < values[-1]] = -float("inf")
            probabilities = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, 1).item())
            generated.append(next_id)
            if next_id == stop_id:
                break

    answer = decode(generated[len(input_ids):], vocab)
    print(answer)


if __name__ == "__main__":
    main()
