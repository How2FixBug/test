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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = json.loads((HERE / "tokenizer_config.json").read_text(encoding="utf-8"))
    if "vocab_hex" not in config or "merges" not in config:
        raise SystemExit("Tokenizer artifacts are outdated. Run python3 tokenizer.py.")
    vocab = [bytes.fromhex(item) if item is not None else None for item in config["vocab_hex"]]
    merges = [tuple(pair) for pair in config["merges"]]
    checkpoint = torch.load(
        HERE / "checkpoints" / "astra-0.1-best.pt",
        map_location="cpu", weights_only=False,
    )
    if checkpoint.get("architecture_version") != 2:
        raise SystemExit(
            "This checkpoint uses the old architecture. Run python3 train.py "
            "to create a fresh checkpoint."
        )
    if checkpoint["model_config"]["vocab_size"] != config["vocab_size"]:
        raise SystemExit(
            "Checkpoint and tokenizer vocabularies do not match. Run tokenizer.py, "
            "then train.py."
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LoopCore(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.manual_seed(args.seed)

    prompt = f"<user>\n{args.prompt}\n</user>\n<astra>\n<think>\n"
    input_ids = encode(prompt, merges)
    generated = list(input_ids)
    think_end = SPECIAL_TOKENS["</think>"]
    astra_end = SPECIAL_TOKENS["</astra>"]
    special_ids = list(SPECIAL_TOKENS.values())
    think_closed = False

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = torch.tensor(
                [generated[-model.max_seq_len:]], dtype=torch.long, device=device
            )
            logits = model(context)[0, -1] / max(args.temperature, 1e-5)
            allowed = astra_end if think_closed else think_end
            for token_id in special_ids:
                if token_id != allowed:
                    logits[token_id] = -float("inf")
            if args.top_k > 0:
                # Preserve the required closing marker even when it is not
                # among the top-k ordinary token candidates.
                required_logit = logits[allowed].clone()
                values, indices = torch.topk(logits, min(args.top_k, logits.numel()))
                filtered = torch.full_like(logits, -float("inf"))
                filtered[indices] = logits[indices]
                filtered[allowed] = required_logit
                logits = filtered
            probabilities = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, 1).item())
            generated.append(next_id)
            if next_id == think_end:
                think_closed = True
            elif next_id == astra_end:
                break

    answer = decode(generated[len(input_ids):], vocab)
    answer = answer.replace("</think>", "\n\n").replace("</astra>", "").strip()
    print(answer)

if __name__ == "__main__":
    main()
