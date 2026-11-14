#!/usr/bin/env python3
"""Generate a direct reply with a trained ASTRA 0.1 checkpoint."""
from __future__ import annotations

import argparse
import codecs
import json
import sys
from pathlib import Path

import torch
from model import LoopCore
from tokenizer import SPECIAL_TOKENS, encode, decode

HERE = Path(__file__).resolve().parent


def sample_next_token(
    logits: torch.Tensor,
    history: list[int],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repeat_window: int,
    allowed_special_ids: set[int],
    special_ids: set[int],
) -> int:
    """Apply repetition penalty, top-k, and nucleus filtering before sampling."""
    logits = logits.clone() / temperature

    if repetition_penalty > 1.0 and repeat_window > 0:
        recent = set(history[-repeat_window:])
        for token_id in recent:
            if token_id not in special_ids and 0 <= token_id < logits.numel():
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

    # Structural tokens are legal only when the current generation phase allows them.
    for token_id in special_ids:
        if token_id not in allowed_special_ids:
            logits[token_id] = -float("inf")

    if top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, -float("inf"))
        filtered[indices] = values
        for token_id in allowed_special_ids:
            filtered[token_id] = logits[token_id]
        logits = filtered

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative_probs - sorted_probs > top_p
        remove[0] = False
        filtered = logits.clone()
        filtered[sorted_indices[remove]] = -float("inf")
        for token_id in allowed_special_ids:
            filtered[token_id] = logits[token_id]
        logits = filtered

    probabilities = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probabilities, 1).item())


def main():
    parser = argparse.ArgumentParser(
        description="Generate a direct reply from a trained ASTRA 0.1 checkpoint."
    )
    parser.add_argument("prompt", nargs="?", default="Hello! What can you do?")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--repeat-window", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-stream", action="store_true", help="Print the completed answer all at once")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")
    if args.top_k < 0:
        parser.error("--top-k cannot be negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the range (0, 1]")
    if args.repetition_penalty < 1:
        parser.error("--repetition-penalty must be at least 1")
    if args.repeat_window < 0:
        parser.error("--repeat-window cannot be negative")

    config = json.loads(
        (HERE / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    if not all(key in config for key in ("vocab_hex", "merges", "fingerprint")):
        raise SystemExit("Tokenizer artifacts are outdated. Run python3 tokenizer.py.")

    vocab = [
        bytes.fromhex(item) if item is not None else None
        for item in config["vocab_hex"]
    ]
    merges = [tuple(pair) for pair in config["merges"]]

    checkpoint = torch.load(
        HERE / "checkpoints" / "astra-0.1-best.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("architecture_version") != 2:
        raise SystemExit(
            "This checkpoint uses the old architecture. Run python3 train.py "
            "to create a fresh checkpoint."
        )
    if (
        checkpoint["model_config"]["vocab_size"] != config["vocab_size"]
        or checkpoint.get("tokenizer_fingerprint") != config["fingerprint"]
    ):
        raise SystemExit(
            "Checkpoint and tokenizer do not match. Run tokenizer.py, then train.py."
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LoopCore(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.manual_seed(args.seed)
    if device.type == "mps":
        torch.mps.manual_seed(args.seed)

    # Match the training format. Generate the think section, but never display it.
    prompt = f"<user>\n{args.prompt}\n</user>\n<astra>\n<think>\n"
    generated = encode(prompt, merges)
    think_end = SPECIAL_TOKENS["</think>"]
    astra_end = SPECIAL_TOKENS["</astra>"]
    special_ids = set(SPECIAL_TOKENS.values())
    answer_start = None
    answer_ids: list[int] = []
    decoder = codecs.getincrementaldecoder("utf-8")()

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            context = torch.tensor(
                [generated[-model.max_seq_len:]],
                dtype=torch.long,
                device=device,
            )
            logits = model(context)[0, -1]
            allowed = {think_end} if answer_start is None else {astra_end}
            next_id = sample_next_token(
                logits,
                generated,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                repeat_window=args.repeat_window,
                allowed_special_ids=allowed,
                special_ids=special_ids,
            )
            # Stop before appending the structural end token, so it never
            # appears in the decoded user-facing answer.
            if answer_start is not None and next_id == astra_end:
                break

            generated.append(next_id)
            if answer_start is None and next_id == think_end:
                answer_start = len(generated)
                continue

            if answer_start is not None:
                answer_ids.append(next_id)
                if not args.no_stream and vocab[next_id] is not None:
                    piece = decoder.decode(vocab[next_id], final=False)
                    if piece:
                        sys.stdout.write(piece)
                        sys.stdout.flush()

    if answer_start is None:
        raise SystemExit(
            "ASTRA did not finish its think section. Try increasing --max-new-tokens."
        )

    if args.no_stream:
        print(decode(answer_ids, vocab).strip())
    else:
        tail = decoder.decode(b"", final=True)
        if tail:
            sys.stdout.write(tail)
        print()


if __name__ == "__main__":
    main()
