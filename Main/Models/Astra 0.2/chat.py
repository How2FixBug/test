#!/usr/bin/env python3
"""Interactive ASTRA 0.2 chat. Loads the model once and keeps session context."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from model import AstraGPT
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
) -> int:
    logits = logits.clone() / temperature
    special_ids = set(SPECIAL_TOKENS.values())

    if repetition_penalty > 1.0 and repeat_window:
        for token_id in set(history[-repeat_window:]):
            if token_id not in special_ids and 0 <= token_id < logits.numel():
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

    for token_id in special_ids:
        if token_id not in allowed_special_ids:
            logits[token_id] = -float("inf")

    if top_k and top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, -float("inf"))
        filtered[indices] = values
        for token_id in allowed_special_ids:
            filtered[token_id] = logits[token_id]
        logits = filtered

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities > top_p
        remove[0] = False
        filtered = logits.clone()
        filtered[sorted_indices[remove]] = -float("inf")
        for token_id in allowed_special_ids:
            filtered[token_id] = logits[token_id]
        logits = filtered

    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())


def make_prompt(history: list[tuple[str, str]], message: str, merges, max_seq_len: int) -> list[int]:
    turns = history + [(message, "")]
    while True:
        transcript = "".join(
            f"<user>\n{user}\n</user>\n<astra>\n{answer}\n</astra>\n"
            if answer else f"<user>\n{user}\n</user>\n<astra>\n"
            for user, answer in turns
        )
        tokens = encode(transcript, merges)
        if len(tokens) <= max_seq_len or len(turns) == 1:
            return tokens[-max_seq_len:]
        turns = turns[1:]


@torch.inference_mode()
def generate_reply(model, device, merges, vocab, history, message, args):
    prompt_ids = make_prompt(history, message, merges, model.max_seq_len)
    generated = list(prompt_ids)
    answer_ids: list[int] = []
    inside_think = False
    inside_getinfo = False
    tool_complete = False

    for _ in range(args.max_new_tokens):
        context = torch.tensor(
            [generated[-model.max_seq_len:]], dtype=torch.long, device=device
        )
        logits = model(context)[0, -1]
        if tool_complete:
            allowed = {SPECIAL_TOKENS["</astra>"]}
        elif inside_think:
            allowed = {SPECIAL_TOKENS["</think>"]}
        elif inside_getinfo:
            allowed = {SPECIAL_TOKENS["</getinfo>"]}
        else:
            allowed = {
                SPECIAL_TOKENS["</astra>"],
                SPECIAL_TOKENS["<think>"],
                SPECIAL_TOKENS["<getinfo>"],
            }

        token = sample_next_token(
            logits, generated,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            repeat_window=args.repeat_window,
            allowed_special_ids=allowed,
        )
        if token == SPECIAL_TOKENS["</astra>"]:
            break

        generated.append(token)
        answer_ids.append(token)
        if token == SPECIAL_TOKENS["<think>"]:
            inside_think = True
        elif token == SPECIAL_TOKENS["</think>"]:
            inside_think = False
        elif token == SPECIAL_TOKENS["<getinfo>"]:
            inside_getinfo = True
        elif token == SPECIAL_TOKENS["</getinfo>"]:
            inside_getinfo = False
            tool_complete = True

    raw = decode(answer_ids, vocab)
    tool_match = re.search(r"<getinfo>([\s\S]*?)</getinfo>", raw)
    if tool_complete and tool_match:
        return "", tool_match.group(1).strip()
    if inside_getinfo:
        partial = re.search(r"<getinfo>([\s\S]*)$", raw)
        return "", "INCOMPLETE: " + (partial.group(1).strip() if partial else "")
    visible = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", raw).strip()
    return visible, None


def main():
    parser = argparse.ArgumentParser(description="Chat with ASTRA 0.2.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--repeat-window", type=int, default=64)
    args = parser.parse_args()

    if args.max_new_tokens < 1 or args.temperature <= 0:
        parser.error("max-new-tokens and temperature must be positive")
    if args.top_k < 0 or not 0 < args.top_p <= 1 or args.repetition_penalty < 1:
        parser.error("Invalid sampling settings")
    if args.repeat_window < 0:
        parser.error("repeat-window cannot be negative")

    config_path = HERE / "tokenizer_config.json"
    checkpoint_path = HERE / "checkpoints" / "astra-0.2-best.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        raise SystemExit(
            "ASTRA artifacts are missing. A matching tokenizer config and trained "
            "checkpoints/astra-0.2-best.pt are required."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not all(key in config for key in ("vocab_hex", "merges", "fingerprint")):
        raise SystemExit("Tokenizer artifacts are outdated.")
    vocab = [bytes.fromhex(item) if item is not None else None for item in config["vocab_hex"]]
    merges = [tuple(pair) for pair in config["merges"]]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture_version") != 3:
        raise SystemExit("The checkpoint architecture is outdated.")
    if (
        checkpoint["model_config"]["vocab_size"] != config["vocab_size"]
        or checkpoint.get("tokenizer_fingerprint") != config["fingerprint"]
    ):
        raise SystemExit("Checkpoint and tokenizer do not match.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AstraGPT(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    history: list[tuple[str, str]] = []
    print("ASTRA 0.2 — interactive chat")
    print(f"Loaded once on {device}. Type /help for commands, /quit to exit.")
    while True:
        try:
            message = input("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nASTRA session ended.")
            break
        if not message:
            continue
        command = message.casefold()
        if command in {"/quit", "/exit"}:
            print("ASTRA session ended.")
            break
        if command == "/reset":
            history.clear()
            print("Conversation context cleared.")
            continue
        if command == "/help":
            print("Commands: /reset clears this conversation; /quit exits.")
            continue

        try:
            reply, tool_request = generate_reply(
                model, device, merges, vocab, history, message, args
            )
        except Exception as exc:
            print(f"Generation error: {exc}")
            continue

        if tool_request is not None:
            print("ASTRA requested external information: " + tool_request)
            history.append((message, "<getinfo>" + tool_request + "</getinfo>"))
        else:
            print("ASTRA: " + (reply or "[No response generated]"))
            history.append((message, reply))


if __name__ == "__main__":
    main()
