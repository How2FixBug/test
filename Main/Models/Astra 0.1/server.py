#!/usr/bin/env python3
"""Local web server for the ASTRA Home interface and ASTRA 0.1 model."""
from __future__ import annotations

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from model import AstraGPT
from tokenizer import encode, SPECIAL_TOKENS
from generate import sample_next_token

HOST = "127.0.0.1"
PORT = 8765
CHECKPOINT_PATH = HERE / "checkpoints" / "astra-0.1-best.pt"
TOKENIZER_PATH = HERE / "tokenizer_config.json"

MODEL = None
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
VOCAB = []
MERGES = []
LOAD_ERROR = "Model files have not been added."


def load_model() -> None:
    global MODEL, VOCAB, MERGES, LOAD_ERROR

    if not CHECKPOINT_PATH.is_file():
        LOAD_ERROR = f"Checkpoint not found: {CHECKPOINT_PATH}"
        return
    if not TOKENIZER_PATH.is_file():
        LOAD_ERROR = f"Tokenizer config not found: {TOKENIZER_PATH}"
        return

    config = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    if not all(key in config for key in ("vocab_hex", "merges", "fingerprint", "vocab_size")):
        LOAD_ERROR = "The tokenizer config is outdated. Add the current BPE tokenizer_config.json."
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture_version") != 3:
        LOAD_ERROR = "This checkpoint is not compatible with ASTRA 0.1's current architecture."
        return
    if (
        checkpoint["model_config"]["vocab_size"] != config["vocab_size"]
        or checkpoint.get("tokenizer_fingerprint") != config["fingerprint"]
    ):
        LOAD_ERROR = "Checkpoint and tokenizer do not match. Use the matching tokenizer_config.json."
        return

    VOCAB = [bytes.fromhex(item) if item is not None else None for item in config["vocab_hex"]]
    MERGES = [tuple(pair) for pair in config["merges"]]
    MODEL = AstraGPT(**checkpoint["model_config"]).to(DEVICE)
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.eval()
    LOAD_ERROR = ""


def generate_reply(message: str, history: list[dict]) -> str:
    if MODEL is None:
        raise RuntimeError(LOAD_ERROR or "ASTRA 0.1 is not loaded.")

    parts = []
    for item in history[-12:]:
        role = item.get("role")
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if role == "user":
            parts.append(f"<user>\\n{text}\\n</user>\\n")
        elif role in ("assistant", "sol", "astra"):
            parts.append(f"<astra>\\n{text}\\n</astra>\\n")

    parts.append(f"<user>\\n{message.strip()}\\n</user>\\n<astra>\\n")
    token_ids = encode("".join(parts), {tuple(pair): rank for rank, pair in enumerate(MERGES)})
    generated = list(token_ids)
    answer_ids = []
    end_id = SPECIAL_TOKENS["</astra>"]
    special_ids = set(SPECIAL_TOKENS.values())

    torch.manual_seed(42)
    if DEVICE.type == "mps":
        torch.mps.manual_seed(42)

    with torch.inference_mode():
        for _ in range(180):
            context = torch.tensor([generated[-MODEL.max_seq_len:]], dtype=torch.long, device=DEVICE)
            logits = MODEL(context)[0, -1]
            next_id = sample_next_token(
                logits,
                generated,
                temperature=0.75,
                top_k=40,
                top_p=0.92,
                repetition_penalty=1.08,
                repeat_window=64,
                allowed_special_ids={end_id},
                special_ids=special_ids,
            )
            if next_id == end_id:
                break
            generated.append(next_id)
            answer_ids.append(next_id)

    output = bytearray()
    for token_id in answer_ids:
        if 0 <= token_id < len(VOCAB) and VOCAB[token_id] is not None:
            output.extend(VOCAB[token_id])
    return output.decode("utf-8", errors="replace").strip() or " "


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            self.send_json(200, {
                "ready": MODEL is not None,
                "model": "astra-0.1",
                "error": LOAD_ERROR,
                "device": str(DEVICE),
            })
            return
        if self.path.startswith("/api/"):
            self.send_json(404, {"error": "Unknown API endpoint."})
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_json(404, {"error": "Unknown API endpoint."})
            return
        if MODEL is None:
            self.send_json(503, {"error": LOAD_ERROR})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            message = str(payload.get("message", "")).strip()
            history = payload.get("history", [])
            if not message:
                self.send_json(400, {"error": "Message cannot be empty."})
                return
            if not isinstance(history, list):
                history = []
            reply = generate_reply(message, history)
            self.send_json(200, {"reply": reply, "model": "astra-0.1"})
        except Exception as error:
            self.send_json(500, {"error": f"Generation failed: {error}"})

    def log_message(self, format, *args):
        print("[ASTRA server] " + (format % args))


if __name__ == "__main__":
    load_model()
    if MODEL is None:
        print("[ASTRA] Waiting for model files:")
        print("  " + str(CHECKPOINT_PATH))
        print("  " + str(TOKENIZER_PATH))
        print("  " + LOAD_ERROR)
    else:
        print(f"[ASTRA] ASTRA 0.1 loaded on {DEVICE}.")
    print(f"[ASTRA] Open http://{HOST}:{PORT}/home/")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
