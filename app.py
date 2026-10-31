"""Local chat server. Uses only Python's standard library plus PyTorch."""
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
from model import TinyGPT, encode, decode

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "checkpoints" / "roger-small.pt"
MEMORY = ROOT / "data" / "memory.jsonl"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model = None
ready_error = None

def load_model():
    global model, ready_error
    if not CHECKPOINT.exists():
        ready_error = "No trained checkpoint yet. Add training text to data/train.txt and run python train.py."
        return
    try:
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        c = saved["config"]
        model = TinyGPT(c["width"], c["layers"], c["heads"], c["context"])
        model.load_state_dict(saved["model"])
        model.to(DEVICE).eval()
        ready_error = None
    except Exception as exc:
        ready_error = f"Could not load checkpoint: {exc}"

def recall(query, limit=3):
    if not MEMORY.exists():
        return []
    words = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
    scored = []
    try:
        with MEMORY.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item.get("text", "")
                overlap = len(words & set(re.findall(r"[a-z0-9]{3,}", text.lower())))
                if overlap:
                    scored.append((overlap, text))
    except (OSError, json.JSONDecodeError):
        return []
    return [text for _, text in sorted(scored, reverse=True)[:limit]]

def remember(user, assistant):
    MEMORY.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"text": f"User: {user}\nAssistant: {assistant}"}, ensure_ascii=False) + "\n")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "null")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/status":
            self.send_json(200, {"ready": model is not None, "device": DEVICE, "error": ready_error})
            return
        if self.path in ("/", "/index.html"):
            raw = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return
        if model is None:
            self.send_json(503, {"error": ready_error or "Model is not loaded."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 20000:
                self.send_json(413, {"error": "Message too large."})
                return
            payload = json.loads(self.rfile.read(size))
            user = str(payload.get("message", "")).strip()[:4000]
            if not user:
                self.send_json(400, {"error": "Type a message first."})
                return
            memories = recall(user)
            history = payload.get("history", [])[-6:]
            transcript = "You are Roger, a helpful, curious assistant. Answer clearly and honestly.\n"
            if memories:
                transcript += "Relevant remembered context:\n" + "\n".join(memories) + "\n"
            for turn in history:
                transcript += f"User: {str(turn.get('user',''))[:1000]}\nAssistant: {str(turn.get('assistant',''))[:1500]}\n"
            transcript += f"User: {user}\nAssistant:"
            ids = torch.tensor([encode(transcript)], dtype=torch.long, device=DEVICE)
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=220, temperature=0.8, top_k=40)
            answer = decode(out[0, ids.shape[1]:].tolist()).strip()
            if not answer:
                answer = "I don't have a useful response yet. The model needs more training data."
            remember(user, answer)
            self.send_json(200, {"answer": answer})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

if __name__ == "__main__":
    load_model()
    print(f"Roger AI server: http://127.0.0.1:8000 | device={DEVICE}")
    if ready_error:
        print(ready_error)
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
