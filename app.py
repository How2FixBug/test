"""Local-only server for the custom desktop simulator and Ollama bridge."""
import json
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
OLLAMA = "http://127.0.0.1:11434"

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, data, content_type="application/json; charset=utf-8"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/models":
            try:
                req = urllib.request.Request(OLLAMA + "/api/tags")
                with urllib.request.urlopen(req, timeout=3) as response:
                    self.reply(200, json.loads(response.read()))
            except Exception as exc:
                self.reply(503, {"error": "Cannot reach local Ollama at 127.0.0.1:11434. Start Ollama first.", "detail": str(exc)})
            return
        if self.path in ("/", "/index.html"):
            self.reply(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/favicon.ico":
            self.reply(204, b"", "image/x-icon")
            return
        self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/ollama":
            self.reply(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 18_000_000:
                self.reply(413, {"error": "Request size is invalid."})
                return
            payload = json.loads(self.rfile.read(size))
            payload["stream"] = False
            req = urllib.request.Request(
                OLLAMA + "/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                self.reply(200, json.loads(response.read()))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            self.reply(exc.code, {"error": detail or str(exc)})
        except Exception as exc:
            self.reply(502, {"error": "Ollama request failed: " + str(exc)})

if __name__ == "__main__":
    print("Local desktop simulator: http://127.0.0.1:8000")
    print("Ollama bridge: http://127.0.0.1:11434")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
