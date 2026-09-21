# -*- coding: utf-8 -*-
"""
Local model server for the demo page.

    python app/server.py              # http://127.0.0.1:8765

The demo page (Marathi-NLP-Demo.html) always works on its own, using the
small in-browser models from src/lite.py. When this server is running, the
page detects it and sends pasted text here instead, so the full pipeline --
MahaBERT topic model, MahaNER, the sentiment model and NLLB-200 -- analyses
it. Nothing else changes on the page.

Endpoints (JSON):
  GET  /api/health                     -> {"ok": true, "device": ...}
  POST /api/analyse  {title, text, translate}
                                       -> the pipeline's Document as a dict
  POST /api/translate {texts: [...]}   -> {"translations": [...]}

Standard library only. It listens on 127.0.0.1, so it is reachable from this
machine and nowhere else.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402

HOST = "127.0.0.1"
PORT = int(os.environ.get("MNIP_PORT", 8765))
MAX_CHARS = 20000

# The models are not thread-safe to load concurrently, and one request at a
# time is plenty for a demo.
_LOCK = threading.Lock()


def analyse(payload: dict) -> dict:
    from src.pipeline import process
    title = str(payload.get("title") or "")[:1000]
    text = str(payload.get("text") or "")[:MAX_CHARS]
    if not text.strip():
        raise ValueError("text is empty")
    with _LOCK:
        doc = process(title, text, translate=bool(payload.get("translate")))
    return doc.to_dict()


def translate(payload: dict) -> dict:
    from src import translate as mt
    texts = [str(t)[:2000] for t in (payload.get("texts") or [])][:40]
    if not texts:
        return {"translations": []}
    with _LOCK:
        return {"translations": mt.translate(texts)}


class Handler(BaseHTTPRequestHandler):
    server_version = "MarathiNLP/1.0"

    def _headers(self, status=200, body_len=0):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        # The page is usually opened from file://, whose origin is "null".
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # Chrome's Private Network Access preflight for pages on https://.
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", str(body_len))
        self.end_headers()

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._headers(status, len(body))
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802
        self._headers(204)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/api/health":
            self._json({"ok": True, "device": config.DEVICE,
                        "models": {"topic": config.MODEL_TOPIC,
                                   "ner": config.MODEL_NER,
                                   "sentiment": config.MODEL_SENTIMENT,
                                   "translation": config.MODEL_TRANSLATE}})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        routes = {"/api/analyse": analyse, "/api/translate": translate}
        fn = routes.get(self.path.rstrip("/"))
        if fn is None:
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            self._json(fn(payload))
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s %s\n" % (self.command, self.path))


def main():
    print(f"Marathi NLP model server on http://{HOST}:{PORT}  (device: {config.DEVICE})")
    print("Open Marathi-NLP-Demo.html -- it will switch to the full models.")
    print("Models load on the first request (the first one takes a while).")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
