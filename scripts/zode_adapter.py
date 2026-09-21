"""OpenAI-compatible non-streaming adapter over zode's SSE-only chat endpoint.

Listens on 127.0.0.1:18780. Accepts POST /chat/completions (non-stream only),
consumes the SSE stream from zode, and returns a single chat.completion JSON.
"""
import json
import os
import time
import uuid

import httpx
from http.server import BaseHTTPRequestHandler, HTTPServer

ZODE_BASE = os.environ.get("ZODE_BASE_URL", "https://zode.qa.qima-inc.com/api/proxy/forward")
ZODE_KEY = os.environ.get("ZODE_API_KEY", "zode_daa9a98d90c2c6e74b9da376ccc9263f")
PORT = int(os.environ.get("ZODE_ADAPTER_PORT", "18780"))


def complete(body):
    """Call zode's SSE-only endpoint and assemble a non-streaming response."""
    upstream = {**body, "stream": True}
    content_parts, reasoning_parts, model_id, usage, finish_reason = [], [], None, {}, None
    with httpx.Client(http2=True, timeout=60) as client:
        with client.stream(
            "POST",
            f"{ZODE_BASE}/chat/completions",
            json=upstream,
            headers={"Authorization": f"Bearer {ZODE_KEY}"},
        ) as response:
            if response.is_error:
                raise RuntimeError(f"zode HTTP {response.status_code}")
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                model_id = chunk.get("model", model_id)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta", {}) or {}
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id or body.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") not in ("/chat/completions", "/v1/chat/completions"):
            self.send_error(404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            result = complete(body)
        except Exception as error:  # noqa: BLE001
            result = {"error": {"message": str(error), "type": "upstream_error"}}
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            payload = json.dumps({"data": [{"id": "deepseek-v4-flash", "type": "model"}]}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # quiet
        pass


if __name__ == "__main__":
    print(f"zode-adapter listening on http://127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()