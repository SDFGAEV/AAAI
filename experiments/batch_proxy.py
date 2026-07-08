"""
Batch VLM Proxy — collects concurrent VLM requests and dispatches them
as a batch to app.py's /batch_chat endpoint for 2-4x GPU throughput.

Runs as a lightweight HTTP server in the experiment runner process.
Workers send individual VLM requests here instead of directly to app.py.
The proxy buffers for 50ms, then sends one batch to the GPU server.

Usage (auto-started by parallel_runner.py):
  python experiments/batch_proxy.py --port 12346 --vlm_url http://127.0.0.1:12345
"""

import json, time, threading, sys, os, argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError


class BatchProxy:
    """Accumulates VLM requests and dispatches in batches."""

    def __init__(self, vlm_url: str, batch_window: float = 0.05):
        self.vlm_url = vlm_url.rstrip("/")
        self.batch_window = batch_window
        self._lock = threading.Lock()
        self._queue: list = []  # [(request_data, result_holder)]
        self._batch_count = 0
        self._request_count = 0

    def submit(self, request_data: dict) -> dict:
        """Submit a single VLM request. Blocks until batch result is ready."""
        result_holder = {"ready": False, "response": None}
        with self._lock:
            self._queue.append((request_data, result_holder))
            self._request_count += 1
            if len(self._queue) == 1:
                # First request in a new batch window — start timer
                threading.Timer(self.batch_window, self._flush).start()

        # Wait for result (with timeout)
        deadline = time.time() + 30
        while not result_holder["ready"] and time.time() < deadline:
            time.sleep(0.005)
        return result_holder["response"]

    def _flush(self):
        """Send all queued requests as a batch to /batch_chat."""
        with self._lock:
            if not self._queue:
                return
            batch = list(self._queue)
            self._queue.clear()

        requests_data = [item[0] for item in batch]
        result_holders = [item[1] for item in batch]

        if len(requests_data) == 1:
            # Only one request — use normal /chat
            resp = self._post_single(requests_data[0])
            result_holders[0]["response"] = resp
            result_holders[0]["ready"] = True
        else:
            # Batch request
            self._batch_count += 1
            responses = self._post_batch(requests_data)
            for i, rh in enumerate(result_holders):
                rh["response"] = responses[i] if i < len(responses) else {"error": "missing"}
                rh["ready"] = True

    def _post_single(self, data: dict) -> dict:
        """Forward a single request to /chat."""
        try:
            body = json.dumps(data).encode()
            req = Request(f"{self.vlm_url}/chat", body,
                         headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"error": str(e)}

    def _post_batch(self, data_list: list) -> list:
        """Send batch to /batch_chat."""
        try:
            body = json.dumps(data_list).encode()
            req = Request(f"{self.vlm_url}/batch_chat", body,
                         headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return [{"error": str(e)}] * len(data_list)

    def stats(self) -> dict:
        return {
            "requests": self._request_count,
            "batches": self._batch_count,
            "avg_batch_size": round(self._request_count / max(self._batch_count, 1), 1),
        }


# ── HTTP handler that exposes the proxy ──
class ProxyHandler(BaseHTTPRequestHandler):
    proxy: BatchProxy = None

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid json"})
            return

        result = self.proxy.submit(data)
        self._respond(200, result)

    def do_GET(self):
        if self.path == "/stats":
            self._respond(200, self.proxy.stats())
        elif self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {})

    def _respond(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # Suppress HTTP access logs


def run_proxy(port: int, vlm_url: str, batch_window: float = 0.05):
    """Start the batch proxy server (blocking)."""
    proxy = BatchProxy(vlm_url, batch_window)
    ProxyHandler.proxy = proxy

    server = HTTPServer(("127.0.0.1", port), ProxyHandler)
    print(f"[BatchProxy] Listening on 127.0.0.1:{port}, forwarding to {vlm_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(f"[BatchProxy] Stopped. Stats: {proxy.stats()}")


def start_proxy_in_thread(port: int, vlm_url: str,
                          batch_window: float = 0.05) -> threading.Thread:
    """Start proxy in a daemon thread. Returns the thread."""
    t = threading.Thread(target=run_proxy, args=(port, vlm_url, batch_window),
                         daemon=True)
    t.start()
    time.sleep(0.5)  # Wait for server to start
    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12346)
    parser.add_argument("--vlm_url", default="http://127.0.0.1:12345")
    parser.add_argument("--window", type=float, default=0.05)
    args = parser.parse_args()
    run_proxy(args.port, args.vlm_url, args.window)
