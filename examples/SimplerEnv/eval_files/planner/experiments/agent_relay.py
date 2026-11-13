"""The frontier-model study's agent arm: the relay on the machine that holds the API key (README.md,
"Frontier-model study").

The agent's eval runs on the GPU server, which holds no key. Each turn of the model, the eval's controller
(planner/agent.py, HttpRelay) sends the turn's request to this relay; the relay sends it to the model with the key of
this machine and gives the model's message back, unchanged.

    python -m examples.SimplerEnv.eval_files.planner.experiments.agent_relay LOG [--port 8765] [--limit-usd USD]

LOG: a file (JSON lines) with one line per request: its date, time, tokens and cost, and no content. The key comes
from the environment (ANTHROPIC_API_KEY), as the SDK reads it; the relay never prints, saves or sends it anywhere
but to the API. The relay listens on this machine only (127.0.0.1). An SSH tunnel that this machine opens gives the
GPU server its end, as a Unix socket that only its owner there can open:

    ssh -N -R SERVER_FOLDER/agent_relay.sock:127.0.0.1:8765 SERVER

The plan's rules, as the relay keeps them:
- Only the study's model at the study's effort: a request for anything else is refused (400), and no request is
  changed.
- The SDK retries a request RETRIES times (connection errors, 408, 409, 429 and 5xx). An error that remains goes back
  as an error (502), which ends the model's part of that episode.
- --limit-usd: when the cost of the requests in LOG reaches it, every further request is refused (402). It is a
  guard against a mistake, not the plan's spending limit: that one decides, before a test set starts, whether the
  whole set runs.
"""

import argparse
import datetime
import http.server
import json
import os
import sys
import threading
import time
from typing import Callable, Optional, Tuple

from .. import frontier

RETRIES = 6
PORT = 8765


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spent_usd(log_path: str) -> float:
    """The cost of the requests in the log so far."""
    if not os.path.exists(log_path):
        return 0.0
    with open(log_path, encoding="utf-8") as f:
        return sum(json.loads(line).get("cost_usd") or 0.0 for line in f if line.strip())


def problem(request) -> Optional[str]:
    """Why a request is not one of the study's; None when it is."""
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        return "the request is not a Messages request"
    if request.get("model") != frontier.MODEL:
        return f"the model must be {frontier.MODEL}, not {request.get('model')!r}"
    if (request.get("output_config") or {}).get("effort") != frontier.EFFORT:
        return f"the effort must be {frontier.EFFORT}"
    return None


class Relay:
    """ask(request) gives the model's message as a dict, or raises one of `errors` when the request failed after the
    client's retries."""

    def __init__(self, ask: Callable[[dict], dict], errors: Tuple[type, ...], log_path: str,
                 limit_usd: Optional[float] = None) -> None:
        self.ask, self.errors, self.log_path, self.limit_usd = ask, errors, log_path, limit_usd
        self.lock = threading.Lock()
        self.spent = spent_usd(log_path)

    def log(self, row: dict) -> None:
        with self.lock:
            self.spent += row.get("cost_usd") or 0.0
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"date": now(), **row}) + "\n")

    def handle(self, request) -> Tuple[int, dict]:
        """(the HTTP status, the answer's JSON) of one request."""
        why = problem(request)
        if why is not None:
            return 400, {"error": why}
        if self.limit_usd is not None and self.spent >= self.limit_usd:
            return 402, {"error": f"the relay's spending limit of {self.limit_usd:.2f} USD is reached"}
        t0 = time.perf_counter()
        try:
            message = self.ask(request)
        except self.errors as e:
            error = f"{type(e).__name__}: {e}"[:500]
            self.log({"seconds": time.perf_counter() - t0, "error": error})
            return 502, {"error": error}
        usage = {f: (message.get("usage") or {}).get(f) or 0 for f in frontier.USAGE_FIELDS}
        self.log({"seconds": time.perf_counter() - t0, "id": message.get("id"), "model": message.get("model"),
                  "stop_reason": message.get("stop_reason"), "turn": sum(m.get("role") == "assistant"
                                                                         for m in request["messages"]) + 1,
                  "usage": usage, "cost_usd": frontier.usage_cost(usage)})
        return 200, message


def handler(relay: Relay):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            except ValueError:
                status, answer = 400, {"error": "the request is not JSON"}
            else:
                status, answer = relay.handle(request)
            body = json.dumps(answer).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # the relay's own log has each request
            pass

    return Handler


def serve(relay: Relay, port: int = PORT) -> http.server.ThreadingHTTPServer:
    """The relay's server, on this machine only."""
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler(relay))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="the relay's log (JSON lines); a run adds to it")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--limit-usd", type=float, default=None, help="refuse requests once the log's cost reaches this")
    a = ap.parse_args()
    try:
        import anthropic
    except ImportError:
        raise SystemExit("the relay needs the Anthropic SDK on the machine that holds the key: "
                         "pip install anthropic") from None
    client = anthropic.Anthropic(max_retries=RETRIES)

    def ask(request: dict) -> dict:
        with client.messages.stream(**request) as stream:  # streamed: the SDK refuses a long request that is not
            return stream.get_final_message().to_dict()

    relay = Relay(ask, (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError), a.log,
                  a.limit_usd)
    server = serve(relay, a.port)
    print(f"the relay listens on 127.0.0.1:{a.port} for {frontier.MODEL} at effort {frontier.EFFORT}; "
          f"{relay.spent:.2f} USD in {a.log} so far"
          + ("" if a.limit_usd is None else f", limit {a.limit_usd:.2f} USD") + "; Ctrl+C stops it")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"stopped; {relay.spent:.2f} USD in {a.log}")
        sys.exit(0)


if __name__ == "__main__":
    main()
