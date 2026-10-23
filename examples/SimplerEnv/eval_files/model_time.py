"""How long an episode waits on the model servers: each request's round trip, from sending it to its answer, summed
per kind of model (vla: the VLA policy, vlm: the VLM, sam: SAM).

The models run in servers on the GPU, and a sim waits for each answer before it goes on. So with one sim at a time
on a GPU that runs nothing else, the round trips are the GPU time the episode costs (plus the short local transfer).
With several sims sharing the servers (K sim clients, or the planner's 4 tasks at once), a request can also wait for
the others' requests, so the round trips are then the episode's waiting time, not its GPU time; the GPU energy of a
whole eval (the runners' dmon.txt, experiments/score_costs.py) is the cost measure that holds either way.

The clients' websocket connections are wrapped (time_requests), so a round trip includes the server's work and the
transfer, but not packing the request or unpacking the answer. One meter per process (METER): a sim process runs one
episode at a time, and the evaluator resets the meter before each episode and writes it after (cost_record).
"""

import time
from typing import Dict, Optional


class Meter:
    """Requests and their round-trip seconds, per kind of model."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls: Dict[str, int] = {}
        self.seconds: Dict[str, float] = {}

    def add(self, kind: str, seconds: float) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1
        self.seconds[kind] = self.seconds.get(kind, 0.0) + seconds

    def snapshot(self) -> Dict[str, dict]:
        """{kind: {"calls": requests, "seconds": their round trips}} so far."""
        return {k: {"calls": self.calls[k], "seconds": self.seconds[k]} for k in sorted(self.calls)}


METER = Meter()


class TimedConnection:
    """A websocket connection whose requests are timed into a meter: from send() to the recv() of the answer. Every
    other attribute is the connection's own."""

    def __init__(self, conn, kind: str, meter: Meter) -> None:
        self._conn, self._kind, self._meter = conn, kind, meter
        self._sent: Optional[float] = None

    def send(self, message, *args, **kwargs):
        self._sent = time.perf_counter()
        return self._conn.send(message, *args, **kwargs)

    def recv(self, *args, **kwargs):
        try:
            return self._conn.recv(*args, **kwargs)
        finally:
            if self._sent is not None:
                self._meter.add(self._kind, time.perf_counter() - self._sent)
                self._sent = None

    def __getattr__(self, name):
        return getattr(self._conn, name)


def time_requests(client, kind: str, meter: Meter = METER):
    """Time a model client's requests (SAM3Client, VLMClientSimpler and WebsocketClientPolicy each talk to their server
    through one websocket connection, `_ws`, made once). Returns the client; None stays None."""
    if client is not None and not isinstance(client._ws, TimedConnection):
        client._ws = TimedConnection(client._ws, kind, meter)
    return client


def cost_record(episode: str, success: bool, steps: int, seconds: float, meter: Meter = METER, **info) -> dict:
    """One episode's cost: its wall-clock seconds (from the model's reset to the last step: simulation and model
    time) and its model requests (meter.snapshot()), with `info` (the task, episode and layout)."""
    return {**info, "episode": episode, "success": bool(success), "steps": int(steps), "seconds": float(seconds),
            "model": meter.snapshot()}
