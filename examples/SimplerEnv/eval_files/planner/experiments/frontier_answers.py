"""The frontier-model study's plan arm: the frontier model's answers (README.md, "Frontier-model study").

Runs on a machine of ours that holds the API key, never on the shared GPU server. It reads the first frames that
frames_eval.sh saved (copied from the server), asks the frontier model the VLM plan's question (vlm_plan.prompt)
about each frame, and saves each answer as one line of the answers file. The planner-only eval then reads that file
in place of the VLM server (planner_eval.sh with VLM_PLAN_ANSWERS).

    python -m examples.SimplerEnv.eval_files.planner.experiments.frontier_answers FRAMES ANSWERS [--limit N]
        [--dry-run]

FRAMES: a run folder of frames_eval.sh. ANSWERS: the answers file (JSON lines); a run adds to it. The key comes from
the environment (ANTHROPIC_API_KEY), as the SDK reads it; this script never prints or saves it.

The plan's rules, as this script keeps them:
- One request per episode: the picture and the question, no tools and no system prompt, the model MODEL at effort
  EFFORT, with no fallback model. The question is vlm_plan.prompt, unchanged.
- Every episode is asked once. An episode that has a line in ANSWERS is never asked again, whatever the line says.
  Before each request the episode is written to ANSWERS.started; an episode that was started and has no answer (the
  run was killed) gets an error line at the next run instead of a second request.
- A refusal, or an error that remains after the client's retries (RETRIES), ends the model's part of the episode:
  its line has an empty text and the reason in "error", and the planner reads it as "not feasible". After an error
  line the run stops, so that an outage costs one episode and not the rest of the set; the next run goes on with the
  next episode.
- Before the first episode of a run, one small request without a picture checks the key, the model's name and the
  settings. When it fails, the run stops and no episode is asked.
"""

import argparse
import base64
import datetime
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

from .. import frontier, vlm_plan

MODEL = "claude-fable-5-2"
EFFORT = "high"
MAX_TOKENS = 32000  # thinking and the answer together; the answer itself is a few lines of JSON
RETRIES = 6  # the SDK's retries of a request (connection errors, 408, 409, 429 and 5xx, with backoff)
# USD per million input and output tokens (the model's list price on 2026-11-13), for the cost that is reported.
PRICES = {"claude-fable-5-2": (8.0, 40.0)}
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def request(instruction: str, png: bytes) -> dict:
    """The request of one episode: the picture and the VLM plan's question, nothing else."""
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "output_config": {"effort": EFFORT},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(png).decode("ascii")}},
            {"type": "text", "text": vlm_plan.prompt(instruction)},
        ]}],
    }


def send(client, kwargs: dict):
    """One request, streamed (the SDK refuses a long request that is not), as its final message."""
    with client.messages.stream(**kwargs) as stream:
        return stream.get_final_message()


def answer_fields(message) -> dict:
    """What a line keeps of the model's message: its text, why it stopped, the model that answered and the tokens.
    A refusal has "error"; its text does not count."""
    text = "".join(block.text for block in message.content if block.type == "text")
    refused = message.stop_reason == "refusal"
    details = getattr(message, "stop_details", None)
    return {
        "text": "" if refused else text,
        "error": "refusal" if refused else None,
        "stop_reason": message.stop_reason,
        "refusal_category": getattr(details, "category", None) if refused else None,
        "model": message.model,
        "id": message.id,
        "usage": {f: getattr(message.usage, f, None) for f in USAGE_FIELDS},
    }


def cost_usd(rows: List[dict]) -> Optional[float]:
    """The list-price cost of these lines' tokens; None when a line's model has no price here."""
    total = 0.0
    for r in rows:
        usage, price = r.get("usage") or {}, PRICES.get(r.get("request", {}).get("model"))
        if price is None:
            return None
        total += ((usage.get("input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)
                  + (usage.get("cache_read_input_tokens") or 0)) * price[0] / 1e6
        total += (usage.get("output_tokens") or 0) * price[1] / 1e6
    return total


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append(path: str, row: dict) -> None:
    """One line at the end of a file, on the disk before the next request."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def base_row(frame: dict) -> dict:
    return {"key": frame["key"], "instruction": frame["instruction"], "frame": frame["frame"],
            "image_sha256": frame["image_sha256"],
            "prompt_sha256": frontier.text_hash(vlm_plan.prompt(frame["instruction"])),
            "request": {"model": MODEL, "effort": EFFORT, "max_tokens": MAX_TOKENS}}


def error_row(frame: dict, error: str) -> dict:
    return {**base_row(frame), "text": "", "error": error, "stop_reason": None, "refusal_category": None,
            "model": None, "id": None, "usage": {}, "seconds": None, "date": now()}


def started_keys(path: str) -> List[Tuple[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [(r["key"], r["instruction"]) for r in map(json.loads, filter(str.strip, f))]


def run(frames: List[dict], answers_path: str, ask: Callable[[dict], object], errors: Tuple[type, ...],
        limit: Optional[int] = None, log: Callable[[str], None] = print) -> Dict[str, int]:
    """Asks the frames that have no answer yet, in order, and adds each answer to the answers file. ask(kwargs) gives
    the model's message or raises one of `errors` when the request failed after the client's retries. Returns the
    counts: asked, refusals, errors (the run stops after one), interrupted (started in an earlier run, no answer)."""
    done = frontier.read_answers(answers_path) if os.path.exists(answers_path) else {}
    by_key = {(f["key"], f["instruction"]): f for f in frames}
    counts = {"asked": 0, "refusals": 0, "errors": 0, "interrupted": 0}
    started_path = answers_path + ".started"
    for k in started_keys(started_path):  # a request went out and its answer was never saved: no second request
        if k not in done and k in by_key:
            row = error_row(by_key[k], "error: the run stopped before the answer was saved")
            append(answers_path, row)
            done[k] = row
            counts["interrupted"] += 1
            log(f"{k[0]}: started in an earlier run without a saved answer: an error line, not a second request")
    for frame in frames:
        k = (frame["key"], frame["instruction"])
        if k in done:
            continue
        if limit is not None and counts["asked"] >= limit:
            break
        with open(frame["path"], "rb") as f:
            png = f.read()
        if frontier.image_hash(frontier.read_png(frame["path"])) != frame["image_sha256"]:
            raise SystemExit(f"{frame['path']}: the file is not the picture that frames.jsonl names")
        append(started_path, {"key": k[0], "instruction": k[1], "date": now()})
        t0 = time.perf_counter()
        try:
            fields = answer_fields(ask(request(frame["instruction"], png)))
        except errors as e:
            fields = None
            row = {**error_row(frame, f"error: {type(e).__name__}: {e}"[:500]), "seconds": time.perf_counter() - t0}
        if fields is not None:
            row = {**base_row(frame), **fields, "seconds": time.perf_counter() - t0, "date": now()}
        append(answers_path, row)
        done[k] = row
        counts["asked"] += 1
        if row["error"] == "refusal":
            counts["refusals"] += 1
        elif row["error"]:
            counts["errors"] += 1
            log(f"{k[0]}: {row['error']}\nThe run stops after an error; the next run goes on with the next episode.")
            break
        log(f"{counts['asked']}: {k[0]}: {row['stop_reason']}, {row['usage'].get('output_tokens')} output tokens, "
            f"{row['seconds']:.0f} s")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", help="a run folder of frames_eval.sh")
    ap.add_argument("answers", help="the answers file (JSON lines); a run adds to it")
    ap.add_argument("--limit", type=int, default=None, help="ask at most this many episodes in this run")
    ap.add_argument("--dry-run", action="store_true", help="count the episodes that would be asked; no request")
    a = ap.parse_args()
    frames = frontier.read_frames(a.frames)
    done = frontier.read_answers(a.answers) if os.path.exists(a.answers) else {}
    left = [f for f in frames if (f["key"], f["instruction"]) not in done]
    print(f"{len(frames)} first frames in {a.frames}; {len(frames) - len(left)} have an answer in {a.answers}, "
          f"{len(left)} do not")
    if done:
        spent = cost_usd(list(done.values()))
        print("the saved answers' tokens at the list price: "
              + ("no price for their model" if spent is None else f"{spent:.2f} USD"))
    if a.dry_run or not left:
        return
    try:
        import anthropic
    except ImportError:
        raise SystemExit("this script needs the Anthropic SDK on the machine that holds the key: "
                         "pip install anthropic") from None
    client = anthropic.Anthropic(max_retries=RETRIES)
    # A response that is not a success, a network failure or a timeout, each after the SDK's retries.
    api_errors = (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError)
    try:  # the key, the model's name and the settings, before any episode
        check = send(client, {"model": MODEL, "max_tokens": MAX_TOKENS, "output_config": {"effort": EFFORT},
                              "messages": [{"role": "user", "content": "Reply with the one word: ready"}]})
    except api_errors as e:
        raise SystemExit(f"the check request failed, so no episode was asked: {type(e).__name__}: {e}") from None
    print(f"check request: answered by {check.model} ({check.stop_reason})")
    counts = run(frames, a.answers, lambda kwargs: send(client, kwargs), api_errors, a.limit)
    rows = list(frontier.read_answers(a.answers).values())
    spent = cost_usd(rows)
    print(f"this run: {counts}; the answers file has {len(rows)} of {len(frames)} episodes"
          + ("" if spent is None else f", {spent:.2f} USD at the list price"))
    if counts["errors"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
