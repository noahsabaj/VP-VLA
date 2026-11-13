"""CPU tests for the frontier-model study's plan arm: the first-frame export and the saved answers
(planner/frontier.py), the script that asks the frontier model (experiments/frontier_answers.py, with a made-up SDK:
no request leaves the machine) and the planner client reading a saved answer."""

import base64
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import frontier, vlm_plan
from examples.SimplerEnv.eval_files.planner.experiments import frontier_answers as fa

README = Path(__file__).parents[1] / "examples/SimplerEnv/eval_files/planner/experiments/README.md"
TASK = "PutCarrotOnPlateInScene-v0"
INSTRUCTION = "put carrot on plate"
PLAN = '{"feasible": true, "steps": [{"object": "carrot", "object_point": [500, 500], "target": "plate"}]}'


def _image(seed=0, shape=(48, 64, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _export(folder, episodes=3, instruction=INSTRUCTION, task=TASK):
    """A task's folder as frames_eval.sh leaves it: each episode's first frame and its line in frames.jsonl."""
    exporter = frontier.FrameExporter(str(folder / task))
    for e in range(episodes):
        exporter.set_episode(f"{task}/e98/{e}")
        exporter.reset(instruction)
        exporter.step(_image(e), instruction)
    return exporter


def _message(text=PLAN, stop_reason="end_turn", category=None):
    """The SDK's final message, as far as the script reads it."""
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=types.SimpleNamespace(category=category) if category else None,
        model=fa.MODEL,
        id="msg_made_up",
        usage=types.SimpleNamespace(
            input_tokens=1000, output_tokens=500, cache_creation_input_tokens=0, cache_read_input_tokens=0
        ),
    )


class Outage(Exception):
    pass


class Asker:
    """Stands in for the API: answers every request, or fails at the requests whose number is in `fail`."""

    def __init__(self, fail=(), refuse=()):
        self.requests, self.fail, self.refuse = [], set(fail), set(refuse)

    def __call__(self, kwargs):
        self.requests.append(kwargs)
        n = len(self.requests)
        if n in self.fail:
            raise Outage("overloaded")
        if n in self.refuse:
            return _message(text="I cannot help with that.", stop_reason="refusal", category="cyber")
        return _message()


def _run(folder, answers, asker, **kwargs):
    return fa.run(frontier.read_frames(str(folder)), str(answers), asker, (Outage,), log=lambda line: None, **kwargs)


def test_a_png_file_gives_back_the_same_picture(tmp_path):
    image = _image()
    frontier.write_png(str(tmp_path / "frame.png"), image)
    back = frontier.read_png(str(tmp_path / "frame.png"))
    assert back.dtype == np.uint8 and np.array_equal(back, image)
    assert frontier.image_hash(back) == frontier.image_hash(image) != frontier.image_hash(_image(1))
    # the same bytes in another shape are another picture
    assert frontier.image_hash(image.reshape(64, 48, 3)) != frontier.image_hash(image)


def test_the_exporter_saves_each_first_frame_once_and_ends_the_episode(tmp_path):
    exporter = _export(tmp_path, episodes=2)
    exporter.set_episode(f"{TASK}/e98/2")
    exporter.reset(INSTRUCTION)
    _, action = exporter.step(_image(2), INSTRUCTION)
    assert action["terminate_episode"][0] == 1 and not action["world_vector"].any()  # the robot does not move
    exporter.step(_image(7), INSTRUCTION)  # a later frame of the same episode is not saved
    rows = frontier.read_frames(str(tmp_path))
    assert [r["key"] for r in rows] == [f"{TASK}/e98/{e}" for e in range(3)]
    assert {r["instruction"] for r in rows} == {INSTRUCTION}
    assert rows[2]["frame"] == f"frames/{TASK}__e98__2.png" and (rows[2]["width"], rows[2]["height"]) == (64, 48)
    for e, row in enumerate(rows):
        assert (
            frontier.image_hash(frontier.read_png(row["path"])) == row["image_sha256"] == frontier.image_hash(_image(e))
        )
    exporter.reset(INSTRUCTION)  # the same episode again: its frame would replace the saved one
    with pytest.raises(ValueError, match="a second first frame"):
        exporter.step(_image(2), INSTRUCTION)


def test_read_frames_stops_on_a_wrong_folder(tmp_path):
    with pytest.raises(SystemExit, match=r"no frames\.jsonl"):
        frontier.read_frames(str(tmp_path))
    _export(tmp_path, episodes=2)
    file = tmp_path / TASK / frontier.FRAMES_FILE
    lines = file.read_text().splitlines()
    file.write_text("\n".join([*lines, lines[0]]) + "\n")
    with pytest.raises(SystemExit, match="a second frame for episode"):
        frontier.read_frames(str(tmp_path))
    file.write_text("\n".join(lines) + "\n")
    (tmp_path / TASK / "frames" / f"{TASK}__e98__1.png").unlink()
    with pytest.raises(SystemExit, match="no file"):
        frontier.read_frames(str(tmp_path))


def test_the_request_holds_the_picture_and_the_question_only(tmp_path):
    _export(tmp_path, episodes=1)
    (frame,) = frontier.read_frames(str(tmp_path))
    png = Path(frame["path"]).read_bytes()
    request = fa.request(INSTRUCTION, png)
    assert set(request) == {"model", "max_tokens", "output_config", "messages"}  # no tools, no system prompt
    assert request["model"] == "claude-fable-5-2" and request["output_config"] == {"effort": "high"}
    (message,) = request["messages"]
    picture, question = message["content"]
    assert message["role"] == "user" and picture["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(picture["source"]["data"]) == png
    assert question == {"type": "text", "text": vlm_plan.prompt(INSTRUCTION)}  # the VLM plan's question, unchanged


def test_every_episode_is_asked_once(tmp_path):
    _export(tmp_path / "frames", episodes=3)
    answers, asker = tmp_path / "answers.jsonl", Asker()
    assert _run(tmp_path / "frames", answers, asker, limit=2) == {
        "asked": 2,
        "refusals": 0,
        "errors": 0,
        "interrupted": 0,
    }
    assert _run(tmp_path / "frames", answers, asker)["asked"] == 1  # the third episode only
    assert _run(tmp_path / "frames", answers, asker)["asked"] == 0 and len(asker.requests) == 3
    rows = frontier.read_answers(str(answers))
    assert list(rows) == [(f"{TASK}/e98/{e}", INSTRUCTION) for e in range(3)]
    row = rows[f"{TASK}/e98/1", INSTRUCTION]
    assert row["text"] == PLAN and row["error"] is None and row["model"] == fa.MODEL
    assert row["request"] == {"model": fa.MODEL, "effort": fa.EFFORT, "max_tokens": fa.MAX_TOKENS}
    assert row["prompt_sha256"] == frontier.text_hash(vlm_plan.prompt(INSTRUCTION))
    assert row["usage"]["output_tokens"] == 500 and row["seconds"] >= 0 and row["date"].endswith("Z")
    assert fa.cost_usd(list(rows.values())) == pytest.approx(3 * (1000 * 8 + 500 * 40) / 1e6)
    assert fa.cost_usd([{**row, "request": {"model": "another-model"}}]) is None


def test_a_refusal_is_an_answer_and_an_error_stops_the_run(tmp_path):
    _export(tmp_path / "frames", episodes=4)
    answers, asker = tmp_path / "answers.jsonl", Asker(fail={2}, refuse={1})
    counts = _run(tmp_path / "frames", answers, asker)
    assert counts == {"asked": 2, "refusals": 1, "errors": 1, "interrupted": 0}  # the run stops after the error
    rows = list(frontier.read_answers(str(answers)).values())
    assert [(r["text"], r["error"]) for r in rows] == [("", "refusal"), ("", "error: Outage: overloaded")]
    assert rows[0]["refusal_category"] == "cyber" and rows[0]["stop_reason"] == "refusal"
    # the next run goes on with the next episode: the failed one is not asked again
    assert _run(tmp_path / "frames", answers, asker) == {"asked": 2, "refusals": 0, "errors": 0, "interrupted": 0}
    assert len(asker.requests) == 4 and len(frontier.read_answers(str(answers))) == 4
    for row in rows:  # both count as "not feasible" for the planner
        assert vlm_plan.parse(row["text"], 64, 48)["feasible"] is False


def test_an_episode_whose_run_was_killed_is_not_asked_again(tmp_path):
    _export(tmp_path / "frames", episodes=2)
    answers = tmp_path / "answers.jsonl"
    # a run that was killed after it sent the first episode's request
    fa.append(str(answers) + ".started", {"key": f"{TASK}/e98/0", "instruction": INSTRUCTION, "date": fa.now()})
    asker = Asker()
    counts = _run(tmp_path / "frames", answers, asker)
    assert counts == {"asked": 1, "refusals": 0, "errors": 0, "interrupted": 1} and len(asker.requests) == 1
    rows = frontier.read_answers(str(answers))
    assert rows[f"{TASK}/e98/0", INSTRUCTION]["error"].startswith("error: the run stopped before")
    assert rows[f"{TASK}/e98/1", INSTRUCTION]["text"] == PLAN


def test_a_changed_frame_file_stops_the_run(tmp_path):
    _export(tmp_path / "frames", episodes=1)
    (frame,) = frontier.read_frames(str(tmp_path / "frames"))
    frontier.write_png(frame["path"], _image(5))
    asker = Asker()
    with pytest.raises(SystemExit, match=r"not the picture that frames\.jsonl names"):
        _run(tmp_path / "frames", tmp_path / "answers.jsonl", asker)
    assert not asker.requests


def _answers(tmp_path, asker=None):
    _export(tmp_path / "frames", episodes=2)
    _run(tmp_path / "frames", tmp_path / "answers.jsonl", asker or Asker())
    return frontier.SavedAnswers(str(tmp_path / "answers.jsonl"))


def test_a_saved_answer_is_tied_to_its_episode_question_and_picture(tmp_path, monkeypatch):
    saved = _answers(tmp_path)
    key = f"{TASK}/e98/1"
    assert saved.answer(key, INSTRUCTION, _image(1)) == {"text": PLAN, "model": fa.MODEL, "error": None}
    with pytest.raises(SystemExit, match="no saved answer for episode"):
        saved.answer(f"{TASK}/e98/2", INSTRUCTION, _image(1))
    with pytest.raises(SystemExit, match="no saved answer for episode"):
        saved.answer(key, "put the carrot on the dish", _image(1))  # another wording
    with pytest.raises(SystemExit, match="first frame is not the picture"):
        saved.answer(key, INSTRUCTION, _image(0))
    monkeypatch.setattr(vlm_plan, "prompt", lambda instruction: f"A new question about {instruction}")
    with pytest.raises(SystemExit, match="question is not the one that was answered"):
        saved.answer(key, INSTRUCTION, _image(1))


def test_a_second_answer_or_an_empty_file_stops_the_load(tmp_path):
    saved = _answers(tmp_path)
    file = Path(saved.path)
    file.write_text(file.read_text() + file.read_text().splitlines()[0] + "\n")
    with pytest.raises(SystemExit, match="a second answer for episode"):
        frontier.SavedAnswers(str(file))
    file.write_text("")
    with pytest.raises(SystemExit, match="no saved answers"):
        frontier.SavedAnswers(str(file))


def test_the_client_plans_from_the_saved_answer(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from examples.SimplerEnv.eval_files.planner.planner_client import PlannerClient

    c = PlannerClient.__new__(PlannerClient)  # no servers: only what the VLM plan reads
    c.vlm_answers, c.vlm_planner = _answers(tmp_path, Asker(refuse={2})), None
    c.task_description, c.steps, c.stats, c.episode_key = INSTRUCTION, None, {}, f"{TASK}/e98/0"
    assert c._vlm_plan(_image(0)) and [(m["object"], m["target"]) for m in c.steps] == [("carrot", "plate")]
    assert c.stats["vlm_plan_answer"] == {"file": "answers.jsonl", "model": fa.MODEL, "error": None}
    assert c.stats["vlm_plan_feasible"] and c.stats["vlm_plan_text"] == PLAN
    # a refusal: no pick-and-place moves, so the episode is "not feasible"
    c.steps, c.stats, c.episode_key = None, {}, f"{TASK}/e98/1"
    assert not c._vlm_plan(_image(1)) and c.steps is None
    assert c.stats["vlm_plan_answer"]["error"] == "refusal" and not c.stats["vlm_plan_feasible"]
    assert c.stats["failure"].startswith("VLM plan: not pick-and-place")


class FakeSDK(types.ModuleType):
    """The parts of the anthropic package that the script uses; `fail` makes every request fail."""

    def __init__(self, fail=False):
        super().__init__("anthropic")
        sdk = self
        self.requests, self.clients = [], []
        self.APIStatusError = type("APIStatusError", (Exception,), {})
        self.APIConnectionError = type("APIConnectionError", (Exception,), {})
        self.APITimeoutError = type("APITimeoutError", (self.APIConnectionError,), {})

        class Stream:
            def __init__(self, kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                sdk.requests.append(self.kwargs)
                if fail:
                    raise sdk.APIStatusError("401 invalid x-api-key")
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return _message()

        class Anthropic:
            def __init__(self, **kwargs):
                sdk.clients.append(kwargs)
                self.messages = types.SimpleNamespace(stream=lambda **kwargs: Stream(kwargs))

        self.Anthropic = Anthropic


def _main(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["frontier_answers", *map(str, argv)])
    fa.main()
    return capsys.readouterr().out


def test_main_checks_the_key_before_the_first_episode(tmp_path, monkeypatch, capsys):
    _export(tmp_path / "frames", episodes=2)
    answers = tmp_path / "answers.jsonl"
    monkeypatch.setitem(sys.modules, "anthropic", None)  # no SDK: a dry run needs none
    out = _main(monkeypatch, capsys, tmp_path / "frames", answers, "--dry-run")
    assert "2 first frames" in out and "0 have an answer" in out and "2 do not" in out
    with pytest.raises(SystemExit, match="pip install anthropic"):
        _main(monkeypatch, capsys, tmp_path / "frames", answers)

    broken = FakeSDK(fail=True)
    monkeypatch.setitem(sys.modules, "anthropic", broken)
    with pytest.raises(SystemExit, match="the check request failed, so no episode was asked"):
        _main(monkeypatch, capsys, tmp_path / "frames", answers)
    assert len(broken.requests) == 1 and not answers.exists()

    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    out = _main(monkeypatch, capsys, tmp_path / "frames", answers)
    assert sdk.clients == [{"max_retries": fa.RETRIES}]  # the key comes from the environment, as the SDK reads it
    check, *episodes = sdk.requests
    assert check["messages"][0]["content"] == "Reply with the one word: ready" and len(episodes) == 2
    assert all(r["model"] == fa.MODEL and r["output_config"] == {"effort": fa.EFFORT} for r in sdk.requests)
    assert f"check request: answered by {fa.MODEL}" in out and "the answers file has 2 of 2 episodes" in out
    assert len(frontier.read_answers(str(answers))) == 2
    out = _main(monkeypatch, capsys, tmp_path / "frames", answers)  # nothing left: no request, not even the check
    assert "2 have an answer" in out and len(sdk.requests) == 3
    assert json.loads(Path(str(answers) + ".started").read_text().splitlines()[0])["key"] == f"{TASK}/e98/0"


def test_the_readme_has_the_plan_arms_commands():
    text = README.read_text(encoding="utf-8")
    block = text[text.index("## Evals (GPU)") : text.index("## Result files (CPU)")]
    for needle in ("frames_eval.sh 0 ", "frontier_answers ", "VLM_PLAN_ANSWERS="):
        assert needle in block, needle
