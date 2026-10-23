"""CPU tests for the per-episode cost records (model_time.py) and their scorer (experiments/score_costs.py)."""

import json
import sys
from types import SimpleNamespace

import pytest

from examples.SimplerEnv.eval_files import model_time as mt
from examples.SimplerEnv.eval_files.planner.experiments import score_costs

SPOON = "PutSpoonOnTableClothInScene-v0"


class FakeConnection:
    def __init__(self):
        self.sent, self.closed = [], False

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        return b"answer"

    def close(self):
        self.closed = True


def test_meter_counts_requests_per_kind():
    m = mt.Meter()
    m.add("sam", 0.5)
    m.add("sam", 0.25)
    m.add("vla", 1.0)
    assert m.snapshot() == {"sam": {"calls": 2, "seconds": 0.75}, "vla": {"calls": 1, "seconds": 1.0}}
    m.reset()
    assert m.snapshot() == {}


def test_timed_connection_times_each_request_from_send_to_answer(monkeypatch):
    ticks = iter([10.0, 12.5, 20.0, 20.25])
    monkeypatch.setattr(mt, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    m = mt.Meter()
    conn = FakeConnection()
    timed = mt.TimedConnection(conn, "vlm", m)
    timed.send(b"q1")
    assert timed.recv() == b"answer"
    timed.send(b"q2")
    timed.recv()
    assert conn.sent == [b"q1", b"q2"]
    assert m.snapshot() == {"vlm": {"calls": 2, "seconds": 2.75}}
    timed.recv()  # an answer with no request (a greeting) is not a request
    assert m.calls == {"vlm": 2}


def test_time_requests_wraps_the_connection_once_and_keeps_its_attributes():
    m = mt.Meter()
    client = SimpleNamespace(_ws=FakeConnection())
    assert mt.time_requests(client, "sam", m) is client
    mt.time_requests(client, "sam", m)
    assert isinstance(client._ws, mt.TimedConnection)
    assert isinstance(client._ws._conn, FakeConnection)  # one layer only
    client._ws.close()
    assert client._ws.closed
    assert mt.time_requests(None, "sam", m) is None


def test_cost_record():
    m = mt.Meter()
    m.add("vla", 2.0)
    r = mt.cost_record(f"{SPOON}/e15/3", True, 120, 33.5, m, env_name=SPOON, obj_episode_id=3)
    assert r == {
        "env_name": SPOON,
        "obj_episode_id": 3,
        "episode": f"{SPOON}/e15/3",
        "success": True,
        "steps": 120,
        "seconds": 33.5,
        "model": {"vla": {"calls": 1, "seconds": 2.0}},
    }
    json.dumps(r)


DMON = (
    "#Date        Time         gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa     fb   bar1   ccpm\n"
    "#YYYYMMDD    HH:MM:SS     Idx      W      C      C      %      %      %      %      %      %     MB     MB     MB\n"
    " 20261023    23:59:55       0    100     40      -     10      5      0      0      0      0   1000      4      0\n"
    " 20261024    00:00:00       0    200     41      -     90     50      0      0      0      0   9000      4      0\n"
    " 20261024    00:00:05       0      -     41      -     90     50      0      0      0      0   9000      4      0\n"
    " 20261024    00:00:10       0    300     42      -     95     60      0      0      0      0   9000      4      0\n"
    " 20261024    00:00:15       0    300     42      -     95     60      0      0      0      0   9000      4      0\n"
)


def test_gpu_energy_integrates_power_over_time(tmp_path):
    path = tmp_path / "dmon.txt"
    path.write_text(DMON)
    assert len(score_costs.read_dmon(str(path))) == 4  # the sample with no power is left out
    joules, seconds = score_costs.gpu_energy(str(path))
    # 100 W for 5 s (over midnight), 200 W for 10 s (up to the next sample with power), 300 W for 5 s, and the last
    # 300 W for the median interval (5 s)
    assert seconds == 25.0
    assert joules == 100 * 5 + 200 * 10 + 300 * 5 + 300 * 5
    old = tmp_path / "old.txt"  # the runners' log before 2026-10-23: no power, no timestamps
    old.write_text("# gpu     sm    mem\n# Idx      %      %\n    0      0      0\n    0      5      1\n")
    assert score_costs.gpu_energy(str(old)) is None
    assert score_costs.gpu_energy(str(tmp_path / "missing.txt")) is None


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _cost(episode, success, seconds, model):
    return {
        "env_name": SPOON,
        "obj_episode_id": episode,
        "success": success,
        "steps": 50,
        "seconds": seconds,
        "model": model,
    }


def test_run_cost_and_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    run = tmp_path / "vlmplan_iid15_harsh"
    sam, vlm = {"calls": 4, "seconds": 0.4}, {"calls": 1, "seconds": 3.0}
    _write_jsonl(
        run / SPOON / "episode_costs.jsonl",
        [_cost(0, True, 10.0, {"sam": sam, "vlm": vlm}), _cost(1, False, 20.0, {"sam": sam})],
    )
    _write_jsonl(run / SPOON / "planner_stats.jsonl", [{"first_look_seconds": 4.0, "first_look_model": {"vlm": vlm}}])
    (run / "dmon.txt").write_text(DMON)
    c = score_costs.run_cost("vlmplan_iid15_harsh")
    assert (c["episodes"], c["success"], c["seconds"]) == (2, 0.5, 15.0)
    assert c["model"] == {"vla": (0.0, 0.0), "vlm": (0.5, 1.5), "sam": (4.0, 0.4)}
    assert c["energy_per_episode"] == 5500 / 2 and c["gpu_seconds"] == 25.0 and c["mean_watts"] == 220.0
    assert c["first_look"] == {
        "episodes": 1,
        "seconds": 4.0,
        "model": {"vla": (0.0, 0.0), "vlm": (1.0, 3.0), "sam": (0.0, 0.0)},
    }
    monkeypatch.setattr(sys, "argv", ["score_costs", "vlmplan_iid15_harsh"])
    score_costs.main()
    out = capsys.readouterr().out
    assert "| vlmplan_iid15_harsh | 2 | 50.0% | 15.0 | 0.0 / 0.00 | 0.5 / 1.50 | 4.0 / 0.40 | 2.75 | 220 | 0 |" in out
    assert "first look" in out


def test_cost_records_refuse_a_second_record_for_an_episode(tmp_path):
    _write_jsonl(tmp_path / SPOON / "eps_0_8" / "episode_costs.jsonl", [_cost(0, True, 1.0, {})])
    _write_jsonl(tmp_path / SPOON / "eps_0_4" / "episode_costs.jsonl", [_cost(0, False, 1.0, {})])
    with pytest.raises(SystemExit):
        score_costs.cost_records(str(tmp_path))
    # the same episode at two depth levels of a closed-loop eval is two episodes
    other = tmp_path / "closed"
    _write_jsonl(other / "harsh" / SPOON / "eps_0_8" / "episode_costs.jsonl", [_cost(0, True, 1.0, {})])
    _write_jsonl(other / "none" / SPOON / "eps_0_8" / "episode_costs.jsonl", [_cost(0, True, 1.0, {})])
    assert len(score_costs.cost_records(str(other))) == 2
