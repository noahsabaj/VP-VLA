"""CPU tests for the energy runs: the GPU energy counter and the null controller (energy.py), the meter's energy and
controller time (model_time.py), and the energy study's scorer (experiments/score_energy.py)."""

import ctypes
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.SimplerEnv.eval_files import energy
from examples.SimplerEnv.eval_files import model_time as mt
from examples.SimplerEnv.eval_files.planner.experiments import score_energy as se

SPOON = "PutSpoonOnTableClothInScene-v0"
CARROT = "PutCarrotOnPlateInScene-v0"


class FakeNVML:
    """The three NVML calls of GPUEnergy; the counter gives the millijoules in `values`, one per read."""

    def __init__(self, values, code=0):
        self.values, self.code, self.index = iter(values), code, None

    def nvmlInit_v2(self):
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, index, handle):
        self.index = index.value
        return self.code

    def nvmlDeviceGetTotalEnergyConsumption(self, handle, value):
        value._obj.value = next(self.values)
        return 0


class FakeCounter:
    def __init__(self, joules, index=0):
        self.values, self.index = iter(joules), index

    def joules(self):
        return next(self.values)


def test_gpu_energy_reads_the_counter_in_joules():
    lib = FakeNVML([1_234_567, 2_000_000])
    counter = energy.GPUEnergy(1, lib)
    assert lib.index == 1 and counter.index == 1
    assert counter.joules() == pytest.approx(1234.567)
    assert counter.joules() == 2000.0
    with pytest.raises(RuntimeError, match="nvmlDeviceGetHandleByIndex_v2"):
        energy.GPUEnergy(0, FakeNVML([], code=999))


def test_energy_from_env_only_in_the_energy_runs(monkeypatch):
    monkeypatch.delenv(energy.GPU_ENV, raising=False)
    assert energy.energy_from_env() is None
    monkeypatch.setenv(energy.GPU_ENV, " ")
    assert energy.energy_from_env() is None


def test_idle_power_is_the_mean_power_over_the_window():
    ticks, slept = iter([100.0, 160.0]), []
    out = energy.idle_power(
        FakeCounter([1000.0, 4000.0], index=0), 60.0, settle=30.0, clock=lambda: next(ticks), sleep=slept.append
    )
    assert slept == [30.0, 60.0]
    assert out == {"gpu": 0, "seconds": 60.0, "joules": 3000.0, "watts": 50.0}


def test_null_controller_does_nothing_and_never_stops():
    c = energy.NullController()
    c.reset("put the spoon on the towel")
    raw, action = c.step(np.zeros((480, 640, 3), np.uint8), "put the spoon on the towel", obs={}, env=None)
    assert not action["world_vector"].any() and not action["rot_axangle"].any()
    assert action["gripper"].tolist() == [1.0] and action["terminate_episode"].tolist() == [0.0]
    assert raw["open_gripper"].tolist() == [1.0]


def test_meter_reads_the_energy_since_reset():
    m = mt.Meter(FakeCounter([10.0, 25.5, 100.0, 130.0]))
    assert m.gpu_joules() == 15.5
    m.reset()
    assert m.gpu_joules() == 30.0
    assert mt.Meter().gpu_joules() is None


def test_meter_times_the_controller(monkeypatch):
    wall, cpu = iter([1.0, 3.5, 10.0, 11.0]), iter([0.5, 1.0, 2.0, 2.25])
    monkeypatch.setattr(mt, "time", SimpleNamespace(perf_counter=lambda: next(wall), process_time=lambda: next(cpu)))
    m = mt.Meter()
    with m.controller():
        pass
    with pytest.raises(ValueError), m.controller():
        raise ValueError("a failed step is timed too")
    assert (m.controller_seconds, m.controller_cpu_seconds) == (3.5, 0.75)
    r = mt.cost_record("e", True, 3, 9.0, m)
    assert r["controller"] == {"seconds": 3.5, "cpu_seconds": 0.75}


# ------------------------------------------------------------------------------------------------- the scorer


def _jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _cost(task, episode, success, steps, joules, controller=(5.0, 2.0)):
    return {
        "env_name": task,
        "obj_episode_id": episode,
        "success": success,
        "steps": steps,
        "seconds": 30.0,
        "model": {},
        "controller": {"seconds": controller[0], "cpu_seconds": controller[1]},
        "gpu_joules": joules,
    }


def _energy_run(root, name, per_step, idle, episodes, plans=None, level=None):
    """An energy run: its episodes (task, index, success, steps, controller joules), its null episodes (120 steps
    at per_step J each) and its idle power."""
    run = root / name
    folder = run / level if level else run
    for task in (SPOON, CARROT):
        mine = [e for e in episodes if e[0] == task]
        recs = [_cost(t, i, y, s, s * per_step + j) for t, i, y, s, j in mine]
        _jsonl(folder / task / "eps_0_3" / "episode_costs.jsonl", recs)
        if plans:
            _jsonl(folder / task / "eps_0_3" / "planner_stats.jsonl", [p for p in plans if p["env_name"] == task])
        null = [_cost(task, i, False, 120, 120 * per_step) for i in range(3)]
        _jsonl(root / f"{name}_null" / task / "eps_0_3" / "episode_costs.jsonl", null)
    (run / "idle_power.json").write_text(json.dumps({"gpu": 0, "seconds": 60.0, "joules": 60 * idle, "watts": idle}))


# The ASM's choices: index 0 both tasks to the planner, index 1 the spoon to the VLA, index 2 both to the VLA.
CHOICES = {
    (SPOON, 0): "planner",
    (CARROT, 0): "planner",
    (SPOON, 1): "vla",
    (CARROT, 1): "planner",
    (SPOON, 2): "vla",
    (CARROT, 2): "vla",
}


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    keys = sorted(CHOICES)
    # VLA alone: 20 kJ of controller energy per episode, 120 steps; it succeeds on index 1 and 2.
    _energy_run(tmp_path, se.VLA_RUN, 10.0, 60.0, [(t, i, i > 0, 120, 20000.0) for t, i in keys])
    # Planner alone (harsh): 500 J per episode, 200 J of it the first look, 40 steps; it succeeds on index 0.
    looks = [{"env_name": t, "obj_episode_id": i, "first_look_gpu_joules": 200.0} for t, i in keys]
    _energy_run(tmp_path, se.PLANNER_RUNS["harsh"], 8.0, 50.0, [(t, i, i == 0, 40, 500.0) for t, i in keys], looks)
    # The closed-loop ASM system: the planner's episode, or the first look and the VLA's episode.
    asm_eps = [
        (
            t,
            i,
            (i == 0) if CHOICES[t, i] == "planner" else True,
            40 if CHOICES[t, i] == "planner" else 120,
            500.0 if CHOICES[t, i] == "planner" else 20200.0,
        )
        for t, i in keys
    ]
    plans = [
        {"env_name": t, "obj_episode_id": i, "selector_choice": CHOICES[t, i], "first_look_gpu_joules": 200.0}
        for t, i in keys
    ]
    _energy_run(tmp_path, se.ASM_RUN, 9.0, 70.0, asm_eps, plans, level="harsh")
    return tmp_path


def test_episodes_subtract_the_simulation(study):
    vla = se.load_run(se.VLA_RUN)
    e = vla[(SPOON, 1)]
    assert (e.joules, e.sim_joules, e.controller_joules) == (21200.0, 1200.0, 20000.0)
    assert e.above_idle_joules == 20000.0 - 60.0 * 5.0
    assert e.cpu_joules == pytest.approx(2.0 * 350 / 32)
    assert se.run_facts(vla) == {"idle_watts": 60.0, "sim_joules_per_step": {SPOON: 10.0, CARROT: 10.0}}
    asm = se.load_run(se.ASM_RUN, "harsh")
    assert asm[(SPOON, 1)].choice == "vla" and asm[(SPOON, 1)].controller_joules == 20200.0


def test_score(study):
    vla, asm = se.load_run(se.VLA_RUN), {"harsh": se.load_run(se.ASM_RUN, "harsh")}
    planner = {"harsh": se.load_run(se.PLANNER_RUNS["harsh"])}
    out = se.score(vla, planner, asm)
    lv = out["levels"]["harsh"]
    assert lv["indices"] == [0, 1, 2] and lv["vla_picks"] == 3
    s = lv["systems"]
    assert s["VLA alone"]["joules"] == 20000.0 and s["Planner alone"]["joules"] == 500.0
    assert s["ASM system"]["joules"] == pytest.approx((500 + 500 + 20200 + 500 + 20200 + 20200) / 6)
    assert s["ASM system (offline)"]["joules"] == pytest.approx(s["ASM system"]["joules"])
    assert s["ASM system"]["saving_vs_vla"] == pytest.approx(1 - 10350 / 20000)
    assert s["VLA alone"]["success"] == pytest.approx(4 / 6)
    assert s["VLA alone"]["joules_per_success"] == pytest.approx(6 * 20000 / 4)
    # over 1 in 40 draws take index 0 only, where the VLA never succeeds: no finite upper end
    assert s["VLA alone"]["joules_per_success_interval"][1] == float("inf")
    assert s["VLA alone"]["kwh_per_million"] == pytest.approx(20000 * 1e6 / 3.6e6)
    # the oracle: the planner on index 0 (it succeeds), the first look and the VLA elsewhere
    assert s["Oracle"]["joules"] == pytest.approx((500 * 2 + 20200 * 4) / 6) and s["Oracle"]["success"] == 1.0
    # per index, ASM - VLA summed over the tasks: -39000, -19300, +400: 4 of the 8 sign patterns are as far out
    t = out["planned"]
    assert (t["lower"], t["higher"], t["episodes"]) == (2, 1, 6)
    assert t["mean"] == pytest.approx(-57900 / 6) and t["p"] == pytest.approx(0.5) and not t["win"]


def test_sign_flip_exact():
    assert se.sign_flip_exact(np.array([-3.0, -2.0, -1.0])) == pytest.approx(2 / 8)
    assert se.sign_flip_exact(np.array([5.0, -5.0])) == 1.0
    many = np.full(24, -2.0)  # past 20 values: the test on whole units
    assert se.sign_flip_exact(many) == pytest.approx(2 / 2**24)


def test_main_prints_the_report_and_writes_json(study, monkeypatch, capsys):
    out = study / "energy.json"
    argv = ["score_energy", "--planner", f"harsh={se.PLANNER_RUNS['harsh']}", "--levels", "harsh", "--json", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    se.main()
    text = capsys.readouterr().out
    assert "Planned test (harsh depth" in text and "not a win" in text
    assert "| ASM system | 6 | 83.3% | 10.35" in text
    assert "| VLA alone | 60.0 | Carrot 10.0, Spoon 10.0 |" in text
    assert json.loads(out.read_text())["planned"]["indices"] == 3


def test_an_episode_without_energy_is_refused(study):
    path = study / se.VLA_RUN / SPOON / "eps_0_3" / "episode_costs.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    recs[0].pop("gpu_joules")
    _jsonl(path, recs)
    with pytest.raises(SystemExit, match="no gpu_joules"):
        se.load_run(se.VLA_RUN)


def test_ctypes_byref_keeps_its_object():  # FakeNVML relies on it
    v = ctypes.c_ulonglong()
    assert ctypes.byref(v)._obj is v
