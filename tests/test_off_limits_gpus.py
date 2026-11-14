"""The off-limits GPUs of the eval runners (experiments/off_limits_gpus.py and GPU_GUARD in common.sh).

Run from the repository root with `pytest tests/test_off_limits_gpus.py`. The tests of the kernel's restriction
itself run on Linux with Landlock only; the others need no GPU and no Linux.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from examples.SimplerEnv.eval_files.planner.experiments import off_limits_gpus as olg

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "examples/SimplerEnv/eval_files/planner/experiments"
needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="the runners are bash scripts"
)
needs_landlock = pytest.mark.skipif(olg.landlock_version() < 1, reason="needs Linux with Landlock")

# A machine with two GPUs whose device minors are not nvidia-smi's indices (as on the server where this was found).
SMI = "0, 00000000:01:00.0\n1, 00000000:E1:00.0\n"
INFORMATION = {
    "0000:01:00.0": "Model: \t\t NVIDIA RTX A6000\nBus Location: \t 0000:01:00.0\nDevice Minor: \t 1\n",
    "0000:e1:00.0": "Model: \t\t NVIDIA RTX A6000\nBus Location: \t 0000:e1:00.0\nDevice Minor: \t 0\n",
}


def test_the_setting_is_a_list_of_gpu_indices():
    assert olg.off_limits("") == ()
    assert olg.off_limits("1") == ("1",)
    assert olg.off_limits(" 1, 3 ,") == ("1", "3")
    with pytest.raises(SystemExit, match="GPU indices"):
        olg.off_limits("GPU-5b4c")
    with pytest.raises(SystemExit, match="twice"):
        olg.off_limits("1,1")


def test_a_gpus_node_comes_from_its_device_minor_not_from_its_index():
    hidden, others = olg.gpu_nodes(SMI, INFORMATION.__getitem__, ("1",))
    assert hidden == ("/dev/nvidia0",)  # GPU 1 has the minor 0
    assert others == ("/dev/nvidia1",)
    hidden, others = olg.gpu_nodes(SMI, INFORMATION.__getitem__, ("0",))
    assert (hidden, others) == (("/dev/nvidia1",), ("/dev/nvidia0",))


def test_a_setting_that_does_not_fit_the_machine_stops_the_run():
    with pytest.raises(SystemExit, match="GPU 2"):
        olg.gpu_nodes(SMI, INFORMATION.__getitem__, ("2",))
    with pytest.raises(SystemExit, match="leaves no GPU"):
        olg.gpu_nodes(SMI, INFORMATION.__getitem__, ("0", "1"))
    with pytest.raises(SystemExit, match="no device minor"):
        olg.gpu_nodes(SMI, lambda address: "Model: NVIDIA RTX A6000\n", ("1",))
    with pytest.raises(SystemExit, match="INDEX, PCI ADDRESS"):
        olg.gpu_nodes("No devices were found\n", INFORMATION.__getitem__, ("1",))


def tree(folders, links):
    """listdir and islink of a made-up file tree: {folder: [entries]} and the set of links."""
    return (lambda folder: folders[folder]), (lambda path: path in links)


def test_every_path_but_the_hidden_ones_gets_a_rule():
    folders = {
        "/": ["bin", "dev", "home", "usr"],
        "/dev": ["null", "nvidia0", "nvidia1", "nvidiactl", "shm", "stdin"],
    }
    listdir, islink = tree(folders, {"/bin", "/dev/stdin"})
    assert olg.allowed(["/dev/nvidia0"], listdir, islink) == [
        "/dev/null",
        "/dev/nvidia1",
        "/dev/nvidiactl",
        "/dev/shm",
        "/home",
        "/usr",
    ]
    # two hidden nodes; a folder above a hidden path gets no rule of its own, so the path stays forbidden
    assert olg.allowed(["/dev/nvidia0", "/dev/nvidia1"], listdir, islink) == [
        "/dev/null",
        "/dev/nvidiactl",
        "/dev/shm",
        "/home",
        "/usr",
    ]
    folders["/home"] = ["a", "b"]
    folders["/home/a"] = ["keep", "secret"]
    assert olg.allowed(["/home/a/secret"], listdir, islink) == ["/dev", "/home/a/keep", "/home/b", "/usr"]
    with pytest.raises(ValueError, match="absolute"):
        olg.allowed(["nvidia0"], listdir, islink)


RESTRICTED = """
import os, subprocess, sys
from examples.SimplerEnv.eval_files.planner.experiments import off_limits_gpus as olg
hidden, other, new = sys.argv[1:4]
olg.restrict([hidden], [other])
for path in (hidden, other):
    try:
        os.close(os.open(path, os.O_RDWR))
        print(os.path.basename(path), "opens")
    except OSError as e:
        print(os.path.basename(path), type(e).__name__)
with open(new, "w") as f:
    f.write("made after the restriction")
print("new file:", open(new).read())
child = subprocess.run([sys.executable, "-c", "open(__import__('sys').argv[1])", hidden], capture_output=True)
print("child:", "PermissionError" if b"PermissionError" in child.stderr else child.returncode)
"""


def restricted(*paths):
    return subprocess.run([sys.executable, "-c", RESTRICTED, *map(str, paths)], cwd=REPO, capture_output=True, text=True)


@needs_landlock
def test_the_kernel_forbids_the_hidden_file_to_the_process_and_its_children(tmp_path):
    nodes, work = tmp_path / "dev", tmp_path / "work"
    nodes.mkdir()
    work.mkdir()
    hidden, other = nodes / "nvidia0", nodes / "nvidia1"
    hidden.write_text("off limits")
    other.write_text("ours")
    out = restricted(hidden, other, work / "new.txt")
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [
        "nvidia0 PermissionError",
        "nvidia1 opens",
        "new file: made after the restriction",
        "child: PermissionError",
    ]
    assert hidden.read_text() == "off limits"  # this process is not restricted
    # the limit that the module's text names: a file made later directly in a folder above a hidden path has no rule
    out = restricted(hidden, other, nodes / "later.txt")
    assert out.returncode != 0 and "PermissionError" in out.stderr


@needs_landlock
def test_the_command_does_not_start_when_the_restriction_cannot_be_proved(tmp_path):
    hidden = tmp_path / "nvidia0"
    hidden.write_text("off limits")
    code = (
        "import sys\n"
        "from examples.SimplerEnv.eval_files.planner.experiments import off_limits_gpus as olg\n"
        "olg.restrict([sys.argv[1]], [sys.argv[2]])\n"
        "print('started')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(hidden), str(tmp_path / "no_such_node")],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0 and "started" not in out.stdout
    assert "cannot be opened" in out.stderr and "was not started" in out.stderr


def test_without_the_setting_or_without_landlock_nothing_starts(monkeypatch):
    monkeypatch.delenv(olg.SETTING, raising=False)
    with pytest.raises(SystemExit, match=olg.SETTING):
        olg.main(["true"])
    monkeypatch.setattr(olg, "landlock_version", lambda: 0)
    with pytest.raises(SystemExit, match="no Landlock"):
        olg.restrict(["/dev/nvidia0"])
    with pytest.raises(SystemExit) as stop:
        olg.main([])
    assert "OFF_LIMITS_GPUS=1 python off_limits_gpus.py COMMAND" in str(stop.value)


FAKE_PYTHON = """#!/bin/bash
# Stands in for every Python of the runners: it writes its arguments to $CALLS. As the launcher it fails when
# FAIL_GUARD=1; otherwise it does what the launcher does after the restriction: it runs the command.
echo "$*" >> "$CALLS"
case $1 in
  */off_limits_gpus.py) [ "${FAIL_GUARD:-0}" = 1 ] && exit 1; shift; exec "$@" ;;
esac
"""
FAKE_NVIDIA = """#!/bin/bash
# Stands in for nvidia-smi and nvidia-cuda-mps-control: it writes its name and arguments to $CALLS (but not the
# count of the GPUs, which examples/eval_env.sh asks for on a machine with nvidia-smi).
[ "$*" = --list-gpus ] || echo "$(basename "$0") $*" >> "$CALLS"
if [ "$(basename "$0")" = nvidia-smi ]; then echo GPU-made-up; fi
"""


def source_common(tmp_path, body, **settings):
    """Runs `body` in bash after `source common.sh` with a made-up environment; (the result, the calls' lines)."""
    fake = tmp_path / "python"
    fake.write_text(FAKE_PYTHON, newline="\n")
    fake.chmod(0o755)
    tools = tmp_path / "bin"
    tools.mkdir(exist_ok=True)
    for name in ("nvidia-smi", "nvidia-cuda-mps-control"):
        (tools / name).write_text(FAKE_NVIDIA, newline="\n")
        (tools / name).chmod(0o755)
    calls = tmp_path / "calls.txt"
    env = {
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "RUNS_DIR": str(tmp_path / "runs"),
        "SIMPLERENV_PATH": str(tmp_path),
        "CALLS": str(calls),
        **{name: str(fake) for name in ("STARVLA_PYTHON", "SAM3_PYTHON", "SAM31_PYTHON", "SIMPLER_PYTHON")},
        **settings,
    }
    script = f"source '{EXPERIMENTS}/common.sh' || exit 9\n{body}\n"
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, cwd=tmp_path)
    return out, calls.read_text().splitlines() if calls.exists() else []


SIM_AND_SERVER = """
sim 0 PutSpoonOnTableClothInScene-v0 0 1 "$RUNS_DIR/run/task" --controller frames || exit 3
SERVER_ENV=(CUDA_VISIBLE_DEVICES=0)
port_in_use() { return 1; }
start_server 10320 "$RUNS_DIR/run/server.log" "$SAM3_PYTHON" a_server.py --port 10320 || exit 4
wait "${SERVER_PIDS[@]}"
"""


@needs_bash
def test_with_the_setting_each_simulator_and_server_starts_through_the_launcher(tmp_path):
    out, calls = source_common(tmp_path, SIM_AND_SERVER, OFF_LIMITS_GPUS="1")
    assert out.returncode == 0, out.stderr
    launcher, python = str(EXPERIMENTS / "off_limits_gpus.py"), str(tmp_path / "python")
    assert calls[0] == f"{launcher} true"  # the restriction is tried before anything starts
    assert calls[1].startswith(f"{launcher} {python} examples/SimplerEnv/eval_files/start_simpler_env_visual_prompt.py ")
    assert calls[2].startswith("examples/SimplerEnv/eval_files/start_simpler_env_visual_prompt.py ")
    assert "--controller frames" in calls[2]
    assert calls[3:] == [f"{launcher} {python} a_server.py --port 10320", "a_server.py --port 10320"]


@needs_bash
def test_without_the_setting_the_runners_start_everything_as_before(tmp_path):
    out, calls = source_common(tmp_path, SIM_AND_SERVER)
    assert out.returncode == 0, out.stderr
    assert len(calls) == 2 and not any("off_limits_gpus" in c for c in calls)
    assert calls[0].startswith("examples/SimplerEnv/eval_files/start_simpler_env_visual_prompt.py ")
    assert calls[1] == "a_server.py --port 10320"


MPS_DAEMON = """
start_mps 0 "$RUNS_DIR/run/mps_log" || exit 5
rm -rf "$MPS_PIPE"
"""


@needs_bash
def test_the_mps_daemon_starts_through_the_launcher_too(tmp_path):
    launcher = EXPERIMENTS / "off_limits_gpus.py"
    smi, daemon = "nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0", "nvidia-cuda-mps-control -d"
    out, calls = source_common(tmp_path, MPS_DAEMON, OFF_LIMITS_GPUS="1")
    assert out.returncode == 0, out.stderr
    assert calls == [f"{launcher} true", smi, f"{launcher} {daemon}", daemon]
    (tmp_path / "calls.txt").unlink()
    out, calls = source_common(tmp_path, MPS_DAEMON)
    assert out.returncode == 0, out.stderr
    assert calls == [smi, daemon]


@needs_bash
def test_a_restriction_that_fails_stops_the_runner_before_anything_starts(tmp_path):
    out, calls = source_common(tmp_path, SIM_AND_SERVER, OFF_LIMITS_GPUS="1", FAIL_GUARD="1")
    assert out.returncode == 1
    assert "the restriction failed; nothing was started" in out.stderr
    assert calls == [f"{EXPERIMENTS / 'off_limits_gpus.py'} true"]


def test_the_readme_names_the_setting_and_the_launcher():
    readme = (EXPERIMENTS / "README.md").read_text(encoding="utf-8")
    assert "## Off-limits GPUs" in readme
    assert "`OFF_LIMITS_GPUS`" in readme and "off_limits_gpus.py" in readme
