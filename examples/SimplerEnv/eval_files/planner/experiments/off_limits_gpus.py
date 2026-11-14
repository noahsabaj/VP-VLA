"""Runs a command that cannot open the GPUs that are off limits (README.md, "Off-limits GPUs").

    OFF_LIMITS_GPUS=1 python off_limits_gpus.py COMMAND [ARGUMENT...]
    OFF_LIMITS_GPUS=1 python off_limits_gpus.py --check

OFF_LIMITS_GPUS: nvidia-smi's indices of the GPUs, with commas ("1", "1,3").

Why: CUDA_VISIBLE_DEVICES keeps CUDA off a GPU, but not a renderer. A simulator's renderer is a Vulkan process, and
the NVIDIA Vulkan driver opens a GPU when the renderer starts, before any device is chosen (on a machine of ours:
always the GPU with the boot display, about 13 MiB of its memory for each simulator). So a simulator that renders on
its own GPU also holds a small graphics context on another one. nvidia-smi's list of compute processes does not show
it; `nvidia-smi pmon` does (type G).

How: the Linux kernel (Landlock, Linux 5.13 or later; no special rights) forbids this process, and every process it
starts, to open the device nodes of those GPUs. It only takes access away from our own processes: nothing changes
for other processes, other users or the machine. With a node that it cannot open, the driver does not list that GPU.
- Landlock allows by list. Opening files for reading or writing stays allowed for every path but the nodes: for
  every entry of / except /dev, and for every entry of /dev except the nodes. Nothing else is restricted (running
  programs, making and removing files and folders, sockets, memory). A file that is made later directly in / or in
  /dev cannot be opened.
- A GPU's node is /dev/nvidiaN with N its "Device Minor", which is not nvidia-smi's index (on a machine of ours they
  are swapped). So the nodes are looked up at each start: nvidia-smi gives each GPU's PCI address, and
  /proc/driver/nvidia/gpus/ADDRESS/information gives its minor.
- The command starts only when the restriction holds: each forbidden node was tried and could not be opened, and
  each other GPU's node could. Without Landlock, or when a GPU of the setting is not there, it does not start.

`--check` prints the kernel's Landlock version and the nodes of the setting, and starts nothing.

Standard library only; Linux only.
"""

import ctypes
import errno
import os
import posixpath
import re
import subprocess
import sys
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

SETTING = "OFF_LIMITS_GPUS"
# The Landlock system calls have the same numbers on every architecture.
SYS_LANDLOCK_CREATE_RULESET, SYS_LANDLOCK_ADD_RULE, SYS_LANDLOCK_RESTRICT_SELF = 444, 445, 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
WRITE_FILE, READ_FILE = 1 << 1, 1 << 2  # LANDLOCK_ACCESS_FS_WRITE_FILE, LANDLOCK_ACCESS_FS_READ_FILE
PR_SET_NO_NEW_PRIVS = 38
O_PATH = 0o10000000


def off_limits(setting: str) -> Tuple[str, ...]:
    """The GPU indices of the setting's text, in its order; a wrong text stops the run."""
    parts = [p.strip() for p in setting.split(",") if p.strip()]
    for p in parts:
        if not p.isdigit():
            raise SystemExit(f"{SETTING} must be nvidia-smi's GPU indices with commas, not {setting!r}")
    if len(set(parts)) != len(parts):
        raise SystemExit(f"{SETTING} names a GPU twice: {setting!r}")
    return tuple(parts)


def gpu_nodes(smi: str, information: Callable[[str], str],
              off: Sequence[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(the nodes of the GPUs in `off`, the nodes of the other GPUs).

    smi: the output of `nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader`. information(ADDRESS): the
    text of /proc/driver/nvidia/gpus/ADDRESS/information. Every GPU of `off` must be there, and one other GPU."""
    hidden, others, seen = [], [], set()
    for line in smi.splitlines():
        if not line.strip():
            continue
        fields = [s.strip() for s in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit():
            raise SystemExit(f"nvidia-smi's line is not INDEX, PCI ADDRESS: {line!r}")
        index, address = fields[0], fields[1][-12:].lower()  # nvidia-smi writes the PCI domain with 8 digits
        minor = re.search(r"Device Minor:\s*(\d+)", information(address))
        if minor is None:
            raise SystemExit(f"no device minor for GPU {index} at {address}")
        seen.add(index)
        (hidden if index in off else others).append(f"/dev/nvidia{minor.group(1)}")
    missing = [i for i in off if i not in seen]
    if missing:
        raise SystemExit(f"{SETTING} names GPU {', '.join(missing)}, which nvidia-smi does not list")
    if not others:
        raise SystemExit(f"{SETTING} leaves no GPU")
    return tuple(hidden), tuple(others)


def allowed(hidden: Iterable[str], listdir: Callable[[str], List[str]] = os.listdir,
            islink: Callable[[str], bool] = os.path.islink) -> List[str]:
    """The paths whose rules allow every path but `hidden` (absolute paths without links): each entry of / and of
    the folders above a hidden path, except those folders and the hidden paths themselves. A link gets no rule: its
    target has one."""
    hidden = {posixpath.normpath(p) for p in hidden}
    for p in hidden:
        if not posixpath.isabs(p):
            raise ValueError(f"not an absolute path: {p}")
    out: List[str] = []

    def walk(folder: str) -> None:
        for entry in sorted(listdir(folder)):
            path = posixpath.join(folder, entry)
            if path in hidden or islink(path):
                continue
            if any(h.startswith(path + "/") for h in hidden):
                walk(path)
            else:
                out.append(path)

    walk("/")
    return out


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def landlock_version() -> int:
    """The kernel's Landlock version; 0 without Landlock (or not on Linux)."""
    if not sys.platform.startswith("linux"):
        return 0
    v = _libc().syscall(SYS_LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0),
                        ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION))
    return max(int(v), 0)


def _stop(what: str) -> None:
    raise SystemExit(f"off_limits_gpus: {what}: {os.strerror(ctypes.get_errno())} (the command was not started)")


def restrict(hidden: Sequence[str], others: Sequence[str] = ()) -> None:
    """Forbids this process and the processes it starts to open the paths in `hidden` for reading or writing, and
    proves it: each hidden path cannot be opened, each path of `others` can. It cannot be undone."""
    if landlock_version() < 1:
        raise SystemExit("off_limits_gpus: this kernel has no Landlock (the command was not started)")
    libc = _libc()
    hidden = [os.path.realpath(p) for p in hidden]
    attr = _RulesetAttr(WRITE_FILE | READ_FILE)
    ruleset = libc.syscall(SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)),
                           ctypes.c_uint32(0))
    if ruleset < 0:
        _stop("the rule set")
    for path in allowed(hidden):
        try:
            fd = os.open(path, O_PATH | os.O_CLOEXEC)
        except OSError:
            continue  # an entry that is gone or cannot be reached stays forbidden
        try:
            rule = _PathBeneathAttr(WRITE_FILE | READ_FILE, fd)
            if libc.syscall(SYS_LANDLOCK_ADD_RULE, ctypes.c_int(ruleset), ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
                            ctypes.byref(rule), ctypes.c_uint32(0)) != 0 and \
                    ctypes.get_errno() not in (errno.EBADFD, errno.EINVAL):  # not a file or a folder: stays forbidden
                _stop(f"the rule for {path}")
        finally:
            os.close(fd)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        _stop("no_new_privs")
    if libc.syscall(SYS_LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset), ctypes.c_uint32(0)) != 0:
        _stop("the restriction")
    os.close(ruleset)
    for path in hidden:
        try:
            os.close(os.open(path, os.O_RDONLY))
        except PermissionError:
            continue
        except FileNotFoundError:
            continue
        raise SystemExit(f"off_limits_gpus: {path} can still be opened (the command was not started)")
    for path in others:
        try:
            os.close(os.open(path, os.O_RDWR))
        except OSError as e:
            raise SystemExit(f"off_limits_gpus: {path} cannot be opened: {e.strerror} "
                             "(the command was not started)") from None


def _information(address: str) -> str:
    path = f"/proc/driver/nvidia/gpus/{address}/information"
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise SystemExit(f"off_limits_gpus: {path}: {e.strerror} (the command was not started)") from None


def nodes(setting: Optional[str] = None) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(the nodes that the setting forbids, the other GPUs' nodes) on this machine."""
    off = off_limits(os.environ.get(SETTING, "") if setting is None else setting)
    if not off:
        raise SystemExit(f"off_limits_gpus: set {SETTING} to nvidia-smi's indices of the GPUs that are off limits")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"off_limits_gpus: nvidia-smi failed: {e} (the command was not started)") from None
    return gpu_nodes(smi, _information, off)


def main(argv: List[str]) -> None:
    if argv == ["--check"]:
        print("Landlock version", landlock_version())
        hidden, others = nodes()
        print("forbidden:", *hidden, "| stay open:", *others)
        return
    if not argv:
        raise SystemExit(__doc__)
    restrict(*nodes())
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main(sys.argv[1:])
