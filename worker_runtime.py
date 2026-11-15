"""Worker memory telemetry and ComfyUI restarts (stdlib only, unit-testable).

Workers die after a handful of renders: the container's RAM fills until the
ComfyUI process is OOM-killed, and every later job then fails with "All
failover attempts exhausted." This module gives the handler two tools:

* ``memory_snapshot`` — the container's cgroup memory (current / max / peak,
  split into anonymous vs. reclaimable file cache), the ComfyUI process RSS,
  and GPU memory, so each job logs whether memory keeps climbing.
* ``RestartPolicy`` + ``restart_comfyui`` — an opt-in guard that restarts the
  ComfyUI process every N jobs (``COMFY_RESTART_EVERY_N_JOBS``, default 0 =
  off), before the next job starts so finished jobs are never delayed.

``start.sh`` writes the exact ComfyUI command line to ``COMFY_ARGS_FILE`` so a
restart reuses the same flags.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

CGROUP_ROOT = Path(os.environ.get("CGROUP_ROOT", "/sys/fs/cgroup"))
COMFY_PID_FILE = Path(os.environ.get("COMFY_PID_FILE", "/tmp/comfyui.pid"))
COMFY_ARGS_FILE = Path(os.environ.get("COMFY_ARGS_FILE", "/tmp/comfyui.args"))
MiB = 1024 * 1024


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if raw == "max":  # cgroup v2: no limit
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_stat(path: Path, keys: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[0] in keys:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


def _mib(value: int | None) -> float | None:
    return None if value is None else round(value / MiB, 1)


def cgroup_memory(root: Path | None = None) -> dict:
    """Container memory in MiB. ``anon`` is what a leak grows; ``file`` is
    page cache (model files) the kernel can reclaim before an OOM kill."""
    root = root or CGROUP_ROOT
    if (root / "memory.current").exists():  # cgroup v2
        stat = _read_stat(root / "memory.stat", ("anon", "file", "shmem"))
        return {
            "cgroup": "v2",
            "current_mib": _mib(_read_int(root / "memory.current")),
            "max_mib": _mib(_read_int(root / "memory.max")),
            "peak_mib": _mib(_read_int(root / "memory.peak")),
            "anon_mib": _mib(stat.get("anon")),
            "file_mib": _mib(stat.get("file")),
            "shmem_mib": _mib(stat.get("shmem")),
        }
    v1 = root / "memory"
    if (v1 / "memory.usage_in_bytes").exists():  # cgroup v1
        limit = _read_int(v1 / "memory.limit_in_bytes")
        stat = _read_stat(v1 / "memory.stat", ("rss", "cache", "shmem"))
        return {
            "cgroup": "v1",
            "current_mib": _mib(_read_int(v1 / "memory.usage_in_bytes")),
            # v1 reports "no limit" as a huge number
            "max_mib": _mib(limit) if limit is not None and limit < 2**60 else None,
            "peak_mib": _mib(_read_int(v1 / "memory.max_usage_in_bytes")),
            "anon_mib": _mib(stat.get("rss")),
            "file_mib": _mib(stat.get("cache")),
            "shmem_mib": _mib(stat.get("shmem")),
        }
    return {"cgroup": None}


def comfy_pid(pid_file: Path | None = None) -> int | None:
    pid = _read_int(pid_file or COMFY_PID_FILE)
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def process_rss_mib(pid: int | None, proc: Path = Path("/proc")) -> float | None:
    if pid is None:
        return None
    try:
        for line in (proc / str(pid) / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)  # kB → MiB
    except (OSError, ValueError, IndexError):
        pass
    return None


def gpu_memory() -> dict:
    """GPU memory from nvidia-smi (MiB); empty when unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
        used, total = (float(v) for v in out[0].split(","))
        return {"gpu_used_mib": used, "gpu_total_mib": total}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def memory_snapshot(job_index: int, phase: str) -> dict:
    pid = comfy_pid()
    return {
        "phase": phase,
        "job_index": job_index,
        "worker_id": os.environ.get("RUNPOD_POD_ID", ""),
        "comfy_pid": pid,
        "comfy_rss_mib": process_rss_mib(pid),
        **cgroup_memory(),
        **gpu_memory(),
    }


def log_line(snapshot: dict) -> str:
    """One greppable line per snapshot for the RunPod log view."""
    return "worker-comfyui: memstats " + json.dumps(snapshot, sort_keys=True)


class RestartPolicy:
    """Count finished jobs; ask for a ComfyUI restart every ``every`` jobs.

    The restart runs before the *next* job so a finished job's response is
    never delayed. ``every`` <= 0 disables the policy.
    """

    def __init__(self, every: int | None = None) -> None:
        if every is None:
            try:
                every = int(os.environ.get("COMFY_RESTART_EVERY_N_JOBS", "0") or 0)
            except ValueError:
                every = 0
        self.every = max(0, every)
        self.jobs_done = 0
        self.pending = False

    def job_finished(self) -> None:
        self.jobs_done += 1
        if self.every and self.jobs_done % self.every == 0:
            self.pending = True

    def take_pending(self) -> bool:
        pending, self.pending = self.pending, False
        return pending


def read_comfy_args(args_file: Path | None = None) -> list[str]:
    """The ComfyUI command line start.sh used (NUL-separated)."""
    try:
        raw = (args_file or COMFY_ARGS_FILE).read_bytes()
    except OSError:
        return []
    return [a.decode() for a in raw.split(b"\0") if a]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # reap if it is our own child, so a zombie does not read as alive
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def stop_process(pid: int, grace_s: float = 30.0) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def wait_until_ready(url: str, timeout_s: float = 300.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def restart_comfyui(ready_url: str, timeout_s: float = 300.0) -> int:
    """Stop the running ComfyUI and start it again with the same arguments.

    Returns the new PID; raises RuntimeError when it can't come back up, so
    the caller fails the job loudly instead of rendering on a dead engine.
    """
    args = read_comfy_args()
    if not args:
        raise RuntimeError(f"no ComfyUI command line recorded at {COMFY_ARGS_FILE}")
    old = comfy_pid()
    if old is not None:
        stop_process(old)
    proc = subprocess.Popen(args)  # inherits stdout/stderr → worker log
    COMFY_PID_FILE.write_text(str(proc.pid))
    if not wait_until_ready(ready_url, timeout_s):
        raise RuntimeError("ComfyUI did not come back after restart")
    return proc.pid
