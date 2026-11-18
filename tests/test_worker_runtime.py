import asyncio
import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import worker_runtime as wr

ROOT = Path(__file__).resolve().parents[1]


class TestCgroupMemory(unittest.TestCase):
    def test_v2_splits_anon_and_file_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "memory.current").write_text(f"{3 * wr.MiB}\n")
            (root / "memory.max").write_text(f"{8 * wr.MiB}\n")
            (root / "memory.peak").write_text(f"{5 * wr.MiB}\n")
            (root / "memory.stat").write_text(
                f"anon {2 * wr.MiB}\nfile {wr.MiB}\nshmem 0\nkernel 123\n"
            )
            mem = wr.cgroup_memory(root)
        self.assertEqual(mem["cgroup"], "v2")
        self.assertEqual(
            (mem["current_mib"], mem["max_mib"], mem["peak_mib"]), (3.0, 8.0, 5.0)
        )
        self.assertEqual((mem["anon_mib"], mem["file_mib"], mem["shmem_mib"]), (2.0, 1.0, 0.0))

    def test_v2_unlimited_and_v1_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "memory.current").write_text("1048576")
            (root / "memory.max").write_text("max")
            self.assertIsNone(wr.cgroup_memory(root)["max_mib"])
        with tempfile.TemporaryDirectory() as d:
            v1 = Path(d) / "memory"
            v1.mkdir()
            (v1 / "memory.usage_in_bytes").write_text(str(4 * wr.MiB))
            (v1 / "memory.limit_in_bytes").write_text(str(2**63 - 4096))  # "no limit"
            (v1 / "memory.stat").write_text(f"rss {3 * wr.MiB}\ncache {wr.MiB}\n")
            mem = wr.cgroup_memory(Path(d))
        self.assertEqual(mem["cgroup"], "v1")
        self.assertIsNone(mem["max_mib"])
        self.assertEqual((mem["current_mib"], mem["anon_mib"], mem["file_mib"]), (4.0, 3.0, 1.0))

    def test_no_cgroup_and_process_rss(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(wr.cgroup_memory(Path(d)), {"cgroup": None})
            proc = Path(d) / "42"
            proc.mkdir()
            (proc / "status").write_text("Name:\tpython\nVmRSS:\t  2048 kB\n")
            self.assertEqual(wr.process_rss_mib(42, Path(d)), 2.0)
            self.assertIsNone(wr.process_rss_mib(43, Path(d)))
        self.assertIsNone(wr.process_rss_mib(None))

    def test_log_line_is_one_greppable_json_line(self):
        line = wr.log_line({"job_index": 3, "phase": "after"})
        self.assertTrue(line.startswith("worker-comfyui: memstats {"))
        self.assertNotIn("\n", line)


class TestRestartPolicy(unittest.TestCase):
    def test_every_n_jobs_requests_one_restart(self):
        policy = wr.RestartPolicy(every=3)
        seen = []
        for _ in range(7):
            seen.append(policy.take_pending())
            policy.job_finished()
        # restart is taken before the 4th and the 7th job
        self.assertEqual(seen, [False, False, False, True, False, False, True])

    def test_disabled_by_default_and_on_bad_values(self):
        for value in (None, "", "0", "-2", "abc"):
            env = {} if value is None else {"COMFY_RESTART_EVERY_N_JOBS": value}
            with self.subTest(value=value), patch.dict(os.environ, env, clear=False):
                if value is None:
                    os.environ.pop("COMFY_RESTART_EVERY_N_JOBS", None)
                policy = wr.RestartPolicy()
                for _ in range(10):
                    policy.job_finished()
                self.assertFalse(policy.take_pending())


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestRestartComfyUI(unittest.TestCase):
    """A stand-in "ComfyUI" (tiny HTTP server) is stopped and started again
    from the recorded command line."""

    def test_restart_replaces_the_process_with_the_same_command(self):
        port = _free_port()
        server = (
            "import http.server,sys;"
            "http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])),"
            "http.server.SimpleHTTPRequestHandler).serve_forever()"
        )
        with tempfile.TemporaryDirectory() as d:
            pid_file, args_file = Path(d) / "comfy.pid", Path(d) / "comfy.args"
            args = [sys.executable, "-c", server, str(port)]
            args_file.write_bytes(b"".join(a.encode() + b"\0" for a in args))
            old = subprocess.Popen(args, cwd=d)
            pid_file.write_text(str(old.pid))
            try:
                with patch.object(wr, "COMFY_PID_FILE", pid_file), patch.object(
                    wr, "COMFY_ARGS_FILE", args_file
                ):
                    self.assertTrue(wr.wait_until_ready(f"http://127.0.0.1:{port}/", 20))
                    self.assertEqual(wr.read_comfy_args(), args)
                    new_pid = wr.restart_comfyui(f"http://127.0.0.1:{port}/", 30)
                self.assertNotEqual(new_pid, old.pid)
                self.assertEqual(int(pid_file.read_text()), new_pid)
                self.assertIsNotNone(old.poll())  # the old process is gone
            finally:
                for pid in (old.pid, int(pid_file.read_text())):
                    try:
                        os.kill(pid, 9)
                    except OSError:
                        pass

    def test_restart_without_recorded_command_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d, patch.object(
            wr, "COMFY_ARGS_FILE", Path(d) / "missing"
        ):
            with self.assertRaisesRegex(RuntimeError, "no ComfyUI command line"):
                wr.restart_comfyui("http://127.0.0.1:9/", 1)


class TestStartScriptArgs(unittest.TestCase):
    def run_filter(self, cli_args_text, *args):
        source = (ROOT / "src/start.sh").read_text()
        fn = "filter_supported_comfy_args() {" + source.split(
            "filter_supported_comfy_args() {", 1
        )[1].split("\n}\n", 1)[0] + "\n}\n"
        with tempfile.TemporaryDirectory() as d:
            cli = Path(d) / "cli_args.py"
            cli.write_text(cli_args_text)
            result = subprocess.run(
                ["/bin/bash", "-c", fn + 'filter_supported_comfy_args "$@"', "x", str(cli), *args],
                capture_output=True, text=True, check=True,
            )
        return result.stdout.splitlines(), result.stderr

    def test_default_args_are_cache_none_and_no_pinned_memory(self):
        source = (ROOT / "src/start.sh").read_text()
        self.assertIn(': "${COMFY_EXTRA_ARGS=--cache-none --disable-pinned-memory}"', source)

    def test_unknown_flags_and_their_values_are_dropped(self):
        cli = 'add_argument("--cache-none")\nadd_argument("--cache-ram")\n'
        kept, err = self.run_filter(cli, "--cache-ram", "4", "--disable-pinned-memory", "--cache-none")
        self.assertEqual(kept, ["--cache-ram", "4", "--cache-none"])
        self.assertIn("does not support --disable-pinned-memory", err)
        kept, err = self.run_filter(cli, "--disable-pinned-memory", "7")
        self.assertEqual(kept, [])  # nothing left → no empty argument

    def test_command_line_is_recorded_for_restarts(self):
        source = (ROOT / "src/start.sh").read_text()
        self.assertIn('''printf '%s\\0' "${comfy_args[@]}" > "${COMFY_ARGS_FILE:-/tmp/comfyui.args}"''', source)


class TestHandlerMemory(unittest.TestCase):
    def load_handler(self):
        runpod = types.ModuleType("runpod")
        runpod.serverless = types.SimpleNamespace(start=lambda config: None)
        aiohttp = types.ModuleType("aiohttp")
        aiohttp.ClientSession = type("ClientSession", (), {})
        boto3 = types.ModuleType("boto3")
        botocore = types.ModuleType("botocore")
        botocore_config = types.ModuleType("botocore.config")
        botocore_config.Config = type("Config", (), {})
        redis = types.ModuleType("redis")
        redis_asyncio = types.ModuleType("redis.asyncio")
        redis_asyncio.from_url = Mock(return_value=Mock(hset=AsyncMock(), expire=AsyncMock()))
        redis.asyncio = redis_asyncio
        modules = {
            "aiohttp": aiohttp, "boto3": boto3, "botocore": botocore,
            "botocore.config": botocore_config, "redis": redis,
            "redis.asyncio": redis_asyncio, "runpod": runpod,
        }
        spec = importlib.util.spec_from_file_location("handler_memory_test", ROOT / "handler.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch.dict(os.environ):
            os.environ.pop("REDIS_URL", None)
            spec.loader.exec_module(module)
        return module

    def test_each_job_reports_memory_and_restarts_follow_the_policy(self):
        h = self.load_handler()
        snaps = []

        def fake_snapshot(job_index, phase):
            snaps.append((job_index, phase))
            return {"job_index": job_index, "phase": phase}

        h.RESTART_POLICY = wr.RestartPolicy(every=2)
        h.handle_workflow_job = AsyncMock(
            return_value={"status": "success", "output": {}, "metadata": {"render_time_sec": 1}}
        )
        h.is_workflow_job = lambda _: True
        restart = Mock(return_value=999)
        with patch.object(h.worker_runtime, "memory_snapshot", fake_snapshot), patch.object(
            h.worker_runtime, "restart_comfyui", restart
        ), patch("builtins.print"):
            r1 = asyncio.run(h.handler({"id": "a", "input": {"workflow": {}}}))
            asyncio.run(h.handler({"id": "b", "input": {"workflow": {}}}))
            restart.assert_not_called()
            h.handle_workflow_job = AsyncMock(side_effect=RuntimeError("All failover attempts exhausted."))
            r3 = asyncio.run(h.handler({"id": "c", "input": {"workflow": {}}}))
        # the restart ran once, before the third job
        restart.assert_called_once()
        self.assertEqual(
            r1["metadata"]["memory"],
            {"before": {"job_index": 1, "phase": "before"}, "after": {"job_index": 1, "phase": "after"}},
        )
        self.assertEqual(r1["metadata"]["render_time_sec"], 1)
        self.assertEqual(r3["status"], "error")
        self.assertEqual(r3["memory"]["after"]["phase"], "after_error")
        self.assertIn((2, "before_restart"), snaps)
        self.assertIn((2, "after_restart"), snaps)

    def test_telemetry_failure_never_fails_a_job(self):
        h = self.load_handler()
        h.handle_workflow_job = AsyncMock(return_value={"status": "success", "metadata": {}})
        h.is_workflow_job = lambda _: True
        with patch.object(h.worker_runtime, "memory_snapshot", side_effect=OSError("boom")), patch(
            "builtins.print"
        ):
            r = asyncio.run(h.handler({"id": "a", "input": {"workflow": {}}}))
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["metadata"]["memory"]["before"]["error"], "boom")


if __name__ == "__main__":
    unittest.main()
