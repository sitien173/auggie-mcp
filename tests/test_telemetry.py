import json
import multiprocessing
import os
import stat
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from auggie_mcp import telemetry


class ConfigTest(unittest.TestCase):
    def test_dir_precedence_explicit(self):
        env = {"AUGGIE_TELEMETRY_DIR": "/a/b", "XDG_STATE_HOME": "/x"}
        self.assertEqual(telemetry.telemetry_dir(env), Path("/a/b"))

    def test_dir_precedence_xdg(self):
        env = {"XDG_STATE_HOME": "/x"}
        self.assertEqual(
            telemetry.telemetry_dir(env), Path("/x/auggie-mcp/telemetry")
        )

    def test_dir_default_home(self):
        self.assertEqual(
            telemetry.telemetry_dir({}),
            Path.home() / ".local" / "state" / "auggie-mcp" / "telemetry",
        )

    def test_is_enabled(self):
        self.assertTrue(telemetry.is_enabled({}))
        self.assertTrue(telemetry.is_enabled({"AUGGIE_TELEMETRY": "1"}))
        self.assertFalse(telemetry.is_enabled({"AUGGIE_TELEMETRY": "0"}))

    def test_retention_days(self):
        key = "AUGGIE_TELEMETRY_RETENTION_DAYS"
        self.assertEqual(telemetry.retention_days({}), 30)
        self.assertEqual(telemetry.retention_days({key: "7"}), 7)
        for bad in ("abc", "0", "-3", ""):
            self.assertEqual(telemetry.retention_days({key: bad}), 30)


NOW = datetime(2026, 11, 12, 15, 4, 5, 123000, tzinfo=UTC)


def _worker(directory, worker_id, count):
    env = {"AUGGIE_TELEMETRY_DIR": directory}
    payload = "x" * 8192
    for i in range(count):
        telemetry.append_event(
            {"worker": worker_id, "i": i, "payload": payload},
            environ=env,
            now=NOW,
        )


class AppendEventTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.dir = self.root / "state" / "telemetry"
        self.env = {"AUGGIE_TELEMETRY_DIR": str(self.dir)}

    def day_file(self, now=NOW):
        return self.dir / f"events-{now.strftime('%Y-%m-%d')}.jsonl"

    def test_kill_switch_writes_nothing(self):
        env = {**self.env, "AUGGIE_TELEMETRY": "0"}
        self.assertFalse(telemetry.append_event({"a": 1}, environ=env, now=NOW))
        self.assertFalse(self.dir.exists())

    def test_kill_switch_prunes_nothing(self):
        self.dir.mkdir(parents=True)
        old = self.dir / "events-2020-01-01.jsonl"
        old.write_text("{}\n")
        env = {**self.env, "AUGGIE_TELEMETRY": "0"}
        telemetry.prune_old_files(environ=env, now=NOW)
        self.assertTrue(old.exists())

    def test_modes(self):
        self.assertTrue(telemetry.append_event({"a": 1}, environ=self.env, now=NOW))
        self.assertEqual(stat.S_IMODE(self.dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.day_file().stat().st_mode), 0o600)

    def test_tightens_preexisting_file(self):
        self.dir.mkdir(parents=True)
        path = self.day_file()
        path.write_text("")
        os.chmod(path, 0o644)
        self.assertTrue(telemetry.append_event({"a": 1}, environ=self.env, now=NOW))
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_one_json_line_per_event(self):
        telemetry.append_event({"a": 1, "u": "é"}, environ=self.env, now=NOW)
        telemetry.append_event({"a": 2}, environ=self.env, now=NOW)
        text = self.day_file().read_text(encoding="utf-8")
        self.assertEqual(text, '{"a":1,"u":"é"}\n{"a":2}\n')

    def test_uses_utc_date_of_now(self):
        late = datetime(2026, 11, 12, 23, 59, tzinfo=UTC)
        telemetry.append_event({"a": 1}, environ=self.env, now=late)
        self.assertTrue((self.dir / "events-2026-11-12.jsonl").exists())

    def test_concurrent_processes_do_not_interleave(self):
        procs = [
            multiprocessing.Process(target=_worker, args=(str(self.dir), n, 200))
            for n in range(4)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
            self.assertEqual(p.exitcode, 0)
        lines = self.day_file().read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 800)
        for line in lines:
            self.assertIn("worker", json.loads(line))

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores modes")
    def test_unwritable_directory_returns_false(self):
        parent = self.dir.parent
        parent.mkdir(parents=True)
        os.chmod(parent, 0o500)
        self.addCleanup(os.chmod, parent, 0o700)
        self.assertFalse(telemetry.append_event({"a": 1}, environ=self.env, now=NOW))

    def test_short_write_returns_false(self):
        from unittest import mock

        with mock.patch("auggie_mcp.telemetry.os.write", return_value=1) as w:
            self.assertFalse(
                telemetry.append_event({"a": 1}, environ=self.env, now=NOW)
            )
        self.assertEqual(w.call_count, 1)

    def test_unserializable_event_returns_false(self):
        self.assertFalse(
            telemetry.append_event({"a": object()}, environ=self.env, now=NOW)
        )


class PruneTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "t"
        self.env = {"AUGGIE_TELEMETRY_DIR": str(self.dir)}

    def make(self, name):
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / name
        path.write_text("{}\n")
        return path

    def name_for(self, days):
        return f"events-{(NOW - timedelta(days=days)).strftime('%Y-%m-%d')}.jsonl"

    def test_default_retention(self):
        old = self.make(self.name_for(31))
        keep = self.make(self.name_for(29))
        other = self.make("notes.txt")
        bad = self.make("events-not-a-date.jsonl")
        telemetry.prune_old_files(environ=self.env, now=NOW)
        self.assertFalse(old.exists())
        self.assertTrue(keep.exists())
        self.assertTrue(other.exists())
        self.assertTrue(bad.exists())

    def test_custom_retention(self):
        env = {**self.env, "AUGGIE_TELEMETRY_RETENTION_DAYS": "5"}
        old = self.make(self.name_for(6))
        keep = self.make(self.name_for(4))
        telemetry.prune_old_files(environ=env, now=NOW)
        self.assertFalse(old.exists())
        self.assertTrue(keep.exists())

    def test_missing_directory_is_fine(self):
        telemetry.prune_old_files(environ=self.env, now=NOW)


class CallRecordTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "t"
        self.env = {"AUGGIE_TELEMETRY_DIR": str(self.dir)}

    def read_events(self):
        files = list(self.dir.glob("events-*.jsonl"))
        self.assertEqual(len(files), 1)
        return [json.loads(line) for line in files[0].read_text().splitlines()]

    def test_finish_writes_schema_v1_event(self):
        rec = telemetry.CallRecord(
            "mcp", "semantic_search", {"pattern": "x"}, environ=self.env
        )
        self.assertTrue(rec.finish("ok", now=NOW))
        (event,) = self.read_events()
        self.assertEqual(
            set(event),
            {
                "v", "ts", "event_id", "process_id", "source", "tool",
                "app_version", "client", "input", "outcome", "error",
                "durations_ms", "runner", "processing",
            },
        )
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["ts"], "2026-11-12T15:04:05.123Z")
        self.assertEqual(event["source"], "mcp")
        self.assertEqual(event["tool"], "semantic_search")
        self.assertEqual(event["input"], {"pattern": "x"})
        self.assertEqual(event["outcome"], "ok")
        self.assertIsNone(event["client"])
        self.assertIsNone(event["error"])
        self.assertIsNone(event["runner"])
        self.assertIsNone(event["processing"])
        self.assertIn("total", event["durations_ms"])
        self.assertEqual(len(event["event_id"]), 32)

    def test_ids(self):
        a = telemetry.CallRecord("cli", "t", {}, client="c", environ=self.env)
        b = telemetry.CallRecord("cli", "t", {}, environ=self.env)
        self.assertNotEqual(a.event_id, b.event_id)
        self.assertEqual(a.process_id, b.process_id)
        self.assertEqual(a.process_id, telemetry.process_id)
        a.finish("ok", now=NOW)
        self.assertEqual(self.read_events()[0]["client"], "c")

    def test_app_version_falls_back(self):
        from unittest import mock

        with mock.patch(
            "auggie_mcp.telemetry.importlib.metadata.version",
            side_effect=RuntimeError("boom"),
        ):
            rec = telemetry.CallRecord("cli", "t", {}, environ=self.env)
        self.assertEqual(rec.app_version, "unknown")

    def test_stage_records_duration_and_reraises(self):
        rec = telemetry.CallRecord("cli", "t", {}, environ=self.env)
        with rec.stage("a"):
            pass
        with self.assertRaises(ValueError), rec.stage("b"):
            raise ValueError("bad")
        self.assertIn("a", rec.durations_ms)
        self.assertIn("b", rec.durations_ms)
        self.assertGreaterEqual(rec.durations_ms["b"], 0)

    def test_stage_bookkeeping_failure_does_not_raise(self):
        class Broken:
            def get(self, *a):
                raise RuntimeError("get")

            def __setitem__(self, *a):
                raise RuntimeError("set")

        rec = telemetry.CallRecord("cli", "t", {}, environ=self.env)
        rec.durations_ms = Broken()
        with rec.stage("a"):
            pass

    def test_stage_bookkeeping_failure_keeps_body_exception(self):
        class Broken:
            def get(self, *a):
                raise RuntimeError("get")

            def __setitem__(self, *a):
                raise RuntimeError("set")

        rec = telemetry.CallRecord("cli", "t", {}, environ=self.env)
        rec.durations_ms = Broken()
        with self.assertRaises(ValueError) as ctx, rec.stage("a"):
            raise ValueError("x")
        self.assertEqual(str(ctx.exception), "x")

    def test_stage_start_failure_still_runs_body(self):
        from unittest import mock

        rec = telemetry.CallRecord("cli", "t", {}, environ=self.env)
        ran = []
        with mock.patch(
            "auggie_mcp.telemetry.time.perf_counter",
            side_effect=RuntimeError("clock"),
        ), rec.stage("a"):
            ran.append(True)
        self.assertEqual(ran, [True])
        self.assertNotIn("a", rec.durations_ms)

    def test_stage_start_failure_keeps_body_exception(self):
        from unittest import mock

        rec = telemetry.CallRecord("cli", "t", {}, environ=self.env)
        with mock.patch(
            "auggie_mcp.telemetry.time.perf_counter",
            side_effect=RuntimeError("clock"),
        ), self.assertRaises(ValueError) as ctx, rec.stage("a"):
            raise ValueError("x")
        self.assertEqual(str(ctx.exception), "x")

    def test_runner_and_processing_blocks(self):
        rec = telemetry.CallRecord("cli", "t", {}, environ=self.env)
        rec.set_runner(exit_code=0, stdout_bytes=10, stderr_bytes=2)
        rec.set_processing(
            {
                "status": "chunked",
                "retention": "archive",
                "estimated_tokens": 99,
                "recovery_path": "/p",
                "chunks": [{"retained": True}, {"retained": False}, {"retained": True}],
            },
            "héllo",
        )
        rec.finish("ok", now=NOW)
        (event,) = self.read_events()
        self.assertEqual(
            event["runner"], {"exit_code": 0, "stdout_bytes": 10, "stderr_bytes": 2}
        )
        self.assertEqual(
            event["processing"],
            {
                "status": "chunked",
                "retention": "archive",
                "estimated_tokens": 99,
                "chunks_total": 3,
                "chunks_retained": 2,
                "output_bytes": 6,
            },
        )

    def test_processing_without_output(self):
        rec = telemetry.CallRecord("cli", "t", {}, environ=self.env)
        rec.set_processing(
            {"status": "s", "retention": "r", "estimated_tokens": 1, "chunks": []}
        )
        self.assertIsNone(rec.processing["output_bytes"])
        self.assertEqual(rec.processing["chunks_total"], 0)

    def test_error_serialised(self):
        rec = telemetry.CallRecord("cli", "t", {}, environ=self.env)
        rec.finish("error", KeyError("k"), now=NOW)
        (event,) = self.read_events()
        self.assertEqual(event["error"], {"type": "KeyError", "message": "'k'"})

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores modes")
    def test_finish_unwritable_does_not_raise(self):
        parent = self.dir.parent
        os.chmod(parent, 0o500)
        self.addCleanup(os.chmod, parent, 0o700)
        rec = telemetry.CallRecord("cli", "t", {}, environ=self.env)
        self.assertFalse(rec.finish("ok", now=NOW))

    def test_finish_never_raises_on_bad_input(self):
        rec = telemetry.CallRecord("cli", "t", {"x": object()}, environ=self.env)
        self.assertFalse(rec.finish("ok", now=NOW))


if __name__ == "__main__":
    unittest.main()
