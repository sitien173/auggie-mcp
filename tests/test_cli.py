"""Tests for the three supported auggie CLI commands."""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from auggie_mcp import cli, config, embed, indexer


class CliTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        self.cfg = config.Config()

    def test_parser_exposes_only_index_status_stats(self):
        parser = cli.build_parser()
        subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        self.assertEqual(set(subparsers.choices), {"index", "status", "stats"})

    def test_index_emits_json_result_and_accepts_json_flag(self):
        expected = {"files_scanned": 2, "files_changed": 1, "files_deleted": 0,
                    "chunks_embedded": 3, "chunks_reused": 0, "duration_ms": 4}
        out = io.StringIO()
        with contextlib.redirect_stdout(out), patch.object(indexer, "index_update", return_value=expected):
            code = cli.main(["index", "--json", str(self.repo)], config=self.cfg)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), expected)

    def test_status_returns_indexer_result(self):
        expected = {"exists": False, "next_step": "index_update"}
        out = io.StringIO()
        with contextlib.redirect_stdout(out), patch.object(indexer, "index_status", return_value=expected):
            code = cli.main(["status", str(self.repo)], config=self.cfg)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), expected)

    def test_stats_uses_loaded_telemetry_directory_and_retention(self):
        telemetry_dir = self.repo / "configured-telemetry"
        telemetry_dir.mkdir()
        (telemetry_dir / "events-2020-01-01.jsonl").write_text(
            '{"v":1,"ts":"2020-01-01T00:00:00.000Z","tool":"old","source":"mcp","outcome":"ok"}\n'
        )
        (telemetry_dir / "events-2099-01-01.jsonl").write_text(
            '{"v":1,"ts":"2099-01-01T00:00:00.000Z","tool":"configured","source":"mcp","outcome":"ok"}\n'
        )
        cfg_file = self.repo / "config.toml"
        cfg_file.write_text(
            f'[telemetry]\nenabled = true\ndir = "{telemetry_dir}"\nretention_days = 1\n'
        )
        cfg = config.load(cfg_file, environ={})
        stdout = io.StringIO()
        with (
            patch.dict(os.environ, {"AUGGIE_TELEMETRY_DIR": str(self.repo / "environment-telemetry")}),
            contextlib.redirect_stdout(stdout),
        ):
            code = cli.main(["stats", "--json"], config=cfg)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["by_tool"], {"configured": 1})
        self.assertFalse((telemetry_dir / "events-2020-01-01.jsonl").exists())
        self.assertFalse((self.repo / "environment-telemetry").exists())

    def test_index_json_emits_structured_embed_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), patch.object(
            indexer, "index_update", side_effect=embed.EmbedError("quota exceeded", status=429)
        ):
            code = cli.main(["index", "--json", str(self.repo)], config=self.cfg)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stdout.getvalue()), {
            "error": {"code": "EMBED_FAILED", "message": "quota exceeded", "http_status": 429}
        })
        self.assertEqual(stderr.getvalue(), "")

    def test_real_cli_index_missing_directory_is_invalid_path(self):
        home = self.repo / "home"
        home.mkdir()
        env = {key: value for key, value in os.environ.items() if not key.startswith("AUGGIE_")}
        env["HOME"] = str(home)
        missing = self.repo / "missing-repository"
        proc = subprocess.run(
            [sys.executable, "-m", "auggie_mcp.cli", "index", "--json", str(missing)],
            check=False, env=env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), {
            "error": {"code": "INVALID_PATH", "message": f"repository directory does not exist: {missing}", "http_status": None}
        })


class CliStatsTest(unittest.TestCase):
    NOW = datetime(2026, 11, 12, 12, tzinfo=UTC)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name) / "telemetry"
        self.directory.mkdir()
        self.cfg = config.Config(telemetry=config.TelemetryConfig(
            enabled=True, dir=self.directory, retention_days=30
        ))

    @staticmethod
    def event(ts, tool, source, outcome, runner, total, processing=None):
        return {
            "v": 1, "ts": ts, "tool": tool, "source": source,
            "outcome": outcome, "durations_ms": {"runner": runner, "total": total},
            "processing": processing,
        }

    def write_fixture(self):
        recent = [
            self.event("2026-11-12T11:30:00.000Z", "codebase_retrieval", "cli", "ok", 100, 120,
                       {"chunks_total": 4, "chunks_retained": 2, "estimated_tokens": 100}),
            self.event("2026-11-12T11:45:00.000Z", "semantic_search", "mcp", "no_match", 10, 20),
            self.event("2026-11-12T11:50:00.000Z", "codebase_retrieval", "mcp", "error", 300, 310,
                       {"chunks_total": 0, "chunks_retained": 0, "estimated_tokens": 50}),
        ]
        lines = [json.dumps(event) for event in recent]
        lines += ["not json", "[1, 2]", json.dumps({"v": 2, "ts": "2026-11-12T11:00:00.000Z"}),
                  json.dumps({"v": 1, "ts": "yesterday-ish", "tool": "x"})]
        (self.directory / "events-2026-11-12.jsonl").write_text("\n".join(lines) + "\n")
        old = self.event("2026-11-10T09:00:00.000Z", "grep_search", "cli", "ok", 50, 60,
                         {"chunks_total": 2, "chunks_retained": 2, "estimated_tokens": 25})
        (self.directory / "events-2026-11-10.jsonl").write_text(json.dumps(old) + "\n")

    def stats(self, *extra, directory=None):
        cfg = config.Config(telemetry=config.TelemetryConfig(
            enabled=True, dir=directory or self.directory, retention_days=30
        ))
        out, err = io.StringIO(), io.StringIO()
        with patch.object(cli, "_utc_now", lambda: self.NOW), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["stats", *extra], config=cfg)
        return code, out.getvalue(), err.getvalue()

    def test_json_window_and_percentiles(self):
        self.write_fixture()
        code, out, err = self.stats("--json")
        self.assertEqual((code, err), (0, ""))
        data = json.loads(out)
        self.assertEqual((data["events"], data["skipped"]), (4, 4))
        self.assertEqual(data["by_tool"], {"codebase_retrieval": 2, "semantic_search": 1, "grep_search": 1})
        self.assertEqual(data["durations_ms"], {
            "runner": {"p50": 50, "p95": 300}, "total": {"p50": 60, "p95": 310}
        })
        self.assertAlmostEqual(data["retained_chunk_ratio_mean"], 0.75)
        self.assertEqual(data["estimated_tokens_total"], 175)

    def test_hour_window_excludes_old_events(self):
        self.write_fixture()
        _, out, _ = self.stats("--since", "1h", "--json")
        data = json.loads(out)
        self.assertEqual(data["events"], 3)
        self.assertEqual(data["durations_ms"]["runner"], {"p50": 100, "p95": 300})

    def test_empty_and_missing_directories_report_zero(self):
        self.assertEqual(json.loads(self.stats("--json")[1])["events"], 0)
        code, out, err = self.stats("--json", directory=self.directory / "missing")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads(out)["events"], 0)

    def test_invalid_and_overflowing_windows_return_json_error(self):
        for value in ("7", "7w", "0d", "1000000000d"):
            with self.subTest(value=value):
                code, out, err = self.stats("--since", value)
                self.assertEqual((code, out), (2, ""))
                self.assertIn("error", json.loads(err))
                self.assertNotIn("Traceback", err)

    def test_malformed_field_types_are_skipped(self):
        good = self.event("2026-11-12T11:30:00Z", "t", "cli", "ok", 1, 2)
        bad = [{**good, "tool": []}, {**good, "source": {}}, {**good, "outcome": [1]},
               {**good, "tool": None}, {key: value for key, value in good.items() if key != "source"}]
        (self.directory / "events-2026-11-12.jsonl").write_text(
            "\n".join(json.dumps(event) for event in bad + [good]) + "\n"
        )
        _, out, _ = self.stats("--json")
        self.assertEqual((json.loads(out)["events"], json.loads(out)["skipped"]), (1, 5))

    def test_human_output_and_no_event_for_stats(self):
        self.write_fixture()
        before = sorted((path.name, path.read_text()) for path in self.directory.glob("events-*.jsonl"))
        code, out, err = self.stats()
        after = sorted((path.name, path.read_text()) for path in self.directory.glob("events-*.jsonl"))
        self.assertEqual((code, err), (0, ""))
        self.assertIn("codebase_retrieval", out)
        self.assertFalse(out.lstrip().startswith("{"))
        self.assertEqual(before, after)
