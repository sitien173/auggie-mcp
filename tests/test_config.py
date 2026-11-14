"""Tests for auggie_mcp.config."""

import dataclasses
import logging
import os
import tempfile
import unittest
from pathlib import Path

from auggie_mcp import config


class ConfigTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "config.toml"

    def write(self, text: str) -> Path:
        self.path.write_text(text, encoding="utf-8")
        return self.path


class PrecedenceTests(ConfigTestCase):
    def test_defaults_when_file_missing_and_env_empty(self):
        cfg = config.load(path=self.dir / "missing.toml", environ={})
        self.assertEqual(cfg.embedding.model, "voyage-code-3")
        self.assertEqual(cfg.embedding.dimensions, 1024)
        self.assertEqual(cfg.embedding.batch_size, 128)
        self.assertEqual(cfg.voyage.timeout_s, 120.0)
        self.assertEqual(cfg.voyage.retries, 5)
        self.assertEqual(cfg.chunking.lines, 30)
        self.assertEqual(cfg.chunking.overlap_lines, 15)
        self.assertEqual(cfg.index.timeout_s, 1800.0)
        self.assertTrue(cfg.telemetry.enabled)
        self.assertIsNone(cfg.telemetry.dir)
        self.assertEqual(cfg.telemetry.retention_days, 30)
        self.assertEqual(cfg.retrieval.max_concurrent, 2)
        self.assertIsNone(cfg.agent.driver)
        self.assertEqual(cfg.agent.args, ())
        self.assertEqual(cfg.agent.timeout_s, 1800.0)

    def test_toml_overrides_defaults(self):
        self.write(
            """
[embedding]
model = "voyage-3"
dimensions = 512
batch_size = 64

[voyage]
timeout_s = 30
retries = 3

[chunking]
lines = 50
overlap_lines = 10

[index]
timeout_s = 600.0

[telemetry]
enabled = false
dir = "/tmp/tele"
retention_days = 7

[retrieval]
max_concurrent = 4
"""
        )
        cfg = config.load(path=self.path, environ={})
        self.assertEqual(cfg.embedding.model, "voyage-3")
        self.assertEqual(cfg.embedding.dimensions, 512)
        self.assertEqual(cfg.embedding.batch_size, 64)
        self.assertEqual(cfg.voyage.timeout_s, 30.0)
        self.assertEqual(cfg.voyage.retries, 3)
        self.assertEqual(cfg.chunking.lines, 50)
        self.assertEqual(cfg.chunking.overlap_lines, 10)
        self.assertEqual(cfg.index.timeout_s, 600.0)
        self.assertFalse(cfg.telemetry.enabled)
        self.assertEqual(cfg.telemetry.dir, Path("/tmp/tele"))
        self.assertEqual(cfg.telemetry.retention_days, 7)
        self.assertEqual(cfg.retrieval.max_concurrent, 4)

    def test_env_overrides_toml(self):
        self.write(
            """
[embedding]
model = "voyage-3"
dimensions = 512

[retrieval]
max_concurrent = 4
"""
        )
        env = {
            "AUGGIE_MODEL": "voyage-3-lite",
            "AUGGIE_DIMENSIONS": "256",
            "AUGGIE_RETRIEVAL_MAX_CONCURRENT": "8",
            "AUGGIE_TELEMETRY": "0",
        }
        cfg = config.load(path=self.path, environ=env)
        self.assertEqual(cfg.embedding.model, "voyage-3-lite")
        self.assertEqual(cfg.embedding.dimensions, 256)
        self.assertEqual(cfg.retrieval.max_concurrent, 8)
        self.assertFalse(cfg.telemetry.enabled)

    def test_os_environ_unchanged_after_loading(self):
        self.write('[embedding]\nmodel = "voyage-3"\n')
        before = dict(os.environ)
        config.load(path=self.path)
        self.assertEqual(dict(os.environ), before)

    def test_config_is_frozen(self):
        cfg = config.load(path=self.dir / "missing.toml", environ={})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.retrieval = config.RetrievalConfig(max_concurrent=9)  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.retrieval.max_concurrent = 9  # type: ignore[misc]


class AgentKeysTests(ConfigTestCase):
    def test_agent_keys_load_from_toml_and_env(self):
        self.write(
            """
[agent]
driver = "codex"
model = "o3"
reasoning = "high"
args = ["--flag", "val"]
executable = "/usr/bin/codex"
timeout_s = 900
"""
        )
        cfg = config.load(
            path=self.path,
            environ={"AUGGIE_AGENT_MODEL": "gpt-5", "AUGGIE_AGENT_ARGS": "--one 'two three'"},
        )
        self.assertEqual(cfg.agent.driver, "codex")
        self.assertEqual(cfg.agent.model, "gpt-5")
        self.assertEqual(cfg.agent.reasoning, "high")
        self.assertEqual(cfg.agent.args, ("--one", "two three"))
        self.assertEqual(cfg.agent.executable, "/usr/bin/codex")
        self.assertEqual(cfg.agent.timeout_s, 900.0)


class RemovedAndValidationTests(ConfigTestCase):
    def test_removed_keys_warn_once_and_are_ignored(self):
        self.write(
            """
[search]
mode = "hybrid"

[confidence]
mode = "absolute"

[rerank]
top_n = 5

[cache]
enabled = false

[watch]
debounce_ms = 200

[jev]
model = "claude"

[voyage]
chars_per_token = 2.0
retries = 7
"""
        )
        with config._REMOVED_WARNING_LOCK:
            config._WARNED_REMOVED_KEYS.clear()
        with self.assertLogs(config.__name__, level=logging.WARNING) as cm:
            cfg = config.load(path=self.path, environ={})
        self.assertEqual(cfg.voyage.retries, 7)
        self.assertEqual(len(cm.records), 7)

    def test_dataclass_has_no_field_for_removed_keys(self):
        top_fields = {f.name for f in dataclasses.fields(config.Config)}
        for removed_section in ("search", "confidence", "rerank", "cache", "watch", "jev"):
            self.assertNotIn(removed_section, top_fields)
        voyage_fields = {f.name for f in dataclasses.fields(config.VoyageConfig)}
        self.assertNotIn("chars_per_token", voyage_fields)

    def test_unknown_section_raises_config_error(self):
        self.write("[unknown_sec]\nfoo = 1\n")
        with self.assertRaises(config.ConfigError):
            config.load(path=self.path, environ={})

    def test_unknown_key_raises_config_error(self):
        self.write("[embedding]\nunknown_key = 1\n")
        with self.assertRaises(config.ConfigError):
            config.load(path=self.path, environ={})

    def test_wrong_type_raises_config_error(self):
        self.write("[chunking]\nlines = true\n")
        with self.assertRaises(config.ConfigError):
            config.load(path=self.path, environ={})

    def test_wrong_agent_args_type_raises_config_error(self):
        self.write('[agent]\nargs = "--not-a-list"\n')
        with self.assertRaises(config.ConfigError):
            config.load(path=self.path, environ={})

    def test_invalid_env_value_raises_config_error(self):
        with self.assertRaises(config.ConfigError):
            config.load(path=self.dir / "missing.toml", environ={"AUGGIE_DIMENSIONS": "nope"})


def test_removed_config_warning_is_deduplicated_across_loads(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[search]\nphase4_fix1_unique = true\n", encoding="utf-8")

    with unittest.TestCase().assertLogs(config.__name__, level=logging.WARNING) as captured:
        config.load(path=path, environ={})
        config.load(path=path, environ={})

    assert len(captured.records) == 1


def test_eval_new_backend_reuses_config_for_prepare_and_search(tmp_path: Path) -> None:
    import importlib.util
    from unittest.mock import patch

    script = Path(__file__).parents[1] / "scripts" / "eval_recall.py"
    spec = importlib.util.spec_from_file_location("eval_recall_config_fix1", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    cfg = config.Config()
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with (
        patch("auggie_mcp.config.load", return_value=cfg) as load_config,
        patch("auggie_mcp.indexer.index_update") as index_update,
        patch("auggie_mcp.embed.VoyageEmbedder", return_value=object()) as embedder,
        patch("auggie_mcp.semantic.semantic_search", return_value={"hits": []}) as search,
    ):
        backend = module.NewBackend()
        backend.prepare(corpus)
        backend.search(corpus, "query")

    assert load_config.call_count == 1
    assert index_update.call_args.kwargs["config"] is cfg
    assert embedder.call_args.args == (cfg,)
    assert search.call_args.kwargs["config"] is cfg
