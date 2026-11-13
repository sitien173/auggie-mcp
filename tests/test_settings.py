"""Tests for the config.toml settings module."""

import os
import tempfile
import unittest
from pathlib import Path

from auggie_mcp import agent_runner, settings


class SettingsTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "config.toml"

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")

    def run_apply(self, text, environ=None):
        self.write(text)
        environ = {} if environ is None else environ
        settings.apply(self.path, environ)
        return environ


class EncodingTests(SettingsTestCase):
    def test_missing_file_changes_nothing(self):
        environ = {"X": "1"}
        settings.apply(self.dir / "absent.toml", environ)
        self.assertEqual(environ, {"X": "1"})

    def test_string(self):
        environ = self.run_apply('[search]\nmode = "hybrid"\n')
        self.assertEqual(environ, {"AUGGIE_SEARCH_MODE": "hybrid"})

    def test_integer(self):
        environ = self.run_apply("[search]\nrrf_k = 7\n")
        self.assertEqual(environ, {"AUGGIE_RRF_K": "7"})

    def test_float(self):
        environ = self.run_apply("[search]\nhybrid_alpha = 0.25\n")
        self.assertEqual(environ, {"AUGGIE_HYBRID_ALPHA": "0.25"})

    def test_integer_accepted_for_float_key(self):
        environ = self.run_apply("[voyage]\ntimeout_s = 30\n")
        self.assertEqual(environ, {"AUGGIE_VOYAGE_TIMEOUT": "30"})

    def test_boolean(self):
        environ = self.run_apply("[telemetry]\nenabled = true\n")
        self.assertEqual(environ, {"AUGGIE_TELEMETRY": "1"})

    def test_path_string(self):
        environ = self.run_apply('[jev]\narchive_dir = "/var/jev"\n')
        self.assertEqual(environ, {"AUGGIE_JEV_ARCHIVE_DIR": "/var/jev"})

    def test_boolean_rejected_for_integer_key(self):
        with self.assertRaises(settings.ConfigError):
            self.run_apply("[search]\nrrf_k = true\n")

    def test_boolean_rejected_for_float_key(self):
        with self.assertRaises(settings.ConfigError):
            self.run_apply("[search]\nhybrid_alpha = false\n")


class PrecedenceTests(SettingsTestCase):
    def test_non_empty_env_wins(self):
        environ = self.run_apply(
            "[search]\nrrf_k = 7\n", {"AUGGIE_RRF_K": "9"}
        )
        self.assertEqual(environ["AUGGIE_RRF_K"], "9")

    def test_empty_env_lets_file_apply(self):
        environ = self.run_apply("[search]\nrrf_k = 7\n", {"AUGGIE_RRF_K": ""})
        self.assertEqual(environ["AUGGIE_RRF_K"], "7")

    def test_cache_disabled_writes_flag(self):
        environ = self.run_apply("[cache]\nenabled = false\n")
        self.assertEqual(environ, {"AUGGIE_CACHE_DISABLED": "1"})

    def test_cache_enabled_writes_nothing(self):
        environ = self.run_apply("[cache]\nenabled = true\n")
        self.assertEqual(environ, {})

    def test_telemetry_disabled_writes_zero(self):
        environ = self.run_apply("[telemetry]\nenabled = false\n")
        self.assertEqual(environ, {"AUGGIE_TELEMETRY": "0"})

    def test_archive_dir_expands_home(self):
        environ = self.run_apply('[jev]\narchive_dir = "~/jev"\n')
        self.assertEqual(
            environ["AUGGIE_JEV_ARCHIVE_DIR"], os.path.expanduser("~/jev")
        )
        self.assertNotIn("~", environ["AUGGIE_JEV_ARCHIVE_DIR"])

    def test_telemetry_dir_expands_home(self):
        environ = self.run_apply('[telemetry]\ndir = "~/t"\n')
        self.assertEqual(environ["AUGGIE_TELEMETRY_DIR"], os.path.expanduser("~/t"))

    def test_env_path_value_not_expanded(self):
        environ = self.run_apply(
            '[jev]\narchive_dir = "~/jev"\n', {"AUGGIE_JEV_ARCHIVE_DIR": "~/env"}
        )
        self.assertEqual(environ["AUGGIE_JEV_ARCHIVE_DIR"], "~/env")


class ErrorTests(SettingsTestCase):
    def assert_error(self, text, message, environ=None):
        environ = {} if environ is None else environ
        before = dict(environ)
        with self.assertRaises(settings.ConfigError) as ctx:
            self.run_apply(text, environ)
        self.assertEqual(str(ctx.exception), f"{self.path}: {message}")
        self.assertEqual(environ, before)

    def test_directory(self):
        with self.assertRaises(settings.ConfigError) as ctx:
            settings.apply(self.dir, {})
        self.assertTrue(
            str(ctx.exception).startswith(f"{self.dir}: cannot read config: ")
        )

    def test_invalid_toml(self):
        self.write("[search\n")
        with self.assertRaises(settings.ConfigError) as ctx:
            settings.apply(self.path, {})
        self.assertTrue(str(ctx.exception).startswith(f"{self.path}: invalid TOML: "))

    def test_invalid_utf8(self):
        self.path.write_bytes(b"\xff\xfe[search]\n")
        with self.assertRaises(settings.ConfigError) as ctx:
            settings.apply(self.path, {})
        self.assertTrue(str(ctx.exception).startswith(f"{self.path}: invalid TOML: "))

    @unittest.skipIf(os.geteuid() == 0, "root bypasses file permissions")
    def test_unreadable_file_in_locked_directory(self):
        locked = self.dir / "locked"
        locked.mkdir()
        path = locked / "config.toml"
        path.write_text("[search]\nrrf_k = 7\n", encoding="utf-8")
        locked.chmod(0o000)
        self.addCleanup(locked.chmod, 0o700)
        environ = {}
        with self.assertRaises(settings.ConfigError) as ctx:
            settings.apply(path, environ)
        self.assertTrue(str(ctx.exception).startswith(f"{path}: cannot read config: "))
        self.assertEqual(environ, {})

    def test_top_level_scalar(self):
        self.assert_error(
            "mode = 1\n", "top-level key mode must be inside a section"
        )

    def test_unknown_section(self):
        self.assert_error("[bogus]\nx = 1\n", "unknown section [bogus]")

    def test_unknown_key(self):
        self.assert_error("[search]\nbogus = 1\n", "unknown key [search] bogus")

    def test_unknown_agent_key(self):
        self.assert_error("[agent]\nbogus = 1\n", "unknown key [agent] bogus")

    def test_wrong_type(self):
        self.assert_error(
            '[search]\nrrf_k = "7"\n', "[search] rrf_k: expected integer, got string"
        )

    def test_boolean_for_integer(self):
        self.assert_error(
            "[search]\nrrf_k = true\n", "[search] rrf_k: expected integer, got boolean"
        )

    def test_boolean_for_float(self):
        self.assert_error(
            "[search]\nhybrid_alpha = true\n",
            "[search] hybrid_alpha: expected float, got boolean",
        )

    def test_float_for_integer(self):
        self.assert_error(
            "[search]\nrrf_k = 1.5\n", "[search] rrf_k: expected integer, got float"
        )

    def test_string_for_boolean(self):
        self.assert_error(
            '[cache]\nenabled = "no"\n', "[cache] enabled: expected boolean, got string"
        )

    def test_number_for_string(self):
        self.assert_error(
            "[search]\nmode = 3\n", "[search] mode: expected string, got integer"
        )

    def test_section_not_a_table_is_top_level_error(self):
        self.assert_error("search = 1\n", "top-level key search must be inside a section")

    def test_shadowed_invalid_value_still_raises(self):
        self.assert_error(
            '[search]\nrrf_k = "bad"\n',
            "[search] rrf_k: expected integer, got string",
            {"AUGGIE_RRF_K": "9"},
        )

    def test_error_leaves_environ_unchanged(self):
        self.assert_error(
            '[search]\nmode = "hybrid"\nrrf_k = "bad"\n',
            "[search] rrf_k: expected integer, got string",
        )

    def test_agent_values_not_type_checked_and_not_exported(self):
        environ = self.run_apply('[agent]\ntimeout_s = "x"\nargs = 3\n')
        self.assertEqual(environ, {})


class SchemaTests(unittest.TestCase):
    def test_agent_keys_match_agent_runner(self):
        self.assertEqual(settings.AGENT_KEYS, agent_runner._KEYS)

    def test_schema_has_39_entries(self):
        self.assertEqual(len(settings.SCHEMA), 39)

    def test_schema_env_vars_unique_and_prefixed(self):
        names = [name for name, _ in settings.SCHEMA.values()]
        self.assertEqual(len(set(names)), len(names))
        self.assertTrue(all(n.startswith("AUGGIE_") for n in names))


if __name__ == "__main__":
    unittest.main()
