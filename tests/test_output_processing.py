import asyncio
import concurrent.futures
import json
import math
import os
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from unittest.mock import patch

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeError,
    Usage,
)


class TestPhase1DependencyAndVersion(unittest.TestCase):
    """Task 1: Project version 0.2.0 and typesafe-sdk dependency."""

    def test_project_version_in_pyproject(self):
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")
        self.assertIn('version = "0.3.0"', content)

    def test_typesafe_sdk_runtime_dependency_declared(self):
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")
        self.assertIn('"typesafe-sdk"', content)

    def test_typesafe_sdk_importable(self):
        import typesafe_sdk
        self.assertTrue(hasattr(typesafe_sdk, "AsyncTypeSafeClient"))


class TestPublicConstantsAndContracts(unittest.TestCase):
    """Task 2: Constants, enums, typed models, and configuration."""

    def test_constants_and_categories(self):
        from auggie_mcp import output_processing

        self.assertEqual(output_processing.PRUNING_THRESHOLD_TOKENS, 10_000)
        self.assertEqual(output_processing.DEFAULT_CHUNK_LINES, 20)
        self.assertEqual(output_processing.MAX_LINE_CHARS, 2_000)
        self.assertEqual(output_processing.MAX_CHUNKS, 200)
        self.assertEqual(output_processing.RETENTION_UNCERTAINTY_FLOOR, 0.1)

        expected_categories = {"reference", "diagnostic", "result", "progress", "unknown"}
        self.assertEqual(set(output_processing.CATEGORIES), expected_categories)

        expected_statuses = {"classified", "unclassified"}
        self.assertEqual(set(output_processing.PROCESSING_STATUSES), expected_statuses)

        expected_retentions = {"below_threshold", "protected", "kept_all", "pruned", "failed_open"}
        self.assertEqual(set(output_processing.RETENTION_STATES), expected_retentions)

    def test_pruning_eligibility_threshold_strictly_greater(self):
        from auggie_mcp.output_processing import is_prune_eligible

        self.assertFalse(is_prune_eligible(0))
        self.assertFalse(is_prune_eligible(9_999))
        self.assertFalse(is_prune_eligible(10_000), "Exactly 10,000 must NOT be prune-eligible")
        self.assertTrue(is_prune_eligible(10_001), "10,001 must be prune-eligible")
        self.assertTrue(is_prune_eligible(20_000))

    def test_batch_size_validator(self):
        from auggie_mcp.output_processing import is_valid_batch_size, MAX_ARCHIVE_WORKERS

        self.assertEqual(MAX_ARCHIVE_WORKERS, 4)
        self.assertTrue(is_valid_batch_size(1))
        self.assertTrue(is_valid_batch_size(50))
        self.assertFalse(is_valid_batch_size(0))
        self.assertFalse(is_valid_batch_size(-1))
        self.assertFalse(is_valid_batch_size(True))
        self.assertFalse(is_valid_batch_size(False))
        self.assertFalse(is_valid_batch_size(1.5))
        self.assertFalse(is_valid_batch_size("50"))
        self.assertFalse(is_valid_batch_size(None))

    def test_typed_envelopes_structure(self):
        from auggie_mcp.output_processing import (
            RetrievalEnvelope,
            SearchEnvelope,
            ProcessingMetadata,
            ChunkMetadata,
        )

        chunk: ChunkMetadata = {
            "id": "result:1",
            "source": "result",
            "start_line": 1,
            "end_line": 20,
            "category": "unknown",
            "protected": True,
            "retained": True,
        }
        self.assertEqual(chunk["id"], "result:1")

        proc: ProcessingMetadata = {
            "status": "classified",
            "retention": "below_threshold",
            "estimated_tokens": 500,
            "recovery_path": None,
            "chunks": [chunk],
        }
        retrieval: RetrievalEnvelope = {
            "tool": "codebase_retrieval",
            "data": {"result": "sample output"},
            "processing": proc,
        }
        self.assertEqual(retrieval["tool"], "codebase_retrieval")

        search: SearchEnvelope = {
            "tool": "codebase_search",
            "data": {"stdout": "out", "stderr": "err", "exit_code": 0},
            "processing": proc,
        }
        self.assertEqual(search["tool"], "codebase_search")

    def test_configuration_defaults(self):
        from auggie_mcp.output_processing import load_config

        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config()
            self.assertIsNone(cfg.api_key)
            self.assertEqual(cfg.model, "jev-latest")
            self.assertEqual(cfg.timeout_seconds, 30.0)
            self.assertEqual(cfg.max_requests, 40)
            expected_archive = Path.home() / ".cache" / "auggie-mcp" / "jev"
            self.assertEqual(cfg.archive_dir, expected_archive)

    def test_configuration_xdg_cache_fallback(self):
        from auggie_mcp.output_processing import load_config

        env = {"XDG_CACHE_HOME": "/tmp/custom_cache"}
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()
            self.assertEqual(cfg.archive_dir, Path("/tmp/custom_cache/auggie-mcp/jev"))

    def test_configuration_explicit_archive_dir(self):
        from auggie_mcp.output_processing import load_config

        env = {
            "AUGGIE_JEV_ARCHIVE_DIR": "/var/custom/jev_archives",
            "XDG_CACHE_HOME": "/tmp/custom_cache",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()
            self.assertEqual(cfg.archive_dir, Path("/var/custom/jev_archives"))

    def test_configuration_custom_env_values(self):
        from auggie_mcp.output_processing import load_config

        env = {
            "TYPESAFE_API_KEY": "sk-secret-123",
            "AUGGIE_JEV_MODEL": "jev-2.0-custom",
            "AUGGIE_JEV_TIMEOUT_SECONDS": "45.5",
            "AUGGIE_JEV_MAX_REQUESTS": "60",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()
            self.assertEqual(cfg.api_key, "sk-secret-123")
            self.assertEqual(cfg.model, "jev-2.0-custom")
            self.assertEqual(cfg.timeout_seconds, 45.5)
            self.assertEqual(cfg.max_requests, 60)

    def test_load_config_rejects_negative_or_zero_timeout(self):
        from auggie_mcp.output_processing import load_config

        for invalid in ["-1", "-0.01", "0", "0.0"]:
            with patch.dict(os.environ, {"AUGGIE_JEV_TIMEOUT_SECONDS": invalid}, clear=True):
                with self.assertRaises(ValueError):
                    load_config()

    def test_load_config_rejects_nan_and_inf_timeout(self):
        from auggie_mcp.output_processing import load_config

        for invalid in ["nan", "NaN", "inf", "-inf", "Infinity"]:
            with patch.dict(os.environ, {"AUGGIE_JEV_TIMEOUT_SECONDS": invalid}, clear=True):
                with self.assertRaises(ValueError):
                    load_config()

    def test_load_config_rejects_invalid_max_requests(self):
        from auggie_mcp.output_processing import load_config

        for invalid in ["0", "-1", "-10", "abc"]:
            with patch.dict(os.environ, {"AUGGIE_JEV_MAX_REQUESTS": invalid}, clear=True):
                with self.assertRaises(ValueError):
                    load_config()

    def test_create_typesafe_client_rejects_invalid_timeout(self):
        from auggie_mcp.output_processing import JevConfig, create_typesafe_client

        for invalid in [-1.0, 0.0, float("nan"), float("inf"), float("-inf")]:
            cfg = JevConfig(api_key="test", timeout_seconds=invalid)
            with self.assertRaises(ValueError):
                create_typesafe_client(cfg)

    def test_create_typesafe_client_rejects_invalid_max_requests(self):
        from auggie_mcp.output_processing import JevConfig, create_typesafe_client

        for invalid in [0, -1, -5]:
            cfg = JevConfig(api_key="test", max_requests=invalid)
            with self.assertRaises(ValueError):
                create_typesafe_client(cfg)


class TestTokenEstimator(unittest.TestCase):
    """Task 3A: Conservative token estimator."""

    def test_estimate_tokens_empty_string(self):
        from auggie_mcp.output_processing import estimate_tokens

        self.assertEqual(estimate_tokens(""), 0)

    def test_estimate_tokens_prose(self):
        from auggie_mcp.output_processing import estimate_tokens

        text = "The quick brown fox jumps over the lazy dog."
        tokens = estimate_tokens(text)
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, len(text))

    def test_estimate_tokens_source_code(self):
        from auggie_mcp.output_processing import estimate_tokens

        code = "def add(x: int, y: int) -> int:\n    return x + y\n"
        tokens = estimate_tokens(code)
        self.assertGreater(tokens, 5)

    def test_estimate_tokens_json(self):
        from auggie_mcp.output_processing import estimate_tokens

        json_blob = json.dumps({"status": "ok", "items": [1, 2, 3], "nested": {"key": "val"}})
        tokens = estimate_tokens(json_blob)
        self.assertGreater(tokens, 10)

    def test_estimate_tokens_deterministic(self):
        from auggie_mcp.output_processing import estimate_tokens

        text = "Repeated text for determinism testing 1234567890 !@#$%^&*()"
        first = estimate_tokens(text)
        second = estimate_tokens(text)
        self.assertEqual(first, second)

    def test_search_tokens_combined_stdout_and_stderr(self):
        from auggie_mcp.output_processing import estimate_search_tokens, estimate_tokens

        stdout = "line 1\nline 2\n"
        stderr = "error on line 3\n"
        expected = estimate_tokens(stdout) + estimate_tokens(stderr)
        self.assertEqual(estimate_search_tokens(stdout, stderr), expected)

    def test_exit_code_never_counted(self):
        from auggie_mcp.output_processing import estimate_search_tokens, estimate_tokens

        stdout = "output text"
        stderr = ""
        # Function only accepts stdout and stderr strings, no exit code
        self.assertEqual(estimate_search_tokens(stdout, stderr), estimate_tokens(stdout))


class TestChunking(unittest.TestCase):
    """Task 3B: Source-aware chunking."""

    def test_chunk_empty_source(self):
        from auggie_mcp.output_processing import chunk_source

        chunks = chunk_source("result", "")
        self.assertEqual(chunks, [])

    def test_chunk_fewer_than_20_lines(self):
        from auggie_mcp.output_processing import chunk_source

        lines = "\n".join(f"line {i}" for i in range(1, 10))
        chunks = chunk_source("result", lines)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].id, "result:1")
        self.assertEqual(chunks[0].source, "result")
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 9)

    def test_chunk_exactly_20_lines(self):
        from auggie_mcp.output_processing import chunk_source

        lines = "\n".join(f"line {i}" for i in range(1, 21))
        chunks = chunk_source("result", lines)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].id, "result:1")
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 20)

    def test_chunk_21_lines(self):
        from auggie_mcp.output_processing import chunk_source

        lines = "\n".join(f"line {i}" for i in range(1, 22))
        chunks = chunk_source("result", lines)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].id, "result:1")
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 20)
        self.assertEqual(chunks[1].id, "result:2")
        self.assertEqual(chunks[1].start_line, 21)
        self.assertEqual(chunks[1].end_line, 21)

    def test_chunk_multiple_sources_preserves_channel_identity(self):
        from auggie_mcp.output_processing import chunk_result

        sources = [
            ("stdout", "out1\nout2\n"),
            ("stderr", "err1\nerr2\n"),
        ]
        chunks = chunk_result(sources)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].id, "stdout:1")
        self.assertEqual(chunks[0].source, "stdout")
        self.assertEqual(chunks[1].id, "stderr:1")
        self.assertEqual(chunks[1].source, "stderr")

    def test_chunk_line_longer_than_2000_chars(self):
        from auggie_mcp.output_processing import chunk_source

        # Line 1 is 4,500 chars, Line 2 is 100 chars
        long_line = "A" * 4500
        text = f"{long_line}\nshort line\n"
        chunks = chunk_source("result", text)
        # Should split line 1 into <= 2,000 char pieces
        for chunk in chunks:
            for piece in chunk.pieces:
                self.assertLessEqual(len(piece.text), 2000)
        # Line numbers map to original logical lines
        self.assertEqual(chunks[0].start_line, 1)
        # Verify no content was lost
        reconstructed = "".join(chunk.text for chunk in chunks)
        self.assertEqual(reconstructed, text)

    def test_chunk_multiple_long_lines(self):
        from auggie_mcp.output_processing import chunk_source

        lines = ["X" * 3000, "Y" * 2500, "Z" * 100]
        text = "\n".join(lines)
        chunks = chunk_source("result", text)
        reconstructed = "".join(chunk.text for chunk in chunks)
        self.assertEqual(reconstructed, text)

    def test_chunk_bounding_collapses_to_at_most_200(self):
        from auggie_mcp.output_processing import chunk_result

        # 5,000 lines would normally be 250 chunks (20 lines each)
        stdout_lines = "\n".join(f"stdout line {i}" for i in range(1, 3001))
        stderr_lines = "\n".join(f"stderr line {i}" for i in range(1, 2001))
        sources = [("stdout", stdout_lines), ("stderr", stderr_lines)]
        chunks = chunk_result(sources)
        self.assertLessEqual(len(chunks), 200)

        # Confirm no content lost after merging
        stdout_chunks = [c for c in chunks if c.source == "stdout"]
        stderr_chunks = [c for c in chunks if c.source == "stderr"]
        self.assertEqual("".join(c.text for c in stdout_chunks), stdout_lines)
        self.assertEqual("".join(c.text for c in stderr_chunks), stderr_lines)


class TestProtection(unittest.TestCase):
    """Task 3C: Local protected-line and structured-document detection."""

    def test_first_and_last_chunk_protected(self):
        from auggie_mcp.output_processing import chunk_source

        # 60 lines => 3 chunks
        lines = "\n".join(f"plain line {i}" for i in range(1, 61))
        chunks = chunk_source("result", lines)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[0].protected, "First chunk must be protected")
        self.assertFalse(chunks[1].protected, "Middle plain chunk should not be protected")
        self.assertTrue(chunks[2].protected, "Last chunk must be protected")

    def test_diagnostic_line_protection_and_context(self):
        from auggie_mcp.output_processing import is_critical_line, chunk_source

        self.assertTrue(is_critical_line("ERROR: connection refused"))
        self.assertTrue(is_critical_line("Traceback (most recent call last):"))
        self.assertTrue(is_critical_line("RuntimeError: null pointer"))
        self.assertTrue(is_critical_line("WARNING: disk space low"))

        # Put an error in chunk 2 (lines 25) of 5 chunks (100 lines)
        lines = [f"plain line {i}" for i in range(1, 101)]
        lines[25] = "FATAL: process terminated unexpectedly"
        chunks = chunk_source("result", "\n".join(lines))
        self.assertTrue(chunks[1].protected, "Chunk containing diagnostic must be protected")

    def test_result_critical_line_protection(self):
        from auggie_mcp.output_processing import is_critical_line, chunk_source

        self.assertTrue(is_critical_line("Tests: 45 passed, 0 failed"))
        self.assertTrue(is_critical_line("Exit code: 0"))
        self.assertTrue(is_critical_line("=== 10 passed in 1.23s ==="))
        self.assertTrue(is_critical_line("Build succeeded."))

        lines = [f"info line {i}" for i in range(1, 101)]
        lines[25] = "Build succeeded."
        chunks = chunk_source("result", "\n".join(lines))
        self.assertTrue(chunks[1].protected, "Chunk containing result line must be protected")

    def test_boundary_adjacency_protection(self):
        from auggie_mcp.output_processing import chunk_source

        # If a critical line is at line 20 (the boundary between chunk 0 and chunk 1):
        lines = [f"line {i}" for i in range(1, 81)]
        lines[19] = "ERROR: something broke at boundary"
        chunks = chunk_source("result", "\n".join(lines))
        # Chunk 0 has the error line. Chunk 1 is adjacent to the boundary line.
        self.assertTrue(chunks[0].protected)
        self.assertTrue(chunks[1].protected, "Chunk adjacent to critical boundary must be protected")

    def test_reference_content_not_accidentally_all_protected(self):
        from auggie_mcp.output_processing import chunk_source

        lines = [f"import os\nx = {i}\n" for i in range(1, 100)]
        chunks = chunk_source("result", "".join(lines))
        # Intermediate chunks should not be protected
        unprotected = [c for c in chunks[1:-1] if not c.protected]
        self.assertGreater(len(unprotected), 0)

    def test_structured_document_valid_json_object(self):
        from auggie_mcp.output_processing import is_structured_document, chunk_source

        doc = json.dumps({"key": "value", "list": [1, 2, 3]}, indent=2)
        self.assertTrue(is_structured_document(doc))

        chunks = chunk_source("result", doc)
        for c in chunks:
            self.assertTrue(c.protected, "All chunks in structured document must be protected")

    def test_structured_document_valid_json_array(self):
        from auggie_mcp.output_processing import is_structured_document, chunk_source

        doc = json.dumps([{"id": i} for i in range(50)], indent=2)
        self.assertTrue(is_structured_document(doc))

        chunks = chunk_source("result", doc)
        for c in chunks:
            self.assertTrue(c.protected)

    def test_structured_document_malformed_json_not_protected(self):
        from auggie_mcp.output_processing import is_structured_document

        bad_json = '{"key": "value", "unclosed: 123}'
        self.assertFalse(is_structured_document(bad_json))


class TestOmissionRendering(unittest.TestCase):
    """Task 4A: Omission rendering."""

    def test_omission_rendering_kept_all(self):
        from auggie_mcp.output_processing import chunk_source, render_omissions

        text = "line 1\nline 2\nline 3\n"
        chunks = chunk_source("result", text)
        for c in chunks:
            c.retained = True
        rendered = render_omissions(chunks)
        self.assertEqual(rendered, text)

    def test_single_dropped_chunk(self):
        from auggie_mcp.output_processing import chunk_source, render_omissions

        lines = "\n".join(f"line {i}" for i in range(1, 61))
        chunks = chunk_source("result", lines)
        self.assertEqual(len(chunks), 3)
        chunks[0].retained = True
        chunks[1].retained = False
        chunks[2].retained = True

        rendered = render_omissions(chunks)
        self.assertIn("line 1", rendered)
        self.assertNotIn("line 25", rendered)
        self.assertIn("line 60", rendered)
        self.assertIn("[omitted", rendered)

    def test_consecutive_dropped_chunks_collapse_to_single_marker(self):
        from auggie_mcp.output_processing import chunk_source, render_omissions

        lines = "\n".join(f"line {i}" for i in range(1, 101))
        chunks = chunk_source("result", lines)
        self.assertEqual(len(chunks), 5)
        chunks[0].retained = True
        chunks[1].retained = False
        chunks[2].retained = False
        chunks[3].retained = False
        chunks[4].retained = True

        rendered = render_omissions(chunks)
        self.assertEqual(rendered.count("[omitted"), 1, "Consecutive dropped chunks must collapse")

    def test_separate_dropped_runs_get_separate_markers(self):
        from auggie_mcp.output_processing import chunk_source, render_omissions

        lines = "\n".join(f"line {i}" for i in range(1, 101))
        chunks = chunk_source("result", lines)
        self.assertEqual(len(chunks), 5)
        chunks[0].retained = True
        chunks[1].retained = False
        chunks[2].retained = True
        chunks[3].retained = False
        chunks[4].retained = True

        rendered = render_omissions(chunks)
        self.assertEqual(rendered.count("[omitted"), 2, "Separate dropped runs must get separate markers")

    def test_source_boundaries_remain_independent(self):
        from auggie_mcp.output_processing import (
            chunk_result,
            render_omissions_by_source,
        )

        sources = [
            ("stdout", "\n".join(f"stdout {i}" for i in range(1, 61))),
            ("stderr", "\n".join(f"stderr {i}" for i in range(1, 61))),
        ]
        chunks = chunk_result(sources)
        for c in chunks:
            c.retained = True
        # Drop last chunk of stdout and first chunk of stderr
        stdout_chunks = [c for c in chunks if c.source == "stdout"]
        stderr_chunks = [c for c in chunks if c.source == "stderr"]
        stdout_chunks[-1].retained = False
        stderr_chunks[0].retained = False

        rendered = render_omissions_by_source(chunks)
        self.assertIn("stdout", rendered)
        self.assertIn("stderr", rendered)
        # Verify stdout rendered text does not contain stderr markers or content
        self.assertNotIn("stderr", rendered["stdout"])
        self.assertNotIn("stdout", rendered["stderr"])

    def test_omission_rendering_very_long_single_line_spans_multiple_chunks(self):
        from auggie_mcp.output_processing import chunk_source, render_omissions

        # 50,000 characters in one logical line splits into 25 pieces (>20 nominal lines)
        long_line = "A" * 50000 + "\n"
        chunks = chunk_source("result", long_line)
        self.assertGreater(len(chunks), 1, "Long line must span multiple chunks")
        for c in chunks:
            c.retained = False

        rendered = render_omissions(chunks)
        # Unique logical line count must be exactly 1, not len(chunks)
        self.assertEqual(rendered, f"[omitted 1 lines ({len(long_line)} characters)]")


class TestArchiveCreation(unittest.TestCase):
    """Task 4B: Private archive creation and security."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.archive_dir = Path(self.temp_dir.name) / "archives"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_archive_json_envelope_and_exact_strings(self):
        from auggie_mcp.output_processing import create_archive, ARCHIVE_FORMAT_VERSION

        data = {
            "stdout": "exact stdout content with unicode: 🚀\n",
            "stderr": "error: none\n",
            "exit_code": 0,
        }
        archive_path = create_archive("codebase_search", data, archive_dir=self.archive_dir)
        self.assertTrue(archive_path.is_absolute())
        self.assertTrue(archive_path.exists())

        loaded = json.loads(archive_path.read_text("utf-8"))
        self.assertEqual(loaded["version"], ARCHIVE_FORMAT_VERSION)
        self.assertEqual(loaded["tool"], "codebase_search")
        self.assertEqual(loaded["data"]["stdout"], data["stdout"])
        self.assertEqual(loaded["data"]["stderr"], data["stderr"])
        self.assertEqual(loaded["data"]["exit_code"], data["exit_code"])

    def test_archive_directory_mode_0700_and_file_mode_0600(self):
        from auggie_mcp.output_processing import create_archive

        data = {"result": "retrieval text"}
        archive_path = create_archive("codebase_retrieval", data, archive_dir=self.archive_dir)

        dir_stat = self.archive_dir.stat()
        file_stat = archive_path.stat()

        dir_mode = stat.S_IMODE(dir_stat.st_mode)
        file_mode = stat.S_IMODE(file_stat.st_mode)

        self.assertEqual(dir_mode, 0o700, f"Archive directory mode must be 0700, got {oct(dir_mode)}")
        self.assertEqual(file_mode, 0o600, f"Archive file mode must be 0600, got {oct(file_mode)}")

    def test_archive_unique_uuid_names_and_concurrency(self):
        from auggie_mcp.output_processing import create_archive

        data = {"result": "sample"}

        def write_one(idx):
            return create_archive("codebase_retrieval", {"result": f"sample {idx}"}, archive_dir=self.archive_dir)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            paths = list(executor.map(write_one, range(20)))

        # All 20 paths should be distinct
        self.assertEqual(len(paths), 20)
        self.assertEqual(len(set(paths)), 20)
        for p in paths:
            self.assertTrue(p.exists())
            self.assertTrue(p.name.endswith(".json"))

    def test_archive_path_outside_repo_by_default(self):
        from auggie_mcp.output_processing import get_default_archive_dir

        repo_root = Path(__file__).resolve().parent.parent
        default_dir = get_default_archive_dir()
        self.assertFalse(default_dir.is_relative_to(repo_root), "Default archive path must be outside repo")

    def test_archive_exclusive_create_collision_prevents_replacement(self):
        import uuid
        from auggie_mcp.output_processing import create_archive

        fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        original_data = {"result": "initial uncorrupted data"}
        attempted_data = {"result": "malicious overwrite attempt"}

        with patch("uuid.uuid4", return_value=fixed_uuid):
            path = create_archive("codebase_retrieval", original_data, archive_dir=self.archive_dir)
            self.assertTrue(path.exists())

            # Attempt to create again with forced UUID collision
            with self.assertRaises(FileExistsError):
                create_archive("codebase_retrieval", attempted_data, archive_dir=self.archive_dir)

            # Assert original content was preserved and not overwritten
            content = json.loads(path.read_text("utf-8"))
            self.assertEqual(content["data"]["result"], "initial uncorrupted data")




# ==============================================================================
# Phase 2: Jev Classification and Retention Pipeline Tests
# ==============================================================================


class FakeTypeSafeClient:
    """Minimal offline async fake recording calls and returning programmable responses."""

    def __init__(self, responses: Optional[Sequence[Any]] = None):
        self.call_count: int = 0
        self.calls: List[Dict[str, Any]] = []
        self._responses: List[Any] = list(responses) if responses is not None else []
        self.on_call = None

    async def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        self.calls.append(
            {
                "state": state,
                "questions": dict(questions),
                "model": model,
                "kwargs": kwargs,
            }
        )
        if self.on_call is not None:
            await self.on_call(self.call_count, state, questions)
        if self._responses:
            resp = self._responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if callable(resp):
                result = resp(state, questions, model=model, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, Exception):
                    raise result
                return result
            return resp

        # Default fallback response based on question type
        first_q = next(iter(questions.values())) if questions else None
        if hasattr(first_q, "type") and first_q.type == "choice":
            ans = {
                qid: ChoiceAnswer(
                    choice="reference",
                    confidence=1.0,
                    probabilities={"reference": 1.0},
                )
                for qid in questions
            }
        else:
            ans = {qid: NoulAnswer(noul=1.0) for qid in questions}
        return SystemOneResponse(
            model=model or "jev-latest",
            usage=Usage(input_tokens=10, output_tokens=5),
            answers=ans,
        )


def make_choice_response(
    answers: Mapping[str, str], model: str = "jev-latest"
) -> SystemOneResponse:
    ans = {
        qid: ChoiceAnswer(
            choice=cat,
            confidence=1.0,
            probabilities={cat: 1.0},
        )
        for qid, cat in answers.items()
    }
    return SystemOneResponse(
        model=model,
        usage=Usage(input_tokens=10, output_tokens=5),
        answers=ans,
    )


def make_noul_response(
    answers: Mapping[str, float], model: str = "jev-latest"
) -> SystemOneResponse:
    ans = {qid: NoulAnswer(noul=float(prob)) for qid, prob in answers.items()}
    return SystemOneResponse(
        model=model,
        usage=Usage(input_tokens=10, output_tokens=5),
        answers=ans,
    )


def make_large_3chunk_text(diagnostic_line: Optional[str] = None) -> str:
    """Generate 60 lines (3 chunks: result:1, result:2, result:3) with > 10,000 estimated tokens."""
    line = " ".join(f"w{j}" for j in range(120))
    lines = [f"{line} line{i}" for i in range(1, 61)]
    if diagnostic_line is not None:
        lines[25] = f"{diagnostic_line} {line}"
    return "\n".join(lines)


def make_large_nchunk_text(n_chunks: int) -> str:
    """Generate n_chunks chunks (20 lines each) with > 10,000 estimated tokens."""
    line = " ".join(f"w{j}" for j in range(120))
    lines = [f"{line} line{i}" for i in range(1, (n_chunks * 20) + 1)]
    return "\n".join(lines)


def make_large_1chunk_text() -> str:
    """Generate 20 lines (1 chunk: result:1) with > 10,000 estimated tokens."""
    line = " ".join(f"w{j}" for j in range(300))
    lines = [f"{line} line{i}" for i in range(1, 21)]
    return "\n".join(lines)


def make_large_structured_doc() -> str:
    """Generate valid JSON document with > 10,000 estimated tokens."""
    doc_dict = {f"key_{i}": " ".join(f"val{j}" for j in range(100)) for i in range(1, 100)}
    return json.dumps(doc_dict, indent=2)


class TestFakeClient(unittest.IsolatedAsyncioTestCase):
    """Task 1: Offline fake client protocol and response programming."""

    async def test_fake_client_records_calls_and_returns_responses(self):
        resp1 = make_choice_response({"result:1": "diagnostic"})
        resp2 = make_noul_response({"result:1": 0.5})
        client = FakeTypeSafeClient([resp1, resp2])

        q1 = {"result:1": Choice(instructions="inst", criteria={"diagnostic": "d"})}
        got1 = await client.system_one(state={"task": "t1"}, questions=q1, model="jev-test")
        self.assertEqual(client.call_count, 1)
        self.assertEqual(got1.choices["result:1"].choice, "diagnostic")
        self.assertEqual(client.calls[0]["state"], {"task": "t1"})
        self.assertEqual(client.calls[0]["model"], "jev-test")

        q2 = {"result:1": Noul(instructions="inst")}
        got2 = await client.system_one(state={"task": "t2"}, questions=q2)
        self.assertEqual(client.call_count, 2)
        self.assertEqual(got2.nouls["result:1"].noul, 0.5)

    async def test_fake_client_raises_programmed_exception(self):
        client = FakeTypeSafeClient([TypeSafeError("SDK failure")])
        with self.assertRaises(TypeSafeError):
            await client.system_one(state={}, questions={})


class TestClassification(unittest.IsolatedAsyncioTestCase):
    """Task 2: Classification of chunks with Choice questions."""

    async def test_one_chunk_each_of_five_categories(self):
        from auggie_mcp.output_processing import (
            CATEGORIES,
            process_retrieval_output,
        )

        for cat in CATEGORIES:
            client = FakeTypeSafeClient([make_choice_response({"result:1": cat})])
            env = await process_retrieval_output(
                query="find test",
                result="def test(): pass\n",
                client=client,
            )
            self.assertEqual(env["processing"]["status"], "classified")
            self.assertEqual(env["processing"]["retention"], "below_threshold")
            self.assertEqual(len(env["processing"]["chunks"]), 1)
            self.assertEqual(env["processing"]["chunks"][0]["category"], cat)

    async def test_multiple_chunks_classified(self):
        from auggie_mcp.output_processing import process_retrieval_output

        # 45 lines creates 3 chunks of 20, 20, 5 lines
        text = "\n".join(f"line {i}" for i in range(1, 46))
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "diagnostic",
                        "result:3": "result",
                    }
                )
            ]
        )
        env = await process_retrieval_output(
            query="find lines",
            result=text,
            client=client,
        )
        self.assertEqual(env["processing"]["status"], "classified")
        chunks = env["processing"]["chunks"]
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["category"], "reference")
        self.assertEqual(chunks[1]["category"], "diagnostic")
        self.assertEqual(chunks[2]["category"], "result")

    async def test_multiple_sources_classified(self):
        from auggie_mcp.output_processing import process_search_output

        stdout_text = "stdout line 1\nstdout line 2\n"
        stderr_text = "stderr line 1\nstderr line 2\n"
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "stdout:1": "result",
                        "stderr:1": "diagnostic",
                    }
                )
            ]
        )
        env = await process_search_output(
            pattern="test",
            stdout=stdout_text,
            stderr=stderr_text,
            exit_code=0,
            client=client,
        )
        self.assertEqual(env["processing"]["status"], "classified")
        chunks = env["processing"]["chunks"]
        self.assertEqual(len(chunks), 2)
        by_source = {c["source"]: c for c in chunks}
        self.assertEqual(by_source["stdout"]["category"], "result")
        self.assertEqual(by_source["stderr"]["category"], "diagnostic")

    async def test_below_threshold_results_still_call_jev(self):
        from auggie_mcp.output_processing import process_retrieval_output

        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
        env = await process_retrieval_output(
            query="short",
            result="small output text\n",
            client=client,
        )
        self.assertEqual(client.call_count, 1)
        self.assertEqual(env["processing"]["status"], "classified")
        self.assertEqual(env["processing"]["retention"], "below_threshold")
        self.assertIsNone(env["processing"]["recovery_path"])

    async def test_every_chunk_gets_exactly_one_category_answer(self):
        from auggie_mcp.output_processing import process_retrieval_output

        text = "\n".join(f"line {i}" for i in range(1, 41))  # 2 chunks
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "progress",
                    }
                )
            ]
        )
        env = await process_retrieval_output(query="q", result=text, client=client)
        for c in env["processing"]["chunks"]:
            self.assertIn(c["category"], ["reference", "progress"])

    async def test_invalid_category_fails_open(self):
        from auggie_mcp.output_processing import process_retrieval_output

        # Mock an invalid category returned by Jev
        ans = {
            "result:1": ChoiceAnswer(
                choice="invalid_category",
                confidence=1.0,
                probabilities={"invalid_category": 1.0},
            )
        }
        resp = SystemOneResponse(
            model="jev-latest",
            usage=Usage(input_tokens=10, output_tokens=5),
            answers=ans,
        )
        client = FakeTypeSafeClient([resp])
        text = "original untouched text\n"
        env = await process_retrieval_output(query="q", result=text, client=client)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")
        self.assertEqual(env["processing"]["chunks"][0]["category"], "unknown")

    async def test_missing_answer_fails_open(self):
        from auggie_mcp.output_processing import process_retrieval_output

        # 2 chunks, but response only has result:1
        text = "\n".join(f"line {i}" for i in range(1, 41))
        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
        env = await process_retrieval_output(query="q", result=text, client=client)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")
        for c in env["processing"]["chunks"]:
            self.assertEqual(c["category"], "unknown")

    async def test_wrong_answer_kind_fails_open(self):
        from auggie_mcp.output_processing import process_retrieval_output

        # Answer is NoulAnswer instead of ChoiceAnswer
        ans = {"result:1": NoulAnswer(noul=0.8)}
        resp = SystemOneResponse(
            model="jev-latest",
            usage=Usage(input_tokens=10, output_tokens=5),
            answers=ans,
        )
        client = FakeTypeSafeClient([resp])
        text = "sample text\n"
        env = await process_retrieval_output(query="q", result=text, client=client)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_choice_lookalike_rejected_fails_open(self):
        from auggie_mcp.output_processing import process_retrieval_output, validate_choice_response

        class ChoiceLookalike:
            def __init__(self, choice: str):
                self.choice = choice

        # Direct unit test on validate_choice_response
        fake_resp = type("FakeResp", (), {"choices": {"result:1": ChoiceLookalike("reference")}})()
        with self.assertRaises(ValueError) as ctx:
            validate_choice_response(fake_resp, ["result:1"])
        self.assertIn("not an instance of ChoiceAnswer", str(ctx.exception))

        # End-to-end integration test
        client = FakeTypeSafeClient([fake_resp])
        text = "sample text\n"
        env = await process_retrieval_output(query="q", result=text, client=client)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_extra_answers_ignored(self):
        from auggie_mcp.output_processing import process_retrieval_output

        # Expected result:1, but also contains surplus result:99
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "surplus:99": "diagnostic",
                    }
                )
            ]
        )
        text = "sample text\n"
        env = await process_retrieval_output(query="q", result=text, client=client)
        self.assertEqual(env["processing"]["status"], "classified")
        self.assertEqual(env["processing"]["chunks"][0]["category"], "reference")


class TestGoalState(unittest.IsolatedAsyncioTestCase):
    """Task 3: Effective goal and Jev state hygiene."""

    async def test_explicit_goal_is_used(self):
        from auggie_mcp.output_processing import process_retrieval_output

        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
        await process_retrieval_output(
            query="query string",
            result="sample output\n",
            goal="custom explicit goal",
            client=client,
        )
        state = client.calls[0]["state"]
        self.assertEqual(state["task"], "custom explicit goal")

    async def test_retrieval_fallback_goal_to_query(self):
        from auggie_mcp.output_processing import process_retrieval_output

        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
        await process_retrieval_output(
            query="fallback retrieval query",
            result="sample output\n",
            goal=None,
            client=client,
        )
        state = client.calls[0]["state"]
        self.assertEqual(state["task"], "fallback retrieval query")

    async def test_search_fallback_goal_to_pattern(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        await process_search_output(
            pattern="fallback search pattern",
            stdout="sample output\n",
            stderr="",
            exit_code=0,
            goal=None,
            client=client,
        )
        state = client.calls[0]["state"]
        self.assertEqual(state["task"], "fallback search pattern")

    async def test_absolute_repo_path_never_enters_state(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        # Actual absolute path inside repository is relativized to safe relative path
        await process_search_output(
            pattern="find_func",
            stdout="match 1\n",
            stderr="",
            exit_code=0,
            path="/home/user/project/src/subfolder",
            repository="/home/user/project",
            flags=["-i"],
            client=client,
        )
        tool_input = client.calls[0]["state"]["tool_input"]
        self.assertEqual(tool_input["path"], "src/subfolder")
        serialized_state = json.dumps(client.calls[0]["state"])
        self.assertNotIn("/home/user/project", serialized_state)

    async def test_actual_absolute_path_omitted_when_no_repository(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        # Actual absolute path with no repository cannot be safely relativized; must be omitted
        await process_search_output(
            pattern="find_func",
            stdout="match 1\n",
            stderr="",
            exit_code=0,
            path="/home/user/project/src/subfolder",
            repository=None,
            client=client,
        )
        tool_input = client.calls[0]["state"]["tool_input"]
        self.assertNotIn("path", tool_input)
        serialized_state = json.dumps(client.calls[0]["state"])
        self.assertNotIn("/home/user/project", serialized_state)

    async def test_actual_absolute_path_omitted_when_outside_repository(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        # Actual absolute path outside repository cannot be relativized; must be omitted
        await process_search_output(
            pattern="find_func",
            stdout="match 1\n",
            stderr="",
            exit_code=0,
            path="/etc/passwd",
            repository="/home/user/project",
            client=client,
        )
        tool_input = client.calls[0]["state"]["tool_input"]
        self.assertNotIn("path", tool_input)
        serialized_state = json.dumps(client.calls[0]["state"])
        self.assertNotIn("/etc/passwd", serialized_state)

    async def test_actual_absolute_flags_sanitized(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        await process_search_output(
            pattern="find_func",
            stdout="match 1\n",
            stderr="",
            exit_code=0,
            flags=["--include-dir", "/home/user/project/tests", "/var/log/syslog"],
            repository="/home/user/project",
            client=client,
        )
        tool_input = client.calls[0]["state"]["tool_input"]
        self.assertEqual(tool_input["flags"], ["--include-dir", "tests"])
        serialized_state = json.dumps(client.calls[0]["state"])
        self.assertNotIn("/home/user/project", serialized_state)
        self.assertNotIn("/var/log", serialized_state)

    async def test_equals_form_flag_inside_repo_relativized(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        await process_search_output(
            pattern="find_func",
            stdout="match 1\n",
            stderr="",
            exit_code=0,
            flags=["--file=/home/user/project/src/lib.py"],
            repository="/home/user/project",
            client=client,
        )
        tool_input = client.calls[0]["state"]["tool_input"]
        self.assertEqual(tool_input["flags"], ["--file=src/lib.py"])
        serialized_state = json.dumps(client.calls[0]["state"])
        self.assertNotIn("/home/user/project", serialized_state)

    async def test_equals_form_flag_outside_repo_omitted(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        await process_search_output(
            pattern="find_func",
            stdout="match 1\n",
            stderr="",
            exit_code=0,
            flags=["--file=/etc/passwd"],
            repository="/home/user/project",
            client=client,
        )
        tool_input = client.calls[0]["state"]["tool_input"]
        self.assertEqual(tool_input.get("flags", []), [])
        serialized_state = json.dumps(client.calls[0]["state"])
        self.assertNotIn("/etc/passwd", serialized_state)

    async def test_equals_form_flag_no_repo_omitted(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        await process_search_output(
            pattern="find_func",
            stdout="match 1\n",
            stderr="",
            exit_code=0,
            flags=["--file=/home/user/project/src/lib.py"],
            repository=None,
            client=client,
        )
        tool_input = client.calls[0]["state"]["tool_input"]
        self.assertEqual(tool_input.get("flags", []), [])
        serialized_state = json.dumps(client.calls[0]["state"])
        self.assertNotIn("/home/user/project", serialized_state)

    async def test_equals_form_flag_relative_preserved(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        await process_search_output(
            pattern="find_func",
            stdout="match 1\n",
            stderr="",
            exit_code=0,
            flags=["--file=src/lib.py", "--exclude=tests/fixtures"],
            repository="/home/user/project",
            client=client,
        )
        tool_input = client.calls[0]["state"]["tool_input"]
        self.assertEqual(tool_input["flags"], ["--file=src/lib.py", "--exclude=tests/fixtures"])

    async def test_archive_path_never_enters_state(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = JevConfig(api_key="test", archive_dir=Path(tmp_dir))
            large_text = "a\n" * 10_001
            # 1 classification response, 1 retention response
            chunks_q = {f"result:{i}": "reference" for i in range(1, 201)}
            chunks_noul = {f"result:{i}": 1.0 for i in range(1, 201)}
            client = FakeTypeSafeClient(
                [
                    make_choice_response(chunks_q),
                    make_noul_response(chunks_noul),
                ]
            )
            await process_retrieval_output(
                query="large query",
                result=large_text,
                client=client,
                config=cfg,
            )
            for call in client.calls:
                serialized = json.dumps(call["state"])
                self.assertNotIn(tmp_dir, serialized)

    async def test_safe_tool_inputs_included(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        await process_search_output(
            pattern="target_pattern",
            stdout="out\n",
            stderr="",
            exit_code=0,
            path="relative/path.py",
            flags=["--count", "-i"],
            client=client,
        )
        state = client.calls[0]["state"]
        self.assertEqual(state["tool"], "codebase_search")
        self.assertEqual(state["tool_input"]["pattern"], "target_pattern")
        self.assertEqual(state["tool_input"]["path"], "relative/path.py")
        self.assertEqual(state["tool_input"]["flags"], ["--count", "-i"])


class TestThresholdLifecycle(unittest.IsolatedAsyncioTestCase):
    """Task 4: Small output vs large output lifecycle."""

    async def test_exactly_10000_tokens_classification_only(self):
        from auggie_mcp.output_processing import process_retrieval_output

        text = "a\n" * 10_000
        client = FakeTypeSafeClient()
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
        )
        self.assertEqual(env["processing"]["estimated_tokens"], 10_000)
        self.assertEqual(env["processing"]["status"], "classified")
        self.assertEqual(env["processing"]["retention"], "below_threshold")
        self.assertIsNone(env["processing"]["recovery_path"])
        # Should only be classification calls, never retention
        for call in client.calls:
            first_q = next(iter(call["questions"].values()))
            self.assertEqual(first_q.type, "choice")

    async def test_above_10000_tokens_classification_archive_retention(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = JevConfig(api_key="test", archive_dir=Path(tmp_dir))
            text = "a\n" * 10_001
            client = FakeTypeSafeClient()
            env = await process_retrieval_output(
                query="test",
                result=text,
                client=client,
                config=cfg,
            )
            self.assertEqual(env["processing"]["estimated_tokens"], 10_001)
            self.assertEqual(env["processing"]["status"], "classified")
            self.assertIsNotNone(env["processing"]["recovery_path"])
            self.assertTrue(Path(env["processing"]["recovery_path"]).exists())
            # Must have issued both Choice and Noul questions
            question_types = {
                q.type for call in client.calls for q in call["questions"].values()
            }
            self.assertIn("choice", question_types)
            self.assertIn("noul", question_types)

    async def test_search_threshold_combines_stdout_and_stderr(self):
        from auggie_mcp.output_processing import JevConfig, process_search_output

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = JevConfig(api_key="test", archive_dir=Path(tmp_dir))
            # 6,000 tokens stdout + 5,000 tokens stderr = 11,000 tokens > 10,000
            stdout_text = "a\n" * 6_000
            stderr_text = "b\n" * 5_000
            client = FakeTypeSafeClient()
            env = await process_search_output(
                pattern="p",
                stdout=stdout_text,
                stderr=stderr_text,
                exit_code=0,
                client=client,
                config=cfg,
            )
            self.assertEqual(env["processing"]["estimated_tokens"], 11_000)
            self.assertIsNotNone(env["processing"]["recovery_path"])


class TestRetentionAndProtections(unittest.IsolatedAsyncioTestCase):
    """Task 5: Retention scoring, threshold floor 0.1, and protection overrides."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.archive_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_p_zero_drops_unprotected_chunk(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                make_noul_response(
                    {
                        "result:1": 0.0,
                        "result:2": 0.0,
                        "result:3": 0.0,
                    }
                ),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        self.assertEqual(env["processing"]["retention"], "pruned")
        chunks = env["processing"]["chunks"]
        self.assertTrue(chunks[0]["retained"], "First chunk must be retained")
        self.assertFalse(chunks[1]["retained"], "Unprotected middle chunk with p=0.0 must be dropped")
        self.assertTrue(chunks[2]["retained"], "Last chunk must be retained")
        self.assertIn("[omitted", env["data"]["result"])

    async def test_p_point_one_drops_unprotected_chunk(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                make_noul_response(
                    {
                        "result:1": 1.0,
                        "result:2": 0.1,  # Exactly 0.1 is <= 0.1, should be dropped
                        "result:3": 1.0,
                    }
                ),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        chunks = env["processing"]["chunks"]
        self.assertFalse(chunks[1]["retained"], "Unprotected chunk with p=0.1 must be dropped")

    async def test_p_above_point_one_retains(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                make_noul_response(
                    {
                        "result:1": 1.0,
                        "result:2": 0.1001,  # > 0.1 must be retained
                        "result:3": 1.0,
                    }
                ),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        chunks = env["processing"]["chunks"]
        self.assertTrue(chunks[1]["retained"], "Chunk with p > 0.1 must be retained")
        self.assertEqual(env["processing"]["retention"], "kept_all")

    async def test_diagnostic_category_retains_even_with_p_zero(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        # Chunk 1 is first chunk (protected), chunk 2 is diagnostic (category protected),
        # chunk 3 is last chunk (protected). All chunks are protected before Noul scoring,
        # so zero Noul retention requests are issued.
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "diagnostic",
                        "result:3": "reference",
                    }
                ),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        self.assertEqual(client.call_count, 1, "Protected chunks must not issue Noul retention calls")
        chunks = env["processing"]["chunks"]
        self.assertTrue(chunks[1]["protected"], "Diagnostic category chunk must be protected")
        self.assertTrue(chunks[1]["retained"], "Diagnostic category chunk must retain")
        self.assertEqual(env["processing"]["retention"], "protected")

    async def test_result_category_retains_even_with_p_zero(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        # All chunks protected: chunk 1 (first), chunk 2 (result), chunk 3 (last)
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "result",
                        "result:3": "reference",
                    }
                ),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        self.assertEqual(client.call_count, 1, "Protected chunks must not issue Noul retention calls")
        chunks = env["processing"]["chunks"]
        self.assertTrue(chunks[1]["protected"], "Result category chunk must be protected")
        self.assertTrue(chunks[1]["retained"], "Result category chunk must retain")
        self.assertEqual(env["processing"]["retention"], "protected")

    async def test_first_and_last_source_chunks_retain_with_p_zero(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                make_noul_response(
                    {
                        "result:2": 1.0,
                    }
                ),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        chunks = env["processing"]["chunks"]
        self.assertTrue(chunks[0]["retained"], "First chunk must retain")
        self.assertTrue(chunks[2]["retained"], "Last chunk must retain")

    async def test_local_protections_retain_with_p_zero(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text(diagnostic_line="FATAL: core dumped at boundary")
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        # Chunk 2 has local diagnostic protection, so all 3 chunks are protected
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        self.assertEqual(client.call_count, 1, "Locally protected chunks must not issue Noul calls")
        chunks = env["processing"]["chunks"]
        self.assertTrue(chunks[1]["protected"])
        self.assertTrue(chunks[1]["retained"])

    async def test_structured_document_remains_intact(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_structured_doc()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        client = FakeTypeSafeClient(
            [
                lambda s, q, **kw: make_choice_response({k: "reference" for k in q}),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        self.assertEqual(client.call_count, 1, "Structured documents must not issue Noul calls")
        chunks = env["processing"]["chunks"]
        self.assertTrue(all(c["protected"] for c in chunks))
        self.assertTrue(all(c["retained"] for c in chunks))

    async def test_protected_chunks_do_not_spend_request_budget(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_nchunk_text(4)
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        # 4 chunks: result:1 (first, protected), result:2 (diagnostic, protected),
        # result:3 (reference, UNPROTECTED), result:4 (last, protected).
        # Only result:3 must be asked a Noul question.
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "diagnostic",
                        "result:3": "reference",
                        "result:4": "reference",
                    }
                ),
                make_noul_response({"result:3": 0.0}),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        self.assertEqual(client.call_count, 2)
        noul_call = client.calls[1]
        self.assertEqual(
            set(noul_call["questions"].keys()),
            {"result:3"},
            "Retention questions must only be asked for unprotected chunks",
        )
        chunks = env["processing"]["chunks"]
        self.assertTrue(chunks[0]["protected"])
        self.assertTrue(chunks[0]["retained"])
        self.assertTrue(chunks[1]["protected"])
        self.assertTrue(chunks[1]["retained"])
        self.assertFalse(chunks[2]["protected"])
        self.assertFalse(chunks[2]["retained"], "Unprotected chunk with p=0.0 must be dropped")
        self.assertTrue(chunks[3]["protected"])
        self.assertEqual(env["processing"]["retention"], "pruned")
        self.assertIn("[omitted", env["data"]["result"])

    async def test_unknown_not_automatically_dropped(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "unknown",
                        "result:3": "reference",
                    }
                ),
                make_noul_response(
                    {
                        "result:1": 1.0,
                        "result:2": 0.5,
                        "result:3": 1.0,
                    }
                ),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        chunks = env["processing"]["chunks"]
        self.assertTrue(chunks[1]["retained"])

    async def test_progress_not_automatically_dropped(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "progress",
                        "result:3": "reference",
                    }
                ),
                make_noul_response(
                    {
                        "result:1": 1.0,
                        "result:2": 0.5,
                        "result:3": 1.0,
                    }
                ),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        chunks = env["processing"]["chunks"]
        self.assertTrue(chunks[1]["retained"])

    def test_unscored_chunk_retained(self):
        from auggie_mcp.output_processing import Chunk, apply_retention_decisions

        chunks = [
            Chunk(id="c1", source="s", start_line=1, end_line=1, category="reference"),
            Chunk(id="c2", source="s", start_line=2, end_line=2, category="reference"),
        ]
        # Only c1 is scored; c2 is unscored
        apply_retention_decisions(chunks, {"c1": 0.05})
        self.assertFalse(chunks[0].retained)
        self.assertTrue(chunks[1].retained, "Unscored chunk must be retained")


class TestValidation(unittest.IsolatedAsyncioTestCase):
    """Task 6: Strict validation of Jev Noul answers."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.archive_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    async def _run_with_noul_val(self, noul_val: Any) -> Any:
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        ans = {
            "result:1": NoulAnswer(noul=1.0),
            "result:2": NoulAnswer(noul=noul_val),
            "result:3": NoulAnswer(noul=1.0),
        }
        resp = SystemOneResponse(
            model="jev-latest",
            usage=Usage(input_tokens=10, output_tokens=5),
            answers=ans,
        )
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                resp,
            ]
        )
        return await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )

    async def test_nan_probability_fails_open(self):
        env = await self._run_with_noul_val(float("nan"))
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_infinite_probability_fails_open(self):
        env = await self._run_with_noul_val(float("inf"))
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

        env2 = await self._run_with_noul_val(float("-inf"))
        self.assertEqual(env2["processing"]["status"], "unclassified")
        self.assertEqual(env2["processing"]["retention"], "failed_open")

    async def test_negative_probability_fails_open(self):
        env = await self._run_with_noul_val(-0.01)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_greater_than_one_probability_fails_open(self):
        env = await self._run_with_noul_val(1.01)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_missing_noul_score_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        # Omit result:2 in Noul response
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                make_noul_response(
                    {
                        "result:1": 1.0,
                        "result:3": 1.0,
                    }
                ),
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_malformed_noul_response_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        ans = {"result:1": ChoiceAnswer(choice="reference", confidence=1.0, probabilities={"reference": 1.0})}
        resp = SystemOneResponse(
            model="jev-latest",
            usage=Usage(input_tokens=10, output_tokens=5),
            answers=ans,
        )
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                resp,
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_noul_lookalike_rejected_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output, validate_noul_response

        class NoulLookalike:
            def __init__(self, noul: float):
                self.noul = noul

        # Direct unit test on validate_noul_response
        fake_resp = type("FakeResp", (), {"nouls": {"result:2": NoulLookalike(0.8)}})()
        with self.assertRaises(ValueError) as ctx:
            validate_noul_response(fake_resp, ["result:2"])
        self.assertIn("not an instance of NoulAnswer", str(ctx.exception))

        # End-to-end integration test
        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                fake_resp,
            ]
        )
        env = await process_retrieval_output(
            query="test",
            result=text,
            client=client,
            config=cfg,
        )
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_batch_size_zero_fails_open_retrieval(self):
        from auggie_mcp.output_processing import process_retrieval_output

        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
        text = "sample result\n"
        env = await process_retrieval_output(query="q", result=text, client=client, batch_size=0)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")
        self.assertTrue(all(c["category"] == "unknown" for c in env["processing"]["chunks"]))
        self.assertEqual(len(client.calls), 0)

    async def test_batch_size_negative_fails_open_retrieval(self):
        from auggie_mcp.output_processing import process_retrieval_output

        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
        text = "sample result\n"
        env = await process_retrieval_output(query="q", result=text, client=client, batch_size=-5)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")
        self.assertTrue(all(c["category"] == "unknown" for c in env["processing"]["chunks"]))
        self.assertEqual(len(client.calls), 0)

    async def test_batch_size_zero_fails_open_search(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        stdout = "sample stdout\n"
        stderr = "sample stderr\n"
        env = await process_search_output(
            pattern="p", stdout=stdout, stderr=stderr, exit_code=0, client=client, batch_size=0
        )
        self.assertEqual(env["data"]["stdout"], stdout)
        self.assertEqual(env["data"]["stderr"], stderr)
        self.assertEqual(env["data"]["exit_code"], 0)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")
        self.assertTrue(all(c["category"] == "unknown" for c in env["processing"]["chunks"]))
        self.assertEqual(len(client.calls), 0)

    async def test_batch_size_negative_fails_open_search(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        stdout = "sample stdout\n"
        stderr = "sample stderr\n"
        env = await process_search_output(
            pattern="p", stdout=stdout, stderr=stderr, exit_code=0, client=client, batch_size=-10
        )
        self.assertEqual(env["data"]["stdout"], stdout)
        self.assertEqual(env["data"]["stderr"], stderr)
        self.assertEqual(env["data"]["exit_code"], 0)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")
        self.assertTrue(all(c["category"] == "unknown" for c in env["processing"]["chunks"]))
        self.assertEqual(len(client.calls), 0)

    async def test_batch_size_boolean_and_non_int_fails_open(self):
        from auggie_mcp.output_processing import process_retrieval_output

        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
        text = "sample text\n"
        env_bool = await process_retrieval_output(query="q", result=text, client=client, batch_size=True)
        self.assertEqual(env_bool["processing"]["status"], "unclassified")
        self.assertEqual(env_bool["processing"]["retention"], "failed_open")

        env_float = await process_retrieval_output(query="q", result=text, client=client, batch_size=12.5)  # type: ignore[arg-type]
        self.assertEqual(env_float["processing"]["status"], "unclassified")
        self.assertEqual(env_float["processing"]["retention"], "failed_open")


class TestTransactions(unittest.IsolatedAsyncioTestCase):
    """Task 7: Transactional fail-open across multiple batches."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.archive_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_first_classification_batch_succeeds_second_fails(self):
        from auggie_mcp.output_processing import OutputProcessor, process_retrieval_output

        text = "\n".join(f"line {i}" for i in range(1, 41))  # 2 chunks
        client = FakeTypeSafeClient(
            [
                make_choice_response({"result:1": "reference"}),
                TypeSafeError("API down on batch 2"),
            ]
        )
        processor = OutputProcessor(client=client, batch_size=1)
        env = await processor.process_retrieval(query="q", result=text)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")
        for c in env["processing"]["chunks"]:
            self.assertEqual(c["category"], "unknown")

    async def test_classification_succeeds_retention_second_batch_fails(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        text = make_large_nchunk_text(4)
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        client = FakeTypeSafeClient(
            [
                # 4 classification batches of size 1
                make_choice_response({"result:1": "reference"}),
                make_choice_response({"result:2": "reference"}),
                make_choice_response({"result:3": "reference"}),
                make_choice_response({"result:4": "reference"}),
                # Retention batch 1 succeeds for result:2
                make_noul_response({"result:2": 0.0}),
                # Retention batch 2 fails for result:3
                TypeSafeError("Network timeout on retention batch 2"),
            ]
        )
        processor = OutputProcessor(client=client, config=cfg, batch_size=1)
        env = await processor.process_retrieval(query="q", result=text)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")
        self.assertNotIn("[omitted", env["data"]["result"])

    async def test_no_partial_omission_marker_escapes_failed_transaction(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", archive_dir=self.archive_dir)

        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                make_noul_response({"result:1": 0.0}),  # Incomplete Noul response (missing 2)
            ]
        )
        processor = OutputProcessor(client=client, config=cfg)
        env = await processor.process_retrieval(query="q", result=text)
        self.assertEqual(env["data"]["result"], text)
        self.assertNotIn("[omitted", env["data"]["result"])

    async def test_returned_categories_all_become_unknown_on_failed_pass(self):
        from auggie_mcp.output_processing import OutputProcessor

        text = "\n".join(f"line {i}" for i in range(1, 41))
        client = FakeTypeSafeClient(
            [
                make_choice_response({"result:1": "diagnostic"}),
                TypeSafeError("Failure"),
            ]
        )
        processor = OutputProcessor(client=client, batch_size=1)
        env = await processor.process_retrieval(query="q", result=text)
        for c in env["processing"]["chunks"]:
            self.assertEqual(c["category"], "unknown")


class TestRequestBudget(unittest.IsolatedAsyncioTestCase):
    """Task 8: Enforcing per-pass hard request budget."""

    async def test_exactly_at_budget_succeeds(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", max_requests=2)
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                make_noul_response({"result:2": 1.0}),
            ]
        )
        processor = OutputProcessor(client=client, config=cfg)
        env = await processor.process_retrieval(query="q", result=text)
        self.assertEqual(env["processing"]["status"], "classified")
        self.assertEqual(client.call_count, 2)

    async def test_next_request_exceeds_budget_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        text = make_large_3chunk_text()
        # Budget allows 1 request. Classification takes 1, retention for result:2 would exceed budget
        cfg = JevConfig(api_key="test", max_requests=1)
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
            ]
        )
        processor = OutputProcessor(client=client, config=cfg)
        env = await processor.process_retrieval(query="q", result=text)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_no_request_issued_after_exhaustion(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", max_requests=1)
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
            ]
        )
        processor = OutputProcessor(client=client, config=cfg)
        await processor.process_retrieval(query="q", result=text)
        self.assertEqual(client.call_count, 1)


class TestTimeout(unittest.IsolatedAsyncioTestCase):
    """Task 9: Total deadline enforcement."""

    async def test_classification_timeout_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.2)
            return make_choice_response({"result:1": "reference"})

        cfg = JevConfig(api_key="test", timeout_seconds=0.05)
        client = FakeTypeSafeClient([slow_response])
        processor = OutputProcessor(client=client, config=cfg)
        text = "original content\n"
        env = await processor.process_retrieval(query="q", result=text)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_retention_timeout_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        async def slow_retention(*args, **kwargs):
            await asyncio.sleep(0.2)
            return make_noul_response({"result:2": 1.0})

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", timeout_seconds=0.05)
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                slow_retention,
            ]
        )
        processor = OutputProcessor(client=client, config=cfg)
        env = await processor.process_retrieval(query="q", result=text)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_total_deadline_spans_multiple_calls(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        async def medium_delay(*args, **kwargs):
            await asyncio.sleep(0.06)
            return make_choice_response(
                {
                    "result:1": "reference",
                    "result:2": "reference",
                    "result:3": "reference",
                }
            )

        async def medium_delay_2(*args, **kwargs):
            await asyncio.sleep(0.06)
            return make_noul_response({"result:2": 1.0})

        text = make_large_3chunk_text()
        # Total timeout 0.1s. Each call takes 0.06s. Total would be 0.12s > 0.1s.
        cfg = JevConfig(api_key="test", timeout_seconds=0.1)
        client = FakeTypeSafeClient([medium_delay, medium_delay_2])
        processor = OutputProcessor(client=client, config=cfg)
        env = await processor.process_retrieval(query="q", result=text)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_negative_timeout_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
        cfg = JevConfig(api_key="test", timeout_seconds=-5.0)
        processor = OutputProcessor(client=client, config=cfg)
        env = await processor.process_retrieval(query="q", result="content\n")
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_zero_timeout_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
        cfg = JevConfig(api_key="test", timeout_seconds=0.0)
        processor = OutputProcessor(client=client, config=cfg)
        env = await processor.process_retrieval(query="q", result="content\n")
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_nan_timeout_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
        cfg = JevConfig(api_key="test", timeout_seconds=float("nan"))
        processor = OutputProcessor(client=client, config=cfg)
        env = await processor.process_retrieval(query="q", result="content\n")
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_infinite_timeout_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
        cfg = JevConfig(api_key="test", timeout_seconds=float("inf"))
        processor = OutputProcessor(client=client, config=cfg)
        env = await processor.process_retrieval(query="q", result="content\n")
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_search_invalid_timeout_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        client = FakeTypeSafeClient([make_choice_response({"stdout:1": "reference"})])
        cfg = JevConfig(api_key="test", timeout_seconds=-10.0)
        processor = OutputProcessor(client=client, config=cfg)
        env = await processor.process_search(pattern="p", stdout="out\n", stderr="", exit_code=0)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_archive_timeout_fails_open_retrieval(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        def slow_archive(*args, **kwargs):
            time.sleep(0.2)
            return Path("/tmp/never_reached.json")

        text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", timeout_seconds=0.05)
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
            ]
        )
        processor = OutputProcessor(client=client, config=cfg)
        with patch("auggie_mcp.output_processing.create_archive", side_effect=slow_archive):
            env = await processor.process_retrieval(query="q", result=text)

        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")
        self.assertIsNone(env["processing"]["recovery_path"])

    async def test_archive_timeout_fails_open_search(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        def slow_archive(*args, **kwargs):
            time.sleep(0.2)
            return Path("/tmp/never_reached.json")

        stdout_text = make_large_3chunk_text()
        cfg = JevConfig(api_key="test", timeout_seconds=0.05)
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "stdout:1": "reference",
                        "stdout:2": "reference",
                        "stdout:3": "reference",
                    }
                ),
            ]
        )
        processor = OutputProcessor(client=client, config=cfg)
        with patch("auggie_mcp.output_processing.create_archive", side_effect=slow_archive):
            env = await processor.process_search(
                pattern="p", stdout=stdout_text, stderr="", exit_code=0
            )

        self.assertEqual(env["data"]["stdout"], stdout_text)
        self.assertEqual(env["data"]["stderr"], "")
        self.assertEqual(env["data"]["exit_code"], 0)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")
        self.assertIsNone(env["processing"]["recovery_path"])


class TestApiKeyAndConfig(unittest.IsolatedAsyncioTestCase):
    """Task 10: Missing credentials and fake client injection."""

    async def test_missing_key_fails_open(self):
        from auggie_mcp.output_processing import OutputProcessor

        with patch.dict(os.environ, {}, clear=True):
            # No client injected and no TYPESAFE_API_KEY
            processor = OutputProcessor(client=None)
            text = "untouched content\n"
            env = await processor.process_retrieval(query="q", result=text)
            self.assertEqual(env["data"]["result"], text)
            self.assertEqual(env["processing"]["status"], "unclassified")
            self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_output_processor_without_client_fails_open_without_creating_client(self):
        from auggie_mcp.output_processing import OutputProcessor

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "sk-real-looking-key"}, clear=True):
            with patch("auggie_mcp.output_processing.create_typesafe_client") as mock_factory:
                processor = OutputProcessor(client=None)
                env = await processor.process_retrieval(query="q", result="content\n")
                self.assertEqual(env["processing"]["status"], "unclassified")
                self.assertEqual(env["processing"]["retention"], "failed_open")
                mock_factory.assert_not_called()

    async def test_fake_client_injection_without_credentials(self):
        from auggie_mcp.output_processing import OutputProcessor

        with patch.dict(os.environ, {}, clear=True):
            client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])
            processor = OutputProcessor(client=client)
            env = await processor.process_retrieval(query="q", result="content\n")
            self.assertEqual(env["processing"]["status"], "classified")


class TestCancellation(unittest.IsolatedAsyncioTestCase):
    """Task 11: Caller cancellation propagation."""

    async def test_cancellation_propagates_without_fail_open_envelope(self):
        from auggie_mcp.output_processing import OutputProcessor

        blocker = asyncio.Event()
        entered = asyncio.Event()

        async def blocked_call(*args, **kwargs):
            entered.set()
            await blocker.wait()
            return make_choice_response({"result:1": "reference"})

        client = FakeTypeSafeClient([blocked_call])
        processor = OutputProcessor(client=client)
        task = asyncio.create_task(processor.process_retrieval(query="q", result="content\n"))

        await entered.wait()
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_archive_cancellation_all_protected_large_output_propagates_cancelled_error(self):
        from auggie_mcp.output_processing import OutputProcessor

        archive_started = threading.Event()
        archive_release = threading.Event()

        def slow_archive(*args, **kwargs):
            archive_started.set()
            archive_release.wait(timeout=5.0)
            return Path("/tmp/mock_archive.json")

        text = make_large_3chunk_text()
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "diagnostic",
                        "result:2": "diagnostic",
                        "result:3": "result",
                    }
                ),
            ]
        )
        processor = OutputProcessor(client=client)

        with patch("auggie_mcp.output_processing.create_archive", side_effect=slow_archive):
            task = asyncio.create_task(processor.process_retrieval(query="q", result=text))
            while not archive_started.is_set():
                await asyncio.sleep(0.001)

            task.cancel()
            archive_release.set()

            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_archive_cancellation_unprotected_large_output_propagates_cancelled_error(self):
        from auggie_mcp.output_processing import OutputProcessor

        archive_started = threading.Event()
        archive_release = threading.Event()

        def slow_archive(*args, **kwargs):
            archive_started.set()
            archive_release.wait(timeout=5.0)
            return Path("/tmp/mock_archive.json")

        text = make_large_3chunk_text()
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "result:1": "reference",
                        "result:2": "reference",
                        "result:3": "reference",
                    }
                ),
                make_noul_response({"result:2": 1.0}),
            ]
        )
        processor = OutputProcessor(client=client)

        with patch("auggie_mcp.output_processing.create_archive", side_effect=slow_archive):
            task = asyncio.create_task(processor.process_retrieval(query="q", result=text))
            while not archive_started.is_set():
                await asyncio.sleep(0.001)

            task.cancel()
            archive_release.set()

            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(len(client.calls), 1)

    async def test_archive_cancellation_search_all_protected_propagates_cancelled_error(self):
        from auggie_mcp.output_processing import OutputProcessor

        archive_started = threading.Event()
        archive_release = threading.Event()

        def slow_archive(*args, **kwargs):
            archive_started.set()
            archive_release.wait(timeout=5.0)
            return Path("/tmp/mock_archive.json")

        stdout_text = make_large_3chunk_text()
        client = FakeTypeSafeClient(
            [
                make_choice_response(
                    {
                        "stdout:1": "diagnostic",
                        "stdout:2": "diagnostic",
                        "stdout:3": "result",
                    }
                ),
            ]
        )
        processor = OutputProcessor(client=client)

        with patch("auggie_mcp.output_processing.create_archive", side_effect=slow_archive):
            task = asyncio.create_task(
                processor.process_search(pattern="p", stdout=stdout_text, stderr="", exit_code=0)
            )
            while not archive_started.is_set():
                await asyncio.sleep(0.001)

            task.cancel()
            archive_release.set()

            with self.assertRaises(asyncio.CancelledError):
                await task


class TestArchiveOrdering(unittest.IsolatedAsyncioTestCase):
    """Task 12: Archive is completed before first retention call."""

    async def test_archive_created_before_first_retention_call(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir)
            cfg = JevConfig(api_key="test", archive_dir=archive_path)

            archive_existed_at_retention = False

            async def on_call_check(call_count, state, questions):
                nonlocal archive_existed_at_retention
                first_q = next(iter(questions.values()))
                if hasattr(first_q, "type") and first_q.type == "noul":
                    archives = list(archive_path.glob("*.json"))
                    if archives:
                        archive_existed_at_retention = True

            client = FakeTypeSafeClient(
                [
                    make_choice_response(
                        {
                            "result:1": "reference",
                            "result:2": "reference",
                            "result:3": "reference",
                        }
                    ),
                    make_noul_response({"result:2": 1.0}),
                ]
            )
            client.on_call = on_call_check

            text = make_large_3chunk_text()
            processor = OutputProcessor(client=client, config=cfg)
            env = await processor.process_retrieval(query="q", result=text)
            self.assertEqual(env["processing"]["status"], "classified")
            self.assertTrue(archive_existed_at_retention, "Archive must exist before first Noul request")


class TestArchiveWorkerCleanupAndConcurrency(unittest.IsolatedAsyncioTestCase):
    """Deterministic tests for archive cleanup on timeout/cancellation and bounded worker concurrency."""

    def setUp(self):
        super().setUp()
        from auggie_mcp.output_processing import reset_archive_executor
        reset_archive_executor(wait=True)

    def tearDown(self):
        super().tearDown()
        from auggie_mcp.output_processing import reset_archive_executor
        reset_archive_executor(wait=True)

    async def test_no_archive_remains_after_delayed_worker_completion_on_timeout(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor, create_archive

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_dir = Path(tmp_dir)
            cfg = JevConfig(api_key="test", timeout_seconds=0.05, archive_dir=archive_dir)

            worker_done = threading.Event()
            orig_create = create_archive

            def delayed_archive(tool, data, archive_dir=None):
                time.sleep(0.12)
                res = orig_create(tool, data, archive_dir=archive_dir)
                worker_done.set()
                return res

            client = FakeTypeSafeClient(
                [
                    make_choice_response(
                        {
                            "result:1": "reference",
                            "result:2": "reference",
                            "result:3": "reference",
                        }
                    ),
                ]
            )
            processor = OutputProcessor(client=client, config=cfg)
            text = make_large_3chunk_text()

            with patch("auggie_mcp.output_processing.create_archive", side_effect=delayed_archive):
                env = await processor.process_retrieval(query="q", result=text)

            self.assertEqual(env["data"]["result"], text)
            self.assertEqual(env["processing"]["status"], "unclassified")
            self.assertEqual(env["processing"]["retention"], "failed_open")
            self.assertIsNone(env["processing"]["recovery_path"])

            # Wait for delayed worker to complete
            await asyncio.to_thread(worker_done.wait, 5.0)
            self.assertTrue(worker_done.is_set())

            # Brief check to ensure unlinking is settled
            for _ in range(50):
                if not list(archive_dir.glob("*.json")):
                    break
                await asyncio.sleep(0.01)

            # Assert no unreferenced archive remains on disk
            self.assertEqual(list(archive_dir.glob("*.json")), [])

    async def test_no_archive_remains_after_delayed_worker_completion_on_cancellation(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor, create_archive

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_dir = Path(tmp_dir)
            cfg = JevConfig(api_key="test", archive_dir=archive_dir)

            archive_started = threading.Event()
            archive_release = threading.Event()
            worker_done = threading.Event()
            orig_create = create_archive

            def delayed_archive(tool, data, archive_dir=None):
                archive_started.set()
                archive_release.wait(timeout=5.0)
                res = orig_create(tool, data, archive_dir=archive_dir)
                worker_done.set()
                return res

            client = FakeTypeSafeClient(
                [
                    make_choice_response(
                        {
                            "result:1": "diagnostic",
                            "result:2": "diagnostic",
                            "result:3": "result",
                        }
                    ),
                ]
            )
            processor = OutputProcessor(client=client, config=cfg)
            text = make_large_3chunk_text()

            with patch("auggie_mcp.output_processing.create_archive", side_effect=delayed_archive):
                task = asyncio.create_task(processor.process_retrieval(query="q", result=text))
                while not archive_started.is_set():
                    await asyncio.sleep(0.001)

                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

                archive_release.set()
                await asyncio.to_thread(worker_done.wait, 5.0)
                self.assertTrue(worker_done.is_set())

            for _ in range(50):
                if not list(archive_dir.glob("*.json")):
                    break
                await asyncio.sleep(0.01)

            self.assertEqual(list(archive_dir.glob("*.json")), [])

    async def test_no_archive_remains_when_timeout_occurs_after_archive_completion(self):
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_dir = Path(tmp_dir)
            cfg = JevConfig(api_key="test", timeout_seconds=0.1, archive_dir=archive_dir)

            async def slow_retention(*args, **kwargs):
                await asyncio.sleep(0.2)
                return make_noul_response({"result:2": 1.0})

            client = FakeTypeSafeClient(
                [
                    make_choice_response(
                        {
                            "result:1": "reference",
                            "result:2": "reference",
                            "result:3": "reference",
                        }
                    ),
                    slow_retention,
                ]
            )
            processor = OutputProcessor(client=client, config=cfg)
            text = make_large_3chunk_text()

            env = await processor.process_retrieval(query="q", result=text)

            self.assertEqual(env["data"]["result"], text)
            self.assertEqual(env["processing"]["status"], "unclassified")
            self.assertEqual(env["processing"]["retention"], "failed_open")
            self.assertIsNone(env["processing"]["recovery_path"])

            # Archive was created in step 3, but timeout happened in retention -> archive removed!
            self.assertEqual(list(archive_dir.glob("*.json")), [])

    async def test_archive_work_concurrency_is_bounded(self):
        from auggie_mcp.output_processing import (
            JevConfig,
            OutputProcessor,
            create_archive,
            MAX_ARCHIVE_WORKERS,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_dir = Path(tmp_dir)
            cfg = JevConfig(api_key="test", archive_dir=archive_dir)

            active_workers = 0
            max_active_workers = 0
            lock = threading.Lock()
            orig_create = create_archive

            def tracked_archive(tool, data, archive_dir=None):
                nonlocal active_workers, max_active_workers
                with lock:
                    active_workers += 1
                    if active_workers > max_active_workers:
                        max_active_workers = active_workers
                time.sleep(0.04)
                try:
                    return orig_create(tool, data, archive_dir=archive_dir)
                finally:
                    with lock:
                        active_workers -= 1

            client = FakeTypeSafeClient(
                [
                    make_choice_response(
                        {
                            "result:1": "diagnostic",
                            "result:2": "diagnostic",
                            "result:3": "result",
                        }
                    )
                ]
                * 8
            )
            processor = OutputProcessor(client=client, config=cfg)
            text = make_large_3chunk_text()

            with patch("auggie_mcp.output_processing.create_archive", side_effect=tracked_archive):
                tasks = [
                    asyncio.create_task(processor.process_retrieval(query=f"q{i}", result=text))
                    for i in range(8)
                ]
                results = await asyncio.gather(*tasks)

            self.assertEqual(len(results), 8)
            for r in results:
                self.assertEqual(r["processing"]["status"], "classified")

            # Concurrency must be bounded by MAX_ARCHIVE_WORKERS
            self.assertLessEqual(max_active_workers, MAX_ARCHIVE_WORKERS)
            self.assertLess(MAX_ARCHIVE_WORKERS, 8)

    async def test_boundary_cancellation_retrieval(self):
        """Worker returns real archive path after final token check, coroutine cancelled before consumption."""
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_dir = Path(tmp_dir)
            cfg = JevConfig(api_key="test", archive_dir=archive_dir)

            gate_entered = asyncio.Event()
            pause_coroutine = asyncio.Event()
            created_paths = []
            orig_wrap_future = asyncio.wrap_future

            def boundary_wrap_future(fut, *, loop=None):
                real_async_fut = orig_wrap_future(fut, loop=loop)

                async def _gate():
                    res = await real_async_fut
                    created_paths.append(res)
                    gate_entered.set()
                    await pause_coroutine.wait()
                    return res

                return asyncio.ensure_future(_gate())

            text = make_large_3chunk_text()
            client = FakeTypeSafeClient(
                [
                    make_choice_response(
                        {
                            "result:1": "reference",
                            "result:2": "reference",
                            "result:3": "reference",
                        }
                    )
                ]
            )
            processor = OutputProcessor(client=client, config=cfg)

            with patch(
                "auggie_mcp.output_processing.asyncio.wrap_future",
                side_effect=boundary_wrap_future,
            ):
                task = asyncio.create_task(
                    processor.process_retrieval(query="q", result=text)
                )
                try:
                    await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

                    self.assertEqual(len(created_paths), 1)
                    self.assertTrue(created_paths[0].exists())

                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                finally:
                    pause_coroutine.set()

            for _ in range(50):
                if not list(archive_dir.glob("*.json")):
                    break
                await asyncio.sleep(0.01)

            self.assertFalse(created_paths[0].exists())
            self.assertEqual(list(archive_dir.glob("*.json")), [])

    async def test_boundary_cancellation_search(self):
        """Search pass: worker returns real archive path after token check, coroutine cancelled before consumption."""
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_dir = Path(tmp_dir)
            cfg = JevConfig(api_key="test", archive_dir=archive_dir)

            gate_entered = asyncio.Event()
            pause_coroutine = asyncio.Event()
            created_paths = []
            orig_wrap_future = asyncio.wrap_future

            def boundary_wrap_future(fut, *, loop=None):
                real_async_fut = orig_wrap_future(fut, loop=loop)

                async def _gate():
                    res = await real_async_fut
                    created_paths.append(res)
                    gate_entered.set()
                    await pause_coroutine.wait()
                    return res

                return asyncio.ensure_future(_gate())

            stdout_text = make_large_3chunk_text()
            client = FakeTypeSafeClient(
                [
                    make_choice_response(
                        {
                            "stdout:1": "reference",
                            "stdout:2": "reference",
                            "stdout:3": "reference",
                        }
                    )
                ]
            )
            processor = OutputProcessor(client=client, config=cfg)

            with patch(
                "auggie_mcp.output_processing.asyncio.wrap_future",
                side_effect=boundary_wrap_future,
            ):
                task = asyncio.create_task(
                    processor.process_search(
                        pattern="p", stdout=stdout_text, stderr="", exit_code=0
                    )
                )
                try:
                    await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

                    self.assertEqual(len(created_paths), 1)
                    self.assertTrue(created_paths[0].exists())

                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                finally:
                    pause_coroutine.set()

            for _ in range(50):
                if not list(archive_dir.glob("*.json")):
                    break
                await asyncio.sleep(0.01)

            self.assertFalse(created_paths[0].exists())
            self.assertEqual(list(archive_dir.glob("*.json")), [])

    async def test_boundary_timeout_retrieval(self):
        """Retrieval pass: worker returns real archive path after token check, timeout strikes before consumption."""
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_dir = Path(tmp_dir)
            cfg = JevConfig(
                api_key="test", timeout_seconds=0.08, archive_dir=archive_dir
            )

            gate_entered = asyncio.Event()
            pause_coroutine = asyncio.Event()
            created_paths = []
            orig_wrap_future = asyncio.wrap_future

            def boundary_wrap_future(fut, *, loop=None):
                real_async_fut = orig_wrap_future(fut, loop=loop)

                async def _gate():
                    res = await real_async_fut
                    created_paths.append(res)
                    gate_entered.set()
                    await pause_coroutine.wait()
                    return res

                return asyncio.ensure_future(_gate())

            text = make_large_3chunk_text()
            client = FakeTypeSafeClient(
                [
                    make_choice_response(
                        {
                            "result:1": "reference",
                            "result:2": "reference",
                            "result:3": "reference",
                        }
                    )
                ]
            )
            processor = OutputProcessor(client=client, config=cfg)

            with patch(
                "auggie_mcp.output_processing.asyncio.wrap_future",
                side_effect=boundary_wrap_future,
            ):
                task = asyncio.create_task(
                    processor.process_retrieval(query="q", result=text)
                )
                try:
                    await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

                    self.assertEqual(len(created_paths), 1)
                    self.assertTrue(created_paths[0].exists())

                    env = await asyncio.wait_for(task, timeout=5.0)
                finally:
                    pause_coroutine.set()

            self.assertEqual(env["data"]["result"], text)
            self.assertEqual(env["processing"]["status"], "unclassified")
            self.assertEqual(env["processing"]["retention"], "failed_open")
            self.assertIsNone(env["processing"]["recovery_path"])

            for _ in range(50):
                if not list(archive_dir.glob("*.json")):
                    break
                await asyncio.sleep(0.01)

            self.assertFalse(created_paths[0].exists())
            self.assertEqual(list(archive_dir.glob("*.json")), [])

    async def test_boundary_timeout_search(self):
        """Search pass: worker returns real archive path after token check, timeout strikes before consumption."""
        from auggie_mcp.output_processing import JevConfig, OutputProcessor

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_dir = Path(tmp_dir)
            cfg = JevConfig(
                api_key="test", timeout_seconds=0.08, archive_dir=archive_dir
            )

            gate_entered = asyncio.Event()
            pause_coroutine = asyncio.Event()
            created_paths = []
            orig_wrap_future = asyncio.wrap_future

            def boundary_wrap_future(fut, *, loop=None):
                real_async_fut = orig_wrap_future(fut, loop=loop)

                async def _gate():
                    res = await real_async_fut
                    created_paths.append(res)
                    gate_entered.set()
                    await pause_coroutine.wait()
                    return res

                return asyncio.ensure_future(_gate())

            stdout_text = make_large_3chunk_text()
            client = FakeTypeSafeClient(
                [
                    make_choice_response(
                        {
                            "stdout:1": "reference",
                            "stdout:2": "reference",
                            "stdout:3": "reference",
                        }
                    )
                ]
            )
            processor = OutputProcessor(client=client, config=cfg)

            with patch(
                "auggie_mcp.output_processing.asyncio.wrap_future",
                side_effect=boundary_wrap_future,
            ):
                task = asyncio.create_task(
                    processor.process_search(
                        pattern="p", stdout=stdout_text, stderr="", exit_code=0
                    )
                )
                try:
                    await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

                    self.assertEqual(len(created_paths), 1)
                    self.assertTrue(created_paths[0].exists())

                    env = await asyncio.wait_for(task, timeout=5.0)
                finally:
                    pause_coroutine.set()

            self.assertEqual(env["data"]["stdout"], stdout_text)
            self.assertEqual(env["processing"]["status"], "unclassified")
            self.assertEqual(env["processing"]["retention"], "failed_open")
            self.assertIsNone(env["processing"]["recovery_path"])

            for _ in range(50):
                if not list(archive_dir.glob("*.json")):
                    break
                await asyncio.sleep(0.01)

            self.assertFalse(created_paths[0].exists())
            self.assertEqual(list(archive_dir.glob("*.json")), [])


class TestRetryAndClientConfig(unittest.TestCase):
    """Task 13: Production client construction helper disables retries."""

    def test_client_creation_helper_disables_retries(self):
        from auggie_mcp.output_processing import JevConfig, create_typesafe_client

        cfg = JevConfig(api_key="sk-test-key-123")
        with patch("auggie_mcp.output_processing.AsyncTypeSafeClient") as mock_client_cls:
            create_typesafe_client(cfg)
            mock_client_cls.assert_called_once_with(
                api_key=cfg.api_key,
                model=cfg.model,
                retry=RetryPolicy(max_retries=0),
                timeout=cfg.timeout_seconds,
            )


class TestAdditionalFailureModes(unittest.IsolatedAsyncioTestCase):
    """Additional failure modes for failure matrix completeness."""

    async def test_archive_failure_fails_open(self):
        from auggie_mcp.output_processing import JevConfig, process_retrieval_output

        text = make_large_1chunk_text()
        client = FakeTypeSafeClient([make_choice_response({"result:1": "reference"})])

        with patch("auggie_mcp.output_processing.create_archive", side_effect=OSError("Disk full")):
            env = await process_retrieval_output(
                query="q",
                result=text,
                client=client,
                config=JevConfig(api_key="test"),
            )
            self.assertEqual(env["data"]["result"], text)
            self.assertEqual(env["processing"]["status"], "unclassified")
            self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_sdk_error_fails_open_retrieval(self):
        from auggie_mcp.output_processing import process_retrieval_output

        client = FakeTypeSafeClient([TypeSafeError("Server 500 error")])
        text = "original text\n"
        env = await process_retrieval_output(query="q", result=text, client=client)
        self.assertEqual(env["data"]["result"], text)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")

    async def test_sdk_error_fails_open_search(self):
        from auggie_mcp.output_processing import process_search_output

        client = FakeTypeSafeClient([TypeSafeError("Server 500 error")])
        stdout = "original stdout\n"
        stderr = "original stderr\n"
        env = await process_search_output(
            pattern="q", stdout=stdout, stderr=stderr, exit_code=0, client=client
        )
        self.assertEqual(env["data"]["stdout"], stdout)
        self.assertEqual(env["data"]["stderr"], stderr)
        self.assertEqual(env["processing"]["status"], "unclassified")
        self.assertEqual(env["processing"]["retention"], "failed_open")


if __name__ == "__main__":
    unittest.main()
