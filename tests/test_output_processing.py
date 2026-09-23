"""Tests for auggie_mcp.output_processing (Phase 1 deterministic foundation)."""

import concurrent.futures
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestPhase1DependencyAndVersion(unittest.TestCase):
    """Task 1: Project version 0.2.0 and typesafe-sdk dependency."""

    def test_project_version_in_pyproject(self):
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")
        self.assertIn('version = "0.2.0"', content)

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


if __name__ == "__main__":
    unittest.main()
