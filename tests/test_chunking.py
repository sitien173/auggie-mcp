"""Tests for auggie_mcp.chunking."""

import unittest

from auggie_mcp import chunking


class ChunkingTests(unittest.TestCase):
    def test_empty_file_yields_no_chunks(self):
        self.assertEqual(chunking.chunk_file("empty.py", ""), [])

    def test_covers_every_line_and_prepends_path(self):
        lines = [f"line {i}" for i in range(1, 80)]
        content = "\n".join(lines) + "\n"
        chunks = chunking.chunk_file("src/mod.py", content, max_lines=30, overlap_lines=10)
        self.assertGreater(len(chunks), 1)

        covered = set()
        for idx, ch in enumerate(chunks):
            self.assertEqual(ch.ordinal, idx)
            self.assertLessEqual(ch.start_line, ch.end_line)
            self.assertTrue(ch.text.startswith("src/mod.py\n"))
            covered.update(range(ch.start_line, ch.end_line + 1))

        self.assertEqual(covered, set(range(1, len(lines) + 1)))

    def test_deterministic_output(self):
        content = "\n".join(f"row {i}" for i in range(1, 50))
        first = chunking.chunk_file("a.py", content, max_lines=20, overlap_lines=5)
        second = chunking.chunk_file("a.py", content, max_lines=20, overlap_lines=5)
        self.assertEqual(first, second)

    def test_snaps_end_to_nearest_blank_line_in_window(self):
        lines = [f"code {i}" for i in range(1, 26)]
        lines[14] = ""  # 1-based line 15 is blank inside 1..20 window
        content = "\n".join(lines) + "\n"
        chunks = chunking.chunk_file("snap.py", content, max_lines=20, overlap_lines=5)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 15)
        self.assertEqual(chunks[-1].end_line, 25)

    def test_chunker_version_constant_exists(self):
        self.assertIsInstance(chunking.CHUNKER_VERSION, int)
