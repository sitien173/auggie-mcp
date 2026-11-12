from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import mkdtemp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable))

from auggie.indexer import iter_files


class GitIgnoreIndexingTests(unittest.TestCase):
    def test_iter_files_applies_root_and_nested_gitignore(self) -> None:
        root = Path(mkdtemp(prefix="auggie-gitignore-iter-"))
        (root / ".gitignore").write_text("root_ignored.py\n", encoding="utf-8")
        (root / "keep.py").write_text("print('keep')\n", encoding="utf-8")
        (root / "root_ignored.py").write_text("print('skip')\n", encoding="utf-8")
        (root / "nested").mkdir()
        (root / "nested" / ".gitignore").write_text("child_ignored.py\n!child_kept.py\n", encoding="utf-8")
        (root / "nested" / "child_ignored.py").write_text("print('skip')\n", encoding="utf-8")
        (root / "nested" / "child_kept.py").write_text("print('keep')\n", encoding="utf-8")
        (root / "nested" / "grandchild.py").write_text("print('keep')\n", encoding="utf-8")

        files = sorted(path.relative_to(root).as_posix() for path in iter_files(root))

        self.assertEqual(files, ["keep.py", "nested/child_kept.py", "nested/grandchild.py"])

    def test_iter_files_applies_nested_gitignore_with_negation(self) -> None:
        root = Path(mkdtemp(prefix="auggie-gitignore-negation-"))
        (root / "src").mkdir()
        (root / "src" / ".gitignore").write_text("*.py\n!keep.py\n", encoding="utf-8")
        (root / "src" / "drop.py").write_text("print('drop')\n", encoding="utf-8")
        (root / "src" / "keep.py").write_text("print('keep')\n", encoding="utf-8")
        (root / "src" / "note.txt").write_text("note\n", encoding="utf-8")

        files = sorted(path.relative_to(root).as_posix() for path in iter_files(root, include=["*.txt"]))

        self.assertEqual(files, ["src/keep.py", "src/note.txt"])


if __name__ == "__main__":
    unittest.main()
