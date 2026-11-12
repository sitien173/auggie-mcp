from __future__ import annotations

import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from tempfile import mkdtemp
from unittest import mock

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

    def _git_repo(self, prefix: str, global_config: str = "") -> Path:
        root = Path(mkdtemp(prefix=prefix))
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        config = root.parent / f"{root.name}.gitconfig"
        config.write_text(global_config, encoding="utf-8")
        xdg = root.parent / f"{root.name}.xdg"
        patcher = mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": str(config), "XDG_CONFIG_HOME": str(xdg)})
        patcher.start()
        self.addCleanup(patcher.stop)
        return root

    def test_iter_files_applies_git_info_exclude(self) -> None:
        root = self._git_repo("auggie-gitignore-info-exclude-")
        (root / ".git" / "info").mkdir(exist_ok=True)
        (root / ".git" / "info" / "exclude").write_text(".cache_tool\n", encoding="utf-8")
        (root / "keep.py").write_text("print('keep')\n", encoding="utf-8")
        (root / ".cache_tool").mkdir()
        (root / ".cache_tool" / "data.py").write_text("print('skip')\n", encoding="utf-8")

        files = sorted(path.relative_to(root).as_posix() for path in iter_files(root))

        self.assertEqual(files, ["keep.py"])

    def test_iter_files_applies_global_excludes_file(self) -> None:
        excludes = Path(mkdtemp(prefix="auggie-global-excludes-")) / "ignore"
        excludes.write_text("secret_tool/\n", encoding="utf-8")
        root = self._git_repo("auggie-gitignore-global-", f"[core]\n\texcludesFile = {excludes}\n")
        (root / "keep.py").write_text("print('keep')\n", encoding="utf-8")
        (root / "secret_tool").mkdir()
        (root / "secret_tool" / "data.py").write_text("print('skip')\n", encoding="utf-8")

        files = sorted(path.relative_to(root).as_posix() for path in iter_files(root))

        self.assertEqual(files, ["keep.py"])

    def test_iter_files_gitignore_overrides_info_exclude(self) -> None:
        root = self._git_repo("auggie-gitignore-precedence-")
        (root / ".git" / "info").mkdir(exist_ok=True)
        (root / ".git" / "info" / "exclude").write_text("*.py\n", encoding="utf-8")
        (root / ".gitignore").write_text("!keep.py\n", encoding="utf-8")
        (root / "keep.py").write_text("print('keep')\n", encoding="utf-8")
        (root / "drop.py").write_text("print('drop')\n", encoding="utf-8")

        files = sorted(path.relative_to(root).as_posix() for path in iter_files(root))

        self.assertEqual(files, ["keep.py"])


if __name__ == "__main__":
    unittest.main()
