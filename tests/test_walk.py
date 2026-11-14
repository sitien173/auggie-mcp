"""Tests for auggie_mcp.walk."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from auggie_mcp import walk


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


class WalkGitTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.global_excludes = self.root / "global_excludes"
        self.global_excludes.write_text("*.global_skip\n", encoding="utf-8")
        self.global_gitconfig = self.root / "gitconfig"
        self.global_gitconfig.write_text(
            f"[core]\n\texcludesFile = {self.global_excludes}\n", encoding="utf-8"
        )

        env_patch = mock.patch.dict(
            os.environ,
            {
                "GIT_CONFIG_GLOBAL": str(self.global_gitconfig),
                "GIT_CONFIG_NOSYSTEM": "1",
                "HOME": str(self.home),
            },
        )
        env_patch.start()
        self.addCleanup(env_patch.stop)

        self.repo = self.root / "repo"
        self.repo.mkdir()
        _git(self.repo, "init")

    def test_ignore_sources_binary_oversized_untracked_and_symlink(self):
        # Tracked file
        (self.repo / "kept.py").write_text("print('hi')\n", encoding="utf-8")
        # Untracked non-ignored file
        (self.repo / "untracked.py").write_text("x = 1\n", encoding="utf-8")
        # .gitignore
        (self.repo / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
        (self.repo / "a.ignored").write_text("no\n", encoding="utf-8")
        # .git/info/exclude
        (self.repo / ".git" / "info" / "exclude").write_text(
            "secret.txt\n", encoding="utf-8"
        )
        (self.repo / "secret.txt").write_text("no\n", encoding="utf-8")
        # Global excludes (configured in isolated GIT_CONFIG_GLOBAL)
        (self.repo / "b.global_skip").write_text("no\n", encoding="utf-8")
        # .auggieignore: directory with trailing slash, directory without slash,
        # anchored pattern, and negation re-including a file
        (self.repo / ".auggieignore").write_text(
            "auggie_skip/\n"
            "private\n"
            "/only_root.txt\n"
            "*.log\n"
            "!important.log\n",
            encoding="utf-8",
        )
        (self.repo / "auggie_skip").mkdir()
        (self.repo / "auggie_skip" / "x.py").write_text("no\n", encoding="utf-8")
        (self.repo / "private").mkdir()
        (self.repo / "private" / "secret.txt").write_text("no\n", encoding="utf-8")
        (self.repo / "only_root.txt").write_text("no\n", encoding="utf-8")
        (self.repo / "sub").mkdir()
        (self.repo / "sub" / "only_root.txt").write_text("yes\n", encoding="utf-8")
        (self.repo / "app.log").write_text("no\n", encoding="utf-8")
        (self.repo / "important.log").write_text("yes\n", encoding="utf-8")
        # .auggie/ directory
        (self.repo / ".auggie").mkdir()
        (self.repo / ".auggie" / "index-v2.sqlite").write_text("db\n", encoding="utf-8")
        # Binary file (NUL after 8192 text bytes)
        (self.repo / "bin.dat").write_bytes(b"a" * 8192 + b"\x00def")
        # Oversized file (> default 1 MiB)
        (self.repo / "big.txt").write_bytes(b"a" * (walk.DEFAULT_MAX_BYTES + 1))
        # Symlink pointing outside repository
        outside = self.root / "outside.py"
        outside.write_text("outside = True\n", encoding="utf-8")
        (self.repo / "escape.py").symlink_to(outside)

        _git(self.repo, "add", "kept.py")
        _git(
            self.repo,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        )

        found = sorted(walk.walk_repo(self.repo))

        self.assertIn("kept.py", found)
        self.assertIn("untracked.py", found)
        self.assertIn("sub/only_root.txt", found)
        self.assertIn("important.log", found)
        self.assertNotIn("a.ignored", found)
        self.assertNotIn("secret.txt", found)
        self.assertNotIn("b.global_skip", found)
        self.assertNotIn("auggie_skip/x.py", found)
        self.assertNotIn("private/secret.txt", found)
        self.assertNotIn("only_root.txt", found)
        self.assertNotIn("app.log", found)
        self.assertNotIn(".auggie/index-v2.sqlite", found)
        self.assertNotIn("bin.dat", found)
        self.assertNotIn("big.txt", found)
        self.assertNotIn("escape.py", found)

    def test_tracked_file_under_outside_symlink_dir_is_skipped(self):
        sub = self.repo / "sub"
        sub.mkdir()
        (sub / "tracked.txt").write_text("inside\n", encoding="utf-8")
        _git(self.repo, "add", "sub/tracked.txt")
        _git(
            self.repo,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "add tracked",
        )

        outside_dir = self.root / "outside_dir"
        outside_dir.mkdir()
        (outside_dir / "tracked.txt").write_text("outside\n", encoding="utf-8")

        (sub / "tracked.txt").unlink()
        sub.rmdir()
        sub.symlink_to(outside_dir)

        found = sorted(walk.walk_repo(self.repo))
        self.assertNotIn("sub/tracked.txt", found)


class WalkNonGitTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo = self.root / "plain"
        self.repo.mkdir()

    def test_nongit_fallback_honors_gitignore_and_auggieignore(self):
        (self.repo / ".gitignore").write_text(
            "*.log\n!keep.log\nbuild/\nprivate\n/root_only.txt\n", encoding="utf-8"
        )
        (self.repo / ".auggieignore").write_text("*.tmp\n", encoding="utf-8")
        (self.repo / "main.py").write_text("x = 1\n", encoding="utf-8")
        (self.repo / "app.log").write_text("log\n", encoding="utf-8")
        (self.repo / "keep.log").write_text("keep\n", encoding="utf-8")
        (self.repo / "scratch.tmp").write_text("tmp\n", encoding="utf-8")
        (self.repo / "build").mkdir()
        (self.repo / "build" / "out.py").write_text("out\n", encoding="utf-8")
        (self.repo / "private").mkdir()
        (self.repo / "private" / "secret.txt").write_text("no\n", encoding="utf-8")
        (self.repo / "root_only.txt").write_text("no\n", encoding="utf-8")
        (self.repo / "nested").mkdir()
        (self.repo / "nested" / "root_only.txt").write_text("yes\n", encoding="utf-8")
        (self.repo / ".auggie").mkdir()
        (self.repo / ".auggie" / "state").write_text("x\n", encoding="utf-8")
        outside = self.root / "outside.txt"
        outside.write_text("secret\n", encoding="utf-8")
        (self.repo / "link.txt").symlink_to(outside)

        found = sorted(walk.walk_repo(self.repo))
        self.assertIn("main.py", found)
        self.assertIn("keep.log", found)
        self.assertIn("nested/root_only.txt", found)
        self.assertNotIn("app.log", found)
        self.assertNotIn("scratch.tmp", found)
        self.assertNotIn("build/out.py", found)
        self.assertNotIn("private/secret.txt", found)
        self.assertNotIn("root_only.txt", found)
        self.assertNotIn(".auggie/state", found)
        self.assertNotIn("link.txt", found)

    def test_dir_trailing_slash_does_not_exclude_regular_file(self):
        (self.repo / ".auggieignore").write_text("target/\n", encoding="utf-8")
        (self.repo / "target").write_text("regular file\n", encoding="utf-8")
        found = sorted(walk.walk_repo(self.repo))
        self.assertIn("target", found)


class IgnoreSemanticsMatchGitTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.global_gitconfig = self.root / "gitconfig"
        self.global_gitconfig.write_text("", encoding="utf-8")
        env_patch = mock.patch.dict(
            os.environ,
            {
                "GIT_CONFIG_GLOBAL": str(self.global_gitconfig),
                "GIT_CONFIG_NOSYSTEM": "1",
                "HOME": str(self.home),
            },
        )
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def _git_ignored(self, ignore_text: str, rel_paths: list[str]) -> set[str]:
        git_repo = self.root / "git_ref"
        if not git_repo.exists():
            git_repo.mkdir()
            _git(git_repo, "init")
        (git_repo / ".gitignore").write_text(ignore_text, encoding="utf-8")
        for rel in rel_paths:
            p = git_repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x\n", encoding="utf-8")
        proc = subprocess.run(
            ["git", "-C", str(git_repo), "check-ignore", "--no-index", *rel_paths],
            capture_output=True,
            text=True,
            check=False,
        )
        return set(proc.stdout.splitlines())

    def test_negation_under_excluded_parent_matches_git(self):
        ignore_text = "private/\n!private/secret.txt\n"
        paths = ["private/secret.txt", "private/other.txt", "kept.txt"]
        expected_ignored = self._git_ignored(ignore_text, paths)
        self.assertIn("private/secret.txt", expected_ignored)

        plain = self.root / "plain_excluded_parent"
        plain.mkdir()
        (plain / ".auggieignore").write_text(ignore_text, encoding="utf-8")
        for rel in paths:
            p = plain / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x\n", encoding="utf-8")

        found = set(walk.walk_repo(plain))
        expected_kept = set(paths) - expected_ignored
        self.assertEqual(found & set(paths), expected_kept)

    def test_negation_with_non_excluded_parent_matches_git(self):
        ignore_text = "private/*\n!private/secret.txt\n"
        paths = ["private/secret.txt", "private/other.txt", "kept.txt"]
        expected_ignored = self._git_ignored(ignore_text, paths)
        self.assertNotIn("private/secret.txt", expected_ignored)
        self.assertIn("private/other.txt", expected_ignored)

        plain = self.root / "plain_reincluded"
        plain.mkdir()
        (plain / ".auggieignore").write_text(ignore_text, encoding="utf-8")
        for rel in paths:
            p = plain / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x\n", encoding="utf-8")

        found = set(walk.walk_repo(plain))
        expected_kept = set(paths) - expected_ignored
        self.assertEqual(found & set(paths), expected_kept)

    def test_double_star_positions_match_git(self):
        ignore_text = "c**d\na/**/b\n**/foo\nabc/**\n"
        paths = [
            "c/x/d",
            "cxd",
            "a/b",
            "a/x/y/b",
            "foo",
            "d/foo",
            "abc/one",
            "abc/sub/two",
            "keep.txt",
        ]
        expected_ignored = self._git_ignored(ignore_text, paths)
        self.assertNotIn("c/x/d", expected_ignored)
        self.assertIn("cxd", expected_ignored)
        self.assertIn("a/b", expected_ignored)
        self.assertIn("a/x/y/b", expected_ignored)
        self.assertIn("foo", expected_ignored)
        self.assertIn("d/foo", expected_ignored)
        self.assertIn("abc/one", expected_ignored)
        self.assertIn("abc/sub/two", expected_ignored)

        plain = self.root / "plain_globstar"
        plain.mkdir()
        (plain / ".auggieignore").write_text(ignore_text, encoding="utf-8")
        for rel in paths:
            p = plain / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x\n", encoding="utf-8")

        found = set(walk.walk_repo(plain))
        expected_kept = set(paths) - expected_ignored
        self.assertEqual(found & set(paths), expected_kept)


class WalkStatOnlyTests(unittest.TestCase):
    def test_check_binary_false_skips_opening_repository_files(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("print('hi')\n", encoding="utf-8")
        (repo / "b.bin").write_bytes(b"\x00\x01\x02")
        (repo / "big.txt").write_bytes(b"x" * 100)

        with mock.patch.object(walk, "_is_binary", side_effect=AssertionError("must not read files")):
            found = list(walk.walk_repo(repo, max_bytes=50, check_binary=False))

        self.assertEqual(found, ["a.py", "b.bin"])

    def test_unreadable_candidate_is_not_treated_as_binary(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("print('hi')\n", encoding="utf-8")
        (repo / "secret.py").write_text("secret\n", encoding="utf-8")

        real_open = Path.open

        def fail_secret(path_self, *args, **kwargs):
            if path_self.name == "secret.py":
                raise PermissionError("denied")
            return real_open(path_self, *args, **kwargs)

        with mock.patch.object(Path, "open", fail_secret):
            found = list(walk.walk_repo(repo))

        self.assertEqual(found, ["a.py", "secret.py"])
