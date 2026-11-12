from __future__ import annotations

import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))


# Several test modules stub `tqdm` with a lambda that returns the iterable. That
# stub returns None for the `with tqdm(total=...)` context-manager usage in the
# indexer's detection phase, which crashes when the host `file` binary is present
# (has_file_command() True). Install a context-manager-capable stub first so it
# wins the per-module `sys.modules.setdefault("tqdm", ...)` calls regardless of
# host environment.
class _StubTqdm:
    def __init__(self, iterable=None, **kwargs):
        self._iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self._iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, *args, **kwargs):
        return None

    def close(self):
        return None


def _stub_tqdm(iterable=None, **kwargs):
    return _StubTqdm(iterable, **kwargs)


sys.modules["tqdm"] = types.SimpleNamespace(tqdm=_stub_tqdm)
