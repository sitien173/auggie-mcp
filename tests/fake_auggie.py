#!/usr/bin/env python3
"""Deterministic fake auggie executable for tests (``index`` only)."""

import json
import os
import sys


def fake_index(name):
    """Record ``<name> index <root>`` in $FAKE_INDEX_LOG; fail if $FAKE_INDEX_FAIL is name."""
    log = os.environ.get("FAKE_INDEX_LOG")
    if log:
        with open(log, "a") as handle:
            handle.write(json.dumps([name, sys.argv[1:], os.getcwd()]) + "\n")
    if os.environ.get("FAKE_INDEX_FAIL") == name:
        sys.stderr.write(f"{name} index boom\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(fake_index("auggie"))
