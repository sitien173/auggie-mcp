#!/usr/bin/env python3
"""Deterministic fake ``python -m auggie`` for tests (``index`` and ``reindex``).

Invoked as ``fake_auggie.py <index|reindex> <root> [--json] [--model M] [--dimensions D]``.

Environment:
    FAKE_INDEX_LOG: append ``[verb, argv, cwd]`` as one JSON line.
    FAKE_INDEX_BUSY: report INDEX_BUSY as JSON on stdout and exit 1.
    FAKE_INDEX_FAIL: write ``<verb> index boom`` to stderr and exit 2.
    FAKE_INDEX_SLEEP: seconds to sleep before finishing.
    FAKE_INDEX_PIDFILE: write this pid and a same-group child pid, one per line.
    FAKE_INDEX_LEADER_EXIT: exit right after spawning the child, leaving it in the group.
"""

import json
import os
import subprocess
import sys
import time


def option(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def fake_index(argv):
    verb = argv[0]
    log = os.environ.get("FAKE_INDEX_LOG")
    if log:
        with open(log, "a") as handle:
            handle.write(json.dumps([verb, argv, os.getcwd()]) + "\n")
    pidfile = os.environ.get("FAKE_INDEX_PIDFILE")
    if pidfile:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        with open(pidfile, "w") as handle:
            handle.write(f"{os.getpid()}\n{child.pid}\n")
    if os.environ.get("FAKE_INDEX_LEADER_EXIT"):
        return 0
    sleep = os.environ.get("FAKE_INDEX_SLEEP")
    if sleep:
        time.sleep(float(sleep))
    if os.environ.get("FAKE_INDEX_BUSY"):
        print(
            json.dumps(
                {
                    "error": "index is locked by another auggie process (PID 1)",
                    "error_code": "INDEX_BUSY",
                }
            )
        )
        return 1
    if os.environ.get("FAKE_INDEX_FAIL"):
        sys.stderr.write(f"{verb} index boom\n")
        return 2
    print(
        json.dumps(
            {
                "database": os.path.join(argv[1], ".auggie", "index.sqlite"),
                "model": option(argv, "--model") or "fake-model",
                "files_indexed": 2,
                "files_skipped": 1,
                "files_scanned": 3,
                "chunks_total": 4,
                "chunks_embedded": 4,
                "chunks_reused": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(fake_index(sys.argv[1:]))
