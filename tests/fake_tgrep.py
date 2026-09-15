#!/usr/bin/env python3
"""Deterministic fake tgrep executable for tests."""

import json
import os
import sys
import time


def main():
    explicit_patterns = [
        sys.argv[index + 1]
        for index, value in enumerate(sys.argv[:-1])
        if value in {"-e", "--regexp"}
    ]
    if explicit_patterns:
        pattern = explicit_patterns[-1]
    else:
        pattern = sys.argv[sys.argv.index("--") + 1]
    if pattern == "stall":
        with open(os.path.join(os.getcwd(), "tgrep.pid"), "w") as handle:
            handle.write(str(os.getpid()))
        time.sleep(300)
        return 0
    if pattern == "nomatch":
        sys.stderr.write("warning: no matches\n")
        return 1
    if pattern == "bad[":
        sys.stderr.write("regex parse error\n")
        return 2

    print(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd()}))
    sys.stderr.write("search warning\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
