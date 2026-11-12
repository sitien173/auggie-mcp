#!/usr/bin/env python3
"""A deterministic fake Pi executable for tests.

Reads the prompt from stdin and emits a successful Pi ``--mode json`` event
stream whose final assistant text is ``TEXT``.
"""

import json
import os
import sys

TEXT = "Fake final answer."


def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def message(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}], "timestamp": 1}


def run():
    prompt = sys.stdin.read()
    emit({"type": "session", "version": 3, "id": "fake-session",
          "timestamp": "2026-01-01T00:00:00Z", "cwd": os.getcwd()})
    emit({"type": "agent_start"})
    emit({"type": "turn_start"})
    emit({"type": "message_end", "message": message("user", prompt)})
    emit({"type": "message_end", "message": message("assistant", TEXT)})
    emit({"type": "turn_end", "message": message("assistant", TEXT), "toolResults": []})
    emit({"type": "agent_end", "messages": []})
    return 0


if __name__ == "__main__":
    sys.exit(run())
