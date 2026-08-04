#!/usr/bin/env python3
"""A deterministic fake Pi executable for tests.

Usage: fake_pi.py [flags]

Flags:
  --fail-at <n>        exit 3 with a stderr message after emitting n JSON lines
  --event-count <n>    exit 0 after emitting n JSON lines
  --stall              emit the session header, then wait for stall.lock to
                       appear in cwd before exiting 0
  --chatty             emit 1500 delta lines then exit 0 (bounded-stderr test)
  --malformed          emit a non-JSON line mid-stream
  --empty              emit agent_start then agent_end with no messages
  --no-final           emit turns but no assistant response
  --garbage            emit only non-JSON text

On success emits a minimal Pi-style JSONL event stream ending in an assistant
text response.
"""

import json
import os
import sys
import time

TEXT = "Fake final answer."


def arg_value(name):
    for i, arg in enumerate(sys.argv):
        if arg == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def user_message(text):
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "timestamp": 1,
    }


def assistant_message(text):
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "timestamp": 1,
    }


def session_header():
    emit(
        {
            "type": "session",
            "version": 3,
            "id": "fake-session",
            "timestamp": "2026-01-01T00:00:00Z",
            "cwd": os.getcwd(),
        }
    )


def emit_full_stream():
    session_header()
    emit({"type": "agent_start"})
    emit({"type": "turn_start"})
    emit({"type": "message_start", "message": user_message("query")})
    emit({"type": "message_end", "message": user_message("query")})
    emit({"type": "message_start", "message": {"role": "assistant", "content": []}})
    emit({"type": "message_end", "message": assistant_message(TEXT)})
    emit({"type": "turn_end", "message": assistant_message(TEXT), "toolResults": []})
    emit({"type": "agent_end", "messages": []})
    emit({"type": "agent_settled"})


def emit_query():
    """Emit the stream with the query argument as the final assistant text.

    run_pi appends the fake's extra_args after the query, so ``--echo-query``
    is the last argv element and the query is the one before it.
    """
    query = sys.argv[sys.argv.index("--echo-query") - 1]
    session_header()
    emit({"type": "agent_start"})
    emit({"type": "turn_start"})
    emit({"type": "message_start", "message": user_message("query")})
    emit({"type": "message_end", "message": user_message("query")})
    emit({"type": "message_start", "message": {"role": "assistant", "content": []}})
    emit({"type": "message_end", "message": assistant_message(query)})
    emit({"type": "turn_end", "message": assistant_message(query), "toolResults": []})
    emit({"type": "agent_end", "messages": []})
    emit({"type": "agent_settled"})


def run():
    if "--echo-query" in sys.argv:
        emit_query()
        return 0

    if "--stderr-payload" in sys.argv:
        payload = arg_value("--stderr-payload")
        session_header()
        emit({"type": "agent_start"})
        sys.stderr.write(payload)
        sys.stderr.flush()
        sys.exit(3)

    if "--child-stall" in sys.argv:
        # spawn a grandchild that explicitly ignores SIGTERM and sleeps; write
        # its PID to cwd so tests can track it directly. The leader exits on
        # SIGTERM (default handler) so the grandchild isolates the SIGKILL
        # escalation path.
        import subprocess
        child_code = (
            "import signal, time;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "time.sleep(300)"
        )
        child = subprocess.Popen([sys.executable, "-c", child_code])
        with open(os.path.join(os.getcwd(), "child.pid"), "w") as fh:
            fh.write(str(child.pid))
        session_header()
        while True:
            if os.path.exists(os.path.join(os.getcwd(), "stall.lock")):
                return 0
            time.sleep(0.05)

    if "--stderr-huge" in sys.argv:
        # write far more than STDERR_LIMIT bytes to stderr, then fail
        sys.stderr.write("z" * 32768)
        sys.stderr.flush()
        sys.exit(3)

    if "--stall" in sys.argv:
        # write our own PID so tests can track the leader directly
        with open(os.path.join(os.getcwd(), "child.pid"), "w") as fh:
            fh.write(str(os.getpid()))
        session_header()
        while True:
            if os.path.exists(os.path.join(os.getcwd(), "stall.lock")):
                return 0
            time.sleep(0.05)

    if "--chatty" in sys.argv:
        for _ in range(1500):
            emit(
                {
                    "type": "message_update",
                    "message": {"role": "assistant", "content": []},
                    "assistantMessageEvent": {"type": "text_delta", "delta": "x" * 100},
                }
            )
        return 0

    if "--malformed" in sys.argv:
        session_header()
        emit({"type": "agent_start"})
        sys.stdout.write("this is not json\n")
        sys.stdout.flush()
        emit({"type": "agent_end", "messages": []})
        return 0

    if "--empty" in sys.argv:
        session_header()
        emit({"type": "agent_start"})
        emit({"type": "agent_end", "messages": []})
        return 0

    if "--no-final" in sys.argv:
        session_header()
        emit({"type": "agent_start"})
        emit({"type": "turn_start"})
        emit({"type": "message_start", "message": user_message("query")})
        emit({"type": "message_end", "message": user_message("query")})
        emit({"type": "turn_end", "message": user_message("query"), "toolResults": []})
        emit({"type": "agent_end", "messages": []})
        emit({"type": "agent_settled"})
        return 0

    if "--garbage" in sys.argv:
        session_header()
        sys.stdout.write("NOT JSON AT ALL\n")
        sys.stdout.flush()
        return 0

    fail_at = arg_value("--fail-at")
    event_count = arg_value("--event-count")
    count = 0

    def limited(payload):
        nonlocal count
        emit(payload)
        count += 1
        if fail_at and count == int(fail_at):
            sys.stderr.write("fake pi failure\n")
            sys.exit(3)
        if event_count and count >= int(event_count):
            sys.exit(0)

    limited({"type": "session", "version": 3, "id": "fake-session",
             "timestamp": "2026-01-01T00:00:00Z", "cwd": os.getcwd()})
    limited({"type": "agent_start"})
    limited({"type": "turn_start"})
    limited({"type": "message_start", "message": user_message("query")})
    limited({"type": "message_end", "message": user_message("query")})
    limited({"type": "message_start", "message": {"role": "assistant", "content": []}})
    limited({"type": "message_end", "message": assistant_message(TEXT)})
    limited({"type": "turn_end", "message": assistant_message(TEXT), "toolResults": []})
    limited({"type": "agent_end", "messages": []})
    limited({"type": "agent_settled"})
    return 0


if __name__ == "__main__":
    sys.exit(run())

