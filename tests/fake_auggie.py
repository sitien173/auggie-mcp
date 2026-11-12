#!/usr/bin/env python3
"""Deterministic fake auggie executable for tests (``index`` only)."""

if __name__ == "__main__":
    from fake_tgrep import fake_index

    raise SystemExit(fake_index("auggie"))
