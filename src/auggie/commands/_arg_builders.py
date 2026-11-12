"""
Argument builder functions for CLI commands.

Extracts common argument patterns to reduce duplication between commands.
"""

from __future__ import annotations

import argparse

from ..models import DEFAULT_MODEL


def add_model_args(parser: argparse.ArgumentParser, for_query: bool = False) -> None:
    """
    Add embedding model arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to.
        for_query: If True, adds note about matching indexed model.
    """
    match_note = " (must match indexed model)" if for_query else ""
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        metavar="MODEL",
        help=f"Embedding model or alias: voyage, voyage-code, voyage-3, voyage-lite{match_note}. "
        f"Default: $AUGGIE_MODEL or {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--dimensions",
        "-d",
        type=int,
        default=None,
        metavar="DIM",
        help="Embedding dimensions. Default: $AUGGIE_DIMENSIONS or model default",
    )


def add_indexing_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common indexing arguments shared by index and reindex commands.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=None,
        help="Lines per chunk (default: model-specific, typically 30 for Voyage models)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlapping lines between chunks (default: model-specific, typically 15)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=2_000_000,
        help="Max file size in bytes (default: 2MB)",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Additional exclude patterns (added to defaults)",
    )
    parser.add_argument(
        "--include",
        "-i",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Include patterns (override default excludes). "
        "Example: -i '*.md' to index markdown files",
    )
    parser.add_argument(
        "--no-ast",
        action="store_false",
        dest="ast",
        default=None,  # None = auto-detect (use AST when available)
        help="Disable AST-aware chunking, use line-based chunking instead. "
        "By default, AST chunking is used when tree-sitter is available "
        "(pip install tree-sitter-<language>).",
    )
