"""Semantic search for ``codebase_search`` mode=semantic."""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError

from auggie.commands._common import find_git_root
from auggie.commands.context import _expand_results
from auggie.db import DatabaseError, SchemaOutdatedError, connect_readonly
from auggie.search import query as query_db
from auggie_mcp import index_tools, rg_runner

TOP_K_RANGE = (1, 50)
CONTEXT_LINES_RANGE = (0, 20)
# Candidates fetched per requested hit when a path filter can discard some.
_OVERFETCH = 5
_MAX_CANDIDATES = 250


def validate_params(mode, *, flags, top_k, context_lines) -> None:
    """Reject unknown modes, out-of-range values, and parameters set for the wrong mode."""
    if mode not in ("exact", "semantic"):
        raise ToolError(
            f"mode must be 'exact' or 'semantic', got {mode!r}. "
            "Use mode='exact' for literals and mode='semantic' for concepts."
        )
    if not TOP_K_RANGE[0] <= top_k <= TOP_K_RANGE[1]:
        raise ToolError(
            f"top_k must be between {TOP_K_RANGE[0]} and {TOP_K_RANGE[1]}, got {top_k}. "
            f"Pass an integer from {TOP_K_RANGE[0]} to {TOP_K_RANGE[1]}."
        )
    if not CONTEXT_LINES_RANGE[0] <= context_lines <= CONTEXT_LINES_RANGE[1]:
        raise ToolError(
            f"context_lines must be between {CONTEXT_LINES_RANGE[0]} and "
            f"{CONTEXT_LINES_RANGE[1]}, got {context_lines}. "
            f"Pass an integer from {CONTEXT_LINES_RANGE[0]} to {CONTEXT_LINES_RANGE[1]}."
        )
    if mode == "semantic" and flags:
        raise ToolError(
            "flags apply only to mode=exact. Remove flags, or call with mode='exact'."
        )
    if mode == "exact" and top_k != 10:
        raise ToolError(
            "top_k applies only to mode=semantic. Remove top_k, or call with mode='semantic'."
        )
    if mode == "exact" and context_lines != 0:
        raise ToolError(
            "context_lines applies only to mode=semantic. "
            "Remove context_lines, or call with mode='semantic'."
        )


@contextmanager
def _index_errors(root, db_path):
    """Map read-only index access failures to ``ToolError`` messages with a next step."""
    try:
        yield
    except SchemaOutdatedError as exc:
        raise ToolError(
            f"Index schema at {db_path} is outdated and searching would migrate it ({exc}). "
            f"Call {index_tools._update_hint(root)} to upgrade it, then search again."
        ) from exc
    except DatabaseError as exc:
        raise ToolError(
            f"Index at {db_path} has an unsupported schema ({exc}). "
            f"Call {index_tools._update_hint(root)[:-1]}, rebuild=True) to rebuild it."
        ) from exc
    except sqlite3.OperationalError as exc:
        raise ToolError(
            f"Index at {db_path} needs an update before it can be searched read-only ({exc}). "
            f"Call {index_tools._update_hint(root)}, then search again."
        ) from exc
    except sqlite3.Error as exc:
        raise ToolError(
            f"Index at {db_path} is unreadable ({exc}). "
            f"Call {index_tools._update_hint(root)[:-1]}, rebuild=True) to rebuild it."
        ) from exc


def _display_path(path: str, root: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def run_semantic(search_root, query: str, path, top_k: int, context_lines: int, *, display_root=None) -> dict:
    """Search the semantic index for ``path`` and return stdout, blocks, stderr, exit_code, and index info.

    ``blocks`` holds the rendered hits that ``stdout`` joins with ``"\\n--\\n"``.

    Blocking and read-only: run it through ``asyncio.to_thread``.
    """
    search_root_real = Path(os.path.realpath(search_root))
    try:
        target = rg_runner.resolve_search_path(search_root, path)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    anchor = target if target.is_dir() else target.parent
    if find_git_root(anchor) is None and (target == search_root_real or search_root_real in target.parents):
        anchor = search_root_real
    root, db_path, branch = index_tools._locate(anchor)
    if not db_path.exists():
        raise ToolError(
            f"No index for {root}. Call {index_tools._update_hint(root)} to build it."
        )
    if not os.environ.get("VOYAGE_API_KEY"):
        raise ToolError(
            "VOYAGE_API_KEY is not set. Set it in the MCP server environment, "
            "then call codebase_search again."
        )

    with _index_errors(root, db_path):
        con = connect_readonly(db_path)
        try:
            model, dim = index_tools._stored_model(con)
            modified, deleted = index_tools.stale_files(con, branch)
            (last_indexed,) = con.execute(
                "SELECT MAX(timestamp) FROM index_history"
            ).fetchone()
        finally:
            con.close()

    subtree = target != root
    try:
        with _index_errors(root, db_path):
            hits, _ = query_db(
                db_path=db_path,
                q=query,
                top_k=min(top_k * _OVERFETCH, _MAX_CANDIDATES) if subtree else top_k,
                model=model,
                dimensions=dim,
                mode=None,
                no_cache=True,
                branch=branch,
                readonly=True,
            )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(
            f"Semantic search failed: {exc}. Check that VOYAGE_API_KEY is valid and the "
            f"network is reachable. Then call index_status; if the index is stale or "
            f"unreadable, call {index_tools._update_hint(root)}. Then retry codebase_search."
        ) from exc

    if subtree:
        hits = [h for h in hits if Path(h.path) == target or target in Path(h.path).parents]
    hits = hits[:top_k]

    display_base = Path(os.path.realpath(display_root)) if display_root else root
    stale_modified = set(modified)
    blocks = []
    for hit in hits:
        item = _expand_results(
            [hit], 0 if hit.path in stale_modified else context_lines
        )[0]
        blocks.append(
            f"{_display_path(hit.path, display_base)}:{item['start_line']}-{item['end_line']} "
            f"score={float(hit.score):.3f}\n{str(item['text']).rstrip(chr(10))}"
        )
    return {
        "stdout": "\n--\n".join(blocks),
        "blocks": blocks,
        "stderr": "",
        "exit_code": 0 if blocks else 1,
        "index": {
            "stale_modified": len(modified),
            "stale_deleted": len(deleted),
            "last_indexed": last_indexed,
        },
    }
