# Auggie MCP

A Python MCP server for exact repository search, semantic code search, index maintenance, and read-only codebase retrieval. The server exposes five tools over stdio and stores embeddings in one SQLite file per repository.

## Install

Requires Python 3.12+, `uv`, and ripgrep (`rg`) on `PATH`.

```bash
uv sync
uv run auggie-mcp
```

`python -m auggie_mcp` starts the same MCP server. Set `VOYAGE_API_KEY` to build an index or run semantic search. Configure an agent driver to use `codebase_retrieval`.

## MCP tools

Every tool accepts `repository_path`, except that `index_update` also accepts `rebuild`. Results contain structured content and a text rendering.

| Tool | Parameters | Result shape |
|---|---|---|
| `grep_search` | `repository_path`, `pattern`, `path="."`, `flags=[]`, `max_results=200` | `matches`: `{path, line, text}` items; `truncated`; `archive_path` |
| `semantic_search` | `repository_path`, `query`, `path="."`, `top_k=10` (1-50), `context_lines=0` (0-20) | `hits`: `{path, start_line, end_line, score, stale, text}` items; `index`: `{generation, last_indexed}`; truncation metadata |
| `index_status` | `repository_path` | `{exists, model_key, files, chunks, embeddings, stale, next_step}` |
| `index_update` | `repository_path`, `rebuild=false` | `{files_scanned, files_changed, files_deleted, chunks_embedded, chunks_reused, duration_ms}` |
| `codebase_retrieval` | `repository_path`, `query` | `{answer, driver, duration_ms, repo_modified}` plus truncation metadata |

`grep_search` runs ripgrep and does not require an index. `semantic_search` reads the index only and returns an empty hit list for a search path outside the repository. `index_status` uses file metadata and does not hash files. Only `index_update` writes the index. Tool errors use these codes: `INVALID_PATH`, `INVALID_ARGUMENT`, `INDEX_MISSING`, `INDEX_MODEL_MISMATCH`, `INDEX_BUSY`, `MISSING_API_KEY`, `EMBED_FAILED`, `TIMEOUT`, `RETRIEVAL_BUSY`, `REPO_MODIFIED`, `BINARY_NOT_FOUND`, and `INTERNAL_ERROR`.

### Retrieval safety and limits

`codebase_retrieval` asks a configured agent CLI to inspect files without editing them. The server applies driver-specific read-only flags and compares git status before and after the run. The git check does not cover ignored files or paths outside the repository. The `agy` CLI has no MCP-disable flag, so that driver cannot be prevented from accessing MCP servers through a CLI flag. Treat the git check as a guard, not an operating-system sandbox.

## CLI

The `auggie` command provides three index-operations commands:

```bash
uv run auggie index <repository> [--rebuild] [--json]
uv run auggie status <repository>
uv run auggie stats
```

`index` creates or refreshes the repository index. `status` reports index state. `stats` reports local telemetry counts.

## Configuration

Configuration loads non-empty environment values first, then `~/.auggie/config.toml`, then defaults. Removed keys are ignored with one startup warning per key. Unknown current keys and invalid values stop startup.

| TOML section | Key | Environment variable | Default |
|---|---|---|---|
| `[embedding]` | `model` | `AUGGIE_MODEL` | `voyage-code-3` |
| `[embedding]` | `dimensions` | `AUGGIE_DIMENSIONS` | `1024` |
| `[embedding]` | `batch_size` | `AUGGIE_BATCH_SIZE` | `128` |
| `[voyage]` | `timeout_s` | `AUGGIE_VOYAGE_TIMEOUT` | `120.0` |
| `[voyage]` | `retries` | `AUGGIE_VOYAGE_RETRIES` | `5` |
| `[chunking]` | `lines` | `AUGGIE_CHUNK_LINES` | `30` |
| `[chunking]` | `overlap_lines` | `AUGGIE_OVERLAP_LINES` | `15` |
| `[index]` | `timeout_s` | `AUGGIE_INDEX_TIMEOUT_S` | `1800.0` |
| `[telemetry]` | `enabled` | `AUGGIE_TELEMETRY` | `true` |
| `[telemetry]` | `dir` | `AUGGIE_TELEMETRY_DIR` | `~/.local/state/auggie-mcp/telemetry` or `$XDG_STATE_HOME/auggie-mcp/telemetry` |
| `[telemetry]` | `retention_days` | `AUGGIE_TELEMETRY_RETENTION_DAYS` | `30` |
| `[retrieval]` | `max_concurrent` | `AUGGIE_RETRIEVAL_MAX_CONCURRENT` | `2` |
| `[agent]` | `driver` | `AUGGIE_AGENT_DRIVER` | required for retrieval |
| `[agent]` | `model` | `AUGGIE_AGENT_MODEL` | driver default |
| `[agent]` | `reasoning` | `AUGGIE_AGENT_REASONING` | driver default |
| `[agent]` | `args` | `AUGGIE_AGENT_ARGS` | `[]` |
| `[agent]` | `executable` | `AUGGIE_AGENT_EXECUTABLE` | driver executable |
| `[agent]` | `timeout_s` | `AUGGIE_AGENT_TIMEOUT_S` | `1800.0` |

Example:

```toml
[embedding]
model = "voyage-code-3"
dimensions = 1024

[telemetry]
enabled = false

[agent]
driver = "claude"
timeout_s = 900
```

`VOYAGE_API_KEY` is required for embedding and semantic search. Agent drivers are `pi`, `claude`, `codex`, and `agy`; `driver` has no default.

## Telemetry

Local telemetry events record tool inputs, including queries, search patterns, and repository paths. Events are stored as JSONL files under `~/.local/state/auggie-mcp/telemetry`, or `$XDG_STATE_HOME/auggie-mcp/telemetry` when `XDG_STATE_HOME` is set. Set `AUGGIE_TELEMETRY_DIR` or `[telemetry].dir` to choose another location. Events are retained for 30 days by default; configure `AUGGIE_TELEMETRY_RETENTION_DAYS` or `[telemetry].retention_days` to change the period. Disable telemetry with `AUGGIE_TELEMETRY=0` or `[telemetry] enabled = false`.

## Index migration

The current index is `.auggie/index-v2.sqlite`. It does not reuse earlier index formats, so the first update performs a full re-embed. Older files in `.auggie/` are not deleted automatically. Remove obsolete legacy index files manually after confirming they are no longer needed.

## Development checks

```bash
uv run --frozen pytest tests -q
uv run --frozen ruff check src tests scripts
uv run --frozen ty check src
uv run --frozen python scripts/check_versions.py
```
