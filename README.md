# Auggie MCP

A Python MCP server exposing agent-based codebase retrieval, fast exact repository search, and explicit semantic index search with TypeSafe AI (Jev) output processing.

The server uses the MCP Python SDK v2 (`MCPServer`) over standard input/output transport. Tool outputs are processed through an asynchronous TypeSafe AI pipeline that classifies content and prunes low-utility context from oversized responses.

## Installation

Requires Python 3.12+, `uv`, and ripgrep (`rg`) on `PATH`.

```bash
uv sync
```

Dependencies including `mcp` and `typesafe-sdk` install automatically with no manual HTTP client or transport configuration required.

The `auggie-mcp` console command and `python -m auggie_mcp` both start the stdio server. The separate `auggie` console command exposes the retrieval and search runners and the in-tree semantic index commands on the command line.

## CLI

The `auggie` command reuses the underlying runners without MCP server lifecycle or Jev output processing. Output is JSON by default; pass `--human` for text. Standalone CLI output remains identical to previous versions.

```bash
# Semantic retrieval (uses the configured agent CLI)
uv run auggie retrieve <repository> <query>

# Exact search in the current directory (uses the rg runner)
uv run auggie search <pattern> [rg flags...]
```

- `auggie retrieve <repository> <query>` runs the retrieval runner against `repository` and prints `{"response": "..."}`. With `--human` it prints the response text.
- `auggie search <pattern>` runs ripgrep (`rg`) in the invocation directory. Any remaining arguments are forwarded unchanged to existing rg flag validation. It prints the runner result object `{"stdout": ..., "stderr": ..., "exit_code": ...}`. With `--human` it prints rg standard output.
- A search with no matches is a normal result: it emits the JSON result and exits `1`.
- Argument-validation and runner failures emit a JSON error object to standard error and exit `2`.

Command help is available for both commands:

```bash
uv run auggie --help
uv run auggie retrieve --help
uv run auggie search --help
```

### Semantic index commands

The `auggie` command also carries 13 subcommands of the in-tree semantic search: `index`, `query`, `chunk`, `reset`, `reindex`, `clean`, `cache-report`, `delete`, `log`, `status`, `health`, `tune`, and `watch`. They build and query a per-repository semantic index. `uv run auggie <subcommand> --help` shows the flags of each one.

```bash
uv run auggie index <repository>
uv run auggie query "<natural language query>" --repo-root <repository>
```

## MCP configuration

Configure the server in your MCP client:

```json
{
  "mcpServers": {
    "auggie": {
      "command": "auggie-mcp",
      "args": [],
      "env": {
        "TYPESAFE_API_KEY": "your-typesafe-api-key"
      }
    }
  }
}
```

or with the module entry point:

```json
{
  "mcpServers": {
    "auggie": {
      "command": "uv",
      "args": ["run", "python", "-m", "auggie_mcp"],
      "env": {
        "TYPESAFE_API_KEY": "your-typesafe-api-key"
      }
    }
  }
}
```

### Environment variables

- `TYPESAFE_API_KEY`: API key for TypeSafe AI. If unset or empty, the server automatically fails open: runner output returns unchanged and marked unclassified.
- `AUGGIE_JEV_MODEL`: TypeSafe model identifier (default: `jev-latest`).
- `AUGGIE_JEV_TIMEOUT_SECONDS`: Total processing timeout in seconds for classification and retention (default: `30.0`).
- `AUGGIE_JEV_ARCHIVE_DIR`: Custom directory for storing pre-pruning result archives. If unset, defaults to `$XDG_CACHE_HOME/auggie-mcp/jev` or `~/.cache/auggie-mcp/jev`.
- `AUGGIE_JEV_MAX_REQUESTS`: Maximum TypeSafe API requests allowed per processing pass (default: `40`).
- `VOYAGE_API_KEY`: API key for Voyage embeddings and reranking. Required to index and to run semantic queries.
- `AUGGIE_INDEX_TIMEOUT_S`: Timeout in seconds for one `index_update` call (default: `1800`). Must be a positive number.
- `AUGGIE_MODEL`, `AUGGIE_DIMENSIONS`: Embedding model and dimensions. They must match the indexed model.
- `AUGGIE_SEARCH_MODE`: Default search mode: `semantic`, `fulltext`, or `hybrid`.
- `AUGGIE_RERANK_MODEL`, `AUGGIE_RERANK_TOPN`: Voyage rerank model and candidate count.
- `AUGGIE_HYBRID_ALPHA`, `AUGGIE_FUSION_METHOD`, `AUGGIE_RRF_K`, `AUGGIE_ANN_OVERFETCH`: Hybrid ranking controls.
- `AUGGIE_CONFIDENCE_MODE`, `AUGGIE_CONFIDENCE_HIGH`, `AUGGIE_CONFIDENCE_MEDIUM`, `AUGGIE_CONFIDENCE_LOW`, `AUGGIE_RELATIVE_HIGH`, `AUGGIE_RELATIVE_MEDIUM`, `AUGGIE_RELATIVE_LOW`, `AUGGIE_ABSOLUTE_STRONG`, `AUGGIE_ABSOLUTE_EXPECTED`, `AUGGIE_ABSOLUTE_WEAK`: Confidence labelling thresholds.
- `AUGGIE_CACHE_DISABLED`: Set to `1` to bypass all query caches.
- `AUGGIE_CACHE_L1_MAX`, `AUGGIE_CACHE_L1_TTL`, `AUGGIE_CACHE_L2_MAX`, `AUGGIE_CACHE_L3_MAX`: Query cache sizes and time to live.
- `AUGGIE_VOYAGE_TIMEOUT`, `AUGGIE_VOYAGE_RETRIES`: Voyage request timeout and retry count.
- `AUGGIE_CHUNK_LINES`, `AUGGIE_OVERLAP_LINES`: Lines per chunk and overlap lines between chunks. `auggie tune` can save the optimal chunk size as `AUGGIE_CHUNK_LINES` in a `.env` file.
- `AUGGIE_AST_CHUNKING`: Set to `1`, `true`, `yes`, or `on` to enable AST chunking.
- `AUGGIE_BATCH_SIZE`: Maximum texts per Voyage request.
- `AUGGIE_VOYAGE_CHARS_PER_TOKEN`: Characters-per-token ratio used to estimate tokens when batching Voyage requests.
- `AUGGIE_WATCH_DEBOUNCE_MS`: Debounce delay for `auggie watch`.

### Agent CLI configuration

`codebase_retrieval` and `auggie retrieve` run an agent CLI through the [`coding-agent-cli`](https://pypi.org/project/coding-agent-cli/) package. Built-in drivers are `pi`, `claude`, `codex`, and `agy`. Each setting resolves from the `AUGGIE_AGENT_*` environment variable, then the `[agent]` table in `~/.auggie/config.toml`, then the default.

| Key | Environment variable | Default |
|---|---|---|
| `driver` | `AUGGIE_AGENT_DRIVER` | `pi` |
| `model` | `AUGGIE_AGENT_MODEL` | `gpt-6-luna` for `pi`, unset otherwise |
| `reasoning` | `AUGGIE_AGENT_REASONING` | `high` for `pi`, unset otherwise |
| `args` | `AUGGIE_AGENT_ARGS`, split with shell quoting rules | `--approve --no-session --no-extensions --no-skills --no-prompt-templates` for `pi`, empty otherwise |
| `executable` | `AUGGIE_AGENT_EXECUTABLE` | the driver's CLI on `PATH` |
| `timeout_s` | `AUGGIE_AGENT_TIMEOUT_S` | `1800` |

```toml
[agent]
driver = "claude"
args = ["--permission-mode", "bypassPermissions"]
```

The query is sent to the agent unchanged. No system prompt is added. The library adds no permission or approval flags. Pass them through `args` when the agent needs unattended tool use.

## Tools

The server registers four tools. `index_status` and `index_update` manage the semantic index that `codebase_search` reads in `semantic` mode.

| Tool | Purpose | `readOnlyHint` | `destructiveHint` | `idempotentHint` | `openWorldHint` |
|---|---|---|---|---|---|
| `codebase_search` | Exact (ripgrep) or semantic (index) search | `true` | unset | unset | `true` |
| `codebase_retrieval` | Agent-synthesized answer | `false` | unset | unset | `true` |
| `index_status` | Inspect the index without changing it | `true` | unset | `true` | `false` |
| `index_update` | Build, refresh, or rebuild the index | `false` | `true` | unset | `true` |

### Recommended workflow

1. Use `codebase_search` with `mode="exact"` for known literals and symbols. It needs no index.
2. For concept queries, call `index_status`. If the index is missing or stale, call `index_update`, then call `codebase_search` with `mode="semantic"`.
3. Use `codebase_retrieval` only when you need a synthesized answer. It is slow and runs an agent.

### `codebase_search(query, path=".", mode="exact", flags=None, top_k=10, context_lines=0, goal=None)`

Searches the server startup repository in one of two modes.

- `query` (`string`, required): Regex or literal pattern in `exact` mode. Concept query in `semantic` mode.
- `path` (`string`, optional, default `.`): Relative paths resolve beneath the server startup repository. In `exact` mode, absolute paths may target any accessible file or directory. In `semantic` mode, a path below the repository root limits results to that subtree and can return fewer than `top_k` hits.
- `mode` (`"exact"` or `"semantic"`, optional, default `"exact"`).
- `flags` (`list[string]`, optional): ripgrep-compatible flags such as `-F`, `-t`, `-g`, `-l`, `-C`. `exact` mode only.
- `top_k` (`integer`, optional, default `10`, range 1 to 50): Maximum hits. `semantic` mode only.
- `context_lines` (`integer`, optional, default `0`, range 0 to 20): Source lines around each hit. `semantic` mode only.
- `goal` (`string`, optional): Task context guiding retention scoring. If omitted, falls back to `query`.

Setting a nonempty `flags` list with `mode="semantic"`, or `top_k` or `context_lines` to a non-default value with `mode="exact"`, is a tool error. Out-of-range `top_k` or `context_lines` is a tool error.

`semantic` mode is read-only. It never builds or updates the index, never calls `index_update`, and never migrates an index schema. It requires an existing index and `VOYAGE_API_KEY` in the server environment. The index is the one of the Git root containing `path`. Outside any Git repository, a `path` inside the server startup repository uses the startup repository. An absolute `path` outside both uses the target directory, or the parent directory of a target file. Each hit renders as `path:start-end score=<score>` followed by the code, and blocks are separated by `--`. Hits whose file changed since indexing show no extra context lines.

Errors name the next step:

- No index: call `index_update`.
- Index schema outdated, or an SQLite operational error that needs an update before a read-only search: call `index_update`, then search again.
- Index schema unsupported, or any other unreadable-index SQLite error: call `index_update(rebuild=True)`.
- Query failure: check `VOYAGE_API_KEY` and the network, call `index_status`, call `index_update` if the index is stale or unreadable, then retry.

Returns a `SearchEnvelope`. `data.index` is `null` in `exact` mode. In `semantic` mode it holds `stale_modified`, `stale_deleted`, and `last_indexed`, so a stale index is visible without a second call. An empty result is a normal result with `exit_code` 1.

```json
{
  "tool": "codebase_search",
  "data": {
    "stdout": "...",
    "stderr": "...",
    "exit_code": 0,
    "mode": "exact",
    "index": null
  },
  "processing": {
    "status": "classified",
    "retention": "pruned",
    "estimated_tokens": 18421,
    "recovery_path": "/home/user/.cache/auggie-mcp/jev/uuid.json",
    "chunks": []
  }
}
```

### `codebase_retrieval(repository_path, query, goal=None)`

Runs the configured agent CLI noninteractively in `repository_path` and processes the answer through Jev. The call is long running and typically takes several minutes. The agent can edit files, so the tool is not read-only. It reads files on disk. It does not use or update the semantic index.

- `repository_path` (`string`, required): Any accessible directory path.
- `query` (`string`, required): Natural-language codebase query.
- `goal` (`string`, optional): Task context guiding retention scoring. If omitted, falls back to `query`.

Returns a `RetrievalEnvelope`:

```json
{
  "tool": "codebase_retrieval",
  "data": {
    "result": "..."
  },
  "processing": {
    "status": "classified",
    "retention": "below_threshold",
    "estimated_tokens": 1200,
    "recovery_path": null,
    "chunks": []
  }
}
```

### `index_status(repository_path)`

Inspects the semantic index without changing it. `repository_path` resolves to its Git root, else to the directory itself.

Returns `exists`, `model`, `dim`, `files`, `chunks`, `last_indexed`, `stale`, and `next_step`. `stale` is `null` when no index exists. Otherwise it holds `modified`, `deleted`, and `candidate_added`. `candidate_added` counts files the indexer may add and can overcount binary files. An unreadable index is a tool error that names `index_update` with `rebuild=True`.

### `index_update(repository_path, rebuild=False)`

Builds or refreshes the semantic index. The first call builds it. By default the update is incremental and reuses the indexed model. With `rebuild=True` the existing index is deleted first and then rebuilt, so an interrupted rebuild can leave no index, and `index_status` then reports `exists` as `false`.

- Calls Voyage and incurs charges. Charges already incurred are not recovered if the call is cancelled or times out.
- Requires `VOYAGE_API_KEY` in the server environment.
- Fails if another update is running (see Index lock under Behavior).
- Times out after `AUGGIE_INDEX_TIMEOUT_S` seconds (default `1800`). On timeout the indexer is stopped. An interrupted incremental run keeps its progress.
- Returns the indexer JSON plus `duration_ms`, and reports progress at start and finish.

Each repository keeps its index in a `.auggie/` directory at the repository root. Paths listed in a `.auggieignore` file at the repository root are not indexed.

## Migrating from 0.4

- `codebase_search` parameter `pattern` is now `query`.
- `auggie_query` and `auggie_context` are removed. Use `codebase_search` with `mode="semantic"`, and `context_lines` for surrounding lines.
- `auggie_status` is removed. Use `index_status`.
- `codebase_retrieval` no longer indexes. Build or refresh the index with `index_update`.
- Semantic search no longer updates the index. Call `index_status`, then `index_update` when it is missing or stale.

## Behavior

- **Classification:** Every eligible output chunk is classified into one of five categories: `reference`, `diagnostic`, `result`, `progress`, or `unknown`.
- **Pruning Threshold:** Results are eligible for retention pruning only when estimated tokens strictly exceed 10,000 (`> 10,000`). Results with 10,000 or fewer estimated tokens are never pruned (`retention: "below_threshold"`).
- **Protections:** Diagnostics, results, structured documents (JSON, XML), and first/last chunks from each source are always protected against omission regardless of model scoring.
- **Uncertainty Floor:** Only non-protected chunks scored with retention probability <= `0.1` may be omitted. Omitted contiguous blocks render as deterministic omission markers.
- **Fail-Open Semantics:** Any API failure, invalid response, timeout, or missing API key fails open. The server returns exact original runner output marked `status: "unclassified"` and `retention: "failed_open"`. Jev errors never become MCP tool errors.
- **Exit Codes:** Search exit code `1` (no matches found) remains a normal successful MCP call, not a tool error. Operational runner errors (missing executables, path escapes, invalid flags, process crashes) remain MCP tool errors.
- **No implicit indexing:** No tool refreshes the index as a side effect. Only `index_update` writes it. `codebase_retrieval` does not use the index, and `codebase_search` with `mode="semantic"` only reads it.
- **Index lock:** `auggie index`, `auggie reindex`, `auggie watch`, and the MCP `index_update` (which runs the indexer) take an exclusive OS lock on `index.lock` beside the index database, so only one of them writes a given index at a time. A second one fails fast. Other mutating commands, such as `auggie delete` and `auggie reset`, do not take the lock. `index_update` reports this as `index update already running`, and the CLI emits `error_code` `INDEX_BUSY`. The OS releases the lock when the holder exits.
- **Lifespan Management:** A single `AsyncTypeSafeClient` is created on server lifespan startup and shared across concurrent calls with SDK retries disabled. It closes cleanly on server shutdown alongside the bounded archive executor.

## Privacy

- Absolute search paths allow reading any local file or directory accessible to the server process.
- Eligible output text chunks may be transmitted to TypeSafe AI over the network for classification and retention scoring.
- Absolute local repository paths are sanitized and never intentionally sent to TypeSafe AI.
- No local redaction layer is applied before TypeSafe submission; sensitive strings present in command outputs may leave the local host if an API key is configured.

## Accepted risks

- **Writable tools.** The agent CLI is started with no tool allowlist and no system prompt. Its built-in tools, such as shell and file editing, remain available within the CLI's own permission rules and any `args` you configure. A malicious repository or query could cause the agent to modify files or run commands.
- **Unlimited concurrency.** Requests start immediately; there is no concurrency limit or queue. Concurrent calls may exhaust CPU, memory, or model API quota, and concurrent requests may modify the same repository.
- **Model credentials.** The agent CLI uses the caller's configured provider credentials.
- **`auggie` command name collision.** The Augment `auggie` CLI uses the same command name as this package. If it precedes this package on `PATH`, `auggie query` or `auggie chunk` from the agent subprocess resolve to the wrong binary. `index_update` is not affected: it runs the indexer as `python -m auggie` with the server's interpreter.
- **Previous indexes.** Indexes built by the previous external tool are not read. Call `index_update` or run `auggie index` for each repository.

## Telemetry

The server records one local event for every `codebase_retrieval`, `codebase_search`, `index_status`, and `index_update` call. The `auggie retrieve` and `auggie search` commands record events for `codebase_retrieval` and `codebase_search`. Events stay on this machine. The telemetry code makes no network requests.

Each event records:

- The tool input as called. `codebase_retrieval` records `repository_path`, `query`, and `goal`. `codebase_search` records `query`, `mode`, `path`, `flags`, `goal`, and the startup `repository_path`. The `auggie search` command records `pattern`, `path`, `flags`, `goal`, and `repository_path`, with no `mode`. `index_status` records only `repository_path`. `index_update` records `repository_path` and `rebuild`.
- The error type and message, if the call failed.
- Timings and the outcome. `codebase_retrieval` and `codebase_search` also record runner statistics when the runner stage completes. MCP calls to those two tools record output-processing statistics when the processing stage completes; CLI commands perform no output processing. `index_status` and `index_update` record neither.
- For MCP calls, the client name and version.

Events are appended to daily files named `events-YYYY-MM-DD.jsonl`. The directory is the first of these that applies:

1. `AUGGIE_TELEMETRY_DIR`
2. `$XDG_STATE_HOME/auggie-mcp/telemetry`
3. `~/.local/state/auggie-mcp/telemetry`

The directory has mode 0700 and the files have mode 0600. Files older than 30 days are deleted. Set `AUGGIE_TELEMETRY_RETENTION_DAYS` to a positive integer to change the window. Set `AUGGIE_TELEMETRY=0` to turn telemetry off.

Warning: queries, patterns, and paths are stored as written. Do not search for secrets or personal data while telemetry is on.

Summarize the files with `auggie stats`:

```bash
auggie stats                  # last 7 days, text
auggie stats --since 24h      # Nd or Nh, N greater than 0
auggie stats --json           # one JSON object
```

## Archives and Recovery

- Results exceeding the 10,000 estimated token threshold are archived locally before retention scoring begins.
- Archives are stored in `AUGGIE_JEV_ARCHIVE_DIR` (or `$XDG_CACHE_HOME/auggie-mcp/jev` / `~/.cache/auggie-mcp/jev`).
- The archive directory is restricted to permission mode `0700` (owner read/write/execute). Archive files are created exclusively with permission mode `0600` (owner read/write).
- The archive preserves the complete original decoded text produced by the runner. It stores decoded Python strings rather than raw subprocess bytes.
- When retention pruning occurs, `processing.recovery_path` contains the absolute path to the archive file.
- There is no automatic archive deletion or cleanup mechanism. Long-term usage may accumulate disk storage and persist sensitive data until manually cleaned by the user.

## Testing

Default tests run offline without network access or model credentials using deterministic fakes:

```bash
uv run python -m unittest tests.test_server -v
uv run python -m unittest tests.test_output_processing -v
uv run python -m unittest tests.test_cli -v
uv run python -m unittest discover -s tests -v
uv run pytest tests/semantic -q
```
