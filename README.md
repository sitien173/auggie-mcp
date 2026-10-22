# Auggie MCP

A Python MCP server exposing semantic codebase retrieval and fast exact repository search with TypeSafe AI (Jev) output processing.

The server uses the MCP Python SDK v2 (`MCPServer`) over standard input/output transport. Tool outputs are processed through an asynchronous TypeSafe AI pipeline that classifies content and prunes low-utility context from oversized responses.

## Installation

Requires Python 3.10+ and `uv`.

```bash
uv sync
```

Dependencies including `mcp` and `typesafe-sdk` install automatically with no manual HTTP client or transport configuration required.

The `auggie-mcp` console command and `python -m auggie_mcp` both start the stdio server. The separate `auggie` console command exposes the retrieval and search runners on the command line.

## CLI

The `auggie` command reuses the underlying runners without MCP server lifecycle or Jev output processing. Output is JSON by default; pass `--human` for text. Standalone CLI output remains identical to previous versions.

```bash
# Semantic retrieval (uses the Pi retrieval runner)
uv run auggie retrieve <repository> <query>

# Exact search in the current directory (uses the tgrep runner)
uv run auggie search <pattern> [tgrep flags...]
```

- `auggie retrieve <repository> <query>` runs the retrieval runner against `repository` and prints `{"response": "..."}`. With `--human` it prints the response text.
- `auggie search <pattern>` runs tgrep in the invocation directory. Any remaining arguments are forwarded unchanged to existing tgrep flag validation. It prints the runner result object `{"stdout": ..., "stderr": ..., "exit_code": ...}`. With `--human` it prints tgrep standard output.
- A search with no matches is a normal result: it emits the JSON result and exits `1`.
- Argument-validation and runner failures emit a JSON error object to standard error and exit `2`.

Command help is available for both commands:

```bash
uv run auggie --help
uv run auggie retrieve --help
uv run auggie search --help
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

## Tools

The server exposes two tools, both returning structured replacement envelopes:

### `codebase_retrieval(repository_path, query, goal=None)`

Runs an ephemeral, noninteractive Pi subprocess in `repository_path` and processes the answer through Jev.

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

### `codebase_search(pattern, path=".", flags=None, goal=None)`

Runs fast regex or literal search using `tgrep`. Relative paths resolve beneath the server startup repository. Absolute paths may target any accessible file or directory.

- `pattern` (`string`, required): Regex or literal search pattern.
- `path` (`string`, optional, default `.`): Repository-relative or absolute search path.
- `flags` (`list[string]`, optional): Supported flags (`-F`, `-t`, `-g`, `-l`, `-C`, etc.).
- `goal` (`string`, optional): Task context guiding retention scoring. If omitted, falls back to `pattern`.

Returns a `SearchEnvelope`:

```json
{
  "tool": "codebase_search",
  "data": {
    "stdout": "...",
    "stderr": "...",
    "exit_code": 0
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

## Behavior

- **Classification:** Every eligible output chunk is classified into one of five categories: `reference`, `diagnostic`, `result`, `progress`, or `unknown`.
- **Pruning Threshold:** Results are eligible for retention pruning only when estimated tokens strictly exceed 10,000 (`> 10,000`). Results with 10,000 or fewer estimated tokens are never pruned (`retention: "below_threshold"`).
- **Protections:** Diagnostics, results, structured documents (JSON, XML), and first/last chunks from each source are always protected against omission regardless of model scoring.
- **Uncertainty Floor:** Only non-protected chunks scored with retention probability <= `0.1` may be omitted. Omitted contiguous blocks render as deterministic omission markers.
- **Fail-Open Semantics:** Any API failure, invalid response, timeout, or missing API key fails open. The server returns exact original runner output marked `status: "unclassified"` and `retention: "failed_open"`. Jev errors never become MCP tool errors.
- **Exit Codes:** Search exit code `1` (no matches found) remains a normal successful MCP call, not a tool error. Operational runner errors (missing executables, path escapes, invalid flags, process crashes) remain MCP tool errors.
- **Index Refresh:** Before every call the server incrementally updates the search indexes so results reflect current files on disk. `codebase_search` runs `tgrep index` on the server startup repository. `codebase_retrieval` runs `tgrep index` and `ogrep index` on `repository_path` before starting Pi. Unchanged files are skipped. A failed or missing indexer is an MCP tool error.
- **Lifespan Management:** A single `AsyncTypeSafeClient` is created on server lifespan startup and shared across concurrent calls with SDK retries disabled. It closes cleanly on server shutdown alongside the bounded archive executor.

## Privacy

- Absolute search paths allow reading any local file or directory accessible to the server process.
- Eligible output text chunks may be transmitted to TypeSafe AI over the network for classification and retention scoring.
- Absolute local repository paths are sanitized and never intentionally sent to TypeSafe AI.
- No local redaction layer is applied before TypeSafe submission; sensitive strings present in command outputs may leave the local host if an API key is configured.

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
```

Live tests exercising external APIs are strictly opt-in and require appropriate credentials:

```bash
# Optional live Pi test
AUGGIE_LIVE_PI=1 uv run python -m unittest tests.test_live_pi -v
```
