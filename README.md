# Auggie MCP

A Python MCP server exposing exactly one tool, `codebase_retrieval`. Each call
runs an ephemeral, noninteractive Pi subprocess against a caller-selected
repository directory and returns Pi's final answer with file-line evidence.

The server uses MCP Python SDK v2 (`MCPServer`) over standard input/output
transport. There are no resources, prompts, extra tools, additional transports,
access controls, or concurrency controls.

## Installation

Requires Python 3.10+ and `uv`.

```bash
uv sync
```

The `auggie-mcp` console command and `python -m auggie_mcp` both start the
stdio server. The separate `auggie` console command exposes the retrieval and
search runners on the command line.

## CLI

The `auggie` command reuses the MCP server's runners without changing server
behavior. Output is JSON by default; pass `--human` for text.

```bash
# Semantic retrieval (uses the Pi retrieval runner)
uv run auggie retrieve <repository> <query>

# Exact search in the current directory (uses the tgrep runner)
uv run auggie search <pattern> [tgrep flags...]
```

- `auggie retrieve <repository> <query>` runs the retrieval runner against
  `repository` and prints `{"response": "..."}`. With `--human` it prints the
  response text.
- `auggie search <pattern>` runs tgrep in the invocation directory. Any
  remaining arguments are forwarded unchanged to the existing tgrep flag
  validation. It prints the runner result object
  `{"stdout": ..., "stderr": ..., "exit_code": ...}`. With `--human` it prints
  tgrep standard output.
- A search with no matches is a normal result: it emits the JSON result and
  exits `1`.
- Argument-validation and runner failures emit a JSON error object to standard
  error and exit `2`.

Command help is available for both commands:

```bash
uv run auggie --help
uv run auggie retrieve --help
uv run auggie search --help
```

## MCP configuration

Configure the server in your MCP client using the installed command:

```json
{
  "mcpServers": {
    "auggie": {
      "command": "auggie-mcp",
      "args": []
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
      "args": ["run", "python", "-m", "auggie_mcp"]
    }
  }
}
```

## Tool

`codebase_retrieval(repository_path, query)`

- `repository_path`: any accessible directory. Relative paths resolve against
  the server's working directory. Symlinks are followed.
- `query`: a natural-language codebase question.

The result is Pi's final answer as text, formatted as an `Answer:` explanation
followed by an `Evidence:` section with real file paths and line ranges.

## Behavior

- Every invocation starts one ephemeral Pi subprocess in the requested
  directory with a fixed, confirmed flag set (`--approve --print --no-session
  --mode text --no-context-files --no-extensions --no-skills
  --no-prompt-templates --system-prompt <text> <query>`).
- Pi emits only its final response on stdout. The adapter reads that response
  in fixed-size chunks, avoiding JSONL line-length limits.
- No Pi tool allowlist is passed. All built-in tools, including `bash`, `edit`,
  and `write`) remain available.
- Requests start immediately; there is no concurrency limit or queue.
- Each invocation has a thirty-minute timeout; timeout and cancellation terminate
  the Pi Linux process group (SIGTERM, then SIGKILL after a grace period).
- Invalid directories, missing Pi, nonzero exits, empty responses, timeouts,
  and cancellations surface as MCP tool errors with concise diagnostics.

## Accepted risks

- **Arbitrary paths.** `repository_path` may be any directory accessible to the
  server process.
- **Writable tools.** Pi retains `bash`, `edit`, and `write`. The bundled system
  prompt requests read-only retrieval behavior, but that is a prompt-level
  control only, not a security boundary. A malicious repository or query could
  cause Pi to modify files or run commands.
- **Unlimited concurrency.** Concurrent calls may exhaust CPU, memory, or model
  API quota, and concurrent requests may modify the same repository.
- **Model credentials.** Pi uses the caller's configured provider credentials
  and network access; server operation depends on them.

## Testing

Default tests require no model credentials or network access; they use a fake Pi
executable.

```bash
uv run python -m unittest tests.test_pi_runner -v
uv run python -m unittest tests.test_server -v
uv run python -m unittest discover -s tests -v
```

To exercise real Pi end-to-end (requires credentials, network, and model quota):

```bash
AUGGIE_LIVE_PI=1 uv run python -m unittest tests.test_live_pi -v
```
