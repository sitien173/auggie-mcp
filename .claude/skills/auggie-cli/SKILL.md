---
name: auggie-cli
description: Use auggie CLI for semantic codebase retrieval, exact symbol and literal searches, regex searches, and file discovery. Use when the user requests auggie retrieve or auggie search, or when existing MCP-first instructions permit a CLI fallback. Covers repository scope, supported flags, JSON output, and exit codes.
---

# Auggie CLI

auggie is a command-line interface for semantic codebase retrieval and exact symbol and literal searches. It supports regex searches and file discovery, providing JSON output and clear exit codes for integration into scripts and pipelines.

## 1. Check the command and repository

Run these help commands before using unfamiliar options:

```bash
auggie --help
auggie retrieve --help
auggie search --help
```

Confirm the target repository. Retrieval accepts an explicit directory. Search always uses the invocation directory as its repository root.

Follow existing MCP-first instructions. Use the CLI when explicitly requested or when those instructions permit fallback. Empty or incomplete MCP results do not by themselves justify fallback.

## 2. Select the operation

### Semantic retrieval

Use retrieval for conceptual questions when relevant files or symbols are unknown:

```bash
auggie retrieve "/absolute/path/to/repository" "Where is authentication implemented? Return verified file paths and line numbers."
```

Ask one focused question with enough task context to identify the relevant implementation. The runner starts the configured agent CLI noninteractively in the selected repository. It does not use the MCP server's output-processing pipeline.

Wait for retrieval to finish. Do not duplicate the same investigation with another retrieval call or parallel manual exploration.

### Exact search and file discovery

Run the installed CLI with the target repository as the working directory:

```bash
# Literal symbol with surrounding context.
auggie search "validate_flags" -F -C 3

# Regex restricted to Python files.
auggie search '^class .*Error' -g '*.py'

# Names of files containing a literal.
auggie search "run_agent" -F -l

# File discovery still requires the positional pattern argument.
auggie search "" --files -g '*.py'
```

Search forwards extra arguments to the runner's rg flag validation. Use `-F` for literals, `-g` for file scope, `-l` for matching filenames, and `-C` for context. Quote patterns and globs so the shell does not expand them.

Do not supply an additional positional search path or a `--repository` option. Narrow scope with `-g`. Path-valued flags such as `--file`, `--ignore-file`, and `--index-path` must resolve within the invocation directory.

Do not assume every rg option is accepted. The runner rejects unsupported flags, including `--exclude`. Consult the current runner allowlist before using options not shown here.

## 3. Read output and handle failures

JSON is the default output format:

- Retrieval writes `{"response": "..."}` to standard output.
- Search writes `{"stdout": "...", "stderr": "...", "exit_code": 0}` to standard output. Inspect both output fields.
- Argument-validation and runner failures write `{"error": "..."}` to standard error.

Add `--human` to either subcommand only when plain text is preferable. Retrieval then prints the response text; search prints only rg standard output. Prefer JSON when diagnostics matter.

| Exit code | Meaning | Action |
| --- | --- | --- |
| `0` | Retrieval completed or search succeeded. | Inspect the result and verify relevant evidence. |
| `1` | Search found no matches. | Treat as a normal result, not a tool failure. Check scope or refine the pattern. |
| `2` | Invalid arguments or runner failure. | Read the standard-error JSON and report or correct the specific cause. |

Do not hide failures with `|| true`. Do not invent flags, credentials, or authentication commands to repair them.
