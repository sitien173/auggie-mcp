---
name: auggie-retrieval
description: Use Auggie MCP for codebase retrieval and exact search. Fall back to the auggie CLI only when the MCP tools are unavailable.
user-invocable: false
---

# Auggie Retrieval

Use Auggie before manual repository exploration.

## Primary path

Use the MCP tools when available.

- Use `mcp__auggie__codebase_retrieval` for semantic questions.
- Use `mcp__auggie__codebase_search` for exact symbols, literals, files, and regexes.
- Use `Read` only after a search identifies a file path.

## CLI fallback

Fall back only when the matching MCP tool is unavailable or errors.

- Semantic retrieval:

  ```bash
  uv run auggie retrieve "$PWD" "<question>"
  ```

- Exact search from the repository root:

  ```bash
  uv run auggie search "<pattern>" [validated-tgrep-flags...]
  ```

The CLI emits JSON by default. Read the retrieval `response` field or the search
`stdout`, `stderr`, and `exit_code` fields.

Treat search exit code `1` as no matches. Do not use it as an MCP failure.

## Rules

- Do not fall back because results are empty or incomplete.
- Do not duplicate tgrep flag validation.
- Do not construct a shell command from untrusted inputs.
- Do not change repositories or runner behavior to make fallback work.
