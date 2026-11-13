# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries before the first public release were reconstructed from the Git history. Version dates are the dates of the commits that set each version in `pyproject.toml`.

## [Unreleased]

### Breaking changes

- The server speaks MCP protocol `2026-07-28` only, on `mcp` 2.3.0. A client that opens with `initialize` receives error `-32022`.
- There is no default agent driver. Set `AUGGIE_AGENT_DRIVER` or `driver` in the `[agent]` table of `~/.auggie/config.toml`.
- `AUGGIE_*` settings are read from `config.toml` sections.

### Added

- Plugin packaging for Claude Code, Codex, agy, and pi, with root marketplaces, a shared MCP config, a pi extension, and the `auggie-retrieval` and `auggie-cli` skills under `plugin/skills`.
- The retrieval agent receives the system prompt and index hints.
- `tools/list` responses are cacheable for one hour, and all four tools raise `ToolError` so clients see the real error text.

### Changed

- Embedding requests run in parallel across files.
- The default `AUGGIE_VOYAGE_RETRIES` is 5 attempts.
- The retrieval system prompt requires a fixed Answer, Evidence, and Inferences format.

### Fixed

- Voyage connection errors and HTTP 500 responses are retried with capped exponential backoff.
- The index walk honors `.git/info/exclude` and global git excludes.
- `codebase_retrieval` refuses nested calls from an agent that it started.

## [0.5.0] - 2026-11-13

### Breaking changes

- `codebase_search` parameter `pattern` is renamed to `query`.
- `auggie_query`, `auggie_context`, and `auggie_status` are removed. Use `codebase_search` with `mode=semantic` and `index_status`.
- `codebase_retrieval` no longer refreshes the index. Call `index_update` when `index_status` reports staleness.
- `auggie watch` uses `index.lock` instead of `watch.lock`.

### Added

- `codebase_search` options `mode=exact|semantic`, `top_k`, and `context_lines`. Semantic mode reports index staleness and opens the index read-only.
- `index_status` reports index existence, counts, and staleness without changing the index.
- `index_update` runs the indexer with a timeout, cancellation, and a shared index lock.
- `auggie index`, `reindex`, and `watch` share one `index.lock`. A second writer fails with `INDEX_BUSY`.

## [0.4.0] - 2026-11-12

### Breaking changes

- The agent runs through `coding-agent-cli` and replaces the hand-written Pi runner. Driver, model, reasoning, args, executable, and timeout resolve from `AUGGIE_AGENT_*` variables, then the `[agent]` table in `~/.auggie/config.toml`.
- The read-only system prompt is no longer sent in this release.

### Added

- In-tree semantic search, with 13 `auggie` subcommands for index, query, maintenance, and watch.

### Changed

- `codebase_search` and `auggie search` run ripgrep (`rg`) in place of tgrep. Search no longer refreshes an index.
- Requires Python 3.12+ and `rg` on `PATH`.

## [0.3.0] - 2026-11-12

### Added

- Local JSONL telemetry and the `auggie stats` command.

### Fixed

- The fallback TypeSafe client is closed and scoped to each direct call.

## [0.2.0] - 2026-09-23

### Added

- TypeSafe AI (Jev) output processing that classifies tool output and prunes low-utility context from oversized responses.
- Search accepts absolute paths outside the repository.
- EditorConfig, SonarQube configuration, and coverage reporting.

### Changed

- Pi reasoning is lowered to `high`.

## [0.1.0] - 2026-08-05

### Added

- Initial MCP server exposing the `codebase_retrieval` tool through a Pi subprocess.
- `auggie` command with `retrieve` and `search` subcommands.
- Exact search through a bundled tgrep binary.

[Unreleased]: https://github.com/sitien173/auggie-mcp/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/sitien173/auggie-mcp/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/sitien173/auggie-mcp/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/sitien173/auggie-mcp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/sitien173/auggie-mcp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sitien173/auggie-mcp/releases/tag/v0.1.0
