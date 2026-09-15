"""Stdio MCP server exposing semantic and exact codebase search."""

from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from auggie_mcp import pi_runner, tgrep_runner

_SYSTEM_PROMPT = pi_runner.load_system_prompt()

_TOOL_DESCRIPTION = """

IMPORTANT: This is the primary tool for searching the codebase. Please consider as the **FIRST CHOICE** for any codebase searches.

This MCP tool is Augment's context engine, the world's best codebase context engine. It:

1. Takes in a natural language description of the code you are looking for.
2. Uses a proprietary retrieval/embedding model suite that produces the highest-quality recall of relevant code snippets from across the codebase.
3. Maintains a real-time index of the codebase, so the results are always up-to-date and reflect the current state of the codebase.
4. Can retrieve across different programming languages.
5. Only reflects the current state of the codebase on the disk, and has no information on version control or code history.

## Execution Model

This tool is LONG RUNNING. A single call typically takes several minutes.

- Run it as a background task and wait for it to complete.
- Do NOT explore or try to understand the codebase yourself while this tool is running. Do not read files, tgrep, or spawn search agents in the meantime.
- Do NOT call this tool again for the same question while a call is in flight.
- Wait for the result, then act on it. The returned answer replaces the exploration you would otherwise do.

## When to Use

Use `codebase-retrieval` in the following cases:

- When you don't know which files contain the information you need.
- When you want to gather high-level information about the task you are trying to accomplish.
- When you want to gather information about the codebase in general.

### Good Queries

- "Where is the function that handles user authentication?"
- "What tests are there for the login functionality?"
- "How is the database connected to the application?"

### Bad Queries (use other tools instead)

- "Find definition of constructor of class Foo" — use `tgrep` instead.
- "Find all references to function bar" — use `tgrep` instead.
- "Show me how Checkout class is used in services/payment.py" — use file view instead.
- "Show context of the file foo.py" — use file view instead.

## Tool Selection Rule

**ALWAYS** use `codebase-retrieval` when you're unsure of exact file locations. Use `tgrep` when you want to find ALL occurrences of a known identifier across the codebase, or when searching within specific files.

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repository_path` | `string` | Yes | Absolute path to the directory/repository you want to search. |
| `query` | `string` | Yes | A description of the information you need from the codebase. |"""

_SEARCH_TOOL_DESCRIPTION = (
    "Fast regex search over the repository. ripgrep-compatible flags. "
    "Use -F for literal strings, -t/-g to scope, -l for file names only, "
    "-C N for context."
)


def create_server(
    *,
    executable: str = "pi",
    tgrep_executable: str = "tgrep",
    repository_root=None,
) -> MCPServer:
    """Build the MCP server and capture its repository search root."""
    search_root = Path.cwd() if repository_root is None else Path(repository_root)
    server = MCPServer(
        name="auggie-mcp",
        title="Auggie Codebase Search",
        description="Semantic retrieval and fast exact repository search.",
        version="0.1.0",
    )

    @server.tool(
        name="codebase_retrieval",
        description=_TOOL_DESCRIPTION,
    )
    async def codebase_retrieval(
        repository_path: str, query: str
    ) -> str:
        """Run Pi in ``repository_path`` with ``query`` and return the answer."""
        try:
            return await pi_runner.run_pi(
                repository_path,
                query,
                _SYSTEM_PROMPT,
                executable=executable,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise ToolError(str(exc)) from exc
        except pi_runner.PiRunnerError as exc:
            message = str(exc)
            if isinstance(exc, pi_runner.PiExecutionError) and exc.stderr:
                message = f"{message}: {exc.stderr}"
            raise ToolError(message) from exc
        except Exception as exc:  # timeout, cancellation, unexpected
            raise ToolError(str(exc) or exc.__class__.__name__) from exc

    @server.tool(
        name="codebase_search",
        description=_SEARCH_TOOL_DESCRIPTION,
    )
    async def codebase_search(
        pattern: str,
        path: str = ".",
        flags: list[str] | None = None,
    ) -> tgrep_runner.TgrepResult:
        """Run tgrep beneath the server startup directory."""
        try:
            result = await tgrep_runner.run_tgrep(
                search_root,
                pattern,
                path,
                flags,
                executable=tgrep_executable,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise ToolError(str(exc)) from exc
        if result["exit_code"] not in {0, 1}:
            message = result["stderr"] or (
                f"tgrep exited with status {result['exit_code']}"
            )
            raise ToolError(message)
        return result

    return server


def run() -> None:
    """Run the server on stdio (console entry point and ``python -m``)."""
    create_server().run("stdio")


def main() -> None:
    run()


if __name__ == "__main__":
    run()
