"""Single-tool MCP server exposing ``codebase_retrieval``.

The server registers exactly one tool. Each call runs an ephemeral noninteractive
Pi subprocess (via ``auggie_mcp.pi_runner``) against the caller-selected
directory and returns Pi's final answer as text. Operational failures such as
invalid directories, missing Pi, nonzero exits, malformed output, and timeouts
surface as MCP tool errors. Client cancellation propagates into the adapter.

Transport is stdio only. There are no resources, prompts, extra tools,
transports, access controls, or concurrency controls.
"""

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from auggie_mcp import pi_runner

_SYSTEM_PROMPT = pi_runner.load_system_prompt()


def create_server(*, executable: str = "pi") -> MCPServer:
    """Build the MCP server with the single ``codebase_retrieval`` tool."""
    server = MCPServer(
        name="auggie-mcp",
        title="Auggie Codebase Retrieval",
        description=(
            "Semantic code search and codebase understanding over an arbitrary "
            "repository directory using an ephemeral Pi subprocess."
        ),
        version="0.1.0",
    )

    @server.tool(
        name="codebase_retrieval",
        description=(
            "Answer a natural-language question about a repository by running an "
            "ephemeral Pi subprocess in the given directory. Pi retains all "
            "built-in tools, including write-capable tools. Returns Pi's final "
            "answer with file-line evidence."
        ),
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

    return server


def run() -> None:
    """Run the server on stdio (console entry point and ``python -m``)."""
    create_server().run("stdio")


def main() -> None:
    run()


if __name__ == "__main__":
    run()
