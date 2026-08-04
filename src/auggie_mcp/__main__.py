"""Allow ``python -m auggie_mcp`` to start the stdio MCP server."""

from auggie_mcp.server import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
