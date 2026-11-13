"""Allow ``python -m auggie_mcp`` to start the stdio MCP server."""

from auggie_mcp.launch import server


def main() -> None:
    server()


if __name__ == "__main__":
    main()
