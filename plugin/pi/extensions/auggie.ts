// Pi adapter: registers the auggie-mcp stdio server. Secrets are inherited from the environment.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function auggie(pi: ExtensionAPI) {
  pi.registerMcpServer("auggie", { command: "uvx", timeout: 600, description: "Semantic codebase retrieval and exact repository search", args: ["--from", "git+https://github.com/sitien173/auggie-mcp", "auggie-mcp", ""] });
}
