// Pi adapter: registers the auggie-mcp stdio server. Secrets are inherited from the environment.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function auggie(pi: ExtensionAPI) {
  pi.registerMcpServer("auggie", { command: "auggie-mcp", args: [] });
}
