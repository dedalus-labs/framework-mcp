# Getting Started with Dedalus MCP

Model Context Protocol (MCP) enables AI models to interact with external tools and data sources. This guide helps you understand the difference between building MCP servers and using them as a client.

## Choose Your Path

### I want to BUILD an MCP Server
If you want to create tools and services that AI can use:
- [Building MCP Servers](./server/building-mcp-servers.md) - Learn how to create MCP servers
- [Dedalus Deployment Guide](./server/dedalus-deployment.md) - Deploy your server to Dedalus

### I want to USE MCP Servers
If you want to connect to existing MCP servers as a client:
- [Using Deployed Servers](./client/using-deployed-servers.md) - Connect to MCP servers
- [Dedalus Docs MCP](./client/dedalus-docs-mcp.md) - Access Dedalus documentation via MCP
- [Multi-Agent Workflows](./client/multi-agent-workflows.md) - Coordinate multiple AI agents

## Quick Overview

### What is MCP?
The Model Context Protocol is an open standard by Anthropic that standardizes how AI models communicate with:
- External tools and functions
- Data sources and APIs
- Other AI agents
- Document repositories

### Server vs Client

**MCP Server**: Provides tools and resources
- Exposes functions AI can call
- Serves data and documents
- Handles requests via stdio protocol
- Deployed to platforms like Dedalus

**MCP Client**: Uses servers
- Connects to MCP servers
- Calls tools and retrieves resources
- Integrates with AI models
- Part of applications like Claude Desktop

## For Server Developers

### Minimal Server Example
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool()
def hello(name: str) -> str:
    """Greet someone"""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
```

### Deploy to Dedalus
```bash
# Requires pyproject.toml and proper structure
dedalus deploy . --name "my-server"
```

See [server guides](./server/) for complete documentation.

## For Client Developers

### Using a Server
```python
from dedalus_labs import AsyncDedalus, DedalusRunner

async def main():
    client = AsyncDedalus(api_key="your-key")
    runner = DedalusRunner(client)
    
    result = await runner.run(
        input="Your request",
        model="openai/gpt-4",
        mcp_servers=["org/server-name"]
    )
    print(result.final_output)
```

See [client guides](./client/) for complete documentation.

## Repository Structure

This repository is an example MCP server for documentation:
```
dedalus-docs-mcp-server/
├── src/main.py         # MCP server implementation
├── main.py            # Dedalus entry point
├── pyproject.toml     # Package configuration
├── docs/              # Documentation to serve
│   ├── guides/        # These guides
│   │   ├── server/    # Server development
│   │   └── client/    # Client usage
│   └── hackathon/     # Hackathon info
├── tests/             # Test scripts
└── config/            # Configuration files
```

## Next Steps

**Building a server?**
1. Read [Building MCP Servers](./server/building-mcp-servers.md)
2. Study the example in `src/main.py`
3. Deploy with [Dedalus Deployment Guide](./server/dedalus-deployment.md)

**Using servers?**
1. Read [Using Deployed Servers](./client/using-deployed-servers.md)
2. Connect to [Dedalus Docs MCP](./client/dedalus-docs-mcp.md)
3. Learn [Multi-Agent Workflows](./client/multi-agent-workflows.md)

## Resources

- [MCP Specification](https://modelcontextprotocol.io)
- [Dedalus Platform](https://dedaluslabs.ai)
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [This Repository](https://github.com/yourusername/dedalus-docs-mcp-server)