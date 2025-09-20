# Using Dedalus Documentation MCP Server

This guide shows how to connect to and use the official Dedalus Labs documentation MCP server.

## Overview

Dedalus Labs provides an MCP server that gives AI assistants direct access to their documentation. This enables your AI to search for examples, understand APIs, and get implementation details directly from the source.

## Setup Instructions

### Known Issue
The automated setup via `npx @mintlify/mcp add docs.dedaluslabs.ai` currently fails with:
```
Error: Cannot read properties of undefined (reading 'name')
```
This is because their server is missing the `.well-known/mcp.json` configuration file.

### Manual Setup for Cursor IDE

Add to your `mcp.json` configuration:

```json
{
  "servers": {
    "dedalus-docs": {
      "url": "https://docs.dedaluslabs.ai/mcp"
    }
  }
}
```

### Manual Setup for Claude Desktop

1. Open Claude Desktop
2. Go to Settings → Developer → Connectors
3. Add: `https://docs.dedaluslabs.ai/mcp`

## Available Documentation

The Dedalus MCP server provides access to:

- **Getting Started Guides** - Quick start with Dedalus platform
- **Tool Chaining Examples** - How to chain multiple tools
- **MCP Integration Tutorials** - Building and deploying MCP servers
- **API Authentication** - Setting up API keys and auth
- **Model Handoffs** - Multi-model workflows
- **Streaming Responses** - Real-time output streaming
- **DedalusRunner Examples** - Complete code examples

## Usage Examples

### Search for Documentation

Once connected, you can ask your AI assistant:

```
"Search Dedalus docs for DedalusRunner examples"
```

The AI will receive actual code examples like:

```python
from dedalus_labs import AsyncDedalus, DedalusRunner

async def main():
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    
    result = await runner.run(
        input="What was the score of the 2025 Wimbledon final?",
        model="openai/gpt-5-2025-08-07"
    )
```

### Common Queries

Ask your AI assistant these questions to get information from Dedalus docs:

1. **Basic Setup**:
   - "How do I install dedalus-labs package?"
   - "Show me DedalusRunner initialization examples"
   - "How do I set up API authentication?"

2. **MCP Servers**:
   - "How do I deploy an MCP server to Dedalus?"
   - "What MCP servers are available on Dedalus?"
   - "Show me tool chaining examples"

3. **Advanced Features**:
   - "How do I implement model handoffs?"
   - "Show me streaming response examples"
   - "How do I use multiple MCP servers together?"

4. **Troubleshooting**:
   - "What are common Dedalus deployment errors?"
   - "How do I debug MCP server issues?"
   - "What are the rate limits?"

## Implementation Details

### No Authentication Required

The documentation MCP server doesn't require API authentication - you can query it directly once connected.

### Direct Documentation Access

The server provides direct access to:
- Code examples with syntax highlighting
- API reference documentation
- Best practices and patterns
- Troubleshooting guides

### Real-time Updates

Since it connects to live documentation, you always get the latest information without manual updates.

## Example Interaction Flow

Here's how a typical interaction works:

1. **User asks**: "How do I implement tool chaining with Dedalus?"

2. **AI queries MCP**: The AI sends a search query to the Dedalus docs MCP server

3. **Server responds**: Returns relevant documentation sections with code examples

4. **AI provides answer**: The AI formats and presents the information with working code

## Benefits of Using Dedalus Docs MCP

1. **Always Current**: Get the latest documentation without manual updates
2. **Code Examples**: Access real, tested code examples
3. **Context-Aware**: AI understands the full context of Dedalus platform
4. **No API Key**: Documentation access doesn't require authentication
5. **Direct Integration**: Works seamlessly with your AI coding assistant

## Combining with Your MCP Server

You can use both the Dedalus docs MCP and your own MCP servers together:

```python
from dedalus_labs import AsyncDedalus, DedalusRunner

async def combined_example():
    client = AsyncDedalus(api_key="your-key")
    runner = DedalusRunner(client)
    
    # Use multiple MCP servers
    result = await runner.run(
        input="Search for rate limiting examples and implement one",
        model="claude-3-5-sonnet",
        mcp_servers=[
            "dedalus-labs/docs",      # Official docs
            "your-org/custom-server"  # Your server
        ]
    )
```

## Troubleshooting

### MCP Not Connecting
- Ensure the URL is exactly: `https://docs.dedaluslabs.ai/mcp`
- Check your IDE's MCP configuration
- Restart your IDE after adding the configuration

### No Results
- Try more specific queries
- Ask for "Dedalus" explicitly in your queries
- Check if the docs server is responding

### Outdated Information
- The MCP connects to live docs, so information should be current
- If you see outdated info, report it to Dedalus

## Tips for Best Results

1. **Be Specific**: Ask for specific features or APIs
2. **Request Examples**: Always ask for code examples
3. **Chain Queries**: Build on previous responses for deeper understanding
4. **Combine Sources**: Use alongside your own documentation server

## Related Resources

- [Using Deployed Servers](./using-deployed-servers.md)
- [Multi-Agent Workflows](./multi-agent-workflows.md)
- [Building Your Own Docs Server](../server/building-mcp-servers.md)

## Community Support

If you need help with MCP setup or want example repositories:
- Check the Dedalus Discord
- Visit the YC Agents Hackathon channels
- Review example repos on GitHub

---

*Note: This documentation is based on the current state of the Dedalus docs MCP server. Features and availability may change.*