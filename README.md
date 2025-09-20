# Dedalus MCP Documentation Server

An MCP server for serving and querying documentation with AI capabilities. Built for the YC Agents Hackathon.

## Quick Start (Local Development)

```bash
# Install uv package manager (same as Dedalus uses)
brew install uv  # or pip install uv

# Install dependencies
uv sync --no-dev

# Configure API keys for AI features
cp config/.env.example .env.local
# Edit .env.local and add your OpenAI API key

# Test
uv run python tests/test_server.py

# Run
uv run main
```

## Deploy to Dedalus

### What Dedalus Needs
- `pyproject.toml` - Package configuration with dependencies
- `main.py` (root) - Entry point that Dedalus expects
- `src/main.py` - The actual MCP server code
- `docs/` - Your documentation files

### Deployment Steps

1. **Set Environment Variables in Dedalus UI:**
   - `OPENAI_API_KEY` - Your OpenAI API key (required for AI features)

2. **Deploy:**
```bash
dedalus deploy . --name "your-docs-server"
```

### How Dedalus Runs Your Server
1. Installs dependencies using `uv sync` from `pyproject.toml`
2. Runs `uv run main` to start the server
3. Server runs in `/app` directory in container
4. Docs are served from `/app/docs`

## Features

- Serve markdown documentation
- Search across docs
- AI-powered Q&A (with OpenAI)
- Rate limiting (10 requests/minute) to protect API keys
- Ready for agent handoffs

## Tools Available

- `list_docs()` - List documentation files
- `search_docs()` - Search with keywords
- `ask_docs()` - AI answers from docs
- `index_docs()` - Index documents
- `analyze_docs()` - Analyze for tasks

## Documentation

See `docs/` directory for:
- [Getting Started Guide](docs/guides/getting-started.md)
- [Hackathon Information](docs/hackathon/yc-agents-hackathon.md)
- [Deployment Guide](docs/guides/deployment.md)
- [Examples](examples/)

## License

MIT