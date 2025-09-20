# Dedalus MCP Server Deployment Guide

This guide contains everything we learned about deploying MCP servers to Dedalus Labs platform.

## Key Requirements for Dedalus Deployment

### 1. Project Structure
Dedalus expects a specific structure:
```
your-project/
├── pyproject.toml      # REQUIRED: Package configuration
├── main.py             # REQUIRED: Entry point (root level)
├── src/
│   └── main.py        # Your actual MCP server code
├── docs/              # Your documentation/data files
└── config/            # Configuration files
```

### 2. pyproject.toml (REQUIRED)
Dedalus uses `uv` package manager and requires pyproject.toml:
```toml
[project]
name = "your-mcp-server"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "mcp",
    "python-dotenv",
    "openai",  # or any other dependencies
]

[tool.setuptools]
packages = ["src"]
py-modules = ["main"]

[project.scripts]
main = "main:main"
```

### 3. Root main.py (REQUIRED)
Dedalus runs `uv run main` to start your server. Create this wrapper:
```python
#!/usr/bin/env python
import sys
import os

def main():
    """Main function for script entry point"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from src.main import mcp
        mcp.run()
    except Exception as e:
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 4. Your MCP Server (src/main.py)
Your actual server using FastMCP:
```python
from mcp.server.fastmcp import FastMCP
import os
from pathlib import Path

mcp = FastMCP("Your Server Name")

# Handle container paths
possible_dirs = [
    Path("/app/docs"),  # Dedalus container path
    Path("./docs"),     # Local path
]

DOCS_DIR = None
for dir_path in possible_dirs:
    if dir_path.exists():
        DOCS_DIR = dir_path
        break

@mcp.tool()
def your_tool():
    """Your tool implementation"""
    pass

def main():
    """Entry point for the MCP server"""
    mcp.run()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

## How Dedalus Runs Your Server

1. **Build Phase**:
   - Reads `pyproject.toml`
   - Runs `uv sync` to install dependencies
   - Creates container environment

2. **Runtime**:
   - Working directory: `/app`
   - Runs: `uv run main`
   - Communicates via stdio protocol

3. **File Access**:
   - Your files are in `/app/`
   - Docs in `/app/docs/`
   - Use absolute paths or Path objects

## Environment Variables

Set these in Dedalus UI:
- `OPENAI_API_KEY` - For AI features
- `ANTHROPIC_API_KEY` - For Claude
- Custom variables via `os.getenv()`

## Common Issues and Solutions

### "No pyproject.toml found"
- **Issue**: Dedalus requires pyproject.toml
- **Solution**: Create pyproject.toml with dependencies

### "No module named 'main'"
- **Issue**: Missing root main.py or incorrect pyproject.toml
- **Solution**: Add `py-modules = ["main"]` to pyproject.toml

### Server crashes on startup
- **Issue**: Import errors or path problems
- **Solution**: Use sys.path.insert in main.py wrapper

### Files not found
- **Issue**: Using relative paths that don't work in container
- **Solution**: Check multiple paths like `/app/docs` and `./docs`

## Testing Locally with uv

```bash
# Install uv (same as Dedalus uses)
brew install uv  # or pip install uv

# Install dependencies
uv sync --no-dev

# Test your server
uv run python -c "from src.main import mcp; print('OK')"

# Run like Dedalus does
uv run main
```

## Rate Limiting for API Protection

Protect your API keys from abuse:
```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests=10, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, identifier):
        now = time.time()
        self.requests[identifier] = [
            t for t in self.requests[identifier]
            if now - t < self.window_seconds
        ]
        
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        return False

rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

@mcp.tool()
def protected_tool(user_id: str = None):
    identifier = user_id or "default"
    if not rate_limiter.is_allowed(identifier):
        return {"error": "Rate limit exceeded"}
    # Your tool logic here
```

## Deployment Command

```bash
dedalus deploy . --name "your-server-name"
```

## Complete Checklist

- [ ] Create `pyproject.toml` with all dependencies
- [ ] Create root `main.py` entry point
- [ ] Put server code in `src/main.py`
- [ ] Use `uv` for local testing
- [ ] Handle `/app` container paths
- [ ] Add rate limiting for API protection
- [ ] Set environment variables in Dedalus UI
- [ ] Test with `uv run main` locally
- [ ] Deploy with `dedalus deploy`