# Building MCP Servers

A complete guide to building Model Context Protocol servers for Dedalus deployment.

## What is an MCP Server?

An MCP server exposes tools, resources, and prompts that AI models can use. It communicates via stdio (standard input/output) using JSON-RPC protocol.

## Quick Start with FastMCP

FastMCP simplifies MCP server creation:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool()
def my_tool(param: str) -> str:
    """Tool description for AI"""
    return f"Result: {param}"

if __name__ == "__main__":
    mcp.run()
```

## Core Components

### 1. Tools
Functions that AI can call:
```python
@mcp.tool()
def search_docs(query: str, max_results: int = 5) -> List[Dict]:
    """Search documentation"""
    results = []
    # Your search logic
    return results
```

### 2. Resources
Serve content to clients:
```python
@mcp.resource("docs://{path}")
def get_documentation(path: str) -> str:
    """Serve markdown files"""
    file_path = DOCS_DIR / path
    return file_path.read_text()
```

### 3. Prompts
Templates for AI interactions:
```python
@mcp.prompt()
def query_template(topic: str) -> str:
    """Generate query prompt"""
    return f"Explain {topic} in detail"
```

## Best Practices

### 1. Environment Configuration
```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
```

### 2. Error Handling
```python
@mcp.tool()
def safe_operation(input: str) -> Dict:
    try:
        # Your operation
        result = process(input)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 3. Type Hints
Always use type hints for better AI understanding:
```python
from typing import List, Dict, Optional

@mcp.tool()
def typed_tool(
    required: str,
    optional: Optional[int] = None
) -> Dict[str, Any]:
    """Tool with proper typing"""
    pass
```

### 4. Documentation
Document tools clearly:
```python
@mcp.tool()
def well_documented_tool(param: str) -> str:
    """
    Brief description for AI
    
    Args:
        param: What this parameter does
    
    Returns:
        What the tool returns
    """
    pass
```

## Advanced Features

### 1. Caching
```python
from functools import lru_cache

@mcp.tool()
@lru_cache(maxsize=100)
def cached_operation(query: str) -> Dict:
    """Expensive operation with caching"""
    # This result will be cached
    return expensive_computation(query)
```

### 2. Async Operations
```python
import asyncio

@mcp.tool()
async def async_tool(url: str) -> Dict:
    """Async tool for I/O operations"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### 3. Rate Limiting
```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests=10, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
    def allow(self, key):
        now = time.time()
        self.requests[key] = [
            t for t in self.requests[key] 
            if now - t < self.window
        ]
        if len(self.requests[key]) < self.max_requests:
            self.requests[key].append(now)
            return True
        return False

limiter = RateLimiter()

@mcp.tool()
def rate_limited_tool(user_id: str, query: str) -> Dict:
    if not limiter.allow(user_id):
        return {"error": "Rate limit exceeded"}
    # Process request
    return {"result": "success"}
```

## Example: Documentation Server

Complete example of a documentation MCP server:

```python
from mcp.server.fastmcp import FastMCP
from pathlib import Path
import os

mcp = FastMCP("Documentation Server")

# Configuration
DOCS_DIR = Path("/app/docs") if Path("/app/docs").exists() else Path("./docs")

@mcp.tool()
def list_docs() -> List[Dict]:
    """List all documentation files"""
    docs = []
    for file in DOCS_DIR.rglob("*.md"):
        docs.append({
            "path": str(file.relative_to(DOCS_DIR)),
            "title": file.stem.replace("-", " ").title(),
            "size": file.stat().st_size
        })
    return docs

@mcp.tool()
def search_docs(query: str) -> List[Dict]:
    """Search documentation content"""
    results = []
    query_lower = query.lower()
    
    for file in DOCS_DIR.rglob("*.md"):
        content = file.read_text().lower()
        if query_lower in content:
            # Find snippet
            idx = content.find(query_lower)
            start = max(0, idx - 100)
            end = min(len(content), idx + 100)
            snippet = content[start:end]
            
            results.append({
                "file": str(file.relative_to(DOCS_DIR)),
                "snippet": snippet,
                "score": content.count(query_lower)
            })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)

@mcp.resource("docs://{path}")
def serve_doc(path: str) -> str:
    """Serve documentation file"""
    file_path = DOCS_DIR / path
    if file_path.exists() and file_path.suffix == ".md":
        return file_path.read_text()
    raise ValueError(f"Document not found: {path}")

def main():
    mcp.run()

if __name__ == "__main__":
    main()
```

## Testing Your Server

### Local Testing
```python
# test_server.py
from src.main import list_docs, search_docs

def test_server():
    # Test listing
    docs = list_docs()
    print(f"Found {len(docs)} documents")
    
    # Test search
    results = search_docs("example")
    print(f"Search found {len(results)} results")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_server()
```

### Integration Testing
```bash
# Test with MCP inspector
mcp-inspector ./src/main.py

# Test with stdio
echo '{"method": "tools/list"}' | python src/main.py
```

## Common Patterns

### 1. Multi-Source Data
```python
@mcp.tool()
def aggregate_data(sources: List[str]) -> Dict:
    """Combine data from multiple sources"""
    results = {}
    for source in sources:
        if source == "database":
            results["db"] = query_database()
        elif source == "api":
            results["api"] = fetch_api()
        elif source == "files":
            results["files"] = read_files()
    return results
```

### 2. Validation
```python
@mcp.tool()
def validated_input(
    email: str,
    age: int,
    options: List[str]
) -> Dict:
    """Tool with input validation"""
    import re
    
    # Validate email
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return {"error": "Invalid email"}
    
    # Validate age
    if age < 0 or age > 150:
        return {"error": "Invalid age"}
    
    # Validate options
    valid_options = ["option1", "option2", "option3"]
    for opt in options:
        if opt not in valid_options:
            return {"error": f"Invalid option: {opt}"}
    
    # Process valid input
    return {"success": True}
```

### 3. State Management
```python
class ServerState:
    def __init__(self):
        self.cache = {}
        self.sessions = {}
        self.metrics = defaultdict(int)

state = ServerState()

@mcp.tool()
def stateful_operation(session_id: str, action: str) -> Dict:
    """Operation with state management"""
    if session_id not in state.sessions:
        state.sessions[session_id] = {"created": time.time()}
    
    state.metrics[action] += 1
    
    # Perform action with state
    return {
        "session": state.sessions[session_id],
        "metrics": dict(state.metrics)
    }
```

## Next Steps

- Review [Dedalus Deployment Guide](./dedalus-deployment.md)
- Check [Testing Guide](./testing-mcp-servers.md)
- See [Example Servers](../../examples/)