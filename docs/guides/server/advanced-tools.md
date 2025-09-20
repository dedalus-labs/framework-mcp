# Advanced Tools Development

This guide covers advanced patterns for building sophisticated MCP tools.

## Stateful Tools

MCP servers can maintain state between tool calls within a session:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Stateful Server")

# Session state
sessions = {}

@mcp.tool()
def start_session(user_id: str) -> dict:
    """Initialize a user session"""
    sessions[user_id] = {
        "created_at": datetime.now().isoformat(),
        "actions": []
    }
    return {"status": "session_started", "user_id": user_id}

@mcp.tool()
def track_action(user_id: str, action: str) -> dict:
    """Track user actions in session"""
    if user_id not in sessions:
        return {"error": "No active session"}
    
    sessions[user_id]["actions"].append({
        "action": action,
        "timestamp": datetime.now().isoformat()
    })
    return {"status": "tracked", "total_actions": len(sessions[user_id]["actions"])}
```

## Async Tools

For I/O-bound operations, use async tools:

```python
import asyncio
import aiohttp

@mcp.tool()
async def fetch_multiple(urls: list[str]) -> list[dict]:
    """Fetch multiple URLs concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

async def fetch_url(session, url):
    try:
        async with session.get(url) as response:
            return {
                "url": url,
                "status": response.status,
                "content": await response.text()
            }
    except Exception as e:
        return {"url": url, "error": str(e)}
```

## Tool Composition

Build complex tools by composing simpler ones:

```python
@mcp.tool()
def analyze_website(url: str) -> dict:
    """Comprehensive website analysis"""
    # Fetch content
    content = fetch_content(url)
    
    # Extract metadata
    metadata = extract_metadata(content)
    
    # Analyze SEO
    seo_score = analyze_seo(content, metadata)
    
    # Check performance
    performance = check_performance(url)
    
    return {
        "url": url,
        "metadata": metadata,
        "seo_score": seo_score,
        "performance": performance
    }
```

## Error Handling

Implement robust error handling:

```python
from typing import Union

@mcp.tool()
def safe_operation(
    data: str,
    operation: str
) -> Union[dict, str]:
    """Tool with comprehensive error handling"""
    try:
        # Validate input
        if not data:
            return {"error": "No data provided", "code": "MISSING_DATA"}
        
        # Perform operation
        if operation == "parse":
            result = json.loads(data)
        elif operation == "encode":
            result = base64.b64encode(data.encode()).decode()
        else:
            return {"error": f"Unknown operation: {operation}", "code": "INVALID_OP"}
        
        return {"success": True, "result": result}
        
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}", "code": "PARSE_ERROR"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "code": "INTERNAL_ERROR"}
```

## Tool Validation

Use Pydantic for input validation:

```python
from pydantic import BaseModel, Field, validator

class QueryParams(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

@mcp.tool()
def search_with_validation(params: QueryParams) -> dict:
    """Search with validated parameters"""
    # Parameters are automatically validated
    return perform_search(
        params.query,
        params.limit,
        params.offset
    )
```

## Streaming Responses

For long-running operations, use streaming:

```python
@mcp.tool()
async def process_large_dataset(
    dataset_id: str,
    stream_updates: bool = True
) -> AsyncGenerator[dict, None]:
    """Process dataset with streaming updates"""
    
    total_items = get_dataset_size(dataset_id)
    
    for i, item in enumerate(process_dataset(dataset_id)):
        # Process item
        result = process_item(item)
        
        if stream_updates:
            # Stream progress updates
            yield {
                "type": "progress",
                "current": i + 1,
                "total": total_items,
                "percentage": ((i + 1) / total_items) * 100
            }
        
        # Stream results
        yield {
            "type": "result",
            "item": result
        }
    
    yield {
        "type": "complete",
        "total_processed": total_items
    }
```

## Best Practices

1. **Input Validation**: Always validate inputs
2. **Error Messages**: Provide clear, actionable error messages
3. **Documentation**: Document all parameters and return types
4. **Idempotency**: Make tools idempotent where possible
5. **Rate Limiting**: Implement rate limiting for expensive operations
6. **Caching**: Cache results for expensive computations
7. **Logging**: Log all operations for debugging
8. **Security**: Never expose sensitive data or operations

## Next Steps

- [Agent Handoffs](./agent-handoffs.md) - Learn about multi-agent coordination
- [Performance Optimization](./performance.md) - Optimize your tools
- [Security Best Practices](./security.md) - Secure your MCP server