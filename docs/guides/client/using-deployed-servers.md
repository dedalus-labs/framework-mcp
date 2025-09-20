# Using Deployed MCP Servers

This guide shows how to connect to and use MCP servers that have been deployed to Dedalus or other platforms.

## Prerequisites

- Dedalus API key
- Python 3.10+
- Basic understanding of async programming

## Installation

```bash
pip install dedalus-labs
```

## Basic Usage

### Connect to a Single MCP Server

```python
from dedalus_labs import AsyncDedalus, DedalusRunner
import asyncio

async def use_mcp_server():
    # Initialize client
    client = AsyncDedalus(api_key="your-api-key")
    runner = DedalusRunner(client)
    
    # Call an MCP server
    result = await runner.run(
        input="List all documentation files",
        model="openai/gpt-4o-mini",
        mcp_servers=["your-org/docs-server"],
        stream=False
    )
    
    print(result.final_output)

# Run
asyncio.run(use_mcp_server())
```

## Working with Multiple Servers

### Combine Multiple MCP Servers

```python
async def multi_server_example():
    client = AsyncDedalus(api_key="your-api-key")
    runner = DedalusRunner(client)
    
    result = await runner.run(
        input="Search for Python tutorials and summarize them",
        model="openai/gpt-4",
        mcp_servers=[
            "your-org/docs-server",      # Your documentation
            "dedalus-labs/brave-search"  # Web search
        ],
        stream=False
    )
    
    return result.final_output
```

## Streaming Responses

### Get Real-time Updates

```python
async def streaming_example():
    client = AsyncDedalus(api_key="your-api-key")
    runner = DedalusRunner(client)
    
    # Stream responses
    async for chunk in runner.run(
        input="Analyze this large dataset",
        model="claude-3-5-sonnet",
        mcp_servers=["your-org/analytics-server"],
        stream=True
    ):
        print(chunk, end="", flush=True)
```

## Tool Discovery

### List Available Tools

```python
async def discover_tools():
    client = AsyncDedalus(api_key="your-api-key")
    
    # Get server capabilities
    servers = await client.list_mcp_servers()
    
    for server in servers:
        print(f"Server: {server['name']}")
        print(f"Tools: {server['tools']}")
        print(f"Resources: {server['resources']}")
```

## Error Handling

### Robust Client Implementation

```python
async def robust_client():
    client = AsyncDedalus(api_key="your-api-key")
    runner = DedalusRunner(client)
    
    try:
        result = await runner.run(
            input="Your query",
            model="openai/gpt-4",
            mcp_servers=["your-org/server"],
            timeout=30,  # 30 second timeout
            max_retries=3
        )
        return result.final_output
        
    except TimeoutError:
        print("Request timed out")
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None
```

## Authentication Methods

### API Key Authentication

```python
# From environment variable
import os
os.environ["DEDALUS_API_KEY"] = "your-key"
client = AsyncDedalus()  # Auto-reads from env

# Direct initialization
client = AsyncDedalus(api_key="your-key")

# From config file
from dotenv import load_dotenv
load_dotenv()
client = AsyncDedalus()
```

## Advanced Patterns

### Context Management

```python
class MCPSession:
    def __init__(self, api_key: str):
        self.client = AsyncDedalus(api_key=api_key)
        self.runner = DedalusRunner(self.client)
        self.context = []
    
    async def query(self, input: str, remember: bool = True):
        # Add context from previous queries
        full_input = self._build_context(input)
        
        result = await self.runner.run(
            input=full_input,
            model="openai/gpt-4",
            mcp_servers=["your-org/server"]
        )
        
        if remember:
            self.context.append({
                "query": input,
                "response": result.final_output
            })
        
        return result.final_output
    
    def _build_context(self, input: str) -> str:
        if not self.context:
            return input
        
        context_str = "Previous context:\n"
        for item in self.context[-3:]:  # Last 3 interactions
            context_str += f"Q: {item['query']}\nA: {item['response']}\n"
        
        return f"{context_str}\nCurrent query: {input}"
```

### Batch Processing

```python
async def batch_process(items: list):
    client = AsyncDedalus(api_key="your-api-key")
    runner = DedalusRunner(client)
    
    tasks = []
    for item in items:
        task = runner.run(
            input=f"Process: {item}",
            model="openai/gpt-4o-mini",
            mcp_servers=["your-org/processor"]
        )
        tasks.append(task)
    
    # Process all items concurrently
    results = await asyncio.gather(*tasks)
    
    return [r.final_output for r in results]
```

## Working with Resources

### Access MCP Resources

```python
async def access_resources():
    client = AsyncDedalus(api_key="your-api-key")
    runner = DedalusRunner(client)
    
    # Request specific resource
    result = await runner.run(
        input="Get the contents of docs://guides/getting-started.md",
        model="openai/gpt-4o-mini",
        mcp_servers=["your-org/docs-server"]
    )
    
    print(result.final_output)
```

## Rate Limiting

### Handle Rate Limits Gracefully

```python
import time
from typing import Optional

class RateLimitedClient:
    def __init__(self, api_key: str, requests_per_minute: int = 10):
        self.client = AsyncDedalus(api_key=api_key)
        self.runner = DedalusRunner(self.client)
        self.requests_per_minute = requests_per_minute
        self.request_times = []
    
    async def query(self, input: str) -> Optional[str]:
        # Check rate limit
        self._enforce_rate_limit()
        
        try:
            result = await self.runner.run(
                input=input,
                model="openai/gpt-4",
                mcp_servers=["your-org/server"]
            )
            
            self.request_times.append(time.time())
            return result.final_output
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                print("Rate limited, waiting...")
                await asyncio.sleep(60)
                return await self.query(input)
            raise
    
    def _enforce_rate_limit(self):
        now = time.time()
        # Remove old requests outside the window
        self.request_times = [
            t for t in self.request_times 
            if now - t < 60
        ]
        
        # If at limit, wait
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
```

## Common Use Cases

### 1. Documentation Q&A
```python
result = await runner.run(
    input="How do I implement rate limiting?",
    model="openai/gpt-4",
    mcp_servers=["your-org/docs-server"]
)
```

### 2. Data Analysis
```python
result = await runner.run(
    input="Analyze sales data for Q4",
    model="claude-3-5-sonnet",
    mcp_servers=["your-org/analytics-server"]
)
```

### 3. Code Generation
```python
result = await runner.run(
    input="Generate a Python function to parse CSV files",
    model="claude-3-5-sonnet",
    mcp_servers=["your-org/code-assistant"]
)
```

### 4. Web Research
```python
result = await runner.run(
    input="Research latest AI developments",
    model="openai/gpt-4",
    mcp_servers=["dedalus-labs/brave-search"]
)
```

## Best Practices

1. **Use Environment Variables**: Store API keys in `.env` files
2. **Handle Errors**: Always wrap calls in try-except blocks
3. **Rate Limit**: Respect rate limits to avoid service disruption
4. **Cache Results**: Cache frequently requested data
5. **Use Appropriate Models**: Choose models based on task complexity
6. **Stream Long Responses**: Use streaming for better UX
7. **Log Interactions**: Keep logs for debugging and auditing

## Troubleshooting

### Connection Issues
- Verify API key is correct
- Check network connectivity
- Ensure server is deployed and running

### Tool Not Found
- Verify server name is correct
- Check if tool exists with discovery
- Ensure proper permissions

### Rate Limiting
- Implement exponential backoff
- Use batch processing wisely
- Consider upgrading plan if needed

## Next Steps

- [Advanced Patterns](./advanced-patterns.md)
- [Multi-Agent Workflows](./multi-agent-workflows.md)
- [Performance Optimization](./optimization.md)