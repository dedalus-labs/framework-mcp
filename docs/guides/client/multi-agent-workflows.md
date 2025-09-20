# Multi-Agent Workflows with Dedalus

This guide shows how to coordinate multiple AI agents using Dedalus for complex workflows.

## Understanding Agent Coordination

Multi-agent workflows enable:
- Task specialization by model strengths
- Parallel processing for speed
- Sequential refinement of outputs
- Cost optimization through model selection

## Basic Multi-Model Setup

### Sequential Models

```python
from dedalus_labs import AsyncDedalus, DedalusRunner
import asyncio

async def sequential_workflow():
    client = AsyncDedalus(api_key="your-api-key")
    runner = DedalusRunner(client)
    
    # Step 1: Research with GPT-4
    research = await runner.run(
        input="Research the latest developments in quantum computing",
        model="openai/gpt-4",
        mcp_servers=["dedalus-labs/brave-search"],
        stream=False
    )
    
    # Step 2: Analyze with Claude
    analysis = await runner.run(
        input=f"Analyze these findings: {research.final_output}",
        model="claude-3-5-sonnet-20241022",
        stream=False
    )
    
    # Step 3: Summarize with GPT-4o-mini (cost-effective)
    summary = await runner.run(
        input=f"Create a brief summary: {analysis.final_output}",
        model="openai/gpt-4o-mini",
        stream=False
    )
    
    return summary.final_output
```

## Parallel Processing

### Concurrent Analysis

```python
async def parallel_analysis(document: str):
    client = AsyncDedalus(api_key="your-api-key")
    runner = DedalusRunner(client)
    
    # Launch multiple analyses in parallel
    tasks = [
        runner.run(
            input=f"Technical analysis of: {document}",
            model="claude-3-5-sonnet-20241022"
        ),
        runner.run(
            input=f"Business implications of: {document}",
            model="openai/gpt-4"
        ),
        runner.run(
            input=f"Risk assessment of: {document}",
            model="claude-3-opus-20240229"
        )
    ]
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    
    # Combine insights
    combined = await runner.run(
        input=f"Synthesize these analyses: {[r.final_output for r in results]}",
        model="openai/gpt-4"
    )
    
    return combined.final_output
```

## Model Selection Strategies

### Task-Based Routing

```python
class SmartRouter:
    def __init__(self, api_key: str):
        self.client = AsyncDedalus(api_key=api_key)
        self.runner = DedalusRunner(self.client)
        
        # Model capabilities mapping
        self.model_map = {
            "code": "claude-3-5-sonnet-20241022",
            "creative": "claude-3-opus-20240229",
            "analysis": "openai/gpt-4",
            "simple": "openai/gpt-4o-mini",
            "search": "openai/gpt-4",
            "math": "openai/gpt-4"
        }
    
    async def route(self, task_type: str, input: str):
        model = self.model_map.get(task_type, "openai/gpt-4")
        
        # Add appropriate MCP servers
        servers = []
        if task_type == "search":
            servers.append("dedalus-labs/brave-search")
        
        result = await self.runner.run(
            input=input,
            model=model,
            mcp_servers=servers if servers else None
        )
        
        return result.final_output
```

## Complex Workflows

### Research and Report Generation

```python
async def research_report(topic: str):
    client = AsyncDedalus(api_key="your-api-key")
    runner = DedalusRunner(client)
    
    # Phase 1: Broad research
    print("Phase 1: Researching...")
    research = await runner.run(
        input=f"Research: {topic}",
        model="openai/gpt-4",
        mcp_servers=["dedalus-labs/brave-search"]
    )
    
    # Phase 2: Deep analysis
    print("Phase 2: Analyzing...")
    analysis = await runner.run(
        input=f"Analyze key findings: {research.final_output}",
        model="claude-3-5-sonnet-20241022"
    )
    
    # Phase 3: Generate report sections in parallel
    print("Phase 3: Generating report sections...")
    sections = await asyncio.gather(
        runner.run(
            input=f"Write executive summary: {analysis.final_output}",
            model="claude-3-opus-20240229"
        ),
        runner.run(
            input=f"Write technical details: {analysis.final_output}",
            model="claude-3-5-sonnet-20241022"
        ),
        runner.run(
            input=f"Write recommendations: {analysis.final_output}",
            model="openai/gpt-4"
        )
    )
    
    # Phase 4: Compile final report
    print("Phase 4: Compiling report...")
    report = await runner.run(
        input=f"""Compile these sections into a cohesive report:
        Executive Summary: {sections[0].final_output}
        Technical Details: {sections[1].final_output}
        Recommendations: {sections[2].final_output}""",
        model="claude-3-opus-20240229"
    )
    
    return report.final_output
```

## Consensus Mechanisms

### Multi-Model Voting

```python
async def consensus_decision(question: str):
    client = AsyncDedalus(api_key="your-api-key")
    runner = DedalusRunner(client)
    
    models = [
        "claude-3-5-sonnet-20241022",
        "openai/gpt-4",
        "claude-3-opus-20240229"
    ]
    
    # Get responses from all models
    responses = []
    for model in models:
        result = await runner.run(
            input=question,
            model=model
        )
        responses.append({
            "model": model,
            "response": result.final_output
        })
    
    # Have a judge evaluate
    judge_prompt = f"""
    Question: {question}
    
    Responses:
    {responses}
    
    Evaluate all responses and provide:
    1. The best answer
    2. Why it's the best
    3. Key points of agreement
    4. Points of disagreement
    """
    
    judgment = await runner.run(
        input=judge_prompt,
        model="openai/gpt-4"
    )
    
    return judgment.final_output
```

## Error Recovery

### Fallback Chains

```python
async def robust_query(input: str):
    client = AsyncDedalus(api_key="your-api-key")
    runner = DedalusRunner(client)
    
    # Try primary model
    try:
        result = await runner.run(
            input=input,
            model="claude-3-5-sonnet-20241022",
            timeout=30
        )
        return result.final_output
    except Exception as e:
        print(f"Primary failed: {e}")
    
    # Fallback to secondary
    try:
        result = await runner.run(
            input=input,
            model="openai/gpt-4",
            timeout=30
        )
        return result.final_output
    except Exception as e:
        print(f"Secondary failed: {e}")
    
    # Final fallback
    try:
        result = await runner.run(
            input=input,
            model="openai/gpt-4o-mini",
            timeout=30
        )
        return result.final_output
    except Exception as e:
        return f"All models failed: {e}"
```

## Cost Optimization

### Smart Model Selection

```python
class CostOptimizer:
    def __init__(self, api_key: str):
        self.client = AsyncDedalus(api_key=api_key)
        self.runner = DedalusRunner(self.client)
        
        # Cost per 1K tokens (approximate)
        self.costs = {
            "openai/gpt-4": 0.03,
            "openai/gpt-4o-mini": 0.0002,
            "claude-3-5-sonnet-20241022": 0.003,
            "claude-3-opus-20240229": 0.015
        }
    
    async def optimize(self, input: str, max_cost: float = 0.10):
        # Estimate complexity
        complexity = self._estimate_complexity(input)
        
        # Select model based on complexity and cost
        if complexity == "low" and max_cost >= 0.001:
            model = "openai/gpt-4o-mini"
        elif complexity == "medium" and max_cost >= 0.01:
            model = "claude-3-5-sonnet-20241022"
        elif complexity == "high" and max_cost >= 0.05:
            model = "openai/gpt-4"
        else:
            model = "openai/gpt-4o-mini"  # Default to cheapest
        
        result = await self.runner.run(
            input=input,
            model=model
        )
        
        return {
            "output": result.final_output,
            "model_used": model,
            "estimated_cost": self.costs[model] * len(input) / 1000
        }
    
    def _estimate_complexity(self, input: str) -> str:
        # Simple heuristic
        if len(input) < 100:
            return "low"
        elif len(input) < 500:
            return "medium"
        else:
            return "high"
```

## Workflow Orchestration

### State Machine Pattern

```python
class WorkflowOrchestrator:
    def __init__(self, api_key: str):
        self.client = AsyncDedalus(api_key=api_key)
        self.runner = DedalusRunner(client)
        self.state = "initial"
        self.context = {}
    
    async def execute(self, initial_input: str):
        self.context["input"] = initial_input
        
        while self.state != "complete":
            if self.state == "initial":
                await self._research_phase()
            elif self.state == "analysis":
                await self._analysis_phase()
            elif self.state == "review":
                await self._review_phase()
            elif self.state == "finalize":
                await self._finalize_phase()
        
        return self.context["final_output"]
    
    async def _research_phase(self):
        result = await self.runner.run(
            input=self.context["input"],
            model="openai/gpt-4",
            mcp_servers=["dedalus-labs/brave-search"]
        )
        self.context["research"] = result.final_output
        self.state = "analysis"
    
    async def _analysis_phase(self):
        result = await self.runner.run(
            input=f"Analyze: {self.context['research']}",
            model="claude-3-5-sonnet-20241022"
        )
        self.context["analysis"] = result.final_output
        self.state = "review"
    
    async def _review_phase(self):
        result = await self.runner.run(
            input=f"Review for accuracy: {self.context['analysis']}",
            model="openai/gpt-4"
        )
        self.context["review"] = result.final_output
        self.state = "finalize"
    
    async def _finalize_phase(self):
        result = await self.runner.run(
            input=f"Create final output from: {self.context['review']}",
            model="claude-3-opus-20240229"
        )
        self.context["final_output"] = result.final_output
        self.state = "complete"
```

## Best Practices

1. **Model Selection**: Choose models based on task requirements
2. **Error Handling**: Always have fallback strategies
3. **Cost Management**: Monitor and optimize model usage
4. **Parallel Processing**: Use when tasks are independent
5. **Context Preservation**: Maintain context between agent calls
6. **Rate Limiting**: Respect API limits across all models
7. **Logging**: Track all agent interactions for debugging

## Common Patterns

### Map-Reduce
Process items in parallel, then combine:
```python
# Map: Process each item
results = await asyncio.gather(*[
    runner.run(input=item, model="openai/gpt-4o-mini")
    for item in items
])

# Reduce: Combine results
final = await runner.run(
    input=f"Combine: {results}",
    model="openai/gpt-4"
)
```

### Pipeline
Sequential processing with transformations:
```python
data = initial_input
for step in pipeline_steps:
    data = await runner.run(
        input=f"{step['prompt']}: {data}",
        model=step['model']
    )
```

### Fork-Join
Split work, process, then merge:
```python
# Fork
task_a = runner.run(input=f"Part A: {input}", model="model_a")
task_b = runner.run(input=f"Part B: {input}", model="model_b")

# Join
results = await asyncio.gather(task_a, task_b)
merged = await runner.run(
    input=f"Merge: {results}",
    model="openai/gpt-4"
)
```

## Next Steps

- [Using Deployed Servers](./using-deployed-servers.md)
- [Performance Optimization](./optimization.md)
- [Building Your Own MCP Server](../server/building-mcp-servers.md)