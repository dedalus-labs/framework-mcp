"""
Example client for interacting with the Documentation MCP Server via Dedalus

To use this client:
1. Install the Dedalus client: pip install dedalus-labs
2. Set your API key: export OPENAI_API_KEY="your-key" or ANTHROPIC_API_KEY="your-key"
3. Run the examples
"""

import asyncio
from typing import List, Dict, Any
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()


class DocumentationClient:
    """Client for interacting with documentation MCP server"""
    
    def __init__(self, mcp_server: str = "your-org/docs-server"):
        self.client = AsyncDedalus()
        self.runner = DedalusRunner(self.client)
        self.mcp_server = mcp_server
    
    async def list_all_docs(self) -> List[Dict[str, Any]]:
        """List all available documentation"""
        result = await self.runner.run(
            input="List all available documentation files",
            model="openai/gpt-4o-mini",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        return result.final_output
    
    async def search_documentation(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search documentation for specific topics"""
        result = await self.runner.run(
            input=f"Search documentation for: {query}. Return up to {max_results} results.",
            model="openai/gpt-4o-mini",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        return result.final_output
    
    async def ask_question(
        self,
        question: str,
        context_docs: List[str] = None
    ) -> str:
        """Ask a question about the documentation"""
        prompt = f"Answer this question based on the documentation: {question}"
        
        if context_docs:
            prompt += f"\nUse these specific documents: {', '.join(context_docs)}"
        
        result = await self.runner.run(
            input=prompt,
            model="claude-3-5-sonnet-20241022",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        return result.final_output
    
    async def analyze_documentation(
        self,
        task: str = "find_gaps",
        output_format: str = "summary"
    ) -> Dict[str, Any]:
        """Analyze documentation for various purposes"""
        result = await self.runner.run(
            input=f"Analyze the documentation to {task}. Format: {output_format}",
            model="openai/gpt-4",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        return result.final_output
    
    async def multi_stage_research(
        self,
        topic: str
    ) -> Dict[str, str]:
        """Multi-stage research with different models"""
        
        # Stage 1: Initial search with fast model
        print(f"Stage 1: Searching for {topic}...")
        search = await self.runner.run(
            input=f"Search documentation for information about {topic}",
            model="openai/gpt-4o-mini",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        # Stage 2: Deep analysis with Claude
        print("Stage 2: Analyzing findings...")
        analysis = await self.runner.run(
            input=f"Analyze these search results and extract key insights: {search.final_output}",
            model="claude-3-5-sonnet-20241022",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        # Stage 3: Create summary with GPT-4
        print("Stage 3: Creating comprehensive summary...")
        summary = await self.runner.run(
            input=f"Create a comprehensive summary based on this analysis: {analysis.final_output}",
            model="openai/gpt-4",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        return {
            "search_results": search.final_output,
            "analysis": analysis.final_output,
            "summary": summary.final_output
        }
    
    async def streaming_query(
        self,
        query: str,
        model: str = "openai/gpt-4o-mini"
    ):
        """Stream responses for real-time output"""
        result = self.runner.run(
            input=query,
            model=model,
            mcp_servers=[self.mcp_server],
            stream=True
        )
        
        # Stream the output
        async for chunk in result:
            print(chunk, end="", flush=True)
        print()  # New line at the end


async def main():
    """Example usage of the Documentation Client"""
    
    # Initialize client
    client = DocumentationClient()
    
    print("=== Documentation MCP Client Examples ===\n")
    
    # Example 1: List all documentation
    print("1. Listing all documentation files...")
    docs = await client.list_all_docs()
    print(f"Found documents: {docs}\n")
    
    # Example 2: Search for specific topic
    print("2. Searching for 'agent handoffs'...")
    search_results = await client.search_documentation("agent handoffs", max_results=3)
    print(f"Search results: {search_results}\n")
    
    # Example 3: Ask a question
    print("3. Asking a question about MCP...")
    answer = await client.ask_question(
        "How do I implement streaming responses in MCP?",
        context_docs=["advanced-tools.md"]
    )
    print(f"Answer: {answer}\n")
    
    # Example 4: Analyze documentation
    print("4. Analyzing documentation for gaps...")
    analysis = await client.analyze_documentation(
        task="find_gaps",
        output_format="detailed"
    )
    print(f"Analysis: {analysis}\n")
    
    # Example 5: Multi-stage research
    print("5. Conducting multi-stage research on 'model handoffs'...")
    research = await client.multi_stage_research("model handoffs")
    print(f"Research complete. Summary: {research['summary']}\n")
    
    # Example 6: Streaming response
    print("6. Streaming a response...")
    await client.streaming_query(
        "Explain the concept of agent handoffs in one paragraph"
    )


if __name__ == "__main__":
    asyncio.run(main())