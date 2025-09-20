"""
Advanced workflow examples using the Documentation MCP Server with Dedalus
"""

import asyncio
import json
from typing import List, Dict, Any
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()


class DocumentationWorkflows:
    """Advanced workflows for documentation processing"""
    
    def __init__(self, mcp_server: str = "your-org/docs-server"):
        self.client = AsyncDedalus()
        self.runner = DedalusRunner(self.client)
        self.mcp_server = mcp_server
    
    async def intelligent_routing_workflow(
        self,
        task: str,
        content: str
    ) -> Dict[str, Any]:
        """
        Route tasks to optimal models based on task type
        """
        # Define task-to-model mapping
        model_routing = {
            "code_analysis": {
                "model": "claude-3-5-sonnet-20241022",
                "instructions": "Analyze the code for best practices, potential bugs, and improvements"
            },
            "documentation_generation": {
                "model": "claude-3-opus-20240229",
                "instructions": "Generate comprehensive, well-structured documentation"
            },
            "quick_summary": {
                "model": "openai/gpt-4o-mini",
                "instructions": "Provide a concise summary of the key points"
            },
            "research": {
                "model": "openai/gpt-4.1",
                "instructions": "Research and provide detailed information"
            },
            "creative_expansion": {
                "model": "claude-3-opus-20240229",
                "instructions": "Expand creatively on the given content"
            }
        }
        
        # Get routing configuration
        route_config = model_routing.get(task, model_routing["quick_summary"])
        
        # Execute with selected model
        result = await self.runner.run(
            input=f"{route_config['instructions']}: {content}",
            model=route_config["model"],
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        return {
            "task": task,
            "model_used": route_config["model"],
            "result": result.final_output
        }
    
    async def documentation_improvement_pipeline(
        self,
        doc_path: str
    ) -> Dict[str, Any]:
        """
        Multi-stage pipeline to analyze and improve documentation
        """
        results = {}
        
        # Stage 1: Analyze current documentation
        print(f"Analyzing {doc_path}...")
        analysis = await self.runner.run(
            input=f"Analyze the documentation at {doc_path} for completeness, clarity, and structure",
            model="claude-3-5-sonnet-20241022",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        results["analysis"] = analysis.final_output
        
        # Stage 2: Identify gaps and missing information
        print("Identifying gaps...")
        gaps = await self.runner.run(
            input=f"Based on this analysis, identify gaps and missing information: {analysis.final_output}",
            model="openai/gpt-4",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        results["gaps"] = gaps.final_output
        
        # Stage 3: Generate improvement suggestions
        print("Generating improvements...")
        improvements = await self.runner.run(
            input=f"Generate specific improvement suggestions for these gaps: {gaps.final_output}",
            model="claude-3-opus-20240229",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        results["improvements"] = improvements.final_output
        
        # Stage 4: Create action items
        print("Creating action items...")
        action_items = await self.runner.run(
            input=f"Convert these improvements into specific action items: {improvements.final_output}",
            model="openai/gpt-4o-mini",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        results["action_items"] = action_items.final_output
        
        return results
    
    async def parallel_analysis_workflow(
        self,
        content: str,
        analyses: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run multiple analyses in parallel for faster processing
        """
        if analyses is None:
            analyses = [
                "technical_accuracy",
                "readability",
                "completeness",
                "examples_quality"
            ]
        
        # Create parallel tasks
        tasks = []
        for analysis_type in analyses:
            task = self.runner.run(
                input=f"Analyze this content for {analysis_type}: {content}",
                model="claude-3-5-sonnet-20241022" if "technical" in analysis_type else "openai/gpt-4",
                mcp_servers=[self.mcp_server],
                stream=False
            )
            tasks.append(task)
        
        # Execute in parallel
        print(f"Running {len(analyses)} analyses in parallel...")
        results = await asyncio.gather(*tasks)
        
        # Combine results
        combined = {}
        for analysis_type, result in zip(analyses, results):
            combined[analysis_type] = result.final_output
        
        # Synthesize findings
        print("Synthesizing results...")
        synthesis = await self.runner.run(
            input=f"Synthesize these analysis results into a cohesive report: {json.dumps(combined, indent=2)}",
            model="openai/gpt-4",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        return {
            "individual_analyses": combined,
            "synthesis": synthesis.final_output
        }
    
    async def learning_path_generator(
        self,
        topic: str,
        skill_level: str = "beginner"
    ) -> Dict[str, Any]:
        """
        Generate a personalized learning path from documentation
        """
        # Stage 1: Find relevant documentation
        print(f"Finding documentation for {topic}...")
        docs = await self.runner.run(
            input=f"Find all documentation related to {topic}",
            model="openai/gpt-4o-mini",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        # Stage 2: Analyze complexity and prerequisites
        print("Analyzing complexity...")
        complexity = await self.runner.run(
            input=f"Analyze the complexity and prerequisites for these docs: {docs.final_output}",
            model="claude-3-5-sonnet-20241022",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        # Stage 3: Generate learning path
        print(f"Generating {skill_level} learning path...")
        path = await self.runner.run(
            input=f"Create a {skill_level}-level learning path for {topic} based on: {complexity.final_output}",
            model="claude-3-opus-20240229",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        # Stage 4: Add exercises and examples
        print("Adding exercises...")
        exercises = await self.runner.run(
            input=f"Generate practical exercises for this learning path: {path.final_output}",
            model="openai/gpt-4",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        return {
            "topic": topic,
            "skill_level": skill_level,
            "learning_path": path.final_output,
            "exercises": exercises.final_output,
            "relevant_docs": docs.final_output
        }
    
    async def consensus_review_workflow(
        self,
        content: str,
        question: str
    ) -> Dict[str, Any]:
        """
        Get consensus from multiple models for critical decisions
        """
        models = [
            "claude-3-5-sonnet-20241022",
            "openai/gpt-4",
            "claude-3-opus-20240229"
        ]
        
        # Get responses from all models
        print(f"Getting responses from {len(models)} models...")
        responses = []
        
        for model in models:
            result = await self.runner.run(
                input=f"Based on this content: {content}\n\nAnswer: {question}",
                model=model,
                mcp_servers=[self.mcp_server],
                stream=False
            )
            responses.append({
                "model": model,
                "response": result.final_output
            })
        
        # Have a judge model evaluate consensus
        print("Evaluating consensus...")
        consensus = await self.runner.run(
            input=f"Evaluate these responses and determine the consensus answer:\n{json.dumps(responses, indent=2)}",
            model="openai/gpt-4",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        return {
            "question": question,
            "individual_responses": responses,
            "consensus": consensus.final_output
        }
    
    async def documentation_chat_workflow(
        self,
        conversation_history: List[Dict[str, str]],
        new_question: str
    ) -> str:
        """
        Context-aware chat workflow that maintains conversation history
        """
        # Build context from history
        context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history[-5:]  # Keep last 5 messages
        ])
        
        # Search for relevant documentation
        search = await self.runner.run(
            input=f"Search documentation relevant to: {new_question}",
            model="openai/gpt-4o-mini",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        # Generate response with context
        response = await self.runner.run(
            input=f"""Previous conversation:
{context}

Relevant documentation:
{search.final_output}

User question: {new_question}

Provide a helpful response based on the documentation and conversation context.""",
            model="claude-3-5-sonnet-20241022",
            mcp_servers=[self.mcp_server],
            stream=False
        )
        
        return response.final_output


async def example_workflows():
    """Demonstrate various workflow examples"""
    
    workflows = DocumentationWorkflows()
    
    print("=== Documentation Workflow Examples ===\n")
    
    # Example 1: Intelligent routing
    print("1. Intelligent Routing Workflow")
    routing_result = await workflows.intelligent_routing_workflow(
        task="code_analysis",
        content="def process_data(data): return [x*2 for x in data if x > 0]"
    )
    print(f"Result: {routing_result['result'][:200]}...\n")
    
    # Example 2: Documentation improvement
    print("2. Documentation Improvement Pipeline")
    improvement = await workflows.documentation_improvement_pipeline(
        doc_path="getting-started.md"
    )
    print(f"Action items: {improvement['action_items'][:200]}...\n")
    
    # Example 3: Parallel analysis
    print("3. Parallel Analysis Workflow")
    parallel = await workflows.parallel_analysis_workflow(
        content="MCP servers enable AI models to interact with external tools...",
        analyses=["technical_accuracy", "clarity", "completeness"]
    )
    print(f"Synthesis: {parallel['synthesis'][:200]}...\n")
    
    # Example 4: Learning path generation
    print("4. Learning Path Generator")
    learning_path = await workflows.learning_path_generator(
        topic="agent handoffs",
        skill_level="intermediate"
    )
    print(f"Learning path: {learning_path['learning_path'][:200]}...\n")
    
    # Example 5: Consensus review
    print("5. Consensus Review Workflow")
    consensus = await workflows.consensus_review_workflow(
        content="Agent handoffs enable sophisticated multi-model workflows...",
        question="What is the most important benefit of agent handoffs?"
    )
    print(f"Consensus: {consensus['consensus'][:200]}...\n")
    
    # Example 6: Documentation chat
    print("6. Documentation Chat Workflow")
    chat_response = await workflows.documentation_chat_workflow(
        conversation_history=[
            {"role": "user", "content": "What is MCP?"},
            {"role": "assistant", "content": "MCP is the Model Context Protocol..."}
        ],
        new_question="How do I implement tools in MCP?"
    )
    print(f"Chat response: {chat_response[:200]}...\n")


if __name__ == "__main__":
    asyncio.run(example_workflows())