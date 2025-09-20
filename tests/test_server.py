#!/usr/bin/env python
"""
Test script to verify the MCP server is working correctly
"""

import sys
import os

# Add parent directory to path to find src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_server():
    """Test all server functionality"""
    print("Testing Dedalus Documentation MCP Server\n")
    
    # Import server functions
    try:
        from src.main import list_docs, search_docs, ask_docs, index_docs
        print("[OK] Server modules imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import server: {e}")
        return False
    
    # Test 1: List documents
    print("\nTest 1: Listing documents...")
    docs = list_docs()
    if docs:
        print(f"[OK] Found {len(docs)} documents:")
        for doc in docs[:3]:
            print(f"   - {doc['title']}")
    else:
        print("[ERROR] No documents found")
        return False
    
    # Test 2: Search documents
    print("\nTest 2: Searching documents...")
    results = search_docs("hackathon", max_results=2)
    if results:
        print(f"[OK] Search found {len(results)} results for 'hackathon':")
        for r in results:
            print(f"   - {r['title']} (score: {r.get('relevance_score', 0)})")
    else:
        print("[WARNING] No search results found")
    
    # Test 3: AI Q&A
    print("\nTest 3: AI Question & Answer...")
    api_key_exists = bool(os.getenv("OPENAI_API_KEY"))
    
    if not api_key_exists:
        print("[WARNING] No OpenAI API key found in environment")
        print("   To enable AI responses:")
        print("   1. Copy .env.example to .env.local")
        print("   2. Add your OpenAI API key")
        
        # Test without API key
        result = ask_docs("What is MCP?")
        if "note" in result:
            print("[OK] Server handles missing API key correctly")
            print(f"   Note: {result['note'][:50]}...")
    else:
        print("[OK] OpenAI API key found")
        
        # Test with API key
        result = ask_docs(
            "What is the total prize pool for the hackathon?",
            context_docs=["hackathon/yc-agents-hackathon.md"]
        )
        
        if result.get("model"):
            print("[OK] AI responded successfully:")
            print(f"   Model: {result['model']}")
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Sources: {result.get('sources', [])}")
        else:
            print(f"[ERROR] AI failed to respond: {result.get('error', 'Unknown error')}")
            return False
    
    # Test 4: Index documents
    print("\nTest 4: Document indexing...")
    index_result = index_docs()
    if index_result:
        print(f"[OK] Indexed {index_result['files_indexed']} files")
        print(f"   Total size: {index_result['total_size']} bytes")
    else:
        print("[ERROR] Indexing failed")
        return False
    
    print("\n" + "="*50)
    print("[SUCCESS] All tests passed! Server is ready.")
    print("\nTo deploy to Dedalus:")
    print("  dedalus deploy ./src/server.py --name 'hackathon-docs-server'")
    print("\nTo use with Dedalus SDK:")
    print("  mcp_servers=['your-org/hackathon-docs-server']")
    
    return True

if __name__ == "__main__":
    success = test_server()
    sys.exit(0 if success else 1)