#!/usr/bin/env python
"""
Test rate limiting functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import ask_docs

def test_rate_limiting():
    """Test that rate limiting works correctly"""
    print("Testing Rate Limiting (10 requests per minute)")
    print("=" * 50)
    
    # Test with a specific user ID
    test_user = "test_user_123"
    
    # Make 10 requests (should all succeed)
    print("\nMaking 10 requests (should all succeed)...")
    for i in range(10):
        result = ask_docs(
            question=f"Test question {i+1}",
            user_id=test_user
        )
        if "error" in result and result["error"] == "Rate limit exceeded":
            print(f"  Request {i+1}: FAILED (rate limited) - Unexpected!")
            return False
        else:
            print(f"  Request {i+1}: OK")
    
    # Try one more request (should fail)
    print("\nMaking 11th request (should be rate limited)...")
    result = ask_docs(
        question="This should be rate limited",
        user_id=test_user
    )
    
    if "error" in result and result["error"] == "Rate limit exceeded":
        print("  Request 11: Rate limited as expected!")
        print(f"  Reset in: {result['reset_in_seconds']} seconds")
        print(f"  Message: {result['message']}")
        success = True
    else:
        print("  Request 11: NOT rate limited - This is a problem!")
        success = False
    
    # Test that different users have separate limits
    print("\nTesting different user has separate limit...")
    result = ask_docs(
        question="Different user test",
        user_id="different_user"
    )
    if "error" in result and result["error"] == "Rate limit exceeded":
        print("  Different user: FAILED (should not be rate limited)")
        success = False
    else:
        print("  Different user: OK (not rate limited)")
    
    # Test default user (no user_id provided)
    print("\nTesting default user (no user_id)...")
    result = ask_docs(
        question="Default user test"
    )
    print(f"  Default user: {'Rate limited' if 'error' in result else 'OK'}")
    
    print("\n" + "=" * 50)
    if success:
        print("[SUCCESS] Rate limiting is working correctly!")
    else:
        print("[FAILED] Rate limiting has issues")
    
    return success

if __name__ == "__main__":
    success = test_rate_limiting()
    sys.exit(0 if success else 1)