#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from sql_chain import get_enhanced_sql_chain

def test_sql_queries():
    """Test SQL queries with debugging"""
    print("🔍 Testing SQL Chain Debug")
    
    sql_chain = get_enhanced_sql_chain()
    if not sql_chain:
        print("❌ SQL chain not available")
        return
    
    test_queries = [
        "How many ARGO measurements do we have in total?",
        "What is the temperature range in our dataset?",
        "Show me all unique platform numbers"
    ]
    
    for question in test_queries:
        print(f"\n📝 Testing: {question}")
        print("=" * 50)
        
        try:
            result = sql_chain.invoke({"question": question})
            print(f"✅ Result: {result[:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 50)

if __name__ == "__main__":
    test_sql_queries()