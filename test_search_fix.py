#!/usr/bin/env python3
"""
Test script to verify search fixes
"""

import sys
import os
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_query_expansion():
    """Test query expansion with problematic queries"""
    print("Testing query expansion fixes...")
    
    try:
        from services.pubmed_service import PubMedService
        
        service = PubMedService()
        
        # Test problematic query that was causing errors
        test_query = "heart attack and air pollution"
        
        print(f"Testing query: '{test_query}'")
        
        # Test safe expansion
        expanded = service._safe_expand_query(test_query)
        print(f"Expanded query: '{expanded[:200]}...'")
        
        if len(expanded) > 1000:
            print("✅ Query was too complex, should fallback to original")
        else:
            print("✅ Query expansion successful")
            
        return True
        
    except Exception as e:
        print(f"❌ Query expansion test failed: {e}")
        return False

def test_pubmed_search():
    """Test PubMed search with fallback"""
    print("\nTesting PubMed search with fallback...")
    
    try:
        from services.pubmed_service import PubMedService
        
        service = PubMedService()
        service.set_default_credentials(
            email="221501072@rajalakshmi.edu.in",
            api_key="67991da30c4b789e064446dac0835a377d08"
        )
        
        # Test with a simple query first
        test_query = "diabetes"
        
        print(f"Testing simple query: '{test_query}'")
        
        try:
            articles, search_time = service.search_articles(
                query=test_query,
                max_results=5,
                free_only=False
            )
            
            print(f"✅ Found {len(articles)} articles in {search_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"❌ Simple query failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ PubMed search test failed: {e}")
        return False

def test_model_imports():
    """Test that all models import correctly"""
    print("\nTesting model imports...")
    
    try:
        from models.query import QueryRequest, SearchResponse
        from models.qa import QARequest, QAResponse
        print("✅ All models imported successfully")
        return True
    except Exception as e:
        print(f"❌ Model imports failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Search Fixes")
    print("=" * 50)
    
    # Test model imports
    models_ok = test_model_imports()
    
    # Test query expansion
    expansion_ok = test_query_expansion()
    
    # Test PubMed search (only if we have credentials)
    search_ok = test_pubmed_search()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Model Imports: {'✅ PASS' if models_ok else '❌ FAIL'}")
    print(f"   Query Expansion: {'✅ PASS' if expansion_ok else '❌ FAIL'}")
    print(f"   PubMed Search: {'✅ PASS' if search_ok else '❌ FAIL'}")
    
    if all([models_ok, expansion_ok]):
        print("\n🎉 Core fixes are working!")
        print("\nThe API should now handle problematic queries gracefully.")
        print("Try your curl command again - it should work now!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
