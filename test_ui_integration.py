#!/usr/bin/env python3
"""
Test script to verify UI integration and functionality
"""

import requests
import json
import time

def test_search_endpoint():
    """Test the search endpoint"""
    print("🔍 Testing Search Endpoint...")
    
    url = "http://localhost:8000/api/v1/search"
    
    # Test data
    data = {
        "query": "diabetes treatment",
        "max_results": 20,
        "use_reranking": True,
        "use_flashrank": False,
        "free_only": False,
        "email": "test@example.com"
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Search successful!")
            print(f"   - Query: {result['query']}")
            print(f"   - Total results: {result['total_results']}")
            print(f"   - Search time: {result['search_time']:.2f}s")
            print(f"   - Articles returned: {len(result['articles'])}")
            
            if result['articles']:
                first_article = result['articles'][0]
                print(f"   - Top result: {first_article['title'][:100]}...")
                print(f"   - Score: {first_article['final_score']:.3f}")
            
            return result['articles']
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []

def test_qa_endpoint(articles):
    """Test the QA endpoint"""
    if not articles:
        print("⚠️  Skipping QA test - no articles available")
        return
    
    print("\n🤖 Testing QA Endpoint...")
    
    url = "http://localhost:8000/api/v1/qa"
    
    # Test data
    data = {
        "question": "What are the main treatments for diabetes?",
        "articles": articles[:5],  # Use first 5 articles
        "max_articles": 5,
        "model": "llama3.2"
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ QA successful!")
            print(f"   - Question: {result['question']}")
            print(f"   - Articles used: {result['articles_used']}")
            print(f"   - Processing time: {result['processing_time']:.2f}s")
            print(f"   - Response: {result['response'][:200]}...")
        else:
            print(f"❌ QA failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ QA error: {e}")

def test_summary_endpoint(articles):
    """Test the summary endpoint"""
    if not articles:
        print("⚠️  Skipping summary test - no articles available")
        return
    
    print("\n📋 Testing Summary Endpoint...")
    
    url = "http://localhost:8000/api/v1/summary"
    
    # Test data
    data = {
        "articles": articles[:10],  # Use first 10 articles
        "max_articles": 10,
        "model": "llama3.2"
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Summary successful!")
            print(f"   - Articles used: {result['articles_used']}")
            print(f"   - Processing time: {result['processing_time']:.2f}s")
            print(f"   - Summary: {result['summary'][:300]}...")
        else:
            print(f"❌ Summary failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Summary error: {e}")

def test_health_endpoints():
    """Test health endpoints"""
    print("\n🏥 Testing Health Endpoints...")
    
    endpoints = [
        ("/health", "Main API"),
        ("/api/v1/search/health", "Search Service"),
        ("/api/v1/qa/health", "QA Service")
    ]
    
    for endpoint, service in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {service}: Healthy")
            else:
                print(f"❌ {service}: {response.status_code}")
        except Exception as e:
            print(f"❌ {service}: Error - {e}")

def main():
    """Main test function"""
    print("🧪 Testing PubMed Semantic Search UI Integration")
    print("=" * 60)
    
    # Test health endpoints first
    test_health_endpoints()
    
    # Test search functionality
    articles = test_search_endpoint()
    
    # Test QA functionality
    test_qa_endpoint(articles)
    
    # Test summary functionality
    test_summary_endpoint(articles)
    
    print("\n" + "=" * 60)
    print("🎉 Integration test completed!")
    print("\nIf all tests passed, your UI should be working correctly.")
    print("Visit http://localhost:8000 to use the beautiful interface!")

if __name__ == "__main__":
    main()
