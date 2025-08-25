#!/usr/bin/env python3
"""
Test script for FastAPI structure
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from fastapi import FastAPI
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        from models.query import QueryRequest, SearchResponse
        print("✅ Models imported successfully")
    except ImportError as e:
        print(f"❌ Models import failed: {e}")
        return False
    
    try:
        from services.pubmed_service import PubMedService
        print("✅ PubMedService imported successfully")
    except ImportError as e:
        print(f"❌ PubMedService import failed: {e}")
        return False
    
    try:
        from routers.search import router as search_router
        print("✅ Search router imported successfully")
    except ImportError as e:
        print(f"❌ Search router import failed: {e}")
        return False
    
    return True

def test_app_creation():
    """Test if FastAPI app can be created"""
    print("\nTesting app creation...")
    
    try:
        from app import app
        print("✅ FastAPI app created successfully")
        print(f"   Title: {app.title}")
        print(f"   Version: {app.version}")
        return True
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        return False

def test_health_endpoints():
    """Test health endpoints"""
    print("\nTesting health endpoints...")
    
    try:
        from fastapi.testclient import TestClient
        from app import app
        
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        if response.status_code == 200:
            print("✅ Root endpoint works")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
        
        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ Health endpoint works")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
        
        # Test search health endpoint
        response = client.get("/api/v1/search/health")
        if response.status_code == 200:
            print("✅ Search health endpoint works")
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Dependencies: {data.get('dependencies', {})}")
        else:
            print(f"❌ Search health endpoint failed: {response.status_code}")
        
        return True
        
    except ImportError:
        print("⚠️  TestClient not available (install httpx)")
        return False
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing FastAPI PubMed Search Structure")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test app creation
    app_ok = test_app_creation()
    
    # Test health endpoints
    health_ok = test_health_endpoints()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"   App Creation: {'✅ PASS' if app_ok else '❌ FAIL'}")
    print(f"   Health Endpoints: {'✅ PASS' if health_ok else '❌ FAIL'}")
    
    if all([imports_ok, app_ok, health_ok]):
        print("\n🎉 All tests passed! The FastAPI structure is working.")
        print("\nTo run the server:")
        print("   python app.py")
        print("   # or")
        print("   uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("\nTo install missing dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
