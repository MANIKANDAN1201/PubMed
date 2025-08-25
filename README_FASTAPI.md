# PubMed Semantic Search - FastAPI Backend

A modular FastAPI backend for advanced biomedical literature search with AI-powered semantic understanding.

## 🏗️ Architecture

### **Modular Structure:**
```
PubMed/
├── app.py                 # FastAPI entrypoint
├── routers/              # API route handlers
│   ├── search.py         # /api/v1/search
│   ├── qa.py            # /api/v1/qa
│   ├── summary.py       # /api/v1/summary (redirects to qa)
│   ├── index.py         # /api/v1/index
│   └── benchmark.py     # /api/v1/benchmark
├── services/            # Business logic
│   ├── pubmed_service.py    # PubMed fetching & processing
│   ├── embedding_service.py # Text embeddings
│   └── qa_service.py        # Q&A & summary generation
├── models/              # Pydantic schemas
│   ├── query.py         # Search request/response models
│   └── qa.py           # Q&A request/response models
└── [existing files]     # All original functionality preserved
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the Structure
```bash
python test_fastapi.py
```

### 3. Run the Server
```bash
# Option 1: Using app.py
python app.py

# Option 2: Using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the API
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

### **Search Endpoints**
- `POST /api/v1/search` - Search PubMed articles
- `GET /api/v1/search/health` - Search service health
- `GET /api/v1/search/stats` - Search service statistics

### **Q&A Endpoints**
- `POST /api/v1/qa` - Generate Q&A responses
- `POST /api/v1/qa/summary` - Generate research summaries
- `GET /api/v1/qa/health` - Q&A service health
- `GET /api/v1/qa/models` - Supported LLM models

### **Other Endpoints**
- `GET /` - API information
- `GET /health` - Global health check

## 🔧 Usage Examples

### **Search Articles**
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "diabetes treatment",
    "max_results": 10,
    "use_reranking": true,
    "free_only": false
  }'
```

### **Generate Q&A Response**
```bash
curl -X POST "http://localhost:8000/api/v1/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main symptoms of diabetes?",
    "articles": [...],  # Articles from search
    "max_articles": 10,
    "model": "llama3.2"
  }'
```

### **Generate Summary**
```bash
curl -X POST "http://localhost:8000/api/v1/qa/summary" \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [...],  # Articles from search
    "max_articles": 15,
    "model": "llama3.2"
  }'
```

## 🛡️ Error Handling

The API includes comprehensive error handling:

- **503 Service Unavailable**: When dependencies are missing
- **404 Not Found**: When no articles are found
- **500 Internal Server Error**: For processing failures
- **Graceful Degradation**: Services work even with missing dependencies

## 🔍 Health Checks

### **Global Health**
```bash
curl http://localhost:8000/health
```

### **Service-Specific Health**
```bash
# Search service
curl http://localhost:8000/api/v1/search/health

# Q&A service
curl http://localhost:8000/api/v1/qa/health
```

## 📊 Response Format

### **Search Response**
```json
{
  "query": "diabetes treatment",
  "total_results": 15,
  "articles": [
    {
      "pmid": "12345678",
      "title": "Novel Diabetes Treatment",
      "abstract": "This study examines...",
      "journal": "Journal of Medicine",
      "year": "2023",
      "authors": ["Smith J", "Johnson A"],
      "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
      "final_score": 0.85,
      "semantic_score": 0.78,
      "keyword_score": 0.92,
      "rank": 1
    }
  ],
  "search_time": 2.34,
  "timestamp": "2024-01-15T10:30:00Z",
  "use_reranking": true,
  "use_flashrank": false,
  "free_only": false
}
```

### **Q&A Response**
```json
{
  "response": "Based on the research articles...",
  "question": "What are the main symptoms?",
  "articles_used": 8,
  "processing_time": 2.34,
  "model": "llama3.2",
  "context_length": 15000,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🔧 Configuration

### **Environment Variables**
- `ENTREZ_EMAIL`: PubMed API email (recommended)
- `ENTREZ_API_KEY`: PubMed API key (optional)

### **Service Configuration**
- **Embedding Model**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- **LLM Model**: `llama3.2` (requires Ollama)
- **Vector Store**: FAISS with hybrid search

## 🧪 Testing

### **Run Tests**
```bash
python test_fastapi.py
```

### **Manual Testing**
1. Start the server
2. Visit http://localhost:8000/docs
3. Use the interactive API documentation
4. Test endpoints with sample data

## 🔄 Migration from Streamlit

### **What's Preserved:**
- ✅ All existing functionality
- ✅ PubMed fetching and processing
- ✅ Semantic search with embeddings
- ✅ Hybrid search (semantic + keyword)
- ✅ Reranking capabilities
- ✅ Q&A chatbot functionality
- ✅ Summary generation
- ✅ Evaluation and benchmarking tools

### **What's New:**
- 🆕 RESTful API endpoints
- 🆕 Modular service architecture
- 🆕 Pydantic validation
- 🆕 Comprehensive error handling
- 🆕 Health checks and monitoring
- 🆕 Auto-generated documentation
- 🆕 Production-ready structure

## 🚨 Troubleshooting

### **Common Issues:**

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **PubMed API Limits**
   - Set `ENTREZ_EMAIL` environment variable
   - Consider getting an API key

3. **Ollama Not Available**
   - Install Ollama: https://ollama.ai/
   - Pull model: `ollama pull llama3.2`

4. **Memory Issues**
   - Reduce `max_results` in search requests
   - Use smaller embedding models

### **Logs**
Check the console output for detailed error messages and service status.

## 📈 Performance

### **Optimizations:**
- **Caching**: Embeddings and vector indices
- **Batch Processing**: Efficient embedding generation
- **Async Processing**: Non-blocking API responses
- **Graceful Degradation**: Services work with missing dependencies

### **Expected Performance:**
- **Search**: 2-5 seconds for 20 articles
- **Q&A**: 2-10 seconds depending on context size
- **Summary**: 3-15 seconds for 15 articles

## 🔮 Future Enhancements

- [ ] **Authentication**: JWT-based auth
- [ ] **Rate Limiting**: API usage limits
- [ ] **Caching**: Redis for responses
- [ ] **Monitoring**: Prometheus metrics
- [ ] **Docker**: Containerized deployment
- [ ] **Kubernetes**: Orchestration support

## 📝 License

This project maintains the same license as the original PubMed semantic search project.
