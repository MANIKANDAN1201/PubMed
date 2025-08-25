# PubMed Semantic Search Backend

This folder contains the FastAPI backend for the PubMed Semantic Search System.

## Files

- `main.py` - Main FastAPI application with all endpoints
- `streamlit_app.py` - Streamlit frontend that uses the FastAPI backend
- `start_backend.py` - Startup script for the backend
- `requirements.txt` - Python dependencies

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the backend:**
   ```bash
   python start_backend.py
   ```
   
   Or directly with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Start the Streamlit frontend:**
   ```bash
   streamlit run streamlit_app.py
   ```

## API Endpoints

- `GET /` - Root endpoint with system status
- `GET /health` - Health check
- `POST /api/search` - Basic article search
- `POST /api/search/semantic` - Semantic search with embeddings
- `POST /api/summarize` - Article summarization
- `POST /api/query/expand` - Query expansion
- `POST /api/chat` - Chat functionality
- `GET /api/articles/{pmid}` - Get specific article
- `GET /api/models` - Available models

## Features

- PubMed article search and retrieval
- Semantic search using embeddings
- AI-powered article summarization
- Query expansion with medical terminology
- RESTful API design
- CORS support for frontend integration

## Backend Status

The backend will show warnings for missing components but will continue to run with fallback functionality. Check the startup logs to see which components are available.


