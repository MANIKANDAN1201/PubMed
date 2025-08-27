# PubMed Semantic Search (TEAM 8)

![PubMed Semantic Search](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An advanced biomedical literature search platform powered by AI, enabling semantic search, summarization, and conversational assistance for PubMed articles.

## 🚀 Features

### Core Search
- **Hybrid Search** - Combines semantic (FAISS) and keyword (TF-IDF) search
- **Multiple Embedding Models** - Supports Gemini, Sentence Transformers, PubMedBERT, and BioBERT
- **Query Expansion** - Automatically enhances queries with medical synonyms and MeSH terms
- **Intelligent Reranking** - Prioritizes recent, high-impact papers using advanced algorithms

### User Experience
- **Interactive UI** - Modern, responsive interface with dark/light theme support
- **Research Assistant** - AI-powered chatbot for natural language queries
- **Article Summarization** - Generate concise summaries of search results
- **Export Capabilities** - Download search results in CSV format

### Performance
- **Vector Caching** - Persistent storage of embeddings for faster searches
- **Efficient Indexing** - Optimized FAISS indices for quick retrieval
- **Asynchronous Processing** - Non-blocking UI during search operations

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.10+** - Core programming language
- **Streamlit** - Web application framework
- **FAISS** - Library for efficient similarity search
- **Sentence Transformers** - For generating document embeddings
- **PubMed API** - Access to biomedical literature database
- **Ollama** - Local LLM for the research assistant
- **FlashRank** - Advanced reranking of search results
- **scikit-learn** - For TF-IDF vectorization

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pubmed-semantic-search.git
   cd pubmed-semantic-search
   ```

2. **Set up a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Add your API keys and configuration

5. **Run the application**
   ```bash
   streamlit run app.py
   ```
   The application will be available at `http://localhost:8501`

## 📚 Usage

### Search Interface
1. Enter your medical query in the search bar
2. Adjust search settings as needed
3. View and interact with search results
4. Use filters to refine your search

### Research Assistant
1. Navigate to the Chatbot tab
2. Ask questions about your research topic
3. The AI will provide relevant information from PubMed

### Article Summarization
1. Perform a search
2. Select articles of interest
3. Generate a summary of the selected articles


## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PubMed for providing access to biomedical literature
- Hugging Face for transformer models
- Streamlit for the amazing UI framework
- Ollama for local language model support
   ollama pull llama3.2

## 🚀 Usage
   - Choose between:
     - **💬 Ask Questions**: Interactive Q&A about the research findings
     - **📋 Generate Summary**: Get a comprehensive summary of the top articles
   - Adjust the number of abstracts to use as context
   - Ask questions or generate summaries based on the retrieved research

### Usage

- Enter your medical query in the main text box
- (Optional) Provide your Entrez email and API key in the sidebar for higher PubMed rate limits
- Select embedding model and backend in the sidebar
- Choose how many articles to fetch and how many results to display
- Enable/disable query expansion, reranking, and index persistence as needed
- Click "Search" to view results
- Use "Clear cache" in the sidebar to reset cached data

### Notes

- If you select the "Sentence-Transformers" backend but the chosen model is not a Sentence-Transformers checkpoint, the app will fall back to a Transformers model with mean pooling.
- Embeddings and PubMed responses are cached. Use the "Clear cache" button in the sidebar to reset.
- Saved indices are stored in `vector_cache/` for fast reuse.

### Troubleshooting

- If FAISS installation fails, ensure you are installing `faiss-cpu` (not `faiss`) and that your Python version is supported.
- If model downloads are slow or blocked, try setting the environment variable `HF_HUB_DISABLE_TELEMETRY=1` and retry, or pre-download models via `transformers` CLI.
- For Gemini embeddings, set your `GOOGLE_API_KEY` in `.env`.

### License

This project is provided as-is for educational purposes.

### Usage

- Enter your PubMed query in the main text box
- (Optional) Provide your Entrez email and API key in the sidebar for higher rate limits
- Choose model and backend in the sidebar
- Select how many articles to fetch and how many results to display
- Click "Search"

### Notes

- If you select the "Sentence-Transformers" backend but the chosen model is not a Sentence-Transformers checkpoint, the app will fall back to a Transformers model with mean pooling.
- Embeddings and PubMed responses are cached. Use the "Clear cache" button in the sidebar to reset.

### Troubleshooting

- If FAISS installation fails, ensure you are installing `faiss-cpu` (not `faiss`) and that your Python version is supported.
- If model downloads are slow or blocked, try setting the environment variable `HF_HUB_DISABLE_TELEMETRY=1` and retry, or pre-download models via `transformers` CLI.

### License

This project is provided as-is for educational purposes.
