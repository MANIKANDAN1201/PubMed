## PubMed Semantic Search(TEAM 8)

This Streamlit app enables advanced semantic and hybrid search over PubMed biomedical literature. It leverages transformer-based models (Gemini, Sentence Transformers, PubMedBERT, BioBERT), FAISS for semantic search, TF-IDF for keyword search, and intelligent reranking. The app supports query expansion with medical synonyms and MeSH terms, persistent vector index caching, and modern UI.

### Features

- **Hybrid Search:** Combines semantic (FAISS) and keyword (TF-IDF) search for best result
- **Multiple Embedding Models:** Gemini, Sentence Transformers, PubMedBERT, BioBERT
- **Query Expansion:** Uses medical synonyms and MeSH terms from PubMed
- **Intelligent Reranking:** Boosts recent, high-impact papers
- **Persistent Indexing:** Saves embeddings and indices for fast reuse
- **Enhanced UI:** Card-style results, download options, index statistics
- **Free Full-Text Detection:** PMC and Unpaywall integration
- **Research Assistant Chatbot:** AI-powered conversational interface using Ollama models

### Requirements

- Python 3.10+
- Windows, macOS, or Linux

### Setup

1. **Create and activate a virtual environment**
   - Windows (cmd):
     ```
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - macOS/Linux (bash):
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```
2. **Install dependencies**
   ```
   pip install -U pip
   pip install -r requirements.txt
   ```
3. **Set up environment variables**
   - Copy `env_template.txt` to `.env` and add your `GOOGLE_API_KEY` for Gemini embeddings (optional, but recommended for Gemini model).
4. **Run the app**
   ```
   streamlit run app.py
   ```

### Chatbot Setup (Optional)

The app includes a research assistant chatbot that uses Ollama models for conversational AI:

1. **Install Ollama**
   - Visit [ollama.ai](https://ollama.ai) and download for your platform
   - Or use: `curl -fsSL https://ollama.ai/install.sh | sh`

2. **Start Ollama service**
   ```
   ollama serve
   ```

3. **Download the model**
   ```
   ollama pull llama3.2
   ```

4. **Use the chatbot**
   - Perform a PubMed search first
   - Click on the "🤖 Research Assistant Chatbot" expandable panel
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
