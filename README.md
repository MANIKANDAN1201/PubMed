## Streamlit PubMed Semantic Search

This app lets you search PubMed and retrieve semantically relevant papers using biomedical transformer models (PubMedBERT / BioBERT) and a FAISS vector index.

### Features
- Semantic search over PubMed abstracts using embeddings
- **Models**: PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`) or BioBERT (`dmis-lab/biobert-base-cased-v1.1`)
- **FAISS** (local) vector database for fast similarity search
- Optional simple query expansion with synonyms
- Caching to avoid redundant PubMed requests and re-embeddings
- Clean, card-like Streamlit UI with links to PubMed

### Requirements
- Python 3.10+
- Windows, macOS, or Linux

### Setup
1. Create and activate a virtual environment

Windows (cmd):
```bat
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux (bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -U pip
pip install -r PubMed/requirements.txt
```

3. Run the app
```bash
streamlit run PubMed/app.py
```

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
