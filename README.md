# NLP Mini Project — Marathi News Intelligence Pipeline

A seven-task NLP system for Marathi news (morphology, categorization,
summarization, entity extraction, sentiment, translation, retrieval), built
around a morphological analyser written from scratch.

| Path | What it is |
|---|---|
| [`Marathi-news-intelligence-pipeline/`](Marathi-news-intelligence-pipeline/) | The code — see its [README](Marathi-news-intelligence-pipeline/README.md) and [GUIDE](Marathi-news-intelligence-pipeline/GUIDE.md) |
| [`Marathi-news-intelligence-pipeline/Marathi-NLP-Demo.html`](Marathi-news-intelligence-pipeline/Marathi-NLP-Demo.html) | Standalone demo page — download and double-click, no install needed |
| `NLP_Mini_Project_Deck.pptx`, `nlp ppt.pdf` | Presentation slides |

## Quick start

**Demo page (no Python):** open `Marathi-news-intelligence-pipeline/Marathi-NLP-Demo.html`
in any browser and paste Marathi text into any tab. To have the page use the
full models, run `python app/server.py` from the project folder first.

**Full system (Python 3.10+):**

```bash
cd Marathi-news-intelligence-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py                  # XL-Sum Marathi + MahaNews
python scripts/build_corpus.py --limit 3000      # runs every model over 3,000 articles
python scripts/build_index.py --plain-bm25       # BM25 (+ plain ablation) and dense index
streamlit run app/dashboard.py
```

The first run downloads about 3 GB of models from the Hugging Face Hub. The
code runs on NVIDIA GPUs (CUDA), Apple Silicon (MPS), or CPU, picking one
automatically.
