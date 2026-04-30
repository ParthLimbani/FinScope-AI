# FinScope AI

A production-grade RAG system for querying 20 public fintech research and regulatory documents (Arxiv, RBI, BIS, SEBI) with hybrid retrieval, neural reranking, citation enforcement, and CI-gated evaluation.

---

## Architecture

```
PDFs (data/)
    │
    ├── BM25 Index (rank_bm25)
    └── FAISS Index (sentence-transformers)
              │
         User Query
              │
     Hybrid Retrieval (BM25 + FAISS → RRF, top 20)
              │
     Cross-Encoder Reranking (ms-marco-MiniLM, top 5)
              │
     Groq LLM (LLaMA 3.3 70B) with citation enforcement
              │
     Citation Validation + Structured Response
              │
     FastAPI  (/api/v1/query  /api/v1/ingest  /api/v1/health)
```

---

## Results

Evaluated on 15 hand-crafted Q&A pairs spanning all four document sources using [RAGAS 0.4](https://github.com/explodinggradients/ragas) with LLaMA 3.3 70B as the judge LLM.

| Metric | Score | CI Gate |
|---|---|---|
| Faithfulness | **0.84** | ≥ 0.80 ✓ |
| Answer Relevancy | **0.79** | — |
| Context Recall | **0.73** | — |
| Avg Query Latency | **1.4 s** | — |

> Faithfulness measures whether every claim in the answer is grounded in the retrieved context. The CI gate blocks any PR that drops it below 0.80.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Groq API — LLaMA 3.3 70B |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB | FAISS (local, persisted) |
| Sparse Retrieval | `rank_bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| PDF Parsing | PyMuPDF |
| API Framework | FastAPI + Pydantic v2 |
| Evaluation | RAGAS |
| CI | GitHub Actions |
| Deployment | Render (Python native) |

---

## Running Locally

### 1. Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### 2. Install

```bash
git clone https://github.com/yourusername/finscope-ai
cd finscope-ai
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### 4. Build indexes (first run only — takes ~5 minutes)

```bash
python -m src.ingestion.loader
```

This loads all PDFs, chunks them, and writes `indexes/bm25_index.pkl` and `indexes/faiss_index/`.

### 5. Run the API server

```bash
uvicorn src.api.main:app --reload --port 8000
```

The server starts at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 6. Example query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RBI'\''s approach to CBDC?", "top_k": 5}'
```

### 7. Run unit tests

```bash
pytest -v
```

---

## Running Evaluations

The RAGAS evaluation pipeline measures faithfulness, answer relevancy, and context recall across 15 hand-crafted Q&A pairs.

### Prerequisites

Indexes must be built and `GROQ_API_KEY` must be set.

### Run

```bash
python evaluation/eval_runner.py
```

Output:

```
============================================================
  FinScope AI — RAGAS Evaluation Summary
  CI gate: faithfulness >= 0.80
============================================================
  Metric                    Score   Status
  --------------------------------------------------------
  Faithfulness             0.8400   PASS ✓
  Answer Relevancy         0.7900   PASS ✓
  Context Recall           0.7300   PASS ✓
============================================================
```

Results are saved to `evaluation/eval_results.json` after every run.

### CI Gate

GitHub Actions runs this pipeline on every PR to `main`. Any PR that drops faithfulness below `0.80` is automatically blocked.

---

## Deploying to Render

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/finscope-ai
git push -u origin main
```

### 2. Add GitHub secret

In your repo: **Settings → Secrets and variables → Actions → New repository secret**
- Name: `GROQ_API_KEY`
- Value: your Groq API key

### 3. Deploy on Render

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repo
3. Set:
   - **Environment**: Python
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable `GROQ_API_KEY` in Render's dashboard
5. Click **Deploy**

> **Note**: Pre-build the indexes locally and commit the `indexes/` folder, or set the build command to `pip install -r requirements.txt && python -m src.ingestion.loader`. The free tier has limited RAM — the embedding model (~90 MB) and FAISS index load fine on 512 MB.

### 4. Health check

```bash
curl https://your-service.onrender.com/api/v1/health
# {"status":"healthy","indexes_loaded":true,"model":"llama-3.3-70b-versatile"}
```

---

## API Reference

### `POST /api/v1/query`

```json
// Request
{ "question": "What is RBI's stance on CBDC?", "top_k": 5 }

// Response
{
  "answer": "The RBI has outlined... [chunk_id]...",
  "citations": [
    {
      "chunk_id": "550e8400-...",
      "source": "rbi",
      "filename": "rbi_cbdc_2023.pdf",
      "page_number": 14,
      "snippet": "The Reserve Bank of India has outlined...",
      "ce_score": 0.91,
      "rrf_score": 0.032
    }
  ],
  "citation_validation": { "all_valid": true, "invalid": [], "uncited_chunks": [] },
  "metadata": { "retrieved": 20, "reranked_to": 5, "latency_ms": 3241.5 }
}
```

### `POST /api/v1/ingest`

```json
// Request
{ "data_dir": "data/" }

// Response
{ "status": "ok", "chunks_indexed": 12480, "index_build_time_ms": 284320.0 }
```

### `GET /api/v1/health`

```json
{ "status": "healthy", "indexes_loaded": true, "model": "llama-3.3-70b-versatile" }
```

---

*FinScope AI — Built for production, tested with RAGAS, deployed with Docker.*
