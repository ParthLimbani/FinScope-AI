# FinScope AI

A production-grade RAG system for querying 20 public fintech research and regulatory documents (Arxiv, RBI, BIS, SEBI) with hybrid retrieval, neural reranking, citation enforcement, CI-gated evaluation, and a live PDF upload feature.

---

## Architecture

The system operates in two modes controlled by the `USE_LOCAL_MODELS` environment variable.

### Index Build (offline, always local)

```
PDFs (data/)
    │
    ├── PyMuPDF → page text
    ├── RecursiveCharacterTextSplitter → chunks (~500 tokens, 50-token overlap)
    └── attach_metadata → UUID chunk_id per chunk
              │
    ┌─────────┴──────────┐
    │                    │
BM25Okapi index    FAISS IndexFlatIP
(rank_bm25)        (sentence-transformers all-MiniLM-L6-v2, L2-normalised)
    │                    │
indexes/bm25_index.pkl   indexes/faiss_index/
```

### Query Path (dual-mode)

```
User Query
    │
    ├── [USE_LOCAL_MODELS=true]  Embed locally (SentenceTransformer in-process)
    └── [USE_LOCAL_MODELS=false] Embed via HF Inference API (zero local RAM)
              │
    Hybrid Retrieval — BM25 (top 20) + FAISS (top 20) → RRF fusion (top 20)
              │
    ├── [USE_LOCAL_MODELS=true]  CrossEncoder ms-marco-MiniLM-L-6-v2 (local)
    └── [USE_LOCAL_MODELS=false] Cross-encoder scoring via HF Inference API
              │
    Reranked top 5
              │
    Groq API — LLaMA 3.3 70B (max 1500 tokens)
    Strict system prompt: cite every claim with chunk_id UUID
              │
    Citation Validation — all inline [UUID] refs resolved
              │
    FastAPI response with answer + citations + metadata
```

### Live PDF Upload (incremental indexing)

```
POST /api/v1/upload
    │
    ├── PyMuPDF extracts pages from uploaded file
    ├── Chunk + attach UUID metadata
    ├── BM25: rebuild from all chunks (BM25Okapi not incrementally updatable)
    └── FAISS: index.add() — true incremental vector insertion
              │
    In-memory pipeline updated immediately (no restart needed)
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
| Embeddings (local) | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Embeddings (Render) | HuggingFace Inference API — `InferenceClient.feature_extraction()` |
| Vector DB | FAISS `IndexFlatIP` (cosine via L2-normalised inner product) |
| Sparse Retrieval | `rank_bm25` |
| Reranker (local) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (CrossEncoder) |
| Reranker (Render) | HuggingFace Inference API — `InferenceClient.post()` |
| PDF Parsing | PyMuPDF (`fitz`) |
| API Framework | FastAPI + Pydantic v2 |
| Evaluation | RAGAS 0.4 + InstructorLLM (Groq) |
| CI | GitHub Actions |
| Deployment | Render (Python native, no Docker) |

---

## UI

Three-column layout served as a single static HTML file (`src/static/index.html`):

- **Left panel** — Sample question chips grouped by source (Arxiv, RBI, BIS, SEBI). Click any chip to instantly submit that query.
- **Centre panel** — Chat interface with a thinking indicator, inline citation superscripts, collapsible Sources panel per answer, and per-query latency/rerank metadata in the card footer.
- **Right panel** — Live document browser showing all indexed PDFs. Files uploaded by the user are tagged **"Added by User"** (purple) instead of a source classification. Upload button accepts PDF-only files and streams them into the live index without a restart.
- **Header** — Source corpus badges (Arxiv, RBI, BIS, SEBI). A **User · N files** badge appears dynamically once the first PDF is uploaded.
- **Status pill** — Three states: grey *"Ready — send a query to initialise"* (lazy load not yet triggered), green *"System Online"* (pipeline warm), red *"System Offline"* (server unreachable).

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
# Edit .env — set at minimum GROQ_API_KEY
```

`.env.example` variables:

```
GROQ_API_KEY=your_key_here
EMBED_MODEL=all-MiniLM-L6-v2
HF_TOKEN=your_hf_token          # only needed if USE_LOCAL_MODELS=false
USE_LOCAL_MODELS=true            # set false to use HF Inference API instead of local models
INDEX_DIR=indexes
DATA_DIR=data
TOP_K_RETRIEVE=20
```

### 4. Build indexes (first run only — takes ~5 minutes)

```bash
python -m src.ingestion.loader
```

Loads all PDFs, chunks them, and writes `indexes/bm25_index.pkl` and `indexes/faiss_index/`.

### 5. Run the API server

```bash
uvicorn src.api.main:app --reload --port 8000
```

Server starts at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`. The pipeline loads lazily on the first query (~30 s); the server is available immediately.

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

1. Push the repo to GitHub.
2. Create a new **Web Service** on [Render](https://render.com), connecting the repo.
3. Set the build and start commands:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
4. Add these environment variables in the Render dashboard:

| Variable | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key |
| `HF_TOKEN` | your HuggingFace token |
| `USE_LOCAL_MODELS` | `false` |

Setting `USE_LOCAL_MODELS=false` routes all embedding and reranking through the HuggingFace Inference API. No PyTorch models are loaded into the process — RAM stays under ~80 MB, well within Render's free tier 512 MB limit.

> **Note:** Pre-built indexes (`indexes/`) must be committed to the repo or generated via `POST /api/v1/ingest` after deploy.

---

## API Reference

### `POST /api/v1/query`

```json
// Request
{ "question": "What is RBI's stance on CBDC?", "top_k": 5 }

// Response
{
  "answer": "The RBI has outlined... [550e8400-...]...",
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
  "metadata": { "retrieved": 20, "reranked_to": 5, "latency_ms": 1400.0 }
}
```

### `POST /api/v1/ingest`

Rebuilds both indexes from scratch from a local data directory.

```json
// Request
{ "data_dir": "data/" }

// Response
{ "status": "ok", "chunks_indexed": 12480, "index_build_time_ms": 284320.0 }
```

### `POST /api/v1/upload`

Uploads a PDF, chunks it, and adds it incrementally to the live indexes without a restart.

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@report.pdf"
```

```json
// Response
{ "status": "ok", "filename": "report.pdf", "chunks_added": 143 }
```

### `GET /api/v1/files`

Lists all indexed PDFs with their source classification and file size.

```json
{
  "files": [
    { "filename": "rbi_cbdc_2023.pdf", "source": "rbi", "size_kb": 842.3 },
    { "filename": "report.pdf",        "source": "uploads", "size_kb": 210.1 }
  ],
  "total": 2
}
```

### `GET /api/v1/health`

```json
// Pipeline warm:
{ "status": "healthy", "indexes_loaded": true,  "model": "llama-3.3-70b-versatile" }

// Lazy-load not yet triggered:
{ "status": "loading",  "indexes_loaded": false, "model": "unknown" }
```

---

*FinScope AI — Production RAG, evaluated with RAGAS, deployed on Render.*
