# AcademIQ — RAG-Powered Academic Policy Assistant

A retrieval-augmented assistant that answers student questions **only** from
official college documents, with a citation on every claim and a standardized
NOT-FOUND response whenever the documents don't support an answer. Built to the
zero-hallucination contract in the PRD.

## Quick start (zero keys, zero PDFs)

The app ships in `DEMO_MODE=true`, which runs the full pipeline with deterministic
local providers — no OpenAI/Cohere keys and no network required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 1. Load the bundled sample manual (no PDFs needed)
python -m scripts.load_sample_corpus

# 2. Try the pipeline from the CLI
python -m scripts.demo_query

# 3. Or run the API + demo chat UI
uvicorn app.main:app --reload
# open http://localhost:8000
```

Ingest real PDFs instead of the sample:

```bash
cp your_manuals/*.pdf data/pdfs/
python -m scripts.ingest
```

## Going to production

Set `DEMO_MODE=false` and provide `OPENAI_API_KEY` + `COHERE_API_KEY` in `.env`.
The same code path then uses `text-embedding-3-large`, `gpt-4o`, and Cohere
rerank, exactly as specified in the PRD. OCR needs the system `tesseract` binary
(already in the Docker image).

```bash
docker compose up --build      # http://localhost:8000
```

## How the zero-hallucination contract is enforced

The four hallucination entry points from the PRD are each sealed in code:

1. **Ingestion** — `ingestion/extractor.py` detects scanned pages (text-density
   check) and routes them to OCR so no chunk is silently empty; tables are
   extracted separately and preserved as their own chunks.
2. **Retrieval** — hybrid dense + BM25 with reciprocal-rank fusion and a reranker
   (`services/retrieval.py`) so short, colloquial queries still hit the right
   chunk.
3. **Confidence gating** — `services/confidence.py` runs a dual threshold on the
   top reranker score *before* the LLM is called; below the hard threshold the
   model is never invoked and NOT-FOUND is returned.
4. **Generation** — the master system prompt is deployed verbatim
   (`prompts.py`), and `services/generation.py` rejects any answer lacking a
   `[Source: ...]` citation, downgrading it to NOT-FOUND.

## Engineering decisions (open items in the PRD)

| Ref | Decision | Choice | Why |
|----|----------|--------|-----|
| DR-01 | PDF text engine | **PyMuPDF** primary | Fast; exposes font size + bold flags the chunker needs for heading detection. |
| DR-02 | Table extraction | **pdfplumber** `extract_tables()` | No Ghostscript dependency (Camelot needs it); adequate for v1. Camelot/tabula is the documented upgrade. |
| DR-03 | Chunk boundary | **Section-anchored** + token-cap fallback | Keeps a rule's condition and consequence in one chunk — the top source of policy hallucination. |
| DR-04 | Confidence gate | **Dual threshold** (hard reject + soft warn) on reranker score | Graduated UX; reranker relevance is a stronger signal than raw cosine. Calibrate with `eval/calibrate_threshold.py`. |
| DR-05 | Vector store | **Embedded persistent ChromaDB** | Zero infra for the sprint; cosine space matches the gate. Compose ships a shared-server variant for team ingestion. |
| DR-06 | Scan detection | **Text-density check + tesseract OCR** | Catches image-only PDFs before they poison the store. |
| DR-07 | Short queries | **Query rewrite** (≤6 words) + BM25 | Cheap, ~10 lines; BM25 independently rescues keyword-heavy short queries. |

Two pragmatic deviations for the prototype, both reversible: a single static HTML
chat UI instead of a full Next.js app, and SQLite instead of PostgreSQL for
feedback. Both swap out cleanly.

## Layout

```
app/
  config.py            settings + provider auto-selection
  prompts.py           master system prompt (verbatim) + NOT-FOUND / injection text
  security.py          admin-key auth, rate limiter
  models/schemas.py    request/response models + injection sanitizer
  ingestion/           extractor (PyMuPDF/pdfplumber/OCR), chunker, pipeline
  services/            embedder, vector_store, reranker, retrieval, confidence,
                       llm, generation, cache, feedback
  routers/             query, admin, health
  main.py              FastAPI app, lifespan, static UI mount
eval/                  golden_set.json, run_ragas.py, calibrate_threshold.py
scripts/               ingest, load_sample_corpus, demo_query
tests/                 retrieval, generation, adversarial_suite
static/index.html      demo chat UI
```

## Tests

```bash
pytest                       # unit + adversarial suite (runs in demo mode)
python -m tests.adversarial_suite   # standalone adversarial report
python -m eval.calibrate_threshold  # sweep + suggested gate thresholds
python -m eval.run_ragas            # RAGAS gates (needs OPENAI_API_KEY)
```

## API

- `POST /api/v1/query` — `{ "query": "...", "session_id": "..." }` → answer,
  citations, confidence band, latency.
- `POST /api/v1/feedback` — thumbs up/down to SQLite.
- `POST /api/v1/admin/reingest` / `admin/upload` — re-ingest without a deploy
  (`X-Admin-Key` header).
- `GET /health` — active providers + chunk count.
