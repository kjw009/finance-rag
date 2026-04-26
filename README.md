# Finance RAG

A retrieval-augmented generation system for answering questions about SEC filings and earnings reports. Ask natural language questions about revenue, income, segment performance, and more — the system retrieves the relevant passages from the raw filings and generates a cited answer.

## How it works

```
Query
  ├─► Ticker detection   → pre-filter to one company's chunks
  ├─► Dense retrieval    (ChromaDB + all-MiniLM-L6-v2)  → top-100 candidates
  ├─► BM25 retrieval     (rank_bm25)                     → top-100 candidates
  ├─► RRF fusion         (Reciprocal Rank Fusion)         → top-50 merged
  └─► Cross-encoder      (ms-marco-MiniLM-L-6-v2)        → top-15 final chunks → LLM
```

Dense retrieval finds semantically similar passages; BM25 catches exact keyword matches (tickers, dollar figures); RRF merges both rankings without normalising incompatible scores; the cross-encoder re-reads query + passage together for a more precise final ranking. A company name detected in the query restricts all retrieval to that company's documents before any of these steps run.

The LLM (Llama 3.3 70B via Groq) is instructed to answer only from the retrieved passages and cite every claim by source filename.

## Corpus

8 companies — AAPL, AMZN, AMD, GOOGL, META, MSFT, NVDA, TSLA — across:
- 10-K annual reports (2024 and 2025 fiscal years)
- 10-Q quarterly filings
- 8-K current reports
- Earnings release PDFs and slide decks

~21,000 chunks total after parsing.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

## Usage

**Index documents** (run once, or after adding new filings):
```bash
python main.py index
```

**Ask a single question:**
```bash
python main.py query "What was Meta's total revenue for the year ended December 31, 2025?"
```

**Interactive chat:**
```bash
python main.py chat
```

**Gradio web UI:**
```bash
python main.py serve
# Open http://127.0.0.1:7860
```

## Notebooks

The `notebooks/` directory builds the system step by step:

| Notebook | What it covers |
|---|---|
| `00_edgar_collector` | Downloading filings from SEC EDGAR |
| `01_parsing_strategy` | Parsing HTM and PDF documents into chunks |
| `02_embeddings` | Embedding chunks and building the ChromaDB index |
| `03_retrieval` | Hybrid dense + BM25 + RRF + cross-encoder pipeline |
| `04_generation` | Prompt design and LLM generation |
| `05_evaluation` | Recall@k evaluation on a 20-question eval set |
| `06_gradio_ui` | Gradio chat interface |

## Evaluation

Recall@5 on a 20-question eval set spanning revenue lookups, segment breakdowns, profitability, YoY growth, headcount, and one out-of-scope question (real-time price):

| Metric | Score |
|---|---|
| Source-level Recall@5 | 80% (20/25 expected sources retrieved) |
| Question-level Recall@5 | 74% (14/19 in-scope questions with all sources retrieved) |

Known failure modes:
- **Year disambiguation** — consecutive 10-Ks for the same company (e.g. GOOGL 2025 vs 2026 filing) have near-identical prose; BM25 and dense retrieval can retrieve the wrong year.
- **Earnings slide PDFs** — thin, bullet-heavy slide decks score lower than the 10-K covering the same figures, so they are crowded out of the top-k results.

## Project structure

```
main.py              # CLI entry point (index / query / chat / serve)
utils.py             # Chunk dataclass, ChromaDB helpers
src/
  parser.py          # HTM and PDF parsing → Chunk objects
  retriever.py       # Hybrid retrieval pipeline
  generator.py       # LLM prompting and answer generation
datasets/client/     # Raw SEC filings and PDFs
notebooks/           # Step-by-step development notebooks
chroma_db/           # Persisted ChromaDB vector index (generated)
chunks_cache.pkl     # Serialised chunk list (generated)
```
