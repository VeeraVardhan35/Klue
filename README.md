# AI-Powered Summarization Service

FastAPI service that performs deterministic abstractive summarization using a Hugging Face model, with an optional compressed-summary endpoint.

API contract:
- `POST /summarize` with JSON `{"text": "<string>", "summary_length": <int>}` returns `{"summary": "<string>"}`
- `POST /compress_summary` takes the same payload and returns the compressed summary

## Features
- Model is loaded once at startup and reused for all requests
- Deterministic decoding (`do_sample=False`, beam search, `early_stopping=True`)
- Token-aware chunking with overlap for long inputs, plus an optional second pass over the combined chunk summaries
- Concurrency guard with a semaphore to limit parallel model calls and reduce OOM risk
- Input validation and size limits (422 on invalid inputs)
- Health and readiness probes (`GET /health`, `GET /ready`)
- Optional warmup on startup to reduce first-request latency

## Summary length semantics (words externally, tokens internally)

`summary_length` is treated as an approximate target number of words in the returned summary.

Internally, the model uses token-based generation (`max_new_tokens`), so the service converts the requested word count into a token budget:

- `token_budget = max(16, int(words * 1.6) + 8)`

This is a conservative heuristic because tokens do not map 1:1 to words. After generation, the service enforces the API contract by truncating the output back down to the requested maximum number of words.

Important implications:
- The model may generate slightly more or fewer words than requested, but the final response is hard-capped to `summary_length` words
- The generation token budget exists to reduce under-generation, not to define the public length unit

## Requirements
- Python 3.11+
- `pip install -r requirements.txt`


## Running locally

### Option A: Conda
```bash
cd summarization-service
conda create -n summarization-service python=3.11 -y
conda activate summarization-service
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 5
```

### Option B: Existing Python environment
```bash
cd summarization-service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 5
```

Open:
- UI: http://localhost:8000/
- Swagger: http://localhost:8000/docs
- Readiness: http://localhost:8000/ready

## Docker

### Prereqs
- Install Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- Confirm Docker is running:
```bash
docker version
```

### Build the image
Run from the project root (where the Dockerfile lives):
```bash
docker build -t summarization-api .
```

### Run the container (basic)
```bash
docker run --rm -p 8000:8000 summarization-api
```

Then open:
- UI: http://localhost:8000/
- Swagger: http://localhost:8000/docs
- Readiness: http://localhost:8000/ready

## UI
- Simple HTML UI: http://localhost:8000/ (enter text, set `summary_length` as an approximate number of words, choose compress if needed)
- Includes a Clear button to quickly reset the form (optional)
- Interactive docs: http://localhost:8000/docs

## Configuration (env vars)

Core:
- `MODEL_ID` (default `facebook/bart-large-cnn`)
- `ENABLE_CHUNKING` (default `true`)
- `CHUNK_OVERLAP_TOKENS` (default `96`)
- `SECOND_PASS_SUMMARIZATION` (default `true`)
- `MAX_CONCURRENT_REQUESTS` (default `2`)
- `REQUEST_TIMEOUT_SECONDS` (default `120`)

Input limits:
- `MAX_INPUT_CHARS` (default `20000`) rejects oversized requests with 422 before tokenization
- `MAX_INPUT_TOKENS` (optional) overrides the max source token window used for chunking and truncation

Summary length bounds:
- `MIN_SUMMARY_TOKENS` (default `10`)
- `MAX_SUMMARY_TOKENS` (default `300`)

Note on naming: these environment variable names say "TOKENS" for backward compatibility, but they bound the public `summary_length` value which is interpreted as words. The service converts those words into a token budget internally.

Warmup:
- `WARMUP_ENABLED` (default `true`)

## Behavior notes
- `summary_length` is treated as an approximate number of words
- Internally, the word target is converted to a token budget for generation, and the final output is truncated to the requested maximum word count
- Empty or whitespace-only inputs return an empty summary; other invalid inputs are rejected with 422
- Requests exceeding `MAX_INPUT_CHARS` are rejected with 422 before tokenization
- Requests may time out based on `REQUEST_TIMEOUT_SECONDS` (exact status depends on server/proxy configuration)
- If the model is not ready, `/summarize` and `/compress_summary` return 503
- Generation failures return 500 with a stable message
- Very short or low-quality inputs may return the trimmed input text as the summary to avoid hallucinated outputs

## Tests

### Option A: Unit tests (fast, uses a stubbed summarizer)

Run from the project root:

```bash
cd summarization-service
python -m pytest -q
```

### Option B: Manual API test (PowerShell `Invoke-RestMethod`)

1. **Start the service (Terminal 1)**
   Open a terminal, activate your environment, then start Uvicorn from the project root:

   ```powershell
   cd "W:\MayCooperStation\New Documents\Resumes\Technical interviews\Rad AI\MLOPS\summarization-service"
   conda activate summarization-service
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 5
   ```

   Keep this terminal running.

2. **Send a request (Terminal 2)**
   Open a second terminal (separate window/tab), activate the same environment, then run:

   ```powershell
   cd "W:\MayCooperStation\New Documents\Resumes\Technical interviews\Rad AI\MLOPS\summarization-service"
   conda activate summarization-service

   $body = @{
     text = "This is a long text that should be summarized by the service to a concise form."
     summary_length = 64
   } | ConvertTo-Json -Compress

   Invoke-RestMethod -Uri "http://localhost:8000/summarize" -Method Post -ContentType "application/json" -Body $body
   ```

Expected response shape:

```json
{
  "summary": "<string>"
}
```


## Project layout
```
summarization-service/
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ schemas.py
│  ├─ settings.py
│  ├─ summarizer.py
│  └─ compression.py
├─ tests/
│  ├─ test_api_contract.py
│  └─ test_compression.py
├─ requirements.txt
├─ Dockerfile
└─ README.md
```
