from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.compression import compress_text
from app.schemas import SummarizeRequest, SummarizeResponse
from app.settings import settings
from app.summarizer import SummarizationResult, Summarizer, InvalidInputError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("summarization")

# FastAPI app initialization
app = FastAPI(
    title="AI-Powered Summarization Service",
    version="1.0.0",
    description=(
        "Abstractive text summarization powered by a Hugging Face model. "
        "The API accepts JSON with 'text' and 'summary_length'. "
        "'summary_length' is interpreted as an approximate number of words in the returned summary. "
        "Internally, the service converts that word target into a token budget for generation and then truncates "
        "the output back to the requested word cap."
    ),
    contact={"name": "Summarization Service", "url": "http://localhost:8000/docs"},
)

# Global variables for the summarizer and concurrency control
summarizer: Summarizer | None = None
semaphore: asyncio.Semaphore | None = None
_is_ready: bool = False

# Helper functions

# Clamp requested summary length to service limits
def _clamp_summary_tokens(requested: int) -> int:
    # Convert to int
    req = int(requested)
    # Clamp to configured min/max
    lo = int(settings.min_summary_tokens)
    hi = int(settings.max_summary_tokens)
    return max(lo, min(hi, req))

# Derive a safe min_new_tokens value
def _derive_min_tokens(max_tokens: int) -> int | None:
    """Only apply min_new_tokens when it is safely below max_new_tokens."""
    max_toks = int(max_tokens)

    # For small outputs, do not force a minimum
    if max_toks < 20:
        return None

    # Aim for about half of max, but respect the configured floor
    candidate = max(int(settings.min_summary_tokens), max_toks // 2)

    # Skip if it would collide with max_new_tokens
    if candidate >= max_toks:
        return None

    return candidate

# Enforce input size limits
def _enforce_input_limits(request: SummarizeRequest) -> None:
    # Reject oversized inputs early with a clear 422.
    text = request.text or ""
    limit = int(settings.max_input_chars)

    # Also reject empty/whitespace-only inputs.
    if len(text) > limit:
        raise HTTPException(
            status_code=422,
            detail=f"Input text exceeds MAX_INPUT_CHARS ({limit}).",
        )

# Core summarization logic with concurrency and timeout handling
async def _generate_summary(request: SummarizeRequest) -> SummarizationResult:
    global summarizer, semaphore, _is_ready

    # Grab stable local refs to avoid races with startup failures
    summarizer_local = summarizer
    semaphore_local = semaphore

    # Fail fast if the model is not initialized
    if summarizer_local is None or semaphore_local is None or not _is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    # Map request length to safe generation bounds
    max_tokens = _clamp_summary_tokens(request.summary_length)
    min_tokens = _derive_min_tokens(max_tokens)

    # Offload sync model work to a thread so the event loop stays responsive
    loop = asyncio.get_running_loop()
    start = time.perf_counter()

    def _do_summarize() -> SummarizationResult:
        # Keep this small so it is easy to reason about and test
        return summarizer_local.summarize(
            text=request.text,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            enable_chunking=settings.enable_chunking,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            second_pass=settings.second_pass_summarization,
        )

    async with semaphore_local:
        try:
            fut = loop.run_in_executor(None, _do_summarize)
            result = await asyncio.wait_for(fut, timeout=settings.request_timeout_seconds)
        except InvalidInputError as exc:
            # Input was rejected by deterministic heuristics
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except asyncio.TimeoutError as exc:
            logger.warning("Summarization timed out", extra={"input_chars": len(request.text)})
            raise HTTPException(status_code=504, detail="Summarization timed out") from exc
        except Exception as exc:
            logger.exception("Summarization failed")
            raise HTTPException(status_code=500, detail="Summarization failed") from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "summarize_ok",
        extra={
            "input_chars": len(request.text),
            "summary_words": max_tokens,
            "latency_ms": latency_ms,
            "model_id": settings.model_id,
        },
    )
    return result


@app.on_event("startup")
async def startup_event() -> None:
    global summarizer, semaphore, _is_ready

    # Initialize readiness state and concurrency limit
    _is_ready = False
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    try:
        # Load model once for the lifetime of the process
        summarizer = Summarizer(settings.model_id, max_input_tokens=settings.max_input_tokens)

        # Optional warmup to reduce first request latency
        if settings.warmup_enabled:
            summarizer.warmup()

        _is_ready = True
        logger.info(
            "model_loaded",
            extra={
                "model_id": settings.model_id,
                "max_source_tokens": summarizer.max_source_tokens,
                "device": str(summarizer.device),
            },
        )
    except Exception:
        # Leave service unready if initialization fails
        summarizer = None
        logger.exception("Model initialization failed; service not ready")


@app.get("/health", summary="Health probe", description="Lightweight health check that always returns OK.")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/ready",
    summary="Readiness probe",
    description="Returns 200 only after the model is initialized; otherwise 503.",
)
async def ready() -> dict[str, str]:
    # Readiness depends on a loaded model and a successful startup flag
    if summarizer is None or not _is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}


@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize text",
    description=(
        "Generate an abstractive summary for the provided text. "
        "summary_length is treated as an approximate number of words in the returned summary. "
        "Internally, the service converts the word target into a token budget for generation and truncates the output "
        "back to the requested word cap. "
        "Errors: 422 on validation, 503 if model not ready, 500 on inference failure, 504 on timeout."
    ),
)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    # Validate input size before acquiring concurrency permits
    _enforce_input_limits(request)

    result = await _generate_summary(request)
    return SummarizeResponse(summary=result.summary)


@app.post(
    "/compress_summary",
    response_model=SummarizeResponse,
    summary="Summarize then compress",
    description=(
        "Generate a summary then compress it with run-length encoding at the character level. "
        "summary_length is treated as an approximate number of words in the returned summary. "
        "Uses the same request schema as /summarize. "
        "Errors: 422 on validation, 503 if model not ready, 500 on inference failure, 504 on timeout."
    ),
)
async def compress_summary(request: SummarizeRequest) -> SummarizeResponse:
    # Validate input size before doing any heavy work
    _enforce_input_limits(request)

    summary_result = await _generate_summary(request)
    compressed = compress_text(summary_result.summary)
    return SummarizeResponse(summary=compressed)

# Simple HTML UI for manual testing
@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Simple HTML UI for manual testing."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Summarization Service</title>
  <style>
    :root{
      --bg: #070b14;
      --panel: rgba(255,255,255,0.06);
      --panel-2: rgba(255,255,255,0.08);
      --border: rgba(148,163,184,0.18);
      --border2: rgba(148,163,184,0.30);
      --text: #e5e7eb;
      --muted: #a7b0c0;
      --muted-2:#7f8aa3;
      --accent: #3b82f6;
      --accent-2:#60a5fa;
      --ok: #34d399;
      --err: #fb7185;
      --shadow: 0 18px 50px rgba(0,0,0,0.45);
      --radius: 14px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(1000px 700px at 10% -10%, rgba(96,165,250,0.15), transparent 55%),
                  radial-gradient(900px 650px at 90% 10%, rgba(56,189,248,0.10), transparent 60%),
                  var(--bg);
      color: var(--text);
      font-family: var(--sans);
      line-height: 1.4;
    }
    .wrap { max-width: 980px; margin: 38px auto; padding: 0 18px 50px; }
    .header {
      display: flex; gap: 14px; align-items: center; justify-content: space-between; flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .brand { display: flex; gap: 12px; align-items: center; }
    .logo {
      width: 42px; height: 42px; border-radius: 12px;
      background: linear-gradient(135deg, rgba(59,130,246,0.35), rgba(96,165,250,0.08));
      border: 1px solid var(--border);
      display: grid; place-items: center;
      box-shadow: var(--shadow);
    }
    .titleBlock h1 { margin: 0; font-size: 34px; letter-spacing: -0.02em; }
    .titleBlock p { margin: 4px 0 0; color: var(--muted); max-width: 680px; }
    .pillRow { display: flex; gap: 8px; flex-wrap: wrap; }
    .pill {
      font-size: 12px;
      color: var(--muted);
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      padding: 7px 10px;
      border-radius: 999px;
      backdrop-filter: blur(10px);
    }
    .pill code { font-family: var(--mono); color: var(--text); font-size: 12px; }

    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 18px;
      backdrop-filter: blur(12px);
    }
    .grid { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 14px; margin-bottom: 14px; }
    @media (max-width: 900px){ .grid { grid-template-columns: 1fr; } }

    .sectionTitle { font-weight: 700; margin: 0 0 10px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    .help { margin: 0; color: var(--muted); font-size: 14px; }
    .help strong { color: var(--text); }
    .list { margin: 10px 0 0; padding-left: 18px; color: var(--muted); }
    .list li { margin: 6px 0; }

    label { display: block; margin-top: 14px; font-weight: 700; color: var(--text); }
    textarea {
      width: 100%;
      height: 210px;
      margin-top: 8px;
      padding: 14px 14px;
      font-size: 15px;
      color: var(--text);
      background: rgba(2,6,23,0.40);
      border: 1px solid var(--border);
      border-radius: 12px;
      outline: none;
      resize: vertical;
    }
    textarea:focus { border-color: rgba(96,165,250,0.55); box-shadow: 0 0 0 4px rgba(59,130,246,0.12); }

    .controls {
      margin-top: 14px;
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .field {
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      background: rgba(2,6,23,0.35);
      border: 1px solid var(--border);
      border-radius: 12px;
    }
    .field span { color: var(--muted); font-weight: 600; }
    input[type="number"] {
      width: 120px;
      padding: 9px 10px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(15,23,42,0.55);
      color: var(--text);
      outline: none;
    }
    input[type="number"]:focus { border-color: rgba(96,165,250,0.55); box-shadow: 0 0 0 4px rgba(59,130,246,0.12); }

    .toggle {
      display: flex; align-items: center; gap: 8px;
      padding: 10px 12px;
      background: rgba(2,6,23,0.35);
      border: 1px solid var(--border);
      border-radius: 12px;
      color: var(--muted);
      font-weight: 600;
    }
    .toggle input { accent-color: var(--accent); }

    button {
      padding: 11px 16px;
      border: none;
      border-radius: 12px;
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: white;
      font-weight: 800;
      letter-spacing: 0.01em;
      box-shadow: 0 12px 30px rgba(59,130,246,0.20);
      transition: transform 0.04s ease, filter 0.12s ease;
    }
    button:hover { filter: brightness(1.03); }
    button:active { transform: translateY(1px); }
    button:disabled { opacity: 0.55; cursor: not-allowed; box-shadow: none; }

    button.ghost{
      background: transparent;
      color: var(--text);
      border: 1px solid var(--border2);
      box-shadow: none;
    }
    button.ghost:hover{
      filter: none;
      background: rgba(148,163,184,0.12);
    }

    .status { margin-top: 10px; color: var(--muted); font-size: 13px; }
    .status.ok { color: var(--ok); }
    .status.err { color: var(--err); }

    .result {
      margin-top: 16px;
      background: rgba(2,6,23,0.35);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px;
      min-height: 90px;
      white-space: pre-wrap;
    }
    .resultTitle { font-size: 12px; color: var(--muted-2); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }

    .footer {
      margin-top: 14px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--muted-2);
      font-size: 12px;
    }
    .footer a { color: var(--muted); text-decoration: none; }
    .footer a:hover { color: var(--text); text-decoration: underline; }

    code.inline {
      font-family: var(--mono);
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
      padding: 2px 6px;
      border-radius: 8px;
      color: var(--text);
      font-size: 12px;
    }
    code.endpoint { font-family: var(--mono); color: var(--text); }
  </style>
</head>

<body>
  <div class="wrap">
    <div class="header">
      <div class="brand">
        <div class="logo" aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M13 2L3 14h7l-1 8 12-14h-7l-1-6z" stroke="rgba(229,231,235,0.95)" stroke-width="1.6" stroke-linejoin="round"/>
          </svg>
        </div>
        <div class="titleBlock">
          <h1>Summarization Service</h1>
          <p>
            Abstractive summarization powered by a Hugging Face model.
            <strong>summary_length</strong> is interpreted as <code class="inline"># of words</code> in the summary.
          </p>
        </div>
      </div>

      <div class="pillRow">
        <div class="pill">Endpoints: <code class="endpoint">/summarize</code>, <code class="endpoint">/compress_summary</code></div>
        <div class="pill">Deterministic: <code class="endpoint">do_sample=false</code></div>
        <div class="pill">Chunking + 2nd pass</div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="sectionTitle">How it works</div>
        <p class="help">
          The service tokenizes your input, summarizes deterministically, and (for long inputs) performs token-aware chunking with overlap.
          Optionally, it runs a second-pass summary over the concatenated chunk summaries.
        </p>
        <ul class="list">
          <li><strong>Summarize:</strong> <code class="inline">POST /summarize</code> → <code class="inline">{"summary":"..."}</code></li>
          <li><strong>Compress:</strong> summarizes then run-length compresses consecutive characters (example: <code class="inline">aaabb</code> → <code class="inline">a3b2</code>)</li>
          <li><strong>Guards:</strong> whitespace-only text is rejected; overly large inputs are blocked by <code class="inline">MAX_INPUT_CHARS</code></li>
        </ul>
      </div>

      <div class="card">
        <div class="sectionTitle">API contract</div>
        <p class="help">Request JSON:</p>
        <p class="help"><code class="inline">{"text":"...", "summary_length": 50}</code></p>
        <p class="help" style="margin-top:10px;">Tips:</p>
        <ul class="list">
          <li><code class="inline">summary_length=30</code> ≈ 30 words</li>
          <li><code class="inline">summary_length=120</code> ≈ 120 words</li>
          <li><code class="inline">Ctrl+Enter</code> submits</li>
        </ul>
      </div>
    </div>

    <div class="card">
      <label for="text">Input text</label>
      <textarea id="text" placeholder="Paste text to summarize"></textarea>

      <div class="controls">
        <div class="field">
          <span>Summary length (words)</span>
          <input id="length" type="number" value="50" min="10" max="300" />
        </div>

        <label class="toggle">
          <input id="compress" type="checkbox" />
          Compress summary
        </label>

        <button id="submit" type="button">Summarize</button>
        <button id="clear" type="button" class="ghost">Clear</button>
      </div>

      <div class="status" id="status">Ready.</div>

      <div class="result">
        <div class="resultTitle">Output</div>
        <div id="result"></div>
      </div>

      <div class="footer">
        <div>
          Open <a href="/docs" target="_blank" rel="noreferrer">/docs</a> for interactive testing.
          <span style="margin: 0 8px; opacity: 0.6;">•</span>
          Readiness: <a href="/ready" target="_blank" rel="noreferrer">/ready</a>
        </div>

        <div style="display:flex; gap:10px; align-items:center;">
          <span style="opacity:0.85;">Built by <strong>May Cooper</strong></span>
        </div>
      </div>
    </div>
  </div>

  <script>
    const textEl = document.getElementById('text');
    const lengthEl = document.getElementById('length');
    const compressEl = document.getElementById('compress');
    const statusEl = document.getElementById('status');
    const resultEl = document.getElementById('result');
    const submitBtn = document.getElementById('submit');
    const clearBtn = document.getElementById('clear');

    clearBtn.addEventListener('click', () => {
      textEl.value = '';
      resultEl.textContent = '';
      statusEl.textContent = 'Cleared.';
      statusEl.className = 'status';
      textEl.focus();
    });

    async function callApi() {
      const text = textEl.value;
      const length = Number(lengthEl.value);
      const endpoint = compressEl.checked ? '/compress_summary' : '/summarize';

      statusEl.textContent = 'Submitting...';
      statusEl.className = 'status';
      submitBtn.disabled = true;
      resultEl.textContent = '';

      try {
        const resp = await fetch(endpoint, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ text, summary_length: length })
        });

        const contentType = resp.headers.get('content-type') || '';
        const data = contentType.includes('application/json') ? await resp.json() : { detail: await resp.text() };

        if (!resp.ok) {
          function formatDetail(detail) {
            if (typeof detail === 'string') return detail;

            // FastAPI/Pydantic often returns a list of error objects
            if (Array.isArray(detail)) {
              const msgs = detail.map(e => e?.msg).filter(Boolean);
              if (msgs.length) return msgs.join(' | ');
              return JSON.stringify(detail, null, 2);
            }

            if (detail && typeof detail === 'object') {
              if (detail.msg) return String(detail.msg);
              return JSON.stringify(detail, null, 2);
            }

            return String(detail);
          }

          const rawDetail = (data && data.detail !== undefined) ? data.detail : data;
          const detailText = formatDetail(rawDetail);
          statusEl.textContent = `Error (${resp.status}): ${detailText}`;
          statusEl.className = 'status err';
          return;
        }

        statusEl.textContent = 'Success.';
        statusEl.className = 'status ok';
        resultEl.textContent = data.summary || '';
      } catch (err) {
        statusEl.textContent = 'Request failed.';
        statusEl.className = 'status err';
      } finally {
        submitBtn.disabled = false;
      }
    }

    submitBtn.addEventListener('click', () => callApi());

    // Ctrl+Enter (or Cmd+Enter on macOS) to submit
    textEl.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') callApi();
    });
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
