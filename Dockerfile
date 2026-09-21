FROM python:3.11-slim
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface

WORKDIR /app

# Minimal OS deps (curl used for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create non-root user and cache directory
RUN useradd -m -u 10001 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache

USER app

# Copy application code
COPY --chown=app:app app /app/app
COPY --chown=app:app tests /app/tests
COPY --chown=app:app README.md /app/README.md

EXPOSE 8000

# Allow enough time for first-run model download + load
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/ready || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "5"]
