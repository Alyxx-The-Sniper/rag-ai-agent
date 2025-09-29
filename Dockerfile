FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# If you use pyproject.toml (you do), install your package + deps
# Option 1: pip from pyproject
COPY pyproject.toml uv.lock* ./
# Install uv (fast, respects uv.lock if present) – comment these two lines
RUN pip install --no-cache-dir uv
RUN uv pip install --system -r <(uv export --quiet --format=requirements-txt) || true

# Fallback: if the above didn't install anything (no deps in pyproject),
# you can later add a requirements.txt and replace with:
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (skip .venv and notebooks)
COPY . .
# If your FastAPI app needs env vars, copy .env (optional)
# COPY .env .env

RUN chmod +x ./start.sh

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

EXPOSE 8000 8501
CMD ["./start.sh"]
