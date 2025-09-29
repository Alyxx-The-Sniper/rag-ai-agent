FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps from pyproject/uv.lock if present
COPY pyproject.toml uv.lock* ./
RUN pip install --no-cache-dir uv \
 && (uv export --format=requirements-txt --quiet > /tmp/requirements.txt || true) \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# App code
COPY . .
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
EXPOSE 8000 8501
CMD ["./start.sh"]
