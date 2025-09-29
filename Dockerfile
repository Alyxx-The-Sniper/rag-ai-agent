FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Export deps WITHOUT HASHES to avoid cross-platform hash failures
COPY pyproject.toml uv.lock* ./
RUN pip install --no-cache-dir uv \
 && (uv export --format=requirements-txt --no-hashes --quiet > /tmp/requirements.txt || true) \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# 2) (Optional but recommended) Install CPU-only PyTorch if you use torch
#    Comment out if you don't use torch.
#    This avoids pulling nvidia-cuda* packages.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio || true

# App code
COPY . .
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
EXPOSE 8000 8501
CMD ["./start.sh"]
