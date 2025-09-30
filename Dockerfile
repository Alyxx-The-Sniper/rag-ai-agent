FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy just pyproject first (better layer caching)
COPY pyproject.toml README.md ./

# Generate a requirements file from [project.dependencies] and install
RUN python - <<'PY'
import tomllib, sys
data = tomllib.load(open("pyproject.toml", "rb"))
deps = data.get("project", {}).get("dependencies", [])
open("/tmp/requirements.txt", "w").write("\n".join(deps) + "\n")
print("Deps:", len(deps))
PY
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the rest of the app
COPY . .
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

EXPOSE 8000 8501
CMD ["./start.sh"]
