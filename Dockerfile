FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgeos-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY README.md MANUAL.md ./

RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY tool_catalogs/ tool_catalogs/
COPY schemas/ schemas/
COPY profiles/ profiles/
COPY pipeline_config.yaml .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

VOLUME ["/app/scenarios", "/app/datasets", "/app/reports"]

ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--help"]
