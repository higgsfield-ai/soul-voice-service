FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 sox ffmpeg g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.12.10
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --python /usr/local/bin/python
COPY README.md ./
COPY soul_voice ./soul_voice
COPY src ./src
COPY third_party ./third_party
RUN uv sync --frozen --no-dev --python /usr/local/bin/python
CMD ["python", "-m", "src.main"]
