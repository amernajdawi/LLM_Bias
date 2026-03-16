FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml ./
RUN uv sync --no-dev
COPY src ./src
COPY scripts ./scripts
COPY config.yaml ./

CMD ["sleep", "infinity"]
