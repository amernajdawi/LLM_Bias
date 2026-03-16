FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml ./
RUN uv sync --no-dev
COPY src ./src
COPY scripts ./scripts
COPY config.yaml ./
COPY streamlit_app.py ./

# For local docker-compose: override with "sleep infinity" and exec into the container.
# For Railway/production: run Streamlit on PORT, bind to 0.0.0.0.
ENV STREAMLIT_SERVER_HEADLESS=true
EXPOSE 8501
CMD ["sh", "-c", "uv run streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
