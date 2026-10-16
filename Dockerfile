# Results dashboard (dashboard_v3.py). Small image: no database, no torch.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "streamlit>=1.40" "pandas>=2.0" "plotly>=5.20"
COPY src/analysis/__init__.py src/analysis/metrics.py ./src/analysis/
COPY dashboard_data ./dashboard_data
COPY dashboard_v3.py ./
ENV STREAMLIT_SERVER_HEADLESS=true
EXPOSE 8501
CMD ["sh", "-c", "streamlit run dashboard_v3.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
