# Multi-stage build: compile deps in a builder stage, ship a slim runtime.
FROM python:3.11-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# CPU-only torch from the pytorch wheel index; everything else from PyPI.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.*" faiss-cpu fastapi "uvicorn[standard]" pydantic \
        numpy pandas pyarrow pyyaml httpx

FROM python:3.11-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY ctr ./ctr
COPY configs ./configs
RUN mkdir -p /app/artifacts && chown -R appuser:appuser /app
# Trained models / vocabularies are mounted here at runtime.
VOLUME ["/app/artifacts"]
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1
CMD ["uvicorn", "ctr.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
