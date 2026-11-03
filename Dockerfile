# Group 7 chatbot - container for the Python backend (CPU only).
#
# Build:  docker build -t chatbot .
# Run:    docker run -d --name chatbot --restart unless-stopped -p 8000:8000 -v hf-models:/models chatbot
#         (the API: POST /ask, POST /ask/stream, GET /health - see docs/ask-api-examples.json)
# Test:   docker run --rm -v hf-models:/models -v ~/results:/results chatbot python rag.py --test
# Ask:    docker run --rm -it -v hf-models:/models chatbot python rag.py
#
# The hf-models volume keeps the downloaded models (~3 GB) between runs and rebuilds,
# so they are not baked into the image. Test CSVs are written to the /results mount.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    RESULTS_DIR=/results

# Run as a normal user, not root.
RUN useradd --create-home app && mkdir -p /app /models /results && chown app /app /models /results

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app chatbot/ chatbot/
COPY --chown=app paging/data/chunks.json paging/data/chunks.json

USER app
WORKDIR /app/chatbot

# Build the search index into the image (downloads the 90 MB embedding model into /models).
RUN python build_index.py

EXPOSE 8000

# Docker marks the container "unhealthy" if /health stops answering (the model takes ~1-2 min to load).
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# One worker = one copy of the model in RAM. 4 threads so /health and waiting questions still get answered
# while the model is busy; app.py lets only one question use the model at a time.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "180", \
     "--bind", "0.0.0.0:8000", "--access-logfile", "-", "app:app"]
