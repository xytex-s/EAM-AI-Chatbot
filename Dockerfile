# Group 7 chatbot - container for the Python backend (CPU only).
#
# Build:  docker build -t chatbot .
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

CMD ["python", "rag.py"]
