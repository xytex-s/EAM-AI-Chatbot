"""
app.py - Group 7 chatbot, the web API (Flask).

The chat page sends each question here. The answer comes back in one of two ways:

    POST /ask          {"question": "..."}  ->  one JSON answer when it's finished (the agreed contract)
    POST /ask/stream   {"question": "..."}  ->  the answer typed out live, one JSON object per line
    GET  /health       is it running, and with which settings
    GET  /             a bare test page (or Ahmed's chat.html once it's in templates/)

Every question goes through the same steps:
    1. safety.check()  - crisis words: fixed SHU message and links, the model is skipped
    2. places.lookup() - local food/coffee/shops questions: a list of places, the model is skipped
    3. rag             - search the SHU pages, then the model writes a short answer with its source

Run:
    python app.py                     # development, then open http://localhost:8000
    gunicorn --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:8000 app:app    # on VM 2 (the Docker image does this)

One worker on purpose: each worker loads its own copy of the model (~6 GB of RAM for 1.5B).
The model answers one question at a time (see model_lock); other questions wait their turn.
"""
import json
import logging
import threading
import time
from contextlib import closing
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from rag import RAG

try:                        # Ahmed's modules - the API still works without them while they're being written
    import safety
except ImportError:
    safety = None
try:
    import places
except ImportError:
    places = None

MAX_QUESTION = 500          # characters
WAIT_SECONDS = 120          # how long a question waits for the model before giving up
BAD_REQUEST = "Please enter a question (max 500 characters)."
BUSY = "The assistant is busy with other questions. Please try again in a minute."
SERVER_ERROR = "Something went wrong. Please try again."
PLACES_ANSWER = "Here are some places near City Campus."
NO_PLACES = "Sorry, I couldn't find any places for that near City Campus."

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chatbot")

app = Flask(__name__)
app.json.ensure_ascii = False       # keep "£" as it is
rag = RAG()                         # slow (loads the models), so it happens once when the app starts
model_lock = threading.Lock()       # the model can only write one answer at a time
log.info("Ready: %s (loaded in %s s). safety.py: %s, places.py: %s", rag.settings(), rag.load_seconds,
         "yes" if safety else "MISSING", "yes" if places else "MISSING")


def reply(answer, sources=(), found_places=(), is_safety=False, t0=None):
    """A response in the agreed /ask format."""
    return {"answer": answer, "sources": list(sources), "places": list(found_places),
            "safety": is_safety, "time_ms": round((time.perf_counter() - t0) * 1000)}


def read_question():
    """The question from the request body, or None if it's missing, empty or too long."""
    body = request.get_json(silent=True) or {}
    q = body.get("question")
    if not isinstance(q, str) or not q.strip() or len(q) > MAX_QUESTION:
        return None
    return q.strip()


def answer_without_model(question, t0):
    """Steps 1 and 2: a fixed reply for safety and places questions, or None if it needs the model."""
    if safety:
        hit = safety.check(question)
        if hit:
            return reply(hit["message"], hit["sources"], is_safety=True, t0=t0)
    if places:
        found = places.lookup(question)
        if found is not None:
            if hasattr(places, "summary"):          # Ahmed's wording, e.g. "Here are cafés within ..."
                text = places.summary(question, found)
            else:
                text = PLACES_ANSWER if found else NO_PLACES
            return reply(text, found_places=found, t0=t0)
    return None


def log_answer(kind, result):
    # Only timings and counts are logged, never the question, so no personal data ends up in logs.
    log.info("%s: %s ms, read %s s, write %s s, %s tokens, %s source(s)", kind, result.get("time_ms"),
             result.get("read_s", 0), result.get("write_s", 0), result.get("answer_tokens", 0),
             len(result.get("sources", [])))


@app.post("/ask")
def ask():
    t0 = time.perf_counter()
    question = read_question()
    if question is None:
        return jsonify(error=BAD_REQUEST), 400
    try:
        fixed = answer_without_model(question, t0)
        if fixed:
            log_answer("fixed", fixed)
            return jsonify(fixed)
        if not model_lock.acquire(timeout=WAIT_SECONDS):
            return jsonify(error=BUSY), 503
        try:
            result = rag.answer(question)
        finally:
            model_lock.release()
        log_answer("ask", result)
        return jsonify(reply(result["answer"], result["sources"], t0=t0))
    except Exception:
        log.exception("ask failed")
        return jsonify(error=SERVER_ERROR), 500


@app.post("/ask/stream")
def ask_stream():
    """Like /ask, but sends events as the answer is worked out, one JSON object per line:

        {"type": "status", "stage": "waiting" | "searching" | "reading" | "writing", "text": "..."}
        {"type": "text", "text": "..."}       add this to the answer on screen
        {"type": "done", "answer": ..., "sources": [...], "places": [...], "safety": ..., "time_ms": ...}
        {"type": "error", "error": "..."}     instead of done, if something went wrong

    "done" has the same fields as /ask, and its answer is the final, tidied version:
    the page should replace the typed-out text with it.
    """
    t0 = time.perf_counter()
    question = read_question()
    if question is None:
        return jsonify(error=BAD_REQUEST), 400

    def line(event):
        return json.dumps(event, ensure_ascii=False) + "\n"

    def events():
        try:
            fixed = answer_without_model(question, t0)
            if fixed:
                log_answer("fixed", fixed)
                yield line({"type": "done", **fixed})
                return
            if not model_lock.acquire(blocking=False):
                yield line({"type": "status", "stage": "waiting", "text": "Waiting for another question to finish..."})
                if not model_lock.acquire(timeout=WAIT_SECONDS):
                    yield line({"type": "error", "error": BUSY})
                    return
            try:
                # closing(): if the student leaves the page, the model is told to stop writing.
                with closing(rag.answer_stream(question)) as steps:
                    for e in steps:
                        if e["type"] == "done":
                            log_answer("stream", e)
                            e = {"type": "done", **reply(e["answer"], e["sources"], t0=t0)}
                        yield line(e)
            finally:
                model_lock.release()
        except GeneratorExit:
            log.info("stream: client left before the end")
            raise
        except Exception:
            log.exception("stream failed")
            yield line({"type": "error", "error": SERVER_ERROR})

    return Response(events(), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})     # tells Nginx to pass each line on straight away


@app.get("/health")
def health():
    return jsonify(status="ok", settings=rag.settings(), busy=model_lock.locked(),
                   safety=safety is not None, places=places is not None)


@app.get("/")
def index():
    page = "chat.html" if Path(app.root_path, app.template_folder, "chat.html").exists() else "test_chat.html"
    return render_template(page)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)