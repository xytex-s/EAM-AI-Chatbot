"""
dev_server.py - Ahmed's LOCAL test server for the chat page.

This is NOT the real chatbot. Ethan's app.py is. This file:
  - serves the real page (templates/chat.html + static/)
  - runs the REAL safety.py and places.py, in the same order /ask should:
        1. safety.check()   -> fixed wellbeing response, AI skipped
        2. places.lookup()  -> OpenStreetMap places
        3. search + model   -> FAKED here (Ethan's rag.py does this for real)

Run from the repo root:
    python chatbot/dev_server.py
then open http://127.0.0.1:5000

Test words, for the fake model step only:
    test not found  -> the "couldn't find that" answer
    test 429        -> rate limited (HTML, like Nginx sends)
    test 500        -> server error
    slow            -> waits 20 s (tests the loading message)
    timeout         -> waits 70 s (tests the page's 60 s timeout)
"""

import random
import time

from flask import Flask, jsonify, render_template, request

import places
import safety

app = Flask(__name__)   # finds templates/ and static/ next to this file

FAKE_SOURCES = [{"title": "Example SHU page title", "url": "https://www.shu.ac.uk/example-page"}]


def response(answer, sources, found_places, is_safety, start):
    """Every 200 response has the same five fields (the /ask contract)."""
    return jsonify({
        "answer": answer,
        "sources": sources,
        "places": found_places,
        "safety": is_safety,
        "time_ms": int((time.perf_counter() - start) * 1000),
    })


@app.get("/")
def index():
    return render_template("chat.html")


@app.post("/ask")
def ask():
    start = time.perf_counter()
    data = request.get_json(silent=True) or {}
    question = data.get("question")

    # Same rule as the contract: a non-empty string, max 500 characters
    if not isinstance(question, str) or not question.strip() or len(question) > 500:
        return jsonify({"error": "Please enter a question (max 500 characters)."}), 400
    question = question.strip()
    # Privacy: the question is never printed or logged. Flask's own log only
    # shows "POST /ask 200", because the question is in the body, not the URL.

    # 1. Safety first. If it triggers, nothing else runs.
    hit = safety.check(question)
    if hit:
        return response(hit["message"], hit["sources"], [], True, start)

    # 2. Local places
    found = places.lookup(question)
    if found is not None:
        return response(places.summary(question, found), [], found, False, start)

    # 3. Search + language model: FAKED here
    q = question.lower()
    if "test 429" in q:
        return "<html><body><h1>429 Too Many Requests</h1></body></html>", 429
    if "test 500" in q:
        return jsonify({"error": "Something went wrong. Please try again."}), 500
    if "timeout" in q:
        time.sleep(70)
    elif "slow" in q:
        time.sleep(20)
    else:
        time.sleep(random.uniform(1, 3))      # pretend the model is thinking

    if "test not found" in q:
        return response("I couldn't find that in the SHU pages I have. "
                        "Try contacting Student Support directly.", [], [], False, start)

    return response("FAKE ANSWER from Ahmed's test server. A short, plain-English answer "
                    "written only from the saved SHU pages will appear here once Ethan's "
                    "rag.py is connected.", FAKE_SOURCES, [], False, start)


if __name__ == "__main__":
    # 127.0.0.1 = only this laptop can open it. threaded=True so a slow
    # answer doesn't block the next request.
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
