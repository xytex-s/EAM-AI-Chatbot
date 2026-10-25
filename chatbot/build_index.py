"""
build_index.py - Group 7 chatbot, step 2 of the offline build.

Turns the chunks saved by collect_pages.py into a searchable index. Each chunk
is converted into a vector (a list of 384 numbers) by a small embedding model,
so that chunks with a similar meaning get similar vectors. A question is then
converted the same way, and the closest chunks are the best matches.

    paging/data/chunks.json  ->  build_index.py  ->  chatbot/data/index.pt

Usage:
    python build_index.py                     # build the index
    python build_index.py --test              # build, then run the search test questions
    python build_index.py --ask "How do I get my SHUcard?"   # search an existing index

Needs: pip install sentence-transformers   (PyTorch CPU from the speed test is enough)
The embedding model (~90 MB) downloads automatically the first time.
"""
import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
DEFAULT_CHUNKS = HERE.parent / "paging" / "data" / "chunks.json"
DEFAULT_INDEX = HERE / "data" / "index.pt"

# Built for matching questions to passages, and reads up to 512 word pieces.
# (all-MiniLM-L6-v2 stops at 256, which would cut off the end of most of our ~300-word chunks.)
MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
MAX_TOKENS = 512
TOP_K = 3

# Realistic student questions and the page each should find (part of the URL).
# None = out of scope: the best score should be low, so the bot says it doesn't know.
TEST_QUESTIONS = [
    ("How do I reset my university password?",            ["my-it-account", "enrol/activate", "it-help"]),
    ("How do I connect to the wifi on campus?",           ["wifi"]),
    ("Where can I print on campus?",                      ["print"]),
    ("How do I ask for an extension on my assignment?",   ["extension"]),
    ("What happens if I hand in my coursework late?",     ["late-submission", "submitting-your-work"]),
    ("How do I apply for the hardship fund?",             ["hardship"]),
    ("When do I have to pay my tuition fees?",            ["payment-dates", "fee-payments", "paying-fees"]),
    ("How do I book a counselling appointment?",          ["counselling", "register-for-support"]),
    ("I can't sleep, what can I do?",                     ["sleep"]),
    ("How do I get my student ID card?",                  ["shucard"]),
    ("How do I apply for university accommodation?",      ["accommodation/apply"]),
    ("Can I use ChatGPT for my assignments?",             ["artificial-intelligence"]),
    ("What is the best pizza place in Sheffield?",        None),
]


# ---------- Building ----------

def clean_title(title):
    """Tidy titles for citing, e.g. 'Library Help - Library Help - LibGuides at ...' -> 'Library Help'."""
    title = re.sub(r"\s*[-|]\s*LibGuides at Sheffield Hallam University\s*$", "", title)
    parts = [p.strip() for p in title.split(" - ")]
    unique = list(dict.fromkeys(parts))           # drop repeated parts, keep order
    return " - ".join(unique)


def load_model(name=MODEL_NAME):
    model = SentenceTransformer(name, device="cpu")
    model.max_seq_length = MAX_TOKENS
    return model


def build(chunks_path, index_path, model_name):
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    t0 = time.perf_counter()
    model = load_model(model_name)
    print(f"Loaded model {model_name} in {time.perf_counter() - t0:.1f} s")

    records = []
    for c in chunks:
        records.append({"id": c["id"], "url": c["url"], "title": clean_title(c["title"]), "text": c["text"]})

    # The title goes in front of the text so the page topic counts towards the match.
    texts = [f"{r['title']}. {r['text']}" for r in records]
    too_long = sum(len(model.tokenizer.tokenize(t)) > MAX_TOKENS for t in texts)
    if too_long:
        print(f"Note: {too_long} chunks are longer than {MAX_TOKENS} word pieces; their ends are not searched")

    t0 = time.perf_counter()
    vectors = model.encode(texts, batch_size=32, convert_to_tensor=True,
                           normalize_embeddings=True, show_progress_bar=True)
    print(f"Embedded {len(texts)} chunks in {time.perf_counter() - t0:.1f} s")

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model_name,
        "built": date.today().isoformat(),
        "embeddings": vectors.cpu().float(),        # shape: [number of chunks, 384]
        "chunks": records,
    }, index_path)
    size_mb = index_path.stat().st_size / 1_000_000
    print(f"Saved {index_path} ({size_mb:.1f} MB, vectors {tuple(vectors.shape)})")
    return model


# ---------- Searching (rag.py imports these) ----------

def load_index(index_path=DEFAULT_INDEX):
    return torch.load(index_path, weights_only=True)


def search(index, model, question, k=TOP_K):
    """Return the k best chunks for a question as (score, chunk) pairs, best first.

    Vectors are normalised, so a dot product is the cosine similarity:
    about 1.0 = same meaning, about 0 = unrelated.
    """
    q = model.encode(question, convert_to_tensor=True, normalize_embeddings=True).cpu().float()
    scores = index["embeddings"] @ q
    best = torch.topk(scores, k=min(k, len(scores)))
    return [(round(float(s), 3), index["chunks"][int(i)]) for s, i in zip(best.values, best.indices)]


# ---------- Testing ----------

def show(results):
    for rank, (score, c) in enumerate(results, 1):
        print(f"   {rank}. {score:.3f}  {c['title']}  ({c['id']})")


def run_tests(index, model):
    print(f"\nSearch test: {len(TEST_QUESTIONS)} questions, top {TOP_K} results each\n")
    top1 = top3 = answerable = 0
    t0 = time.perf_counter()
    for question, expected in TEST_QUESTIONS:
        results = search(index, model, question)
        print(f"Q: {question}")
        show(results)
        if expected is None:
            print(f"   (out of scope: best score {results[0][0]:.3f})\n")
            continue
        answerable += 1
        hits = [any(e in c["url"] for e in expected) for _, c in results]
        top1 += hits[0]
        top3 += any(hits)
        print(f"   {'PASS' if hits[0] else 'top 3' if any(hits) else 'MISS'}\n")
    avg_ms = (time.perf_counter() - t0) / len(TEST_QUESTIONS) * 1000
    print("=" * 60)
    print(f"Right page first: {top1}/{answerable}   Right page in top {TOP_K}: {top3}/{answerable}   "
          f"Search time: {avg_ms:.0f} ms per question")


def main():
    ap = argparse.ArgumentParser(description="Build and test the chatbot's search index.")
    ap.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    ap.add_argument("--out", default=str(DEFAULT_INDEX))
    ap.add_argument("--model", default=MODEL_NAME, help="embedding model to use")
    ap.add_argument("--test", action="store_true", help="run the test questions after building")
    ap.add_argument("--ask", help="search the existing index for one question (no rebuild)")
    args = ap.parse_args()

    if args.ask:
        index = load_index(args.out)
        model = load_model(index["model"])          # always search with the model that built the index
        print(f"Q: {args.ask}")
        show(search(index, model, args.ask))
        return

    model = build(args.chunks, args.out, args.model)
    if args.test:
        run_tests(load_index(args.out), model)


if __name__ == "__main__":
    main()
