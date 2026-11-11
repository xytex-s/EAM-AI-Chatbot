"""
build_index.py - Group 7 chatbot, step 2 of the offline build.

Turns the chunks saved by collect_pages.py into a searchable index. Each chunk
is converted into a vector (a list of 384 numbers) by a small embedding model,
so that chunks with a similar meaning get similar vectors. A question is then
converted the same way, and the closest chunks are the best matches.

    paging/data/chunks.json  ->  build_index.py  ->  chatbot/data/index.pt

Searching is "hybrid": the meaning match above, plus a small bonus for sharing actual
words with the page (most of all with its title). Meaning alone missed "hand in my
coursework late", because no page says it that way; the word "late" is in the title
of "Regulation 9 - Late Submission", so the bonus lifts that page up.

Usage:
    python build_index.py                     # build the index
    python build_index.py --test              # build, then run the search test questions
    python build_index.py --ask "How do I get my SHUcard?"   # search an existing index
    python build_index.py --compare           # compare keyword bonus sizes on the existing index

Needs: pip install sentence-transformers   (PyTorch CPU from the speed test is enough)
The embedding model (~90 MB) downloads automatically the first time.
"""
import argparse
import json
import math
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

# How much the word-match bonus counts next to the meaning match (0 = meaning only, as before 11 Nov).
# The bonus is between 0 and 1; meaning scores for good matches are about 0.3-0.6.
# 0.3 chosen on 11 Nov (--compare on the laptop): "hand in my coursework late" moved from 10th to 2nd,
# so it's inside the 2 extracts rag.py reads, and none of the other 12 test questions dropped from 1st.
KEYWORD_WEIGHT = 0.3
RAG_EXTRACTS = 2                        # how many results rag.py gives the model (its TOP_K)
TITLE_MATCH, TEXT_MATCH = 1.0, 0.5      # a word in the page title counts double
STOP_WORDS = set("""
    the and for are you your yours can could would should how what when where which who whom why
    does did doing has have had having this that these those with from into about after before
    there their them they then than out our ours its it's not but all any some get got will just
    want need know like also very more most much many make made way one two use using used
    i'm im can't cant don't dont i've ive i'd id please help tell find there's here university
    """.split())

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
    ("How do I apply for university accommodation?",      ["accommodation/apply", "accommodation/support-and-information/frequently-asked-questions"]),
    ("Can I use ChatGPT for my assignments?",             ["artificial-intelligence"]),
    ("What is the best pizza place in Sheffield?",        None),
]

# Extra questions for --compare only (rag.py's answer test keeps the 13 above, so runs stay comparable).
EXTRA_QUESTIONS = [
    ("I submitted my assignment after the deadline, what happens?", ["late-submission", "submitting-your-work"]),
    ("I'm struggling with money for food and rent",       ["financial-difficulties", "cost-of-living", "budgeting"]),
    ("I've run out of money, can the uni help?",          ["financial-difficulties", "cost-of-living"]),
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


def search(index, model, question, k=TOP_K, keyword_weight=None):
    """Return the k best chunks for a question as (score, chunk) pairs, best first.

    Chunks are ranked by meaning + KEYWORD_WEIGHT x word match. The score returned is the
    meaning score alone (the cosine similarity: about 1.0 = same meaning, 0 = unrelated),
    so rag.py's "nothing relevant" cut-off of 0.2 still means the same as before.
    """
    if keyword_weight is None:
        keyword_weight = KEYWORD_WEIGHT
    q = model.encode(question, convert_to_tensor=True, normalize_embeddings=True).cpu().float()
    meaning = index["embeddings"] @ q
    ranking = meaning + keyword_weight * keyword_scores(index, question) if keyword_weight else meaning
    best = torch.topk(ranking, k=min(k, len(ranking))).indices
    return [(round(float(meaning[i]), 3), index["chunks"][int(i)]) for i in best]


# ---------- Word matching (the "hybrid" part of search) ----------

def words(text):
    """Lower-case words of 3+ letters, without common words, with simple endings removed."""
    out = set()
    for w in re.findall(r"[a-z0-9']+", text.lower()):
        w = w.strip("'")
        if len(w) < 3 or w in STOP_WORDS:
            continue
        for ending in ("ing", "ed", "s"):          # "fees" -> "fee", "printing" -> "print"
            if w.endswith(ending) and len(w) - len(ending) >= 3 and not w.endswith("ss"):
                w = w[:-len(ending)]
                break
        out.add(w)
    return out


_word_cache = {}

def _page_words(index):
    """Each chunk's title words and text words, and how rare each word is (worked out once)."""
    key = id(index)
    if key not in _word_cache:
        titles = [words(c["title"]) for c in index["chunks"]]
        texts = [words(c["text"]) for c in index["chunks"]]
        n = len(texts)
        counts = {}
        for t, x in zip(titles, texts):
            for w in t | x:
                counts[w] = counts.get(w, 0) + 1
        # A word in nearly every chunk ("student") is worth ~0; a word in few chunks ("late") is worth a lot.
        rarity = {w: math.log(n / c) for w, c in counts.items()}
        _word_cache[key] = (titles, texts, rarity)
    return _word_cache[key]


def keyword_scores(index, question):
    """For every chunk, 0-1: how many of the question's (rare) words it contains, title words counting double."""
    titles, texts, rarity = _page_words(index)
    q = [w for w in words(question) if rarity.get(w, 0) > 0]      # ignore words no page uses
    total = sum(rarity[w] for w in q)
    if not total:
        return torch.zeros(len(texts))
    return torch.tensor([
        sum(rarity[w] * (TITLE_MATCH if w in t else TEXT_MATCH if w in x else 0) for w in q) / total
        for t, x in zip(titles, texts)
    ])


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


def compare(index, model, weights=(0, 0.1, 0.2, 0.3, 0.4)):
    """Where the right page ranks for each test question, for several keyword bonus sizes."""
    questions = TEST_QUESTIONS + EXTRA_QUESTIONS
    print(f"\nRank of the right page (top 10) for each keyword weight. Current setting: {KEYWORD_WEIGHT}\n")
    print(f"{'Question':52}" + "".join(f"{w:>6}" for w in weights))
    totals = {w: [0, 0, 0] for w in weights}
    for question, expected in questions:
        row = f"{question[:50]:52}"
        for w in weights:
            results = search(index, model, question, k=10, keyword_weight=w)
            if expected is None:                      # out of scope: show the best meaning score instead
                row += f"{max(r[0] for r in results[:TOP_K]):>6.2f}"
                continue
            rank = next((r for r, (_, c) in enumerate(results, 1) if any(e in c["url"] for e in expected)), None)
            totals[w][0] += rank == 1
            totals[w][1] += rank is not None and rank <= RAG_EXTRACTS
            totals[w][2] += rank is not None and rank <= TOP_K
            row += f"{rank if rank else '-':>6}"
        print(row)
    answerable = sum(e is not None for _, e in questions)
    print("-" * (52 + 6 * len(weights)))
    print(f"{'Right page first (out of ' + str(answerable) + ')':52}" + "".join(f"{totals[w][0]:>6}" for w in weights))
    print(f"{'Right page in top ' + str(RAG_EXTRACTS) + ' (what rag.py reads)':52}" + "".join(f"{totals[w][1]:>6}" for w in weights))
    print(f"{'Right page in top ' + str(TOP_K):52}" + "".join(f"{totals[w][2]:>6}" for w in weights))
    print("\nFor the out-of-scope question the number is its best meaning score (should stay low).")


def main():
    ap = argparse.ArgumentParser(description="Build and test the chatbot's search index.")
    ap.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    ap.add_argument("--out", default=str(DEFAULT_INDEX))
    ap.add_argument("--model", default=MODEL_NAME, help="embedding model to use")
    ap.add_argument("--test", action="store_true", help="run the test questions after building")
    ap.add_argument("--ask", help="search the existing index for one question (no rebuild)")
    ap.add_argument("--compare", action="store_true", help="compare keyword weights on the existing index (no rebuild)")
    args = ap.parse_args()

    if args.compare:
        index = load_index(args.out)
        compare(index, load_model(index["model"]))
        return

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
