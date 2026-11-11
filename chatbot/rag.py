"""
rag.py - Group 7 chatbot: answer a question from the saved SHU pages.

    question -> search the index (build_index.py) -> top 3 extracts
             -> small language model writes a short answer -> answer + source link

The language model runs locally on CPU with Hugging Face transformers (PyTorch).

Usage:
    python rag.py "How do I get my SHUcard?"        # one question
    python rag.py                                    # type questions; answers are typed out as they're written
    python rag.py --test                             # run the test questions, save a CSV of results
    python rag.py --test --model Qwen/Qwen2.5-0.5B-Instruct

Speed settings (try them with --test and compare the summary lines):
    --max-tokens 80     longest answer allowed; every extra word costs time
    --top-k 3           more extracts = more to answer from, but slower to read (default 2)
    --threads 4         CPU threads PyTorch uses (default: PyTorch's own choice)

The model can also be set with the MODEL_NAME environment variable (used on the VMs:
VM 2 runs the 1.5B model, the standby VM 3 the 0.5B model).

The safety check is NOT done here: app.py runs safety.py first and only calls
answer() for questions that pass it.

Needs: everything build_index.py needs, plus: pip install transformers accelerate
"""
import argparse
import csv
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from build_index import DEFAULT_INDEX, TEST_QUESTIONS, load_index, load_model, search

HERE = Path(__file__).parent
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_ANSWER_TOKENS = 80       # about 2 sentences plus the source line; each extra token costs ~0.5 s on CPU
TOP_K = 2                    # extracts given to the model: 2 was as accurate as 3 and faster to read (28 Oct test)
MIN_SCORE = 0.2              # below this, the question is clearly unrelated to any page: skip the model.
                             # Evidence (29 Oct): "hello" 0.155, "Premier League" 0.129; lowest correct match so far 0.293

SYSTEM_PROMPT = """You are the Hallam Student Assistant. You answer questions from Sheffield Hallam University students.
Use ONLY the numbered extracts from SHU web pages that you are given.
Rules:
- Answer in at most 2 short, plain-English sentences. Never use lists or numbered steps.
- Only state facts that are written in the extracts. Do not add steps, links, phone numbers, fees or details of your own.
- If the extracts only partly answer the question, give the part they do answer.
- On the last line, write "Source:" and the number of the extract you used, for example: Source: 2
- Only if none of the extracts are about the question, reply with exactly: NO_ANSWER"""

# Repeated after the question, because small models follow the most recent instruction best.
REMINDER = "Answer in at most 2 sentences using only the extracts, then write the Source line."

# Fixed text, never written by the model.
NOT_FOUND = ("Sorry, I couldn't find that in the SHU pages I have. Hallam Help can answer most "
             "questions: https://www.shu.ac.uk/myhallam/help-and-support")

# Greetings and thanks get a fixed reply: no search result is relevant, so no source link.
GREETING = re.compile(r"^\W*(hi|hiya|hello|hey|good (morning|afternoon|evening))\b\W*(there)?\W*$", re.I)
THANKS = re.compile(r"^\W*(thanks?|thank you|cheers|ta)\b[\w\s]{0,15}\W*$", re.I)
GREETING_REPLY = ("Hello! I'm the Hallam Student Assistant. Ask me about studying at Sheffield Hallam, "
                  "for example IT, money, wellbeing, accommodation or exams.")
THANKS_REPLY = "You're welcome! Ask me anything else about studying at Sheffield Hallam."
BYE = re.compile(r"^\W*(bye|goodbye|good bye|see (you|ya)|cya)\b[\w\s]{0,15}\W*$", re.I)
BYE_REPLY = "Bye! Good luck with your studies."

# Small models sometimes say "I don't know" in their own words instead of NO_ANSWER.
NOT_FOUND_PHRASES = re.compile(
    r"NO_ANSWER|\bI (do not|don't) know\b|"
    r"\b(extracts?|information|text) (do|does)(n't| not) (provide|mention|contain|say|include|answer|cover)",
    re.I)
# "Source: 2", "**Source:** [2]", "The source is Extract 2" - on its own line or at the end of the answer.
SOURCE_MARK = re.compile(r"(\*\*)?\b(the\s+)?sources?(\s+line)?(\*\*)?(\s*[:\-]|\s+is\b)\s*", re.I)
# A "Source: 1" left in the text, e.g. when the model writes it at the start instead of the end (9 Nov).
CITED = re.compile(SOURCE_MARK.pattern + r"(extract\s*)?\[?\d+\]?\.?\s*", re.I)
# A last sentence that only points at extracts, e.g. "Use Extracts 1 & 2." or "(Extract 2)".
EXTRACT_TAIL = re.compile(r"\s*\(?[^.\n()]*\bextracts?\s*\[?\d+[^.\n]*\.?\s*$", re.I)
ONLY_A_LABEL = re.compile(r"\W*(extract|source)?\s*\[?\d*\]?\W*", re.I)  # e.g. "[Extract 1]" and nothing else

# While streaming, text is held back until it's safe to show (see RAG._visible):
HOLD_START = 40     # characters to wait for before showing anything: "NO_ANSWER" / "I don't know" show up early
HOLD_BACK = 12      # characters kept back at the end, in case "Source: 1" or "[1]" is still being written


# ---------- The language model ----------

class StopFlag:
    """Lets the code stop the model early, e.g. if the student closes the page mid-answer."""

    def __init__(self):
        self.event = threading.Event()

    def __call__(self, input_ids, scores, **kwargs):  # transformers asks this after every token
        return self.event.is_set()


class LanguageModel:
    """Runs the model inside this Python process with PyTorch, on CPU."""

    def __init__(self, model_name, max_tokens=MAX_ANSWER_TOKENS, threads=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.max_tokens = torch, max_tokens
        if threads:
            torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()
        # Not used: PyTorch dynamic int8 quantisation (quantize_dynamic) made Qwen2.5-1.5B 4-5x faster
        # at writing but its answers became nonsense in testing (28 Oct), so the model stays float32.

    def stream(self, messages, info):
        """Yield the answer piece by piece as the model writes it.

        The model runs in a background thread and hands each new piece of text to a
        TextIteratorStreamer, which this loop reads from. When it finishes, `info` is
        filled with the token counts and timing ('reading' = before the first word,
        'writing' = the rest) and whether the answer was cut off at the limit.
        """
        from transformers import StoppingCriteriaList, TextIteratorStreamer
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300)
        stop, result = StopFlag(), {}

        def run():
            try:
                with self.torch.inference_mode():
                    result["out"] = self.model.generate(
                        **inputs, max_new_tokens=self.max_tokens, do_sample=False,
                        streamer=streamer, stopping_criteria=StoppingCriteriaList([stop]))
            except Exception as e:                 # pass the error to the reading loop
                result["error"] = e
                streamer.end()

        t0 = time.perf_counter()
        first = None
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            for piece in streamer:
                if piece and first is None:
                    first = time.perf_counter()
                yield piece
        finally:                                   # also runs if the reader stops early
            stop.event.set()
            worker.join()
        if "error" in result:
            raise result["error"]
        t1 = time.perf_counter()
        prompt_tokens = int(inputs["input_ids"].shape[1])
        answer_tokens = int(result["out"].shape[1]) - prompt_tokens
        first = first or t1
        info.update(prompt_tokens=prompt_tokens, answer_tokens=answer_tokens,
                    read_s=round(first - t0, 2), write_s=round(t1 - first, 2),
                    cut_off=answer_tokens >= self.max_tokens)


def last_full_sentence(text):
    """If an answer was cut off at the token limit, drop the unfinished part at the end."""
    text = re.sub(r"(^|\n|\s)\d+\.\s*$", "", text)  # a dangling list number such as "7."
    ends = [m.end() for m in re.finditer(r"[.!?](\s|$)", text)
            if not re.search(r"(^|\s)\d+$", text[:m.start()])]   # "4." is a list number, not a sentence end
    if ends:
        return text[:ends[-1]].strip()
    cut = max(text.rfind(","), text.rfind(";"))       # one long unfinished sentence
    return text[:cut].strip() + " ..." if cut > 0 else text


# ---------- Load once, answer many ----------

class RAG:
    """Loads the index, the embedding model and the language model once (slow),
    then answers any number of questions (fast)."""

    def __init__(self, model=None, top_k=TOP_K, max_tokens=MAX_ANSWER_TOKENS, threads=None,
                 index_path=DEFAULT_INDEX):
        self.model_name = model or os.environ.get("MODEL_NAME") or DEFAULT_MODEL
        self.top_k, self.max_tokens = top_k, max_tokens
        t0 = time.perf_counter()
        self.index = load_index(index_path)
        self.embedder = load_model(self.index["model"])
        self.llm = LanguageModel(self.model_name, max_tokens, threads)
        self.load_seconds = round(time.perf_counter() - t0, 1)

    def settings(self):
        import torch
        n = torch.get_num_threads()
        return (f"{self.model_name}, top {self.top_k} extracts, max {self.max_tokens} tokens, "
                f"{n} CPU thread{'' if n == 1 else 's'}")

    def answer(self, question):
        """Return a dict with the answer, its source and timing (the final 'done' event)."""
        for event in self.answer_stream(question):
            pass
        event.pop("type")
        return event

    def answer_stream(self, question):
        """Answer a question step by step, yielding events as dicts:

            {"type": "status", "stage": "searching" | "reading" | "writing", "text": ..., ("pages": [...])}
            {"type": "text", "text": "..."}        the next piece of the answer, safe to show
            {"type": "done", "answer": ..., "sources": [...], "time_ms": ..., ...}   always last

        The streamed text is a preview: the 'done' answer is the cleaned, final version
        (source line removed, cut-off sentence trimmed, "don't know" replaced), so a
        chat page should replace what it showed with done["answer"].
        """
        t0 = time.perf_counter()
        if GREETING.match(question):
            yield {"type": "done", **self._fixed(GREETING_REPLY, t0)}
            return
        if THANKS.match(question):
            yield {"type": "done", **self._fixed(THANKS_REPLY, t0)}
            return
        if BYE.match(question):
            yield {"type": "done", **self._fixed(BYE_REPLY, t0)}
            return

        yield {"type": "status", "stage": "searching", "text": "Searching SHU pages..."}
        results = search(self.index, self.embedder, question, k=self.top_k)
        best_score = max(score for score, _ in results)   # best meaning match (the top result may be there for its words)
        if best_score < MIN_SCORE:                   # nothing in the pages is about this question
            yield {"type": "done", **self._fixed(NOT_FOUND, t0, best_score)}
            return

        pages = list({c["url"]: {"title": c["title"], "url": c["url"]} for _, c in results}.values())
        yield {"type": "status", "stage": "reading",
               "text": "Reading: " + ", ".join(p["title"] for p in pages), "pages": pages}

        extracts = "\n\n".join(f"[{n}] {c['title']}\n{c['text']}" for n, (_, c) in enumerate(results, 1))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extracts:\n\n{extracts}\n\nQuestion: {question}\n\n{REMINDER}"},
        ]
        raw, shown, info = "", "", {}
        for piece in self.llm.stream(messages, info):
            if not piece:
                continue
            if not raw:
                yield {"type": "status", "stage": "writing", "text": "Writing the answer..."}
            raw += piece
            visible = self._visible(raw)
            if len(visible) > len(shown) and visible.startswith(shown):
                yield {"type": "text", "text": visible[len(shown):]}
                shown = visible

        raw = raw.strip()
        text, source = self._parse(raw, results, question)
        if info.get("cut_off") and source:
            text = last_full_sentence(text)
        yield {
            "type": "done",
            "answer": text,
            "sources": [{"title": source["title"], "url": source["url"]}] if source else [],
            "time_ms": round((time.perf_counter() - t0) * 1000),
            "top_score": best_score,             # logged for analysis
            "raw": raw,                          # what the model actually wrote, for testing
            "prompt_tokens": info.get("prompt_tokens", 0), "answer_tokens": info.get("answer_tokens", 0),
            "read_s": info.get("read_s", 0.0), "write_s": info.get("write_s", 0.0),
        }

    @staticmethod
    def _visible(raw):
        """The part of the answer so far that is safe to show while it's still being written."""
        if len(raw) < HOLD_START or NOT_FOUND_PHRASES.search(raw):
            return ""                            # may still turn into NO_ANSWER / "I don't know"
        start = CITED.match(raw.lstrip())                 # "Source: 1" written first, then the answer
        if start:
            raw = raw.lstrip()[start.end():]
        mark = SOURCE_MARK.search(raw)
        text = raw[:mark.start()] if mark else raw[:-HOLD_BACK]
        text = re.sub(r"\s*\[\d+\]", "", text)     # inline [1] markers
        text = re.sub(r"\s*\[\d*$", "", text)       # a marker that is only half written
        return text.lstrip("!?.,:;- \n").rstrip()   # rstrip: the next piece may start with "." not " "

    @staticmethod
    def _fixed(text, t0, top_score=0.0):
        """A reply written by the code, not the model: no source, no model time."""
        return {"answer": text, "sources": [], "time_ms": round((time.perf_counter() - t0) * 1000),
                "top_score": round(top_score, 3), "raw": "", "prompt_tokens": 0, "answer_tokens": 0,
                "read_s": 0.0, "write_s": 0.0}

    @staticmethod
    def _parse(raw, results, question=""):
        """Split the model's reply into the answer text and the extract it says it used."""
        if not raw or NOT_FOUND_PHRASES.search(raw):
            return NOT_FOUND, None

        # Find the last "Source ..." mention and take it out of the answer text.
        text, ref = raw, ""
        marks = list(SOURCE_MARK.finditer(raw))
        if marks:
            m = marks[-1]
            line_end = raw.find("\n", m.end())
            line_end = len(raw) if line_end == -1 else line_end
            ref = raw[m.end():line_end]
            text = raw[:m.start()] + raw[line_end:]
        tail = EXTRACT_TAIL.search(text.strip())
        if tail and len(tail.group()) < 60:                    # e.g. "Use Extracts 1 & 2."
            ref = ref or tail.group()
            text = text.strip()[:tail.start()].rstrip()
            if text and text[-1] not in ".!?":
                text += "."

        n = 1                                     # default: cite the best search match
        number = re.search(r"\d+", ref)
        if number:
            n = int(number.group())
        else:                                     # "Source: Hallam Hardship Fund" - match the title
            for i, (_, c) in enumerate(results, 1):
                if ref and c["title"].lower() in ref.lower():
                    n = i
                    break
        if not 1 <= n <= len(results):
            n = 1

        text = CITED.sub("", text)                               # any other "Source: 1" left in the text
        text = re.sub(r"\s*\[\d+\]", "", text).strip()         # inline [1] markers; the page shows the source
        text = text.lstrip("!?.,:;- ").strip()                   # stray punctuation at the start, e.g. "! This is..."
        q = question.strip().rstrip("?").lower()
        if q and text.lower().startswith(q):                     # model repeated the question first
            text = text[len(q):].lstrip(" ?.:-").strip()
            text = text[:1].upper() + text[1:]
        if ONLY_A_LABEL.fullmatch(text) or text.rstrip(" .").endswith(":"):
            return NOT_FOUND, None                               # a label, or "follow these steps:" with no steps
        return text, results[best_source(text, results, n) - 1][1]


def best_source(text, results, n):
    """Which extract the answer really came from.

    The small model sometimes names the wrong extract in its Source line (9 Nov: a grades
    answer cited "Preparing for study" when it came from "Release of Module Results").
    So count how many of the answer's words appear in each extract, and switch to another
    extract only if it clearly shares more words than the one the model named.
    """
    words = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}
    if not words:
        return n
    overlap = [len(words & set(re.findall(r"[a-z0-9]+", c["text"].lower()))) for _, c in results]
    best = max(range(len(results)), key=lambda i: overlap[i]) + 1
    return best if overlap[best - 1] >= overlap[n - 1] + 2 else n


# ---------- Command line ----------

def show(result):
    if result["answer"] is not None:                # None: already typed out by show_stream()
        print(f"\n{result['answer']}")
    for s in result["sources"]:
        print(f"Source: {s['title']} - {s['url']}")
    if not result["raw"]:                         # fixed reply from the code: the model wasn't used
        print(f"({result['time_ms'] / 1000:.1f} s, answered without the model)\n")
        return
    print(f"({result['time_ms'] / 1000:.1f} s: reading {result['read_s']} s, writing {result['write_s']} s "
          f"for {result['answer_tokens']} tokens; best match score {result['top_score']:.3f})\n")


def show_stream(events):
    """Print status steps, then type the answer out as it's written."""
    shown = ""
    for e in events:
        if e["type"] == "status":
            print(f"  ... {e['text']}", flush=True)
        elif e["type"] == "text":
            if not shown:
                print()
            print(e["text"], end="", flush=True)
            shown += e["text"]
        elif e["type"] == "done":
            if shown and e["answer"].strip() != shown.strip():
                print("\n\n[Tidied final answer]")      # e.g. a cut-off sentence was trimmed
                shown = ""
            if not shown:
                print(f"\n{e['answer']}", end="")
            print()
            show({**e, "answer": None})


def run_tests(rag):
    rows, times, right = [], [], 0
    answerable = sum(1 for _, e in TEST_QUESTIONS if e)
    for n, (question, expected) in enumerate(TEST_QUESTIONS, 1):
        r = rag.answer(question)
        src = r["sources"][0] if r["sources"] else {"title": "", "url": ""}
        if expected:
            ok = any(e in src["url"] for e in expected)
        else:
            ok = not r["sources"]                 # out of scope: should say it doesn't know
        right += ok and bool(expected)
        times.append(r["time_ms"] / 1000)
        print(f"Q{n}: {question}")
        show(r)
        rows.append([question, src["title"], src["url"], "yes" if ok else "no", r["time_ms"] / 1000,
                     r["read_s"], r["write_s"], r["prompt_tokens"], r["answer_tokens"],
                     r["top_score"], r["answer"], r["raw"]])

    reads = [row[5] for row in rows]
    per_token = sum(row[6] for row in rows) / max(1, sum(row[8] for row in rows))
    print("=" * 60)
    print(rag.settings())
    print(f"Right source {right}/{answerable} | average {sum(times) / len(times):.1f} s, "
          f"slowest {max(times):.1f} s | reading {sum(reads) / len(reads):.1f} s on average, "
          f"writing {per_token:.2f} s per token | load {rag.load_seconds} s")

    name = f"rag_test_{rag.model_name.split('/')[-1]}_k{rag.top_k}_t{rag.max_tokens}.csv"
    out = Path(os.environ.get("RESULTS_DIR", HERE / "data")) / name   # RESULTS_DIR is set in Docker
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "source_title", "source_url", "right_source_or_said_dont_know", "seconds",
                    "read_s", "write_s", "prompt_tokens", "answer_tokens", "top_score", "answer", "raw_model_output"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["settings", rag.settings(), "load_s", rag.load_seconds,
                    "run", datetime.now().isoformat(timespec="minutes")])
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(description="Answer student questions from the saved SHU pages.")
    ap.add_argument("question", nargs="?", help="a question (leave out to type several)")
    ap.add_argument("--model", help=f"Hugging Face model name (default {DEFAULT_MODEL})")
    ap.add_argument("--test", action="store_true", help="run the test questions and save a CSV")
    ap.add_argument("--max-tokens", type=int, default=MAX_ANSWER_TOKENS, help="longest answer allowed")
    ap.add_argument("--top-k", type=int, default=TOP_K, help="number of extracts given to the model")
    ap.add_argument("--threads", type=int, help="CPU threads for PyTorch")
    args = ap.parse_args()

    print("Loading...")
    rag = RAG(args.model, args.top_k, args.max_tokens, args.threads)
    print(f"Ready: {rag.settings()} (loaded in {rag.load_seconds} s)")

    if args.test:
        run_tests(rag)
    elif args.question:
        show(rag.answer(args.question))
    else:
        while (q := input("Ask a question (blank to quit): ").strip()):
            show_stream(rag.answer_stream(q))


if __name__ == "__main__":
    main()

