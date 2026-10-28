"""
rag.py - Group 7 chatbot: answer a question from the saved SHU pages.

    question -> search the index (build_index.py) -> top 3 extracts
             -> small language model writes a short answer -> answer + source link

The language model runs locally on CPU with Hugging Face transformers (PyTorch).

Usage:
    python rag.py "How do I get my SHUcard?"        # one question
    python rag.py                                    # type questions until you enter a blank line
    python rag.py --test                             # run the test questions, save a CSV of results
    python rag.py --test --model Qwen/Qwen2.5-0.5B-Instruct

Speed settings (try them with --test and compare the summary lines):
    --max-tokens 80     longest answer allowed; every extra word costs time
    --top-k 2           fewer extracts = less for the model to read before answering
    --int8              8-bit compressed model weights (PyTorch dynamic quantisation): faster, less RAM
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
import time
from datetime import datetime
from pathlib import Path

from build_index import DEFAULT_INDEX, TEST_QUESTIONS, load_index, load_model, search

HERE = Path(__file__).parent
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_ANSWER_TOKENS = 80       # about 2 sentences plus the source line; each extra token costs ~0.5 s on CPU
TOP_K = 3                    # extracts given to the model; fewer = faster to read, but less to answer from

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

# Small models sometimes say "I don't know" in their own words instead of NO_ANSWER.
NOT_FOUND_PHRASES = re.compile(
    r"NO_ANSWER|\bI (do not|don't) know\b|"
    r"\b(extracts?|information|text) (do|does)(n't| not) (provide|mention|contain|say|include|answer|cover)",
    re.I)
# "Source: 2", "**Source:** [2]", "The source is Extract 2" - on its own line or at the end of the answer.
SOURCE_MARK = re.compile(r"(\*\*)?\b(the\s+)?sources?(\s+line)?(\*\*)?(\s*[:\-]|\s+is\b)\s*", re.I)
# A last sentence that only points at extracts, e.g. "Use Extracts 1 & 2." or "(Extract 2)".
EXTRACT_TAIL = re.compile(r"\s*\(?[^.\n()]*\bextracts?\s*\[?\d+[^.\n]*\.?\s*$", re.I)
ONLY_A_LABEL = re.compile(r"\W*(extract|source)?\s*\[?\d*\]?\W*", re.I)  # e.g. "[Extract 1]" and nothing else


# ---------- The language model ----------

class FirstTokenTimer:
    """Notes when the first word of the answer appears, splitting the time into
    'reading' (the model taking in the extracts) and 'writing' (the answer)."""

    def __init__(self):
        self.calls, self.first_token_at = 0, None

    def put(self, value):                       # transformers calls this: first the prompt, then each new token
        self.calls += 1
        if self.calls == 2:
            self.first_token_at = time.perf_counter()

    def end(self):
        pass


class LanguageModel:
    """Runs the model inside this Python process with PyTorch, on CPU."""

    def __init__(self, model_name, max_tokens=MAX_ANSWER_TOKENS, int8=False, threads=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.max_tokens = torch, max_tokens
        if threads:
            torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()
        if int8:
            # Store the big weight matrices as 8-bit numbers instead of 32-bit: a quarter of the
            # memory to read for every token, which is what limits speed on a CPU.
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

    def generate(self, messages):
        """Return (text, cut_off, timing) for one prompt."""
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        timer, t0 = FirstTokenTimer(), time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False, streamer=timer)
        t1 = time.perf_counter()
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        first = timer.first_token_at or t1
        timing = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "answer_tokens": len(new_tokens),
                  "read_s": round(first - t0, 2), "write_s": round(t1 - first, 2)}
        cut_off = len(new_tokens) >= self.max_tokens
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), cut_off, timing


def last_full_sentence(text):
    """If an answer was cut off at the token limit, drop the unfinished part at the end."""
    text = re.sub(r"(^|\n|\s)\d+\.\s*$", "", text)  # a dangling list number such as "7."
    ends = [m.end() for m in re.finditer(r"[.!?](\s|$)", text)]
    if ends:
        return text[:ends[-1]].strip()
    cut = max(text.rfind(","), text.rfind(";"))       # one long unfinished sentence
    return text[:cut].strip() + " ..." if cut > 0 else text


# ---------- Load once, answer many ----------

class RAG:
    """Loads the index, the embedding model and the language model once (slow),
    then answers any number of questions (fast)."""

    def __init__(self, model=None, top_k=TOP_K, max_tokens=MAX_ANSWER_TOKENS, int8=False, threads=None,
                 index_path=DEFAULT_INDEX):
        self.model_name = model or os.environ.get("MODEL_NAME") or DEFAULT_MODEL
        self.top_k, self.max_tokens, self.int8 = top_k, max_tokens, int8
        t0 = time.perf_counter()
        self.index = load_index(index_path)
        self.embedder = load_model(self.index["model"])
        self.llm = LanguageModel(self.model_name, max_tokens, int8, threads)
        self.load_seconds = round(time.perf_counter() - t0, 1)

    def settings(self):
        import torch
        n = torch.get_num_threads()
        return (f"{self.model_name}, top {self.top_k} extracts, max {self.max_tokens} tokens"
                f"{', int8' if self.int8 else ''}, {n} CPU thread{'' if n == 1 else 's'}")

    def answer(self, question):
        """Return a dict with the answer, its source and timing, ready for app.py."""
        t0 = time.perf_counter()
        results = search(self.index, self.embedder, question, k=self.top_k)
        extracts = "\n\n".join(f"[{n}] {c['title']}\n{c['text']}" for n, (_, c) in enumerate(results, 1))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extracts:\n\n{extracts}\n\nQuestion: {question}\n\n{REMINDER}"},
        ]
        raw, cut_off, timing = self.llm.generate(messages)
        text, source = self._parse(raw, results, question)
        if cut_off and source:
            text = last_full_sentence(text)
        return {
            "answer": text,
            "sources": [{"title": source["title"], "url": source["url"]}] if source else [],
            "time_ms": round((time.perf_counter() - t0) * 1000),
            "top_score": results[0][0],          # logged for analysis, not used as a cut-off
            "raw": raw,                          # what the model actually wrote, for testing
            **timing,                            # prompt/answer token counts, reading and writing time
        }

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

        text = re.sub(r"\s*\[\d+\]", "", text).strip()         # inline [1] markers; the page shows the source
        q = question.strip().rstrip("?").lower()
        if q and text.lower().startswith(q):                     # model repeated the question first
            text = text[len(q):].lstrip(" ?.:-").strip()
            text = text[:1].upper() + text[1:]
        if ONLY_A_LABEL.fullmatch(text) or text.rstrip(" .").endswith(":"):
            return NOT_FOUND, None                               # a label, or "follow these steps:" with no steps
        return text, results[n - 1][1]


# ---------- Command line ----------

def show(result):
    print(f"\n{result['answer']}")
    for s in result["sources"]:
        print(f"Source: {s['title']} - {s['url']}")
    print(f"({result['time_ms'] / 1000:.1f} s: reading {result['read_s']} s, writing {result['write_s']} s "
          f"for {result['answer_tokens']} tokens; best match score {result['top_score']:.3f})\n")


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

    name = f"rag_test_{rag.model_name.split('/')[-1]}_k{rag.top_k}_t{rag.max_tokens}{'_int8' if rag.int8 else ''}.csv"
    out = HERE / "data" / name
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
    ap.add_argument("--int8", action="store_true", help="8-bit compressed weights: faster, less RAM")
    ap.add_argument("--threads", type=int, help="CPU threads for PyTorch")
    args = ap.parse_args()

    print("Loading...")
    rag = RAG(args.model, args.top_k, args.max_tokens, args.int8, args.threads)
    print(f"Ready: {rag.settings()} (loaded in {rag.load_seconds} s)")

    if args.test:
        run_tests(rag)
    elif args.question:
        show(rag.answer(args.question))
    else:
        while (q := input("Ask a question (blank to quit): ").strip()):
            show(rag.answer(q))


if __name__ == "__main__":
    main()