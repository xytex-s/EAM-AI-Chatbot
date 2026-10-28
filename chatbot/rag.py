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
MAX_ANSWER_TOKENS = 120      # about 3 sentences plus the source line; longer answers are slower
TOP_K = 3                    # speed test: 3 extracts keeps the prompt short enough for CPU

SYSTEM_PROMPT = """You are the Hallam Student Assistant. You answer questions from Sheffield Hallam University students.
Use ONLY the numbered extracts from SHU web pages that you are given.
Rules:
- Answer in at most 3 short, plain-English sentences. Never use lists or numbered steps.
- Only state facts that are written in the extracts. Do not add steps, links, phone numbers, fees or details of your own.
- If the extracts only partly answer the question, give the part they do answer.
- On the last line, write "Source:" and the number of the extract you used, for example: Source: 2
- Only if none of the extracts are about the question, reply with exactly: NO_ANSWER"""

# Repeated after the question, because small models follow the most recent instruction best.
REMINDER = "Answer in at most 3 sentences using only the extracts, then write the Source line."

# Fixed text, never written by the model.
NOT_FOUND = ("Sorry, I couldn't find that in the SHU pages I have. Hallam Help can answer most "
             "questions: https://www.shu.ac.uk/myhallam/help-and-support")

# Small models sometimes say "I don't know" in their own words instead of NO_ANSWER.
NOT_FOUND_PHRASES = re.compile(
    r"NO_ANSWER|\bI (do not|don't) know\b|"
    r"\b(extracts?|information|text) (do|does)(n't| not) (provide|mention|contain|say|include|answer|cover)",
    re.I)
SOURCE_LINE = re.compile(r"^\W*sources?\b(.*)$", re.I | re.M)       # "Source: 2", "**Source:** [2]" ...
ONLY_A_LABEL = re.compile(r"\W*(extract|source)?\s*\[?\d*\]?\W*", re.I)  # e.g. "[Extract 1]" and nothing else


# ---------- The language model ----------

class LanguageModel:
    """Runs the model inside this Python process with PyTorch, on CPU."""

    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()

    def generate(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_ANSWER_TOKENS, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        cut_off = len(new_tokens) >= MAX_ANSWER_TOKENS
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), cut_off



def last_full_sentence(text):
    """If an answer was cut off at the token limit, drop the unfinished sentence at the end."""
    text = re.sub(r"(^|\n|\s)\d+\.\s*$", "", text)  # a dangling list number such as "7."
    ends = [m.end() for m in re.finditer(r"[.!?](\s|$)", text)]
    return text[:ends[-1]].strip() if ends else text


# ---------- Load once, answer many ----------

class RAG:
    """Loads the index, the embedding model and the language model once (slow),
    then answers any number of questions (fast)."""

    def __init__(self, model=None, index_path=DEFAULT_INDEX):
        self.model_name = model or os.environ.get("MODEL_NAME") or DEFAULT_MODEL
        t0 = time.perf_counter()
        self.index = load_index(index_path)
        self.embedder = load_model(self.index["model"])
        self.llm = LanguageModel(self.model_name)
        self.load_seconds = round(time.perf_counter() - t0, 1)

    def answer(self, question):
        """Return a dict with the answer, its source and timing, ready for app.py."""
        t0 = time.perf_counter()
        results = search(self.index, self.embedder, question, k=TOP_K)
        extracts = "\n\n".join(f"[{n}] {c['title']}\n{c['text']}" for n, (_, c) in enumerate(results, 1))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extracts:\n\n{extracts}\n\nQuestion: {question}\n\n{REMINDER}"},
        ]
        raw, cut_off = self.llm.generate(messages)
        text, source = self._parse(raw, results)
        if cut_off and source:
            text = last_full_sentence(text)
        return {
            "answer": text,
            "sources": [{"title": source["title"], "url": source["url"]}] if source else [],
            "time_ms": round((time.perf_counter() - t0) * 1000),
            "top_score": results[0][0],          # logged for analysis, not used as a cut-off
            "raw": raw,                          # what the model actually wrote, for testing
        }

    @staticmethod
    def _parse(raw, results):
        """Split the model's reply into the answer text and the extract it says it used."""
        if not raw or NOT_FOUND_PHRASES.search(raw):
            return NOT_FOUND, None

        n = 1                                     # default: cite the best search match
        line = SOURCE_LINE.search(raw)
        if line:
            number = re.search(r"\d+", line.group(1))
            if number:
                n = int(number.group())
            else:                                 # "Source: Hallam Hardship Fund" - match the title
                for i, (_, c) in enumerate(results, 1):
                    if c["title"].lower() in line.group(1).lower():
                        n = i
                        break
        if not 1 <= n <= len(results):
            n = 1

        text = SOURCE_LINE.sub("", raw)
        text = re.sub(r"\s*\[\d+\]", "", text).strip()   # drop inline [1] markers; the page shows the source
        if ONLY_A_LABEL.fullmatch(text):          # no real answer, just a label like "[Extract 1]"
            return NOT_FOUND, None
        return text, results[n - 1][1]


# ---------- Command line ----------

def show(result):
    print(f"\n{result['answer']}")
    for s in result["sources"]:
        print(f"Source: {s['title']} - {s['url']}")
    print(f"({result['time_ms'] / 1000:.1f} s, best match score {result['top_score']:.3f})\n")


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
        rows.append([question, src["title"], src["url"], "yes" if ok else "no",
                     r["time_ms"] / 1000, r["top_score"], r["answer"], r["raw"]])

    print("=" * 60)
    print(f"{rag.model_name}: right source {right}/{answerable}, "
          f"average {sum(times) / len(times):.1f} s, slowest {max(times):.1f} s, load {rag.load_seconds} s")

    out = HERE / "data" / f"rag_test_{rag.model_name.split('/')[-1]}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "source_title", "source_url", "right_source_or_said_dont_know",
                    "seconds", "top_score", "answer", "raw_model_output"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["model", rag.model_name, "load_s", rag.load_seconds,
                    "run", datetime.now().isoformat(timespec="minutes")])
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(description="Answer student questions from the saved SHU pages.")
    ap.add_argument("question", nargs="?", help="a question (leave out to type several)")
    ap.add_argument("--model", help=f"Hugging Face model name (default {DEFAULT_MODEL})")
    ap.add_argument("--test", action="store_true", help="run the test questions and save a CSV")
    args = ap.parse_args()

    print("Loading...")
    rag = RAG(args.model)
    print(f"Ready: {rag.model_name} (loaded in {rag.load_seconds} s)")

    if args.test:
        run_tests(rag)
    elif args.question:
        show(rag.answer(args.question))
    else:
        while (q := input("Ask a question (blank to quit): ").strip()):
            show(rag.answer(q))


if __name__ == "__main__":
    main()