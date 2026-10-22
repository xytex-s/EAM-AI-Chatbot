"""
Speed test for the Group 7 chatbot language model (CPU only).

Builds prompts the same size as the real RAG prompts (top-k extracts of
~500 words each), asks a set of student questions, and records how long
each answer takes. Results are printed and saved to a CSV for the
troubleshooting log and the tech stack evaluation.

Usage:
    python speed_test.py                                   # Qwen2.5-1.5B, 5 extracts
    python speed_test.py Qwen/Qwen2.5-0.5B-Instruct        # the backup model
    python speed_test.py --top-k 3 --max-new 120           # smaller, faster prompt

NOTE: the extract text is placeholder test data, not real SHU information.
"""
import argparse
import csv
import platform
import resource
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same instruction rag.py will use.
SYSTEM_PROMPT = (
    "You are a Sheffield Hallam University student support assistant. "
    "Answer ONLY from the extracts provided. Keep the answer short and plain. "
    "Say which extract title you used. If the extracts do not answer the "
    "question, say you don't know and suggest contacting Student Support."
)

# ~50 words of filler, repeated to pad each extract to roughly 500 words,
# because prompt length has a big effect on CPU speed.
FILLER = (
    "This is placeholder text standing in for a saved university web page. "
    "It is not real Sheffield Hallam information. It exists only so that the "
    "prompt is about as long as a real retrieved extract would be, because the "
    "length of the prompt has a large effect on how fast a model runs on a CPU. "
)

# Each question gets one short, clearly-fake "relevant" line so the model has
# something to answer from. The last question has none, to test "I don't know".
QUESTIONS = [
    ("How do I reset my university password?",
     "TEST DATA: Students reset their password on the example self-service page."),
    ("When is the library open during exams?",
     "TEST DATA: The example library is open 24 hours a day during exam weeks."),
    ("How do I apply for university accommodation?",
     "TEST DATA: Students apply for accommodation through the example housing portal."),
    ("What wellbeing support can I get?",
     "TEST DATA: The example wellbeing team offers drop-in sessions and appointments."),
    ("How do I pay my tuition fees in instalments?",
     "TEST DATA: Fees can be split into example instalments through the student account page."),
    ("What is the Wi-Fi password for the Students' Union bar?",
     None),
]


def build_extracts(relevant_line, top_k, words_per_chunk):
    """Return top_k extracts of about words_per_chunk words each."""
    repeats = max(1, words_per_chunk // 50)
    body = FILLER * repeats
    extracts = []
    for i in range(top_k):
        text = body
        if i == 0 and relevant_line:
            text = relevant_line + " " + body
        extracts.append(f"[Extract {i + 1}: Example SHU page {i + 1}]\n{text}")
    return "\n\n".join(extracts)


def peak_ram_gb():
    # ru_maxrss is in kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def ask(model, tok, question, relevant_line, args):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_extracts(relevant_line, args.top_k, args.chunk_words)
                                     + f"\n\nQuestion: {question}"},
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    prompt_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
    seconds = time.perf_counter() - start

    new_ids = out[0][prompt_tokens:]
    answer = tok.decode(new_ids, skip_special_tokens=True).strip()
    return prompt_tokens, len(new_ids), seconds, answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--top-k", type=int, default=5, help="extracts per prompt")
    p.add_argument("--chunk-words", type=int, default=500, help="words per extract")
    p.add_argument("--max-new", type=int, default=200, help="max answer tokens")
    p.add_argument("--bf16", action="store_true", help="load in bfloat16 (half the RAM, may be slower)")
    args = p.parse_args()

    print(f"Model:       {args.model}")
    print(f"CPU threads: {torch.get_num_threads()}   Python {platform.python_version()}   torch {torch.__version__}")
    print(f"Prompt:      {args.top_k} extracts x ~{args.chunk_words} words, max {args.max_new} new tokens\n")

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"Load time:   {load_s:.1f} s   (RAM so far {peak_ram_gb():.1f} GB)\n")

    print("Warm-up run (not counted)...")
    ask(model, tok, "Hello", None, argparse.Namespace(top_k=1, chunk_words=50, max_new=10))

    rows = []
    for n, (q, line) in enumerate(QUESTIONS, 1):
        prompt_toks, new_toks, secs, answer = ask(model, tok, q, line, args)
        tps = new_toks / secs if secs else 0
        rows.append([q, prompt_toks, new_toks, round(secs, 1), round(tps, 1), answer])
        print(f"Q{n}: {q}")
        print(f"    {secs:.1f} s | prompt {prompt_toks} tok | answer {new_toks} tok | {tps:.1f} tok/s")
        print(f"    A: {answer[:200]}{'...' if len(answer) > 200 else ''}\n")

    times = [r[3] for r in rows]
    print("=" * 60)
    print(f"Average {sum(times) / len(times):.1f} s   Slowest {max(times):.1f} s   "
          f"Peak RAM {peak_ram_gb():.1f} GB   (target: under 15 s per answer)")

    out_file = f"speed_{args.model.split('/')[-1]}_k{args.top_k}.csv"
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "prompt_tokens", "answer_tokens", "seconds", "tokens_per_sec", "answer"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["model", args.model, "load_s", round(load_s, 1),
                    "peak_ram_gb", round(peak_ram_gb(), 1), "threads", torch.get_num_threads()])
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
