"""
collect_pages.py - Group 7 chatbot, step 1 of the offline build.

Reads the URLs in pages.txt, downloads each public page politely, strips the
menus, headers, footers and cookie banners, and splits the main text into
chunks of about 300 words. Every chunk keeps its page URL and title, so the
chatbot can say which page an answer came from.

Output (in the data/ folder next to this script):
    chunks.json   list of chunks, read by build_index.py
    review.txt    the same chunks laid out for a person to skim and check
    report.csv    one row per URL: what happened, word and chunk counts

Usage:
    python collect_pages.py                  # reads pages.txt next to this script
    python collect_pages.py other_list.txt   # reads a different list
    python collect_pages.py --limit 3        # quick test on the first 3 URLs

Needs: pip install requests beautifulsoup4
"""
import argparse
import csv
import json
import re
import time
from datetime import date
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ---------- Settings (explained in the tech reference) ----------
USER_AGENT = "Group7-SHU-Chatbot-StudentProject/1.0 (Sheffield Hallam EAM module)"
DELAY_S = 1.5          # pause between pages so we never load the SHU site heavily
TIMEOUT_S = 60         # some SHU pages are slow to respond
CHUNK_WORDS = 300      # speed test: 3 chunks of ~300 words keeps answers near 15 s
OVERLAP_WORDS = 50     # repeat up to ~50 words between chunks so facts on a boundary aren't split
MIN_PAGE_WORDS = 50    # pages with less real text than this are skipped

HERE = Path(__file__).parent

# Parts of a page that are never useful answers.
REMOVE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form",
               "aside", "svg", "iframe", "button", "template", "picture"]
# Elements whose class or id mentions these are menus, banners or widgets.
NOISE = re.compile(r"cookie|breadcrumb|skip-link|social|share-|newsletter|feedback|"
                   r"megamenu|site-nav|site-header|site-footer|consent", re.I)
# Elements that hold readable text.
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "p", "li", "tr", "dt", "dd", "blockquote"]
TITLE_SUFFIX = re.compile(r"\s*[|\-–]\s*Sheffield Hallam University\s*$", re.I)


# ---------- Step 1: read the URL list ----------

def read_url_list(path):
    urls, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("http"):
            continue                      # blank line or # comment
        url = line.split()[0]             # ignore anything after the URL
        key = url.rstrip("/").lower()
        if key not in seen:               # ignore repeats
            seen.add(key)
            urls.append(url)
    return urls


# ---------- Step 2: download politely ----------

_robots = {}

def allowed_by_robots(url):
    """Check the site's robots.txt (cached once per site)."""
    host = urlparse(url)
    root = f"{host.scheme}://{host.netloc}"
    if root not in _robots:
        rp = robotparser.RobotFileParser(root + "/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None                     # robots.txt unreachable: treat as allowed
        _robots[root] = rp
    rp = _robots[root]
    return rp is None or rp.can_fetch(USER_AGENT, url)


def download(session, url):
    """Return the response, retrying once on a timeout or connection error."""
    for attempt in (1, 2):
        try:
            return session.get(url, timeout=TIMEOUT_S)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
            time.sleep(5)


# ---------- Step 3: clean the page ----------

def page_title(soup, main):
    h1 = main.find("h1") if main else None
    if h1 and h1.get_text(strip=True):
        return " ".join(h1.get_text(" ", strip=True).split())
    if soup.title and soup.title.string:
        return TITLE_SUFFIX.sub("", soup.title.string.strip())
    return "Untitled page"


def clean_page(html):
    """Return (title, list of text blocks) for the main content of a page."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(REMOVE_TAGS):
        tag.decompose()
    for el in soup.find_all(True):
        if el.attrs is None:              # already removed with its parent
            continue
        label = " ".join(el.get("class", [])) + " " + (el.get("id") or "")
        if NOISE.search(label):
            el.decompose()

    # Prefer the page's main content area; fall back to the whole body.
    main = (soup.find("main") or soup.find(attrs={"role": "main"}) or soup.find("article")
            or soup.find(id=re.compile("content", re.I)) or soup.body or soup)
    title = page_title(soup, main)

    blocks, seen = [], set()
    for el in main.find_all(BLOCK_TAGS):
        if el.find(BLOCK_TAGS):           # skip wrappers; their inner blocks are read instead
            continue
        if el.name == "tr":               # keep a table row together: "Monday | 9am to 5pm"
            cells = [" ".join(c.get_text(" ", strip=True).split()) for c in el.find_all(["td", "th"])]
            text = " | ".join(c for c in cells if c)
        else:
            text = " ".join(el.get_text(" ", strip=True).split())
        if text and text not in seen:     # some sites repeat text for mobile layouts
            seen.add(text)
            blocks.append(text)

    # Some pages keep text in plain <div>s. If we found almost nothing, take all lines.
    if sum(len(b.split()) for b in blocks) < MIN_PAGE_WORDS:
        lines = (" ".join(l.split()) for l in main.get_text("\n").splitlines())
        blocks = [l for l in dict.fromkeys(lines) if l]

    return title, blocks


# ---------- Step 4: split into chunks ----------

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def make_chunks(blocks, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Pack whole sentences into chunks of about `size` words.

    - Chunks only break between sentences, never in the middle of one.
    - The last ~`overlap` words of each chunk are repeated at the start of the
      next, so a fact near a boundary is found whichever chunk is searched.
    - A small leftover at the end of a page is added to the previous chunk
      instead of becoming a tiny chunk of its own.
    """
    # Each unit is (sentence, starts_a_new_paragraph).
    units = []
    for block in blocks:
        first = True
        for sentence in SENTENCE_END.split(block):
            words = sentence.split()
            for i in range(0, len(words), size):     # rare: a "sentence" longer than a chunk
                units.append((" ".join(words[i:i + size]), first))
                first = False

    chunks, current, count, fresh = [], [], 0, []
    for unit in units:
        n = len(unit[0].split())
        if fresh and count + n > size:               # chunk full: start the next one
            chunks.append(current)
            carry, carried = [], 0
            for prev in reversed(current):           # overlap from the end of this chunk
                m = len(prev[0].split())
                if carried + m > overlap:
                    break
                carry.insert(0, prev)
                carried += m
            current, count, fresh = carry, carried, []
        current.append(unit)
        fresh.append(unit)
        count += n

    if fresh:
        leftover = sum(len(u[0].split()) for u in fresh)
        if chunks and leftover < size // 3:
            chunks[-1] = chunks[-1] + fresh          # too small to stand alone
        else:
            chunks.append(current)

    def join(chunk_units):
        text = ""
        for sentence, new_paragraph in chunk_units:
            if text:
                text += "\n" if new_paragraph else " "
            text += sentence
        return text

    return [join(c) for c in chunks]


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Collect and chunk SHU pages for the chatbot.")
    ap.add_argument("url_list", nargs="?", default=str(HERE / "pages.txt"))
    ap.add_argument("--out", default=str(HERE / "data"), help="output folder")
    ap.add_argument("--limit", type=int, help="only collect the first N URLs (testing)")
    args = ap.parse_args()

    urls = read_url_list(args.url_list)[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(urls)} URLs to collect -> {out}\n")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    all_chunks, report, done_urls, done_text = [], [], set(), set()

    for n, url in enumerate(urls, 1):
        row = {"url": url, "final_url": "", "status": "", "http": "", "title": "",
               "words": 0, "chunks": 0, "note": ""}
        print(f"[{n}/{len(urls)}] {url}")
        try:
            if not allowed_by_robots(url):
                row["status"] = "skipped"
                row["note"] = "blocked by robots.txt"
                continue

            resp = download(session, url)
            row["http"], row["final_url"] = resp.status_code, resp.url
            if resp.status_code != 200:
                row["status"], row["note"] = "error", f"HTTP {resp.status_code}"
                continue
            if "html" not in resp.headers.get("Content-Type", ""):
                row["status"], row["note"] = "skipped", "not an HTML page"
                continue
            if resp.url.rstrip("/").lower() in done_urls:
                row["status"], row["note"] = "skipped", "redirects to a page already collected"
                continue

            title, blocks = clean_page(resp.text)
            text = "\n".join(blocks)
            words = len(text.split())
            row["title"], row["words"] = title, words
            if words < MIN_PAGE_WORDS:
                row["status"], row["note"] = "skipped", f"only {words} words of text"
                continue
            if text in done_text:
                row["status"], row["note"] = "skipped", "same text as another page"
                continue

            chunks = make_chunks(blocks)
            page_id = f"p{len(done_urls) + 1:03d}"
            for i, chunk in enumerate(chunks, 1):
                all_chunks.append({
                    "id": f"{page_id}-c{i:02d}",
                    "url": resp.url,
                    "title": title,
                    "chunk": i,
                    "words": len(chunk.split()),
                    "collected": date.today().isoformat(),
                    "text": chunk,
                })
            done_urls.add(resp.url.rstrip("/").lower())
            done_text.add(text)
            row["status"], row["chunks"] = "ok", len(chunks)
        except Exception as e:            # one bad page should never stop the run
            row["status"], row["note"] = "error", f"{type(e).__name__}: {e}"
        finally:
            report.append(row)
            note = f" ({row['note']})" if row["note"] else ""
            print(f"    {row['status']}: {row['words']} words, {row['chunks']} chunks{note}")
            if n < len(urls):
                time.sleep(DELAY_S)

    # ---------- Save ----------
    (out / "chunks.json").write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out / "report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]) if report else ["url"])
        writer.writeheader()
        writer.writerows(report)

    with open(out / "review.txt", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(f"===== {c['id']} | {c['title']} | {c['words']} words\n{c['url']}\n\n{c['text']}\n\n")

    ok = [r for r in report if r["status"] == "ok"]
    print(f"\nDone: {len(ok)} of {len(urls)} pages collected, {len(all_chunks)} chunks.")
    for status in ("skipped", "error"):
        rows = [r for r in report if r["status"] == status]
        if rows:
            print(f"{len(rows)} {status} - see report.csv")
    print(f"Check {out / 'review.txt'} for menus or junk text before building the index.")


if __name__ == "__main__":
    main()
