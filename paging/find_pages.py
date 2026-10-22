"""
Find candidate shu.ac.uk pages for pages.txt.

Method 1 (tried first): read SHU's sitemap.xml and keep student-support pages.
Method 2 (automatic fallback if the sitemap is slow or fails): open each page
already in pages.txt and collect the shu.ac.uk links on it (one level deep).

Both methods keep only the path prefixes below, skip anything robots.txt
blocks, and write pages_candidates.txt grouped by section. Copy the useful
URLs into pages.txt.

Usage:
    python find_pages.py              # sitemap first, links from pages.txt if that fails
    python find_pages.py --links      # skip the sitemap, go straight to method 2
Needs: Python 3 only (standard library).
"""
import sys
import time
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

SITE = "https://www.shu.ac.uk"
USER_AGENT = "Group7-SHU-Chatbot-StudentProject/1.0 (Sheffield Hallam EAM module)"
DELAY_S = 1.0          # pause between requests, to be polite
TIMEOUT_S = 120        # the sitemap can be slow to generate
MAX_SITEMAPS = 50      # safety limit if the sitemap index is very large

# Path prefixes worth keeping (lower case). Edit freely.
PREFIXES = [
    "/myhallam/",
    "/wellbeing",
    "/welcome/",
    "/funding/",
    "/study-here/fees-and-funding/",
    "/study-here/accommodation/",
    "/disabled-student-support",
    "/shucard",
]
SKIP_IF_CONTAINS = ["/campaigns/", "/-/media/", ".pdf", ".docx", ".jpg", ".png", "?", "mailto:"]

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def fetch(url, tries=2):
    """Download a URL, retrying once on a timeout or connection error."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                if resp.url != url:
                    print(f"    (redirected to {resp.url})")
                return resp.read()
        except (TimeoutError, OSError) as e:
            print(f"    attempt {attempt} failed: {e}")
            if attempt == tries:
                raise
            time.sleep(5)


# ---------- Method 1: sitemap ----------

def read_sitemap(url, seen):
    """Return all page URLs in a sitemap, following nested sitemap indexes."""
    if url in seen or len(seen) >= MAX_SITEMAPS:
        return []
    seen.add(url)
    print(f"Reading {url}")
    root = ET.fromstring(fetch(url))
    time.sleep(DELAY_S)
    if root.tag.endswith("sitemapindex"):
        urls = []
        for loc in root.findall("sm:sitemap/sm:loc", NS):
            urls += read_sitemap(loc.text.strip(), seen)
        return urls
    return [loc.text.strip() for loc in root.findall("sm:url/sm:loc", NS)]


# ---------- Method 2: links from the pages already in pages.txt ----------

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def read_seeds():
    for place in (Path(__file__).parent / "pages.txt", Path("pages.txt")):
        if place.exists():
            lines = place.read_text(encoding="utf-8").splitlines()
            return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    sys.exit("pages.txt not found - put it next to this script.")


def links_from_seeds():
    seeds = read_seeds()
    found = set(seeds)
    for url in seeds:
        print(f"Reading links on {url}")
        try:
            html = fetch(url).decode("utf-8", errors="replace")
        except Exception as e:
            print(f"    skipped: {e}")
            continue
        parser = LinkParser()
        parser.feed(html)
        for href in parser.links:
            found.add(urljoin(url, href).split("#")[0])
        time.sleep(DELAY_S)
    return list(found)


# ---------- Filter and save ----------

def normalise(url):
    """Treat shu.ac.uk and www.shu.ac.uk as the same site; drop other sites."""
    p = urlparse(url)
    if p.netloc in ("shu.ac.uk", "www.shu.ac.uk"):
        return SITE + p.path.rstrip("/")
    return None


def main():
    robots = urllib.robotparser.RobotFileParser(SITE + "/robots.txt")
    robots.read()

    all_urls = []
    if "--links" not in sys.argv:
        try:
            all_urls = read_sitemap(SITE + "/sitemap.xml", set())
            print(f"\n{len(all_urls)} URLs in the sitemap")
        except Exception as e:
            print(f"\nSitemap failed ({e}). Falling back to links from pages.txt.\n")
    if not all_urls:
        all_urls = links_from_seeds()
        print(f"\n{len(all_urls)} links found")

    groups = defaultdict(set)
    for raw in all_urls:
        url = normalise(raw)
        if not url:
            continue
        path = urlparse(url).path.lower()
        if any(s in raw.lower() for s in SKIP_IF_CONTAINS):
            continue
        prefix = next((p for p in PREFIXES if path.startswith(p)), None)
        if prefix and robots.can_fetch(USER_AGENT, url):
            groups[prefix].add(url)

    total = 0
    with open("pages_candidates.txt", "w", encoding="utf-8") as f:
        f.write("# Candidates for pages.txt - copy the useful ones across\n")
        for prefix in PREFIXES:
            urls = sorted(groups.get(prefix, []))
            total += len(urls)
            print(f"{prefix:35} {len(urls)} pages")
            f.write(f"\n# {prefix} ({len(urls)} pages)\n")
            f.writelines(u + "\n" for u in urls)
    print(f"\nWrote {total} candidates to pages_candidates.txt")


if __name__ == "__main__":
    main()
