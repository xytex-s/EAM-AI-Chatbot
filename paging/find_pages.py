"""
Find candidate shu.ac.uk pages for pages.txt from SHU's sitemap.

Reads https://www.shu.ac.uk/sitemap.xml (following sitemap indexes), keeps
student-support pages under the prefixes below, drops anything robots.txt
blocks, and writes pages_candidates.txt grouped by section. Pick the 50-150
most useful URLs from that file and copy them into pages.txt.

Usage:  python find_pages.py
Needs:  Python 3 only (standard library).
"""
import time
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import defaultdict

SITE = "https://www.shu.ac.uk"
USER_AGENT = "Group7-SHU-Chatbot-StudentProject/1.0 (Sheffield Hallam EAM module)"
DELAY_S = 1.0          # pause between sitemap downloads, to be polite
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
SKIP_IF_CONTAINS = ["/campaigns/", "/-/media/", ".pdf", ".docx", "?"]

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


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


def main():
    robots = urllib.robotparser.RobotFileParser(SITE + "/robots.txt")
    robots.read()

    all_urls = read_sitemap(SITE + "/sitemap.xml", set())
    print(f"\n{len(all_urls)} URLs in the sitemap")

    groups = defaultdict(set)
    for url in all_urls:
        path = url.replace(SITE, "", 1).lower()
        if any(s in path for s in SKIP_IF_CONTAINS):
            continue
        prefix = next((p for p in PREFIXES if path.startswith(p)), None)
        if prefix and robots.can_fetch(USER_AGENT, url):
            groups[prefix].add(url.split("#")[0])

    total = 0
    with open("pages_candidates.txt", "w") as f:
        f.write("# Candidates from the SHU sitemap - copy the useful ones into pages.txt\n")
        for prefix in PREFIXES:
            urls = sorted(groups.get(prefix, []))
            total += len(urls)
            print(f"{prefix:35} {len(urls)} pages")
            f.write(f"\n# {prefix} ({len(urls)} pages)\n")
            f.writelines(u + "\n" for u in urls)
    print(f"\nWrote {total} candidates to pages_candidates.txt")


if __name__ == "__main__":
    main()
