# Paging: SHU page collection

This is step 1 of the chatbot's offline build. It downloads public Sheffield Hallam student-support pages, removes the menus and footers, and splits the text into chunks that `build_index.py` turns into a searchable index.

```
pages.txt ──> collect_pages.py ──> data/chunks.json ──> build_index.py (next step)
```

## Files

| File | What it does |
| --- | --- |
| `pages.txt` | The 149 pages we collect, grouped by category. Lines starting with `#` are ignored. |
| `collect_pages.py` | Downloads each page, cleans it and splits it into ~300-word chunks. |
| `find_pages.py` | Finds candidate pages from SHU's sitemap (used to build `pages.txt`). |

## Setup

Requires Python 3.10 or later.

```
pip install requests beautifulsoup4
```

## Usage

```
python collect_pages.py --limit 3     # quick test on the first 3 pages
python collect_pages.py               # full run, about 5-8 minutes
```

To rebuild the page list from scratch, run `python find_pages.py` (or `python find_pages.py --links` if the sitemap is slow). It writes `pages_candidates.txt`, which you then trim by hand into `pages.txt`.

## Output (`data/`)

| File | Contents |
| --- | --- |
| `chunks.json` | One entry per chunk: `id`, `url`, `title`, `chunk`, `words`, `collected`, `text`. Read by `build_index.py`. |
| `review.txt` | Every chunk laid out for reading. Skim it after each run to check for menu or cookie text. |
| `report.csv` | One row per URL: status (`ok`, `skipped` or `error`), word and chunk counts, and the reason for any skip. |

## How it works

1. **Fetch politely.** Checks each site's `robots.txt`, identifies itself as a student project, waits 1.5 s between pages, and retries once on a timeout.
2. **Clean.** Removes scripts, navigation, headers, footers, cookie banners, breadcrumbs and share buttons, keeping only the page's main content. Table rows stay together (e.g. `Monday | 9am to 5pm`).
3. **Chunk.** Packs whole sentences into chunks of about 300 words, with about 50 words of overlap, so a fact near a boundary appears in both chunks. Every chunk keeps its source URL and title so answers can cite it.
4. **Skip bad pages.** Pages that are blocked, missing, not HTML, under 50 words, or duplicates of a page already collected are logged in `report.csv`, and the run carries on.

Settings such as chunk size and delay are constants at the top of `collect_pages.py`. Chunks are 300 words because our speed tests showed the prompt length was the main cost on CPU.

## Design decisions

- **Public pages only.** No logins and no personal data. `robots.txt` is respected; SHU's blocks only `/campaigns/` and a few admin pages.
- **Crisis and safeguarding pages are not collected.** Pages on topics such as suicide, eating disorders and abuse are listed as comments in `pages.txt` but left out of the index, so the AI can never summarise them. Instead, `safety.py` catches these questions first and returns a fixed SHU message with direct links to those pages.
- **Curated, not crawled.** `pages.txt` was trimmed from 606 sitemap candidates to pages that answer common student questions. This keeps out irrelevant chunks that could be picked by mistake.

## First run (23 Oct 2026)

144 of 149 pages were collected into 273 chunks (median 287 words), with no errors. Five link-only hub pages were skipped automatically, and no navigation or cookie text was found in the output.
