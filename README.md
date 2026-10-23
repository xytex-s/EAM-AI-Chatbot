# EAM AI Chatbot: Hallam Student Assistant

A proof-of-concept chatbot that answers Sheffield Hallam University student-support questions using SHU's own public web pages, and cites the page each answer came from. It runs on free, open-source software and a local AI model, on virtual machines under Proxmox VE, so no student questions are sent to a third-party AI service.

Group 7 project for the Enterprise Application Management (EAM) module, Sheffield Hallam University, 2026–27.

> **Status:** in development. Page collection is done; search, answering and the web API are next. See [Progress](#progress).

## What it does

- Answers general student questions (IT, study and assessment, extensions, wellbeing, accommodation, fees, enrolment) from about 150 saved SHU pages, with a link to the source page.
- Says it doesn't know, rather than guessing, when the pages don't cover a question.
- Recommends nearby food, coffee and shops using OpenStreetMap.
- Detects crisis or safeguarding messages and replies with a fixed SHU wellbeing message and links to SHU's support pages, without using the AI model.

It is a demo, so it has no logins, does not use personal data and stores no chat history.

## How it works

The chatbot uses **retrieval-augmented generation (RAG)**: it searches the saved pages for the most relevant extracts and asks a small language model to answer using only those extracts.

```
Offline (run when pages change)
  pages.txt -> collect_pages.py -> chunks.json -> build_index.py -> index.pt

Per question
  browser -> Nginx (VM 1) -> Flask /ask (VM 2)
               1. safety check   -> crisis? fixed message + SHU links, stop
               2. places check   -> local food/shops? OpenStreetMap results
               3. search index   -> top 3 matching extracts
               4. language model -> short answer + source link
```

### Infrastructure

Three Ubuntu Server 24.04 VMs run on one Proxmox VE host (lab PC: 4-core CPU, 64 GB RAM).

| VM | Role | Runs |
| --- | --- | --- |
| VM 1 | Website | Nginx (HTTPS, rate limiting), keepalived |
| VM 2 | Main AI | Flask and Gunicorn, the search index and the language model |
| VM 3 | Standby | A copy of the app with a smaller model, a standby Nginx and keepalived. It takes over automatically if VM 1 or VM 2 fails. |

## Tech stack

| Part | Technology |
| --- | --- |
| Language | Python 3.12 |
| Page collection | requests, BeautifulSoup |
| Search | sentence-transformers (`all-MiniLM-L6-v2`), PyTorch (CPU) |
| Answers | Hugging Face transformers with Qwen2.5-Instruct (1.5B or 0.5B, chosen after testing) |
| Web API | Flask, Gunicorn |
| Web server | Nginx, keepalived |
| Local places | OpenStreetMap Overpass API |
| Hosting | Proxmox VE, Ubuntu Server 24.04 LTS |

All of it is free, and it runs on CPU only.

## Repository layout

```
paging/                   Page collection (done)
  pages.txt               149 SHU pages to collect
  collect_pages.py        Downloads, cleans and chunks the pages
  find_pages.py           Finds candidate pages from SHU's sitemap
  README.md               Details of the collection step
```

Planned, and added as each part is built:

```
build_index.py            Embeds chunks into a searchable index
rag.py                    Search and answer generation
app.py                    Flask app: chat page and POST /ask
safety.py                 Crisis and safeguarding check
places.py                 OpenStreetMap lookup
templates/chat.html       Chat page
deploy/                   Nginx, keepalived and systemd configuration
docs/                     API contract, speed tests, troubleshooting log
```

## Getting started

Page collection works today:

```
cd paging
pip install requests beautifulsoup4
python collect_pages.py
```

This writes `paging/data/chunks.json`. See [`paging/README.md`](paging/README.md) for the details.

Setup steps for the full chatbot will be added once `app.py` exists.

## API

The chat page talks to the backend through one endpoint: `POST /ask` with `{"question": "..."}`. It returns the answer, source links, any local places, a safety flag and the response time. The full contract, with example responses for every case, is in `ask-api-examples.json`.

## Design decisions

- **Local model, not a paid API.** Questions stay on our own servers and the project costs nothing. Speed tests showed that a 1.5B model on CPU answers in about 15 seconds when the prompt is kept to 3 extracts of about 300 words.
- **Crisis content never goes to the model.** Safeguarding pages are deliberately left out of the index. Those questions get a fixed message with links to SHU's own pages, so the AI can't misstate a helpline or leave out a warning.
- **Polite, curated scraping.** Public pages only, `robots.txt` respected, a delay between requests, and a hand-picked page list instead of a full crawl.
- **Graceful degradation.** If the main AI VM fails, the standby keeps answering, on a smaller and faster model.

## Team

| Member | Responsibility |
| --- | --- |
| Ethan | Python backend: page collection, indexing, RAG, Flask API; tech stack evaluation |
| Waiyan | Proxmox infrastructure, VMs, Nginx, backups and failover |
| Ahmed | Chat page, safety check, local places lookup |
| Junaid | Project management, weekly meetings, testing and results |

## Progress

| Milestone | Date | Status |
| --- | --- | --- |
| Page collection | 24 Oct 2026 | Done (144 pages, 273 chunks) |
| First answer with a source link | 28 Oct 2026 | In progress |
| Site live over HTTPS | 4 Nov 2026 | Planned |
| Backup, restore and failover demos | 11 Nov 2026 | Planned |
| Testing against Hallam Helper | w/c 18 Nov 2026 | Planned |
| Final presentation | w/c 11 Jan 2027 | Planned |

## AI use

Parts of this project's code and documentation were written with help from Claude (Anthropic). Commits containing AI-assisted work include a `Co-Authored-By` line. Further details are in the report's AI transparency appendix.
