# EAM AI Chatbot (Group 7)

A proof-of-concept chatbot that answers SHU student-support questions from SHU's own public pages and links to the page each answer came from. It is free and runs locally: a small AI model on Proxmox VMs, with no paid APIs.

## How it works

1. `paging/collect_pages.py` saves and chunks about 150 SHU pages.
2. `build_index.py` makes the chunks searchable.
3. For each question, `/ask` runs the safety check, then searches for the best extracts and has the model write a short answer with a source link.

Crisis and safeguarding questions skip the AI and get a fixed SHU message with links.

## Status

| Part | Owner | Status |
| --- | --- | --- |
| Page collection | Ethan | Done: 144 pages, 273 chunks |
| Index, RAG, Flask `/ask` | Ethan | In progress, first answer due 28 Oct |
| Chat page, safety, places | Ahmed | In progress |
| VMs, Nginx, failover | Waiyan | In progress |
| Testing and meetings | Junaid | Testing w/c 18 Nov |

## Links

- Tech reference doc: architecture, API and VM sizing
- `ask-api-examples.json`: the `/ask` request and response format
