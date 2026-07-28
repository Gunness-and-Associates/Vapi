# HQ Lead Engine — Project Context & Handoff

> Read this first. It's the full picture of what we're building, where it stands, and how to run it. The working program is in this same folder. Detailed running history also lives in Claude Code's project memory.

## What this is
A **self-hosted AI voice + WhatsApp lead-conversion platform** for **HQ Learning Hub** — a 30-day online program that helps people break into the Canadian job market. It replaces a costly Vapi setup (~$4/call) with a fully owned stack (~$0.14/lead target).

## The funnel (how it works)
1. **Outbound cold call** — an AI voice agent ("Sofia") calls people who signed up showing interest.
2. She warms them up, explains who HQ is + the value, then **hands off to WhatsApp**.
3. On **WhatsApp**, an AI chatbot + inbound voice agent help them **enrol** (walk through the site).
4. **Stripe** payment link → **SuiteCRM** for status/records.
The phone call's only job is to move them to WhatsApp; the selling/enrolling happens there (messaging is ~free inside WhatsApp's 24h window).

## Stack
| Layer | Tech |
|---|---|
| Voice orchestration | **Pipecat** (Python, self-hosted) |
| STT / LLM / TTS | Deepgram nova-3 / OpenAI gpt-4.1 / ElevenLabs (voice `cgSgspJ2msm6clMCkdW9`, "Jessica") |
| Telephony | **Telnyx** (to buy) |
| WhatsApp | **360dialog** OR **Telnyx-as-BSP** (decision open) |
| Orchestration / CRM / Payments | n8n / SuiteCRM / Stripe |
| Hosting | InterServer VPS (to buy), Ubuntu 24.04 |
| Control panel | Custom FastAPI + HTML dashboard |

## Where it stands (27 Jul 2026)
Runs **locally**. Three processes:
- **Dashboard + API** — `uvicorn admin:app` on **:8080** (also serves the browser test client at `/test/`)
- **Voice engine** — `bot.py` on **:7860**

**Done & working:**
- Call-quality tuning (natural turn-taking, no mid-sentence cutoffs, warm voice, one-question-at-a-time, clean single/mutual goodbye)
- End-of-call intelligence — outcome / sentiment / lead-score / summary / next-step per call (stored in SQLite)
- Call Logs + Analytics pages
- RAG knowledge base (Sofia answers program questions accurately; ingests pdf/docx/txt/xlsx)
- Per-call variables ({{first_name}}/{{leadEmail}} fill from lead data)
- Tools wired to live n8n webhooks (enrollment + advisor)
- Glassmorphism UI pass (in progress)

**In progress:** UI polish (glassmorphism) + call-log management (delete / edit-outcome+tag / filter+search / detail drill-down).

**Not started (next):** VPS deploy, Telnyx telephony, n8n dialer → engine, WhatsApp integration, login/auth.

## Key files (`voice-agent/sofia-bot/server/`)
- **bot.py** — the Pipecat voice engine: STT→LLM→TTS pipeline, tool/function calling → n8n webhooks, RAG lookup, per-call variable substitution (`render_vars`), end-of-call classifier (`classify_call`), call logging.
- **admin.py** — FastAPI dashboard API: assistants CRUD, knowledge-base upload/index, call logs (SQLite `calls.db`), `/api/analytics`, serves the dashboard and the `/test` client.
- **rag.py** — knowledge base: extract (pdf/docx/txt/csv/xlsx) → chunk → embed (OpenAI) → cosine search.
- **dashboard/index.html** — the control-panel UI (Assistants / Tools / Knowledge / Call Logs / Analytics).
- **client/index.html** — browser voice test client (served at `:8080/test/`).
- **assistants/<id>/** — `assistant.json` (models, voice, tools, dialing, integrations) + `prompt.txt` (Sofia's script) + `knowledge/` (KB docs).
- **.env.example** — required keys.

## Run it locally
```bash
cd voice-agent/sofia-bot/server
cp .env.example .env          # then add API keys (NOT included in this snapshot)
uv sync
# Terminal 1:
uv run uvicorn admin:app --host 0.0.0.0 --port 8080
# Terminal 2:
uv run bot.py
```
Open **http://localhost:8080** (dashboard). Talk to Sofia at **http://localhost:8080/test/** (use headphones). Requires `uv` (https://docs.astral.sh/uv/).

## Repo
GitHub: **https://github.com/Gunness-and-Associates/Vapi** — latest work pushed. (Should be made private; runtime data + `.env` are git-ignored.)

## Open decisions
- WhatsApp provider: **360dialog** vs **Telnyx-as-BSP**.
- Domain name for the public engine endpoint.
- Two phone numbers (one voice, one WhatsApp) — see `docs/HQ_Lead_Engine_Setup_Checklist.docx`.

## Not in this snapshot (add before running)
- **`.env`** with the real API keys (Deepgram, OpenAI, ElevenLabs) — kept out for security.
- `.venv/`, `calls.db` (call data), `knowledge_index.json` (regenerated on `uv sync` + re-index).

## Docs in `docs/`
- `HQ_Lead_Engine_Project_Plan.docx` — phased plan
- `HQ_Lead_Engine_Setup_Checklist.docx` — VPS + Telnyx + WhatsApp procurement/setup
