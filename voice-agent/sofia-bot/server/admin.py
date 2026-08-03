#
# HQ Lead Engine — Assistant Manager (admin API + dashboard)
#
# A Vapi-style control panel to create / edit / manage voice assistants.
# Reads & writes the same ./assistants/<id>/ configs the voice engine (bot.py) runs on,
# so anything you change here is what the live agent uses on its next call.
#
# Run:  uv run uvicorn admin:app --host 0.0.0.0 --port 8080
# Open: http://localhost:8080
#

import asyncio
import json
import os
import re
import shutil
import sqlite3
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag
from tool_runtime import ALLOWED_HTTP_METHODS, execute_http_tool, normalize_http_method

load_dotenv()  # so the embedding step has OPENAI_API_KEY

BASE = os.path.dirname(__file__)
ASSISTANTS_DIR = os.path.join(BASE, "assistants")
DASHBOARD_DIR = os.path.join(BASE, "dashboard")
CLIENT_DIR = os.path.join(BASE, "..", "client")  # branded test client, served at /test
os.makedirs(ASSISTANTS_DIR, exist_ok=True)

app = FastAPI(title="HQ Lead Engine — Assistant Manager")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# What the dashboard offers in dropdowns. Single provider each (OpenAI / Deepgram /
# ElevenLabs), with the compatible model choices for each. Extend anytime.
OPTIONS = {
    "llm": {
        "openai": [
            "gpt-5.6-sol",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
        ],
    },
    "stt": {
        "deepgram": [
            "nova-3",
            "nova-3-general",
            "nova-3-medical",
            "nova-2",
            "nova-2-general",
            "nova-2-phonecall",
            "nova-2-conversationalai",
        ]
    },
    "tts": {
        "elevenlabs": [
            "eleven_flash_v2_5",
            "eleven_multilingual_v2",
            "eleven_v3",
            "eleven_flash_v2",
        ]
    },
}

# Type-specific starting points for a new assistant.
TYPE_DEFAULTS = {
    "outbound": {
        "greeting": "Greet them warmly by name, say in one line why you're calling, and ask if it's a good time for a quick chat.",
        "prompt": "You are a warm, friendly OUTBOUND rep making a live call. Keep replies SHORT (one sentence), casual and human — react first (\"oh nice,\" \"gotcha\"), then your point. You called them, so open warmly and get to the point without rushing. Never sound scripted. Ask a question often.",
    },
    "inbound": {
        "greeting": "Warmly greet the caller, introduce yourself, and ask how you can help today.",
        "prompt": "You are a warm, friendly INBOUND assistant answering a call someone placed to you. Keep replies SHORT (one sentence), casual and human — react first, then your point. Listen for what they need and help step by step. Never sound scripted. Ask a question often.",
    },
}

DEFAULT_CONFIG = {
    "name": "New Assistant",
    "type": "outbound",
    "stt": {"provider": "deepgram", "model": "nova-3"},
    "llm": {"provider": "openai", "model": "gpt-4.1"},
    "tts": {
        "provider": "elevenlabs",
        "voice_id": "FGY2WhTYpPnrIDTdsKH5",
        "model": "eleven_flash_v2_5",
        "stability": 0.4,
        "similarity_boost": 0.8,
        "style": 0.3,
        "speed": 0.9,
    },
    "greeting": "Greet the person warmly and ask how you can help.",
    "vad_stop_secs": 0.5,
    "tools": [
        {
            "name": "end_call",
            "description": "End the call after a warm goodbye.",
            "webhook_url": "",
            "enabled": True,
            "properties": {},
            "required": [],
        }
    ],
    "dialing": {
        "initial_delay_min": 10,
        "max_attempts": 3,
        "retry_interval_min": 60,
        "pool_size": 3,
        "business_hours": "09:00-19:00",
    },
    "integrations": {
        "result_webhook_url": "",
        "suitecrm": {"base_url": "", "module": "GA_HQ_Students", "status_field": "status"},
    },
}
DEFAULT_PROMPT = "You are a warm, friendly voice assistant. Keep replies short, casual and human — one sentence, react first, then let them talk."


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return s or "assistant"


def _paths(aid: str):
    base = os.path.join(ASSISTANTS_DIR, aid)
    return base, os.path.join(base, "assistant.json"), os.path.join(base, "prompt.txt")


def read_assistant(aid: str):
    base, cfg_path, prompt_path = _paths(aid)
    if not os.path.isfile(cfg_path):
        raise HTTPException(404, f"Assistant '{aid}' not found")
    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f)
    prompt = ""
    if os.path.isfile(prompt_path):
        with open(prompt_path, encoding="utf-8") as f:
            prompt = f.read()
    return {"id": aid, "config": config, "prompt": prompt}


def write_assistant(aid: str, config: dict, prompt: str):
    base, cfg_path, prompt_path = _paths(aid)
    os.makedirs(base, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt or "")


class AssistantPayload(BaseModel):
    config: dict
    prompt: str = ""


class ToolTestPayload(BaseModel):
    webhook_url: str
    arguments: dict = {}
    headers: dict = {}
    method: str = "POST"
    tool_name: str = "tool_test"
    timeout_secs: int = 15
    retries: int = 0


def validate_tools(config: dict):
    """Reject tool definitions that cannot be safely exposed to the model."""
    seen = set()
    for tool in config.get("tools", []):
        name = (tool.get("name") or "").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
            raise HTTPException(400, f"Invalid tool name '{name}'. Use letters, numbers, and underscores.")
        if name in seen:
            raise HTTPException(400, f"Tool name '{name}' is duplicated.")
        seen.add(name)
        if name != "end_call" and tool.get("enabled", True):
            url = (tool.get("webhook_url") or "").strip()
            if not re.match(r"^https?://", url, re.I):
                raise HTTPException(400, f"Enabled tool '{name}' needs an HTTP or HTTPS webhook URL.")
        try:
            normalize_http_method(tool.get("method", "POST"))
        except ValueError:
            allowed = ", ".join(sorted(ALLOWED_HTTP_METHODS))
            raise HTTPException(400, f"Tool '{name}' must use one of: {allowed}.") from None
        props = tool.get("properties") or {}
        if not isinstance(props, dict):
            raise HTTPException(400, f"Tool '{name}' has invalid parameters.")
        for key, schema in props.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key):
                raise HTTPException(400, f"Tool '{name}' has invalid input name '{key}'.")
            if not isinstance(schema, dict) or schema.get("type", "string") not in {
                "string", "number", "boolean"
            }:
                raise HTTPException(400, f"Tool '{name}' has an invalid type for '{key}'.")
        required = tool.get("required") or []
        if any(key not in props for key in required):
            raise HTTPException(400, f"Tool '{name}' has an unknown required parameter.")
        timeout = tool.get("timeout_secs", 15)
        if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 60:
            raise HTTPException(400, f"Tool '{name}' timeout must be between 1 and 60 seconds.")
        retries = tool.get("retries", 0)
        if not isinstance(retries, int) or not 0 <= retries <= 2:
            raise HTTPException(400, f"Tool '{name}' retries must be between 0 and 2.")
        headers = tool.get("headers") or {}
        if not isinstance(headers, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) or "\r" in k + v or "\n" in k + v
            for k, v in headers.items()
        ):
            raise HTTPException(400, f"Tool '{name}' has invalid request headers.")


@app.get("/api/options")
def get_options():
    return OPTIONS


@app.get("/api/assistants")
def list_assistants():
    out = []
    for aid in sorted(os.listdir(ASSISTANTS_DIR)):
        cfg_path = os.path.join(ASSISTANTS_DIR, aid, "assistant.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                tools = [t for t in cfg.get("tools", []) if t.get("enabled", True) and t.get("name") != "end_call"]
                kb = rag.index_stats(os.path.join(ASSISTANTS_DIR, aid))
                out.append(
                    {
                        "id": aid,
                        "name": cfg.get("name", aid),
                        "type": cfg.get("type", "outbound"),
                        "llm": cfg.get("llm", {}).get("model", ""),
                        "voice": cfg.get("tts", {}).get("voice_id", ""),
                        "tools": len(tools),
                        "kb_chunks": kb.get("chunks", 0),
                        "folder": cfg.get("folder", ""),
                    }
                )
            except Exception:
                out.append({"id": aid, "name": aid, "llm": "", "voice": "", "folder": ""})
    return out


# ── Folders (organize assistants) ──
FOLDERS_FILE = os.path.join(ASSISTANTS_DIR, "_folders.json")


def _read_folders() -> list:
    if os.path.isfile(FOLDERS_FILE):
        try:
            with open(FOLDERS_FILE, encoding="utf-8") as f:
                return json.load(f).get("folders", [])
        except Exception:
            return []
    return []


def _write_folders(folders: list):
    with open(FOLDERS_FILE, "w", encoding="utf-8") as f:
        json.dump({"folders": folders}, f, indent=2, ensure_ascii=False)


def _set_assistant_folder(aid: str, folder: str):
    cfg_path = os.path.join(ASSISTANTS_DIR, aid, "assistant.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["folder"] = folder
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)


@app.get("/api/folders")
def list_folders():
    return {"folders": _read_folders()}


@app.post("/api/folders")
def create_folder(payload: dict):
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Folder name required")
    folders = _read_folders()
    if name not in folders:
        folders.append(name)
        _write_folders(folders)
    return {"folders": folders}


@app.delete("/api/folders/{name}")
def delete_folder(name: str):
    folders = [f for f in _read_folders() if f != name]
    _write_folders(folders)
    for aid in os.listdir(ASSISTANTS_DIR):
        try:
            cfg_path = os.path.join(ASSISTANTS_DIR, aid, "assistant.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("folder") == name:
                    _set_assistant_folder(aid, "")
        except Exception:
            pass
    return {"folders": folders}


@app.post("/api/assistants/{aid}/move")
def move_assistant(aid: str, payload: dict):
    if not os.path.isfile(os.path.join(ASSISTANTS_DIR, aid, "assistant.json")):
        raise HTTPException(404, "Assistant not found")
    folder = (payload.get("folder") or "").strip()
    _set_assistant_folder(aid, folder)
    return {"ok": True, "folder": folder}


@app.get("/api/assistants/{aid}")
def get_assistant(aid: str):
    return read_assistant(aid)


@app.post("/api/assistants")
def create_assistant(payload: AssistantPayload):
    validate_tools(payload.config)
    name = payload.config.get("name") or "New Assistant"
    aid = slugify(name)
    base = os.path.join(ASSISTANTS_DIR, aid)
    if os.path.exists(base):
        i = 2
        while os.path.exists(f"{base}_{i}"):
            i += 1
        aid = f"{aid}_{i}"
    write_assistant(aid, payload.config, payload.prompt or DEFAULT_PROMPT)
    return {"id": aid, **read_assistant(aid)}


@app.put("/api/assistants/{aid}")
def update_assistant(aid: str, payload: AssistantPayload):
    base, cfg_path, _ = _paths(aid)
    if not os.path.isfile(cfg_path):
        raise HTTPException(404, f"Assistant '{aid}' not found")
    validate_tools(payload.config)
    write_assistant(aid, payload.config, payload.prompt)
    return read_assistant(aid)


@app.post("/api/tools/test")
async def test_tool(payload: ToolTestPayload):
    """Send a sample request to a connected action before it is used on a call."""
    url = payload.webhook_url.strip()
    if not re.match(r"^https?://", url, re.I):
        raise HTTPException(400, "Enter a valid HTTP or HTTPS request URL.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", payload.tool_name):
        raise HTTPException(400, "Enter a valid tool name before testing.")
    try:
        timeout_secs = max(1, min(payload.timeout_secs, 60))
        headers = {**payload.headers, "X-HQ-Tool-Name": payload.tool_name}
        status, body = await execute_http_tool(
            url=url,
            arguments=payload.arguments,
            headers=headers,
            method=payload.method,
            timeout_secs=timeout_secs,
            retries=payload.retries,
        )
        if status < 200 or status >= 300:
            raise HTTPException(502, f"Tool endpoint returned HTTP {status}: {body[:300]}")
        return {"ok": True, "status": status, "response": body[:2000]}
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        raise HTTPException(504, f"Tool endpoint did not respond within {timeout_secs} seconds.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Could not reach tool endpoint: {exc}")


@app.get("/api/assistants/{aid}/tool-runs")
def list_tool_runs(aid: str, limit: int = 25):
    """Return recent tool executions without exposing webhook credentials."""
    base, cfg_path, _ = _paths(aid)
    if not os.path.isfile(cfg_path):
        raise HTTPException(404, "Assistant not found")
    path = os.path.join(base, "tool_runs.jsonl")
    if not os.path.isfile(path):
        return {"runs": []}
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()[-max(1, min(limit, 100)):]
    runs = []
    for line in reversed(lines):
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"runs": runs}


@app.delete("/api/assistants/{aid}")
def delete_assistant(aid: str):
    base, cfg_path, _ = _paths(aid)
    if not os.path.isdir(base):
        raise HTTPException(404, f"Assistant '{aid}' not found")
    shutil.rmtree(base)
    return {"deleted": aid}


@app.get("/api/defaults")
def get_defaults(type: str = "outbound"):
    t = type if type in TYPE_DEFAULTS else "outbound"
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["type"] = t
    config["name"] = f"New {t.capitalize()} Assistant"
    config["greeting"] = TYPE_DEFAULTS[t]["greeting"]
    return {"config": config, "prompt": TYPE_DEFAULTS[t]["prompt"]}


# ── Knowledge base (per-assistant document uploads; RAG indexing = next build) ──
def _kb_dir(aid: str) -> str:
    d = os.path.join(ASSISTANTS_DIR, aid, "knowledge")
    os.makedirs(d, exist_ok=True)
    return d


def _assistant_dir(aid: str) -> str:
    return os.path.join(ASSISTANTS_DIR, aid)


@app.get("/api/assistants/{aid}/knowledge")
def list_knowledge(aid: str):
    d = _kb_dir(aid)
    files = [
        {"name": f, "size": os.path.getsize(os.path.join(d, f))}
        for f in sorted(os.listdir(d))
        if os.path.isfile(os.path.join(d, f))
    ]
    return {"files": files, **rag.index_stats(_assistant_dir(aid))}


@app.post("/api/assistants/{aid}/knowledge")
async def upload_knowledge(aid: str, file: UploadFile = File(...)):
    dest = os.path.join(_kb_dir(aid), os.path.basename(file.filename))
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    stats = await asyncio.to_thread(rag.index, _assistant_dir(aid))  # (re)build the vector index
    return {"name": os.path.basename(file.filename), "size": os.path.getsize(dest), **stats}


@app.post("/api/assistants/{aid}/knowledge/reindex")
async def reindex_knowledge(aid: str):
    return await asyncio.to_thread(rag.index, _assistant_dir(aid))


@app.delete("/api/assistants/{aid}/knowledge/{filename}")
async def delete_knowledge(aid: str, filename: str):
    p = os.path.join(_kb_dir(aid), os.path.basename(filename))
    if os.path.isfile(p):
        os.remove(p)
    await asyncio.to_thread(rag.index, _assistant_dir(aid))
    return {"deleted": filename}


# ── Call logs (engine posts a record at call end; dashboard reads them) ──
# ── Call logs — SQLite (built for volume: 1000+ calls, concurrent writes) ──
# A flat JSON file corrupts when two calls end at the same moment (interleaved
# read-modify-write) and gets slow as it grows. SQLite is atomic per INSERT,
# handles concurrent writers (WAL), and reads fast via an index — so nothing is
# lost and the dashboard stays snappy at scale.
CALLS_DB = os.path.join(BASE, "calls.db")
CALLS_FILE = os.path.join(BASE, "calls.json")  # legacy — migrated in once
_calls_write_lock = threading.Lock()
_CALL_FIELDS = (
    "time", "assistant", "direction", "duration", "turns", "outcome",
    "sentiment", "lead_score", "summary", "next_step", "objection", "transcript",
    "assistant_id", "recording_file",
)
_EXTRA_COLS = ("sentiment", "lead_score", "summary", "next_step", "objection", "assistant_id", "recording_file")


def _calls_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(CALLS_DB, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads while writing
    conn.execute("PRAGMA synchronous=NORMAL")  # durable enough, much faster
    return conn


def _init_calls_db():
    with _calls_conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS calls(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   time TEXT, assistant TEXT, direction TEXT, duration TEXT,
                   turns INTEGER, outcome TEXT, transcript TEXT,
                   sentiment TEXT, lead_score TEXT, summary TEXT,
                   next_step TEXT, objection TEXT,
                   assistant_id TEXT, recording_file TEXT)"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_calls_id ON calls(id DESC)")
        # migrate older DBs that predate the intelligence / recording columns
        existing = {r[1] for r in c.execute("PRAGMA table_info(calls)").fetchall()}
        for col in _EXTRA_COLS:
            if col not in existing:
                c.execute(f"ALTER TABLE calls ADD COLUMN {col} TEXT")
        # one-time migration from the legacy calls.json (only if DB is empty)
        n = c.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        if n == 0 and os.path.isfile(CALLS_FILE):
            try:
                with open(CALLS_FILE, encoding="utf-8") as f:
                    for r in json.load(f):
                        c.execute(
                            "INSERT INTO calls(time,assistant,direction,duration,turns,outcome,transcript)"
                            " VALUES(?,?,?,?,?,?,?)",
                            (r.get("time"), r.get("assistant"), r.get("direction"),
                             r.get("duration"), r.get("turns"), r.get("outcome"), r.get("transcript")),
                        )
            except Exception:
                pass


_init_calls_db()


@app.get("/api/calls")
def list_calls(limit: int = 500, offset: int = 0):
    limit = max(1, min(limit, 2000))
    cols = ",".join(_CALL_FIELDS)
    with _calls_conn() as c:
        rows = c.execute(
            f"SELECT id,{cols} FROM calls ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/calls")
async def add_call(payload: dict):
    cols = ",".join(_CALL_FIELDS)
    ph = ",".join("?" * len(_CALL_FIELDS))
    row = tuple(payload.get(k) for k in _CALL_FIELDS)
    with _calls_write_lock:  # serialize writes from FastAPI's threadpool
        with _calls_conn() as c:
            c.execute(f"INSERT INTO calls({cols}) VALUES({ph})", row)
    return {"ok": True}


@app.get("/api/calls/{call_id}/recording")
def get_call_recording(call_id: int):
    """Stream back the .wav file for a call, if one was captured. Looks the call up
    by its database id (returned alongside every row from /api/calls) rather than by
    filename, so the frontend never has to know the on-disk path."""
    with _calls_conn() as c:
        row = c.execute(
            "SELECT assistant_id, recording_file FROM calls WHERE id=?", (call_id,)
        ).fetchone()
    if not row or not row["recording_file"]:
        raise HTTPException(404, "No recording was captured for this call.")
    aid = os.path.basename(row["assistant_id"] or "")
    filename = os.path.basename(row["recording_file"] or "")
    path = os.path.join(ASSISTANTS_DIR, aid, "recordings", filename)
    if not aid or not filename or not os.path.isfile(path):
        raise HTTPException(404, "Recording file not found on disk.")
    return FileResponse(path, media_type="audio/wav", filename=filename)


@app.get("/api/analytics")
def analytics():
    """Aggregate call stats for the Analytics page. SQL GROUP BY so it stays fast at volume."""
    def dur_secs(d):
        try:
            m, s = str(d).split(":")
            return int(m) * 60 + int(s)
        except Exception:
            return 0

    with _calls_conn() as c:
        total = c.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

        def group(col):
            rows = c.execute(
                f"SELECT COALESCE(NULLIF(REPLACE(LOWER({col}),' ','_'),''),'(none)') k, COUNT(*) n "
                f"FROM calls GROUP BY k ORDER BY n DESC"
            ).fetchall()
            return [{"key": r["k"], "count": r["n"]} for r in rows]

        by_outcome = group("outcome")
        by_sentiment = group("sentiment")
        by_score = group("lead_score")
        avg_turns = c.execute("SELECT AVG(turns) FROM calls").fetchone()[0] or 0
        by_day = [
            {"day": r["d"], "count": r["n"]}
            for r in c.execute(
                "SELECT substr(time,1,10) d, COUNT(*) n FROM calls "
                "GROUP BY d ORDER BY d DESC LIMIT 14"
            ).fetchall()
        ][::-1]
        durations = [r["duration"] for r in c.execute("SELECT duration FROM calls").fetchall()]

    ds = [dur_secs(d) for d in durations if d]
    return {
        "total": total,
        "avg_turns": round(avg_turns, 1),
        "avg_duration_secs": (sum(ds) // len(ds)) if ds else 0,
        "by_outcome": by_outcome,
        "by_sentiment": by_sentiment,
        "by_score": by_score,
        "by_day": by_day,
    }


# Serve the dashboard (mounted last so /api/* wins)
# Branded test client — mounted BEFORE "/" so /test wins. Always available while
# the dashboard is up, so the "Talk to test" button never dead-ends on a missing server.
if os.path.isdir(CLIENT_DIR):
    app.mount("/test", StaticFiles(directory=CLIENT_DIR, html=True), name="test")

if os.path.isdir(DASHBOARD_DIR):
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")