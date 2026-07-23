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

import json
import os
import re
import shutil

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = os.path.dirname(__file__)
ASSISTANTS_DIR = os.path.join(BASE, "assistants")
DASHBOARD_DIR = os.path.join(BASE, "dashboard")
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
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5",
            "gpt-5-mini",
        ],
    },
    "stt": {
        "deepgram": [
            "nova-3",
            "nova-3-general",
            "nova-2",
            "nova-2-general",
            "nova-2-phonecall",
            "nova-2-conversationalai",
        ]
    },
    "tts": {
        "elevenlabs": [
            "eleven_flash_v2_5",
            "eleven_turbo_v2_5",
            "eleven_multilingual_v2",
            "eleven_flash_v2",
            "eleven_turbo_v2",
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
                out.append(
                    {
                        "id": aid,
                        "name": cfg.get("name", aid),
                        "type": cfg.get("type", "outbound"),
                        "llm": cfg.get("llm", {}).get("model", ""),
                        "voice": cfg.get("tts", {}).get("voice_id", ""),
                    }
                )
            except Exception:
                out.append({"id": aid, "name": aid, "llm": "", "voice": ""})
    return out


@app.get("/api/assistants/{aid}")
def get_assistant(aid: str):
    return read_assistant(aid)


@app.post("/api/assistants")
def create_assistant(payload: AssistantPayload):
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
    write_assistant(aid, payload.config, payload.prompt)
    return read_assistant(aid)


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


# Serve the dashboard (mounted last so /api/* wins)
if os.path.isdir(DASHBOARD_DIR):
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
