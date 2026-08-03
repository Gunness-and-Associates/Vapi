#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HQ Lead Engine — multi-assistant Pipecat voice agent.

The engine is GENERIC. Each assistant lives in ./assistants/<id>/ as:
  - assistant.json  (models, voice, greeting, VAD settings)
  - prompt.txt      (the system prompt)

Pick which assistant to run with the ASSISTANT_ID env var
(default: hq_learning_hub). Shared provider API keys live in .env.

Add a new assistant = create ./assistants/<new-id>/ with those two files.

Cascade pipeline: Speech-to-Text (Deepgram) -> LLM (OpenAI) -> Text-to-Speech (ElevenLabs)

Run with::

    uv run bot.py                 # runs hq_learning_hub
    ASSISTANT_ID=other uv run bot.py
"""

import asyncio
import json
import os
import re
import time
import wave
from datetime import datetime

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from tool_runtime import execute_http_tool, extract_tool_result
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

import rag

load_dotenv(override=True)

# ── Load the selected assistant ───────────────────────────────────────────────
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "hq_learning_hub")
_ASSISTANTS_DIR = os.path.join(os.path.dirname(__file__), "assistants")

# ── Call recording ─────────────────────────────────────────────────────────────
RECORDINGS_DIR_NAME = "recordings"


def save_recording_wav(assistant_dir: str, audio: bytes, sample_rate: int, num_channels: int) -> str:
    """Write raw PCM audio captured during a call to a timestamped .wav file under
    assistants/<id>/recordings/. Returns just the filename (not the full path) so it
    can be stored on the call record and served later by admin.py."""
    rec_dir = os.path.join(assistant_dir, RECORDINGS_DIR_NAME)
    os.makedirs(rec_dir, exist_ok=True)
    filename = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    path = os.path.join(rec_dir, filename)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
    return filename


def load_assistant(assistant_id: str) -> dict:
    """Load an assistant's config (assistant.json) + system prompt (prompt.txt)."""
    base = os.path.join(_ASSISTANTS_DIR, assistant_id)
    with open(os.path.join(base, "assistant.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(base, "prompt.txt"), encoding="utf-8") as f:
        cfg["system_prompt"] = f.read().strip()
    return cfg


CALLS_ENDPOINT = os.getenv("CALLS_ENDPOINT", "http://localhost:8080/api/calls")
CLASSIFY_MODEL = os.getenv("CLASSIFY_MODEL", "gpt-4o-mini")


def resolve_assistant_id(body: dict) -> str:
    """Resolve the assistant for a session, preferring the explicit client choice."""
    requested = str(body.get("assistant_id") or "").strip()
    for assistant_id in (requested, ASSISTANT_ID):
        if assistant_id and os.path.isfile(
            os.path.join(_ASSISTANTS_DIR, assistant_id, "assistant.json")
        ):
            return assistant_id

    available = [
        name
        for name in os.listdir(_ASSISTANTS_DIR)
        if os.path.isfile(os.path.join(_ASSISTANTS_DIR, name, "assistant.json"))
    ]
    if len(available) == 1:
        return available[0]
    raise FileNotFoundError(
        "No valid assistant was selected. Create an assistant in the dashboard, "
        "then launch its test session from the configuration panel."
    )


async def classify_call(transcript: str) -> dict:
    """LLM pass over the transcript at call end → structured outcome / sentiment /
    lead score / summary / next step / objection. Best-effort: returns {} on any
    failure so call logging never breaks."""
    if not transcript.strip():
        return {}
    schema = (
        '{"outcome": one of ["whatsapp_optin","interested","advisor_booked","enrolled",'
        '"callback_requested","not_interested","wrong_number","no_conversation","other"],'
        ' "sentiment": one of ["positive","neutral","negative"],'
        ' "lead_score": one of ["hot","warm","cold"],'
        ' "summary": "one or two sentences", "next_step": "short phrase",'
        ' "objection": "main objection, or empty string"}'
    )
    sys = (
        "You analyze outbound sales-call transcripts for HQ Learning Hub, a Canadian "
        "job-market training program. The agent's goal is to move the lead onto WhatsApp, "
        "send the enrolment link, or book an advisor. Read the transcript and return ONLY "
        "JSON matching: " + schema
    )
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = await client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": transcript[:6000]},
            ],
            response_format={"type": "json_object"},
            max_tokens=250,
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        keys = ("outcome", "sentiment", "lead_score", "summary", "next_step", "objection")
        return {k: data[k] for k in keys if data.get(k)}
    except Exception as e:
        logger.error(f"classify_call failed: {e}")
        return {}
logger.info("Voice engine ready; assistant configuration loads per session")


def render_vars(text: str, variables: dict) -> str:
    """Substitute {{ key }} and {{ key | default: "x" }} placeholders with per-call
    values. Empty/missing values fall back to the default filter, else "" — so an
    unfilled {{leadEmail}} never reaches the model or a tool."""
    def repl(m):
        key = m.group(1).strip()
        default = m.group(2) if m.group(2) is not None else ""
        val = variables.get(key)
        return str(val) if val not in (None, "") else default

    return re.sub(
        r"\{\{\s*([\w.]+)\s*(?:\|\s*default:\s*[\"']?([^\"'}]*?)[\"']?\s*)?\}\}",
        repl,
        text or "",
    )


def build_system_instruction(assistant: dict, call_vars: dict) -> str:
    """Build the per-call instruction and add CRM context automatically."""
    default_prompt = (
        "You are a helpful voice assistant on a live phone call. Respond naturally, "
        "keep answers concise, ask one question at a time, and never invent facts."
    )
    instruction = render_vars(
        assistant.get("system_prompt") or default_prompt, call_vars
    )
    contact_name = call_vars.get("leadName") or (
        call_vars.get("first_name") if call_vars.get("first_name") != "there" else ""
    )
    contact_context = [
        ("Contact name", contact_name),
        ("Email", call_vars.get("leadEmail")),
        ("Phone", call_vars.get("leadPhone")),
    ]
    contact_context = [(label, value) for label, value in contact_context if value]
    if contact_context:
        instruction += (
            "\n\nCONTACT CONTEXT (provided securely by the calling integration):\n"
            + "\n".join(f"- {label}: {value}" for label, value in contact_context)
            + "\nUse these details naturally when relevant. Do not recite the context, "
            "and confirm sensitive information before taking an external action."
        )
    return instruction


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Run one voice session for the selected assistant."""
    # Reload for every new session so dashboard saves take effect without restarting
    # the worker. An in-progress call keeps the snapshot it started with.
    # Per-call variables (lead name/email/phone) come from runner_args.body — set by
    # the WebRTC offer now, and the telephony dialer later. They're substituted into
    # the prompt + greeting, and used as the source of truth for tool args so an
    # unfilled {{leadEmail}} can never leak out to n8n.
    _body = getattr(runner_args, "body", None)
    _body = _body if isinstance(_body, dict) else {}
    # Load a fresh snapshot for each session so saved model and voice changes apply
    # to the next call while an active call remains stable.
    assistant_id = resolve_assistant_id(_body)
    assistant = load_assistant(assistant_id)
    assistant_dir = os.path.join(_ASSISTANTS_DIR, assistant_id)
    logger.info(f"Starting bot: {assistant.get('name', assistant_id)}")
    call_vars = {
        "first_name": _body.get("first_name") or _body.get("firstName") or "there",
        "leadName": _body.get("leadName") or _body.get("first_name") or _body.get("firstName") or "",
        "leadEmail": _body.get("leadEmail") or _body.get("email") or "",
        "leadPhone": _body.get("leadPhone") or _body.get("phone") or "",
    }
    for _k, _v in _body.items():
        call_vars.setdefault(_k, _v)
    _present = [k for k, v in call_vars.items() if v and v != "there"]
    if _present:
        logger.info(f"Call variables provided: {_present}")  # keys only, no PII

    stt_cfg = assistant.get("stt", {})
    llm_cfg = assistant.get("llm", {})
    tts_cfg = assistant.get("tts", {})

    # Knowledge base: if this assistant has an index, expose a lookup tool + nudge the prompt.
    kb_enabled = rag.index_stats(assistant_dir).get("chunks", 0) > 0
    system_instruction = build_system_instruction(assistant, call_vars)
    if kb_enabled:
        system_instruction += (
            "\n\nKNOWLEDGE BASE: You have documents about the program. When asked something "
            "specific you're not fully sure of, call query_knowledge_base with the question FIRST, "
            "then answer naturally from what it returns. Never make facts up."
        )

    # Speech-to-Text — Deepgram. Model from the assistant config (default nova-3).
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTSettings(model=stt_cfg.get("model", "nova-3")),
    )

    # LLM — OpenAI. Model comes from the assistant config (default gpt-4.1).
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model=llm_cfg.get("model", os.getenv("OPENAI_MODEL", "gpt-4.1")),
            system_instruction=system_instruction,
            max_completion_tokens=220,  # ceiling only — brevity comes from the prompt; high enough
                                        # that a normal reply never truncates mid-sentence
            temperature=0.6,
        ),
    )

    # Text-to-Speech — ElevenLabs. Voice + model + tuning from the assistant config.
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        sample_rate=24000,  # matches transport audio_out_sample_rate (fixes audio track error)
        settings=ElevenLabsTTSService.Settings(
            voice=tts_cfg.get("voice_id") or os.getenv("ELEVENLABS_VOICE_ID"),
            model=tts_cfg.get("model", "eleven_flash_v2_5"),
            # Lower stability + some style = more expressive, less monotone (more human).
            stability=tts_cfg.get("stability", 0.4),
            similarity_boost=tts_cfg.get("similarity_boost", 0.8),
            style=tts_cfg.get("style", 0.3),
            speed=tts_cfg.get("speed", 1.0),
            use_speaker_boost=True,
        ),
    )

    # Call recording — captures both sides of the audio (caller + agent) as the call
    # happens. Saved to a .wav file when the call ends; see save_recording_wav() and
    # the on_audio_data handler below.
    audiobuffer = AudioBufferProcessor()

    # Tools (function calling to HTTP actions). Only advertise tools that are
    # actually executable: end_call, or a tool with a webhook_url configured.
    tool_defs = [
        t
        for t in assistant.get("tools", [])
        if t.get("enabled", True) and (t.get("name") == "end_call" or t.get("webhook_url"))
    ]
    tool_schemas = [
        FunctionSchema(
            name=t["name"],
            description=t.get("description", ""),
            properties=t.get("properties", {}),
            required=t.get("required", []),
        )
        for t in tool_defs
    ]

    if kb_enabled:
        tool_schemas.append(
            FunctionSchema(
                name="query_knowledge_base",
                description="Look up specific facts about the program from the uploaded documents. Use whenever you're asked something specific you're not fully sure of, before answering.",
                properties={"query": {"type": "string", "description": "what to look up"}},
                required=["query"],
            )
        )

    context = LLMContext(tools=tool_schemas) if tool_schemas else LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                # Tuned against false barge-in: require sustained speech (start_secs) and
                # higher confidence to interrupt, so echo / background noise doesn't cut
                # her off mid-sentence. stop_secs = silence before she's sure you're done.
                params=VADParams(
                    stop_secs=assistant.get("vad_stop_secs", 0.8),
                    start_secs=0.3,
                    confidence=0.7,
                )
            ),
        ),
    )

    # Pipeline - assembled from reusable components
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            audiobuffer,
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
    )

    # --- Tool handlers: each connected action sends its arguments to the configured
    # HTTP endpoint and returns the response to the model. end_call closes the session. ---
    def record_tool_run(tool_name: str, status: str, started: float, detail: str = ""):
        entry = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tool": tool_name,
            "status": status,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "detail": detail[:240],
        }
        try:
            with open(os.path.join(assistant_dir, "tool_runs.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"Could not write tool run log: {exc}")

    def make_webhook_handler(tool: dict):
        url, tool_name = tool["webhook_url"], tool["name"]
        async def handler(params):
            started = time.monotonic()
            args = dict(params.arguments or {})
            # Per-call CRM data is the source of truth. If a workflow declares one of
            # these context fields and the model leaves it empty, fill it from the call.
            for k in ("leadEmail", "leadName", "leadPhone", "first_name"):
                v = args.get(k)
                if (v in (None, "") or (isinstance(v, str) and "{{" in v)) and call_vars.get(k):
                    args[k] = call_vars[k]
            # Guard: still never forward an unfilled placeholder or a malformed email —
            # if there's genuinely no real value, ask the caller instead of mailing junk.
            if any("{{" in str(v) for v in args.values()):
                await params.result_callback(
                    "I don't have that on file — I need to ask the caller for it before sending."
                )
                record_tool_run(tool_name, "blocked", started, "Missing required call context")
                return
            em = args.get("leadEmail")
            if em is not None and ("@" not in str(em) or "." not in str(em).split("@")[-1]):
                await params.result_callback(
                    "That email doesn't look complete — reconfirm it with the caller, then try again."
                )
                record_tool_run(tool_name, "blocked", started, "Invalid email")
                return
            try:
                timeout_secs = max(1, min(int(tool.get("timeout_secs", 15)), 60))
                headers = {**(tool.get("headers") or {}), "X-HQ-Tool-Name": tool_name}
                status, text = await execute_http_tool(
                    url=url,
                    arguments=args,
                    headers=headers,
                    method=tool.get("method", "POST"),
                    timeout_secs=timeout_secs,
                    retries=tool.get("retries", 0),
                )
                if status < 200 or status >= 300:
                    logger.error(f"Tool '{tool_name}' returned HTTP {status}: {text[:300]}")
                    record_tool_run(tool_name, "failed", started, f"HTTP {status}: {text[:160]}")
                    await params.result_callback(tool.get("failure_message") or "The action failed. Let the caller know it could not be completed and offer another option.")
                    return
                result = extract_tool_result(text, tool.get("response_path", ""))
                success = tool.get("success_message") or ""
                record_tool_run(tool_name, "succeeded", started, f"HTTP {status}")
                await params.result_callback(f"{success}\n{result}".strip() if success else str(result or "Done."))
            except Exception as e:
                logger.error(f"Tool '{tool_name}' webhook failed ({url}): {e}")
                record_tool_run(tool_name, "failed", started, str(e))
                await params.result_callback(tool.get("failure_message") or "The action could not be completed. Apologize briefly and offer to try again or use another option.")

        return handler

    async def end_call_handler(params):
        # End silently — the agent already said its goodbye out loud in its own turn,
        # so the tool must NOT add a second one (that caused the double "take care").
        await params.result_callback("")
        await audiobuffer.stop_recording()
        await log_call()  # log now — on_client_disconnected may not fire after EndFrame
        await worker.queue_frames([EndFrame()])

    for t in tool_defs:
        name = t["name"]
        if name == "end_call":
            llm.register_function("end_call", end_call_handler)
        elif t.get("webhook_url"):
            llm.register_function(name, make_webhook_handler(t))

    if kb_enabled:
        async def kb_handler(params):
            q = (params.arguments or {}).get("query", "")
            try:
                passages = await asyncio.to_thread(rag.search, assistant_dir, q, 4)
                result = "\n\n---\n\n".join(passages) if passages else "Nothing relevant found in the knowledge base."
            except Exception as e:
                logger.error(f"KB search failed: {e}")
                result = "Knowledge base lookup didn't work."
            await params.result_callback(result)

        llm.register_function("query_knowledge_base", kb_handler)

    registered = [t["name"] for t in tool_defs] + (["query_knowledge_base"] if kb_enabled else [])
    if registered:
        logger.info(f"Registered tools: {registered}")

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Kick off the conversation with this assistant's greeting
        default_greeting = (
            "Greet the contact warmly, briefly explain why you are calling, and ask "
            "whether now is a good time to speak."
            if assistant.get("type", "outbound") == "outbound"
            else "Welcome the caller warmly and ask how you can help."
        )
        greeting_text = render_vars(
            assistant.get("greeting") or default_greeting,
            call_vars,
        )
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Your very first message in this call must be exactly the following text, "
                    "word for word, with nothing added before or after it. Do not paraphrase it, "
                    "shorten it, or treat it as already spoken — say only this, verbatim:\n\n"
                    + greeting_text
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    call_state = {"start": None, "logged": False, "recording_file": None}

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        try:
            filename = save_recording_wav(assistant_dir, audio, sample_rate, num_channels)
            call_state["recording_file"] = filename
            logger.info(f"Saved call recording: {filename}")
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")

    async def log_call():
        """At call end, post a record (transcript, duration, outcome) to the dashboard.
        Fires from end_call OR on_client_disconnected — guarded so it logs exactly once."""
        if call_state.get("logged"):
            return
        call_state["logged"] = True
        try:
            start = call_state.get("start")
            secs = int(time.time() - start) if start else 0
            turns = [
                m
                for m in context.get_messages()
                if m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str)
                and m["content"].strip()
            ]
            transcript = "\n".join(
                f'{"Caller" if m["role"] == "user" else "Agent"}: {m["content"]}' for m in turns
            )
            record = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "assistant": assistant.get("name", assistant_id),
                "direction": assistant.get("type", "outbound"),
                "duration": f"{secs // 60}:{secs % 60:02d}",
                "outcome": "completed" if turns else "no_conversation",
                "turns": len(turns),
                "transcript": transcript,
                "recording_file": call_state.get("recording_file") or "",
            }
            # End-of-call intelligence: classify outcome/sentiment/score/summary.
            if turns:
                record.update(await classify_call(transcript))
            async with aiohttp.ClientSession() as s:
                await s.post(CALLS_ENDPOINT, json=record, timeout=aiohttp.ClientTimeout(total=8))
            logger.info(f"Logged call: {secs}s, {len(turns)} turns")
        except Exception as e:
            logger.error(f"log_call failed: {e}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        call_state["start"] = time.time()
        await audiobuffer.start_recording()
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await audiobuffer.stop_recording()
        await log_call()
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,  # matches TTS sample_rate so the browser track doesn't error
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()