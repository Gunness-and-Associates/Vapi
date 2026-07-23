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

import json
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# ── Load the selected assistant ───────────────────────────────────────────────
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "hq_learning_hub")
_ASSISTANTS_DIR = os.path.join(os.path.dirname(__file__), "assistants")


def load_assistant(assistant_id: str) -> dict:
    """Load an assistant's config (assistant.json) + system prompt (prompt.txt)."""
    base = os.path.join(_ASSISTANTS_DIR, assistant_id)
    with open(os.path.join(base, "assistant.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(base, "prompt.txt"), encoding="utf-8") as f:
        cfg["system_prompt"] = f.read().strip()
    return cfg


ASSISTANT = load_assistant(ASSISTANT_ID)
logger.info(f"Loaded assistant '{ASSISTANT_ID}': {ASSISTANT.get('name', ASSISTANT_ID)}")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Run one voice session for the selected assistant."""
    logger.info(f"Starting bot: {ASSISTANT.get('name', ASSISTANT_ID)}")

    llm_cfg = ASSISTANT.get("llm", {})
    tts_cfg = ASSISTANT.get("tts", {})

    # Speech-to-Text — Deepgram (default model is nova-3, the current best).
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # LLM — OpenAI. Model comes from the assistant config (default gpt-4.1).
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model=llm_cfg.get("model", os.getenv("OPENAI_MODEL", "gpt-4.1")),
            system_instruction=ASSISTANT["system_prompt"],
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

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                # stop_secs = silence before the agent decides you're done. Lower = snappier.
                params=VADParams(
                    stop_secs=ASSISTANT.get("vad_stop_secs", 0.5),
                    start_secs=0.2,
                    confidence=0.6,
                )
            )
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

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Kick off the conversation with this assistant's greeting
        context.add_message(
            {
                "role": "developer",
                "content": ASSISTANT.get(
                    "greeting", "Greet the person warmly and ask how you can help."
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
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
