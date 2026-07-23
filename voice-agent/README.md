# HQ Lead Engine — Voice Agent (Sofia) · Phase 1 Prototype

Goal of this phase: run our own voice agent locally and **talk to Sofia in the browser** to prove the loop and measure real latency. No VPS or telephony yet — that's Phase 3.

Stack for the prototype (using existing subscriptions): **Deepgram** (STT) · **OpenAI GPT-4.1-mini** (LLM) · **ElevenLabs** (TTS), wired with **Pipecat**, browser mic via SmallWebRTC.

---

## Step 1 — Add your existing API keys

You already have all three. Copy `.env.example` → `.env` and paste them in:

| Service | Existing sub | Put in `.env` as |
|---|---|---|
| Deepgram (STT) | ✅ | `DEEPGRAM_API_KEY` |
| OpenAI (LLM) | ✅ | `OPENAI_API_KEY` (model = `gpt-4.1-mini`, **not** gpt-5) |
| ElevenLabs (TTS) | ✅ | `ELEVENLABS_API_KEY` (voice pre-set to Sofia's "Laura") |

## Step 2 — Put uv on PATH (this terminal session)

```powershell
$env:Path += ";$env:APPDATA\Python\Python314\Scripts"
uv --version   # should print a version
```

## Step 3 — Scaffold the official quickstart (guaranteed-correct code)

```powershell
cd C:\Users\Hp\Desktop\HQ-Lead-Engine\voice-agent
uv tool install "pipecat-ai[cli]"
pipecat init quickstart
```

This creates a `quickstart/` folder with a correct, runnable `bot.py` for your installed Pipecat version.

## Step 4 — Hand it back to Claude

Tell Claude "scaffold done" and it will:
- rewrite `bot.py` to our stack (Deepgram + GPT-4.1-mini + ElevenLabs),
- load Sofia's prompt from `sofia_system_prompt.txt`,
- pin Python 3.12 (avoids the Python 3.14 wheel issue).

## Step 5 — Run & talk

```powershell
cd quickstart
uv run bot.py
```

Open **http://localhost:7860/client** in your browser, allow the mic, and talk to Sofia. Watch the console for latency metrics.

---

**Why this order:** Pipecat's transport/runner boilerplate changes between versions, so we let the official CLI generate the version-correct skeleton, then swap in our components (which are stable). This avoids chasing import errors.

**On the stack:** we're using your existing Deepgram/OpenAI/ElevenLabs subscriptions — all first-class in Pipecat. ElevenLabs is marginally pricier than Cartesia at high volume; swapping is a one-line change later if needed. No rush.
