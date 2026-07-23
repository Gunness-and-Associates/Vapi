# HQ Lead Engine

Self-hosted AI voice + WhatsApp lead-conversion platform for HQ Learning Hub.
Replaces the old ~$4/call system with a ~$0.14/lead funnel — no Vapi, no platform fee, fully owned.

## What it does
Short outbound teaser call → WhatsApp (AI text chatbot enrolls them + free inbound AI voice) → Stripe → SuiteCRM.

## Stack (locked)
| Layer | Tech |
|---|---|
| Voice orchestrator | Pipecat (Python, self-hosted) |
| Telephony | Telnyx |
| WhatsApp | 360dialog |
| STT | Soniox stt-rt-v5 |
| LLM (voice + chatbot) | Claude Haiku 4.5 |
| TTS | Cartesia Sonic |
| Orchestration / CRM / Pay | n8n · SuiteCRM · Stripe |
| Hosting | InterServer VPS (US-East), Ubuntu 24.04 |

## Folder structure
```
HQ-Lead-Engine/
├─ README.md          ← this file
├─ docs/              ← finalized plan + engineering build docs
└─ voice-agent/       ← the Pipecat voice agent (Phase 1 starts here)
```

## Start here
Phase 1 = run the voice agent locally and talk to Sofia in the browser.
See **`voice-agent/README.md`** for the step-by-step.

---

## Reliability & scale — design principles

Built to run at volume, stay up, and stay cheap:

1. **Managed APIs do the heavy lifting.** We do NOT self-host STT/LLM/TTS models — Soniox, Anthropic and Cartesia handle their own scaling and uptime. Our server only orchestrates (lightweight), so there are far fewer crash surfaces (no GPU/model-load failures). This is why our path is more reliable than the fully-local, open-source-model approach.
2. **Process isolation per call.** Each call is an isolated async task/worker — one bad call can never take down the others.
3. **Horizontal scaling.** Workers are stateless; add more containers/servers behind a load balancer as concurrency grows. Scale out, not up.
4. **Auto-restart + health checks.** Docker restart policies + a supervisor respawn a crashed worker in seconds; the system self-heals.
5. **Concurrency caps per worker.** Bounded active calls per worker prevents the audio degradation we saw on the old trunk; overflow queues instead of dropping.
6. **Graceful fallbacks.** Retry on transient API errors; degrade to advisor/human handoff rather than failing the lead.
7. **Externalized state + monitoring.** Transcripts/outcomes persisted to SuiteCRM; uptime + error alerts so we know before users do.
