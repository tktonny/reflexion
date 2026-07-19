# Qwen Realtime — LLM layer (replaces OpenAI WebRTC)

The OpenAI Realtime (WebRTC) LLM layer was replaced with the **April Qwen Omni Realtime**
implementation, ported from REFLEXION (`realtime_service.py` / `realtime_orchestrator.py`).

## What changed
- **Removed:** `src/hooks/useOpenAIRealtimeConversation.ts`, `src/api/realtime.ts`,
  `app/api/realtime/client-secret+api.ts`, `src/constants/ai.ts`, `react-native-webrtc`.
- **Added (Node relay, self-contained, all-Node — no Python):** `server/`
  - `index.mjs` — standalone `http.Server` + `ws.Server` on the HTTP `upgrade` event at
    `/api/clinic/realtime/ws`. (Expo Router `+api.ts` routes are Request→Response and cannot
    do a WS 101 upgrade, so the relay is a custom Node entry — the target-architecture shape.)
  - `relay.mjs` — 1:1 port of `_relay_live_qwen_session`: bidirectional pump, event whitelist,
    `session.update`, server-VAD, dynamic Cherry/Roy/Kiki voice switching, wrap-up, and the
    **China-backup handshake retry on HTTP 401/403**.
  - `orchestrator.mjs` + `conversationFlow.json` — hidden 4-stage plan, opening messages,
    no-markdown / no-premature-goodbye rules, 6-language greetings.
  - `voice.mjs`, `qwenConfig.mjs` — voice profiles, transcript language detection, config.
- **Added (client):** `src/hooks/useQwenRealtimeConversation.ts` (web Web-Audio capture at
  16 kHz, gapless 24 kHz playback, half-duplex capture hold), `src/constants/realtime.ts`,
  and a verification screen `app/realtime-test.tsx`.
- **Rewired:** `app/conversation.tsx` now uses `useQwenRealtimeConversation`.

## Run (web, fastest to verify)
1. `cp .env.example .env` and set `QWEN_API_KEY` (or `DASHSCOPE_API_KEY`).
2. `npm install`
3. Terminal A — relay:  `npm run relay`   (serves `ws://localhost:8787/api/clinic/realtime/ws`)
4. Terminal B — web app: `npm run web`
5. Open `http://localhost:8081/realtime-test`, press **Start**, allow the microphone, and talk.
   Aria opens the check-in; server VAD drives turn-taking.

## Headless check (no mic)
With the relay running: `node server/smoke.mjs` — connects like the browser and asserts the
Qwen upstream session comes up (`reflexion.session.ready → session.created/updated → response.audio.delta`).
Exits 0 on PASS.

## Notes / next
- **Native (Android) audio** is not wired yet — the browser path uses Web Audio. Android needs
  a native PCM module (`MirrorAudio`), per the blueprint. `useQwenRealtimeConversation` throws a
  clear message on native for now.
- The relay currently loads no identity/memory (first-time patient). Wiring `IdentityProfile`
  (preferred name + memory) into `buildLiveInstructions` is the next increment.
- Voice defaults to `Cherry`; it auto-switches to `Roy` (Minnan) / `Kiki` (Cantonese) from
  transcript language signals, matching the April implementation.
