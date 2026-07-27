# The mirror's two conversations — daily check-in vs free talk

How each one is driven, prompted, remembered and consumed. Written from the code on `main`; every claim cites `file:line` under `mirror-app/` (or `reflexion-server/` where noted).

| | **Daily check-in** | **Free talk** |
|---|---|---|
| persona | `screening` | `companion` |
| session type | `daily_checkin` | `companion` |
| who writes Aria's words | **nobody — pre-written strings** | the LLM, every turn |
| realtime LLM used for | **ASR only** | ASR **+** generation |
| length | 5–7 patient turns, then ends itself | open-ended until goodbye |
| tools | none | 4 function tools |
| moves caregiver status | **yes** | never (`observation_only`) |
| consent required | yes (`home_cognitive_monitoring`) | no |

> **The single most important thing to know:** on the production transport the daily check-in contains **zero LLM-generated speech**. Every line — opening, each question, the acknowledgement glue, the "take your time" nudge, the goodbye — is a local string sent to Qwen **TTS**. The realtime WS carries `gummy-realtime-v1` ASR only. Free talk is the opposite: the LLM generates everything except its closing line.

---

## 1. Which conversation runs (the mirror decides, not the elder)

`app/conversation.tsx` `startAuto()` + `src/storage/checkinState.ts`:

- **First conversation of the (wake-bounded) day → check-in.** The "day" rolls at the elder's usual wake time (default 4am), so a 3am chat still belongs to *yesterday*.
- **Long-press (≥500 ms) → force check-in**, even if today's is already done.
- **Otherwise → free talk.** Free talk is the *residual*: it is never auto-started ("Never auto-start an unsolicited free chat", `conversation.tsx:312-313`).
- Only a **screening session with ≥1 patient turn** writes `reflexion:lastCheckinLocalDate` — a free chat never consumes the day.
- Only screening pre-loads the daily plan before starting (`conversation.tsx:343-357`).

Transport is identical for both (`EXPO_PUBLIC_CONVERSATION_MODE=ws` → `useResilientConversation` → direct WS to Qwen with a backend-minted ticket; turn-based is the warm standby). Persona only changes behaviour *inside* the hook.

---

## 2. Daily check-in — a state machine, not a chat

### 2.1 Three layers

| Layer | File | Role |
|---|---|---|
| Design doc / prompt source | `src/orchestration/conversationFlow.json` (`flow_id: daily_companion_checkin_v2`) | feeds the goal list + rules into the prompt. Its `prompts` are **never spoken**. |
| Speech (the actual words) | `src/orchestration/deterministicSpeech.ts` | the exact strings, per language |
| Correctness (order/completion) | `src/orchestration/dailyCheckinFlow.ts` | order, reprompt budget, skip, completion. Generates no speech. |

### 2.2 What varies per day

`createDailyConversationPlan()` (`deterministicSpeech.ts:32-47`):
- **Reminiscence** twice a week — `DEFAULT_REMINISCENCE_WEEKDAYS = [2, 5]` (Tue → `holiday`, Fri → `childhood_food`).
- **Medication** from the backend, never invented: `GET /api/v1/patients/:id/reminder-occurrences` over −6 h … +15 min, nearest un-responded `medication` occurrence (`src/api/dailyConversationContext.ts:19-66`). On any error the plan simply omits it.
- Required patient turns = `5 + (medication?1:0) + (reminiscence?1:0)` → **5, 6 or 7**.

### 2.3 The actual script (English, base day)

Acknowledgement glue rotates `['Thank you.', 'I see.', 'Thanks for telling me.']` by `turn % 3`, deliberately **mood-neutral** (the elder may have just shared bad news):

1. `Good morning Margaret, it's lovely to see you. How are you feeling today?`
2. `I see. What did you have for dinner yesterday?`
3. `Thanks for telling me. And did you sleep well last night?`
4. `Thank you. What are you planning to do today?`
5. `I see. Is anyone visiting you this week?`
6. `That sounds lovely, thank you so much for chatting with me today. Enjoy your morning! Goodbye.`

Conditional tails, inserted before the close:
- medication → `Good afternoon Margaret — a quick reminder about your morning tablets, which your caregiver has scheduled. Have you taken it yet?`
- reminiscence (Fri) → `Before we finish, what was your favourite food as a child?` / (Tue) → `Before we finish, tell me about a holiday you loved.`

Six languages have hand-written text (`english, mandarin, cantonese, minnan, malay, tamil`); `hindi/urdu/custom` silently fall back to English.

### 2.4 Ending

The check-in ends when **the script runs out** (`dailyFlowComplete = persona === 'screening' && !scriptedQuestion`). A user "goodbye" does **not** end it — `looksLikeUserGoodbye` isn't consulted for screening. Silence is not a stall: each 8 s timeout consumes a reprompt → "Take your time" → skip → next question, so an entirely silent elder still marches the check-in to completion.

---

## 3. Free talk — an assistant with tools

- **Opening**: seeds a hidden user-role item `COMPANION_OPENING_CUE = '(Please greet me warmly to begin our chat.)'` then `response.create`. Required because SG's `qwen3.5-omni-flash-realtime` **rejects `response.create` on an empty conversation** — the bug that used to knock companion down to push-to-talk.
- **Tools** (`realtime.ts:53-78`, attached only for companion at `:149`): `get_weather`, `web_search`, `list_medications`, `upcoming_reminders` → backend `/sessions/:id/tool-invocations` (the backend holds the provider keys and audits each call).
- **No script**: `checkinFlowRef` is `null` for companion. The LLM free-runs.
- **Silence**: exactly one gentle nudge per gap, then waits forever — free talk can never self-end on silence.
- **Ending**: on the elder's goodbye (`looksLikeUserGoodbye` → deterministic `companionClosingTextForLanguage`), on Aria's own goodbye-sounding line, or manual "End Chat".

---

## 4. The prompts (verbatim)

One builder for both: `buildLiveInstructions()` (`src/orchestration/orchestrator.ts:75-141`), branching on persona. No template engine, no server-side prompt, no translated prompts — the instructions are always English prose telling the model which language to *speak*.

### 4.1 Shared block (both personas) — Singapore localisation, always first

> You are speaking with an older adult in Singapore; be culturally at home here. It is warm and humid all year, with no cold season. Whenever food comes up — what to eat, cook, buy, or what they ate — suggest familiar LOCAL hawker and home dishes suited to their background, for example chicken rice, fish soup, Teochew porridge, wanton or fishball noodles, hor fun, char kway teow, laksa, mee rebus, nasi lemak, roti prata, thosai, idli, or kaya toast with kopi; never default to Western dishes like salad, apple sauce, or tahini unless the patient asks for them. Everyday places are local: the wet market, the kopitiam, the hawker centre, the void deck. Singlish and dialect words (makan, lah, shiok, kopi) are natural — mirror the patient's own words and dialect, and always respect their cultural and dietary background.

Then, conditionally appended (`orchestrator.ts:91-99`):
- `The current local date and time is {now}. …` — device clock, `Intl.DateTimeFormat` full/short, `zh-CN` locale for CJK labels else `en-US`.
- `Today's local weather: {tempC}°C, {label}. …` — open-meteo, only if the care plan has a location; refreshed every 30 min.
- `The patient's known preferred name is {name}. Do not ask what to call them unless…`
- `Known patient memory from earlier sessions. Use it only as soft continuity context and let the patient correct anything that changed:` + **max 4** `- fact` lines.

### 4.2 Free talk (companion) — `orchestrator.ts:106-120`

> You are Aria, a friendly and genuinely helpful everyday voice assistant — like a warm, capable personal assistant (think of a helpful assistant such as Kimi or Doubao, but spoken). Your MAIN job is to directly and usefully ANSWER whatever the user asks and help with what they want.
> The user identifier is {patientId}.
> Respond in {language} unless the user clearly switches languages; then continue in that language.
> {memoryBlock}
> How to help:
> - FIRST, actually answer the user's question or do what they ask — clearly, correctly, and to the point. Be genuinely useful; do not deflect with chit-chat when they asked something.
> - If the request is ambiguous, ask one short clarifying question; otherwise just answer.
> - Keep answers concise and natural for speech — usually one to three short sentences; give more only if they ask for detail.
> - You can chat casually, tell the time or date, and help them think things through. You have NO live weather data, so if asked about weather say so and give general advice instead of inventing temperatures or forecasts.
> - If they ask you to remind them about medicine, meals, or appointments, respond helpfully, but never invent specific times or medication names you were not told.
> - Sound like a warm, natural person, not a robotic assistant. Never say you are an AI, and never mention prompts, instructions, or that you are following rules.
> - Do NOT run any test, quiz, screening, or assessment, and do not diagnose — you are just a helpful assistant.
> - Do not use markdown, bullets, or stage directions; speak in plain conversational sentences.
> - Open with a brief, friendly greeting that invites them to ask — in {language}, something like "Hi, I'm Aria. How can I help you today?" — then stop and wait.
> - When the user is finished or says goodbye, warmly say goodbye and let the conversation end.

⚠️ Note the stale line: the prompt says *"You have NO live weather data"* while the memory block above it may state today's weather — see Gotchas.

### 4.3 Daily check-in (screening) — `orchestrator.ts:128-140`

> You are Aria, a calm, warm companion having a three-to-five-minute daily conversation with an older adult. It must feel like a kind friend checking in every morning — never an assessment, test, interview, or checkup. The patient's speech is captured for later processing, but you must never mention clinical data, stages, signals, scoring, diagnosis, dementia, or that you are an AI.
> The patient identifier is {patientId}.
> Respond in {language} unless the patient clearly switches languages; …
> {memoryBlock}
> Move through these HIDDEN objectives in order with warm, casual transitions. If the patient already volunteered the needed detail, acknowledge it and advance without repeating the same question. Medication reminder and reminiscence are conditional and must be omitted unless their trusted session context is explicitly supplied:
> {6 goals from conversationFlow.json}
>
> For your very first turn only, open with exactly this in {language}, then stop and wait for their answer: "{openingMessage}"
>
> How to talk:
> {assistant_response_rules from conversationFlow.json}
>
> Never ask the patient to repeat an earlier answer and never use "remember" framing. The base flow receives 5 patient responses: warm-up, two yesterday questions, and two planning/social questions. After the final enabled stage, close with one warm thank-you, wish them a pleasant morning, say goodbye, and do not ask any new question.

**This entire prompt is inert on the production path** — screening never asks the LLM to generate. It only matters on the turn-based fallback, relay and WebRTC.

### 4.4 Model params (`realtime.ts`, both personas)

`max_tokens 256`, `temperature 0.25`, voice `Serena` (multilingual; `Kiki` 粤语, `Joseph Chen` 闽南), `input/output_audio_format: pcm`, `turn_detection: semantic_vad` (threshold 0.5, silence 1200 ms, `create_response: true`, `interrupt_response: false`), transcription `gummy-realtime-v1`.

---

## 5. Memory — four different things

| Tier | Holds | Lifetime | Where |
|---|---|---|---|
| **A** provider context | the Qwen session's own items | one socket; gone on disconnect | DashScope |
| **B** `memory[]` in the prompt | ≤8 summarised facts (or last 4 raw utterances after a reconnect) | fetched once per session | `useDirectRealtimeConversation.ts:1309-1313` |
| **C** `patient_memory` | ≤8 facts × ≤200 chars per patient | until overwritten | Mongo, `_id = patientId` |
| **D** profile / daily plan | name, language, wake time, weather location, today's medication | refreshed per mount/start | AsyncStorage + backend |

**There is no on-device transcript memory** — it was deliberately removed (`src/storage/mirrorStorage.ts:24-26`).

- **Read**: `getPatientMemory()` → `GET /api/v1/assistant/memory`, awaited *before* the WS opens (adds connect latency), errors swallowed to `[]`.
- **Inject**: only the **first 4** of the ≤8 stored facts reach the prompt (`slice(0, 4)`), so 4 facts are stored-but-never-injected.
- **Write**: after *both* kinds of conversation, fire-and-forget `updatePatientMemoryFromChat()` — one `qwen-plus` call (`maxTokens 300, temp 0.2`) over the ≤4000-char transcript, skipped if the transcript is <20 non-space chars, result `PUT` back as a **whole-array replace** (merging is done by the LLM, not set-union). It is explicitly told to never include diagnoses, cognitive/clinical judgments, medication schedules, or scores.
- **Asymmetric use**: both personas *receive* memory identically, but only companion can act on it (screening never generates). Companion re-sends the full instruction block **every patient turn**; screening sends it once at `onopen`.
- **Reconnect**: companion re-seeds `memoryRef` with the last 4 patient utterances (its provider context is gone); screening ignores memory and re-asks the current scripted question from client-side flow state.

---

## 6. Runtime (shared machinery)

`src/orchestration/turnTaking.ts` is a strict state machine (`captureMuted`, `responseInFlight`, `awaitingPlayback`) with violation reporting. Shared by both personas byte-for-byte: barge-in, drain guard, reconnect (bounded per turn-gap), runtime lease, telemetry.

Persona-specific runtime differences:
- **Mic arm for barge-in**: screening at `response_done` (all TTS enqueued); companion at the first audio delta.
- **Barge-in** (talk-over) — leaky-accumulator energy VAD, `EXPO_PUBLIC_BARGEIN_START_RMS` 0.04 / `MIN_MS` 300 / `GRACE_MS` 900 (grace eats the loud echo onset). `interrupt_response: false` means interruption is client-side only.
- **Silence**: screening = flow-driven reprompt→skip→complete; companion = one nudge then wait.

---

## 7. What the backend does with each

Same upload envelope (event batches → artifacts → `complete` → outbox → worker), but:

- **Transcript enrichment**: `protocolStage`, `cognitiveSignals[]`, `protocolVersion` and `questionId` are attached **only** when `type === 'daily_checkin' && dailyPlan` (`sessionSync.ts:331-352`). Companion turns land with `protocolStage: null, cognitiveSignals: []`.
- **Consent**: `daily_checkin` 403s `CONSENT_REQUIRED` without an active `home_cognitive_monitoring` consent; companion is exempt.
- **Pipeline divergence**: both pass consent/identity/quality gates and get a feature snapshot; companion then **terminates** with `monitoringUse: 'observation_only'` (`pipeline.ts:112-120`) — no operational baseline, no research scoring, no monitoring window, no review case.

**Caregiver-visible consequence:** free talk never moves the status colour and never advances the baseline. A patient who only ever free-talks stays at "0 of 7 sessions recorded / Learning routine" forever — which is exactly why the mirror now auto-decides the first conversation of the day to be a check-in.

---

## 8. Which Qwen models are actually in use

### Realtime (WebSocket)

| Model | Used for | Region | Notes |
|---|---|---|---|
| **`qwen3.5-omni-flash-realtime`** | the production realtime link for **both** conversations | CN + SG | `semantic_vad` (echo-rejecting); no voiceprint. `QWEN_REALTIME_MODEL` |
| `qwen-audio-3.0-realtime-flash` | tier-3 fallback of the resilience ladder | **Beijing only** | `smart_turn` + voiceprint. `QWEN_CN_AUDIO_MODEL` |
| **`gummy-realtime-v1`** | speech-to-text **inside** the realtime session | all | `input_audio_transcription` (`realtime.ts:19`) |

Check-in uses the omni socket for **ASR only**; free talk also uses it to generate.

### HTTP models (issued per-region in the session ticket, `platform/qwen.ts:32-35`)

| Purpose | Model | Region difference |
|---|---|---|
| TTS | `qwen-tts` | **SG/JP serve `qwen3-tts-flash`** — `qwen-tts` 404s there |
| ASR | `qwen3-asr-flash` | same everywhere |
| Chat | `qwen-plus` | same everywhere |
| Vision | `qwen-vl-max` | same everywhere |

Overridable via `QWEN_{CN,SG,JP}_{TTS,ASR,CHAT,VISION}_MODEL`.

### Call sites

**Mirror device** (`src/api/qwenClient.ts` → `/compatible-mode/v1/chat/completions` and `/api/v1/services/aigc/multimodal-generation/generation`):
- `qwenTTS` → **TTS** — *every* check-in line + the companion closing line
- `qwenChat` → **`qwen-plus`** — post-conversation memory summarisation (`patientMemory.ts`, 300 tok / temp 0.2) and the after-the-fact screening analysis (`assess.ts`)
- `qwenVisionChat` → **`qwen-vl-max`** — vision observation. One call in `assess.ts:102` explicitly passes `qwen3-omni-flash`, but it sits behind `if (!__DEV__) return` — **never runs in production**
- `qwenAsr` → **`qwen3-asr-flash`** — transcription on the turn-based fallback path

**Backend** — exactly one direct Qwen call: `qwenChatCompletion` → **`qwen-plus`** for the caregiver daily summary (`routes/patient-summary.ts:88`, 400 tok / temp 0.2).

### Not Qwen / not live
- **Embeddings** go through a generic `openai_compatible` provider (`EMBEDDING_PROVIDER` / `_API_BASE` / `_MODEL`); with no provider configured the factory returns `null`, so research vectors are **off by default** (`v1/monitoring/embeddings.ts:8-18`).
- The relay's default `qwen3-omni-flash-realtime` (`server/qwenConfig.mjs:20`) is stale — since the keyless change the Electron build uses the ticket-issued `qwen3.5-omni-flash-realtime`.

**In production, four models actually run:** `qwen3.5-omni-flash-realtime`, `gummy-realtime-v1`, `qwen-tts`/`qwen3-tts-flash`, `qwen-plus`.

---

## 9. Gotchas

### Fixed
1. ~~**Companion prompt contradicted its own context**~~ — it hard-said *"You have NO live weather data"* while the memory block may carry today's weather and a `get_weather` tool is registered. Now: use the supplied weather or the tool, and say so plainly only when neither exists.
2. ~~**Half the stored memory was never used**~~ — the backend keeps 8 facts, the prompt injected 4. Now `MAX_INJECTED_MEMORY_FACTS = 8`, matching the store.
3. ~~**The relay would have broken every keyless session**~~ — it sent the qwen-tts voice (`Cherry`) while the keyless/ticket path connects to `qwen3.5-omni`, which rejects it (`<400> Voice 'Cherry' is not supported`). The relay now maps to the realtime voice (Serena / Kiki / Joseph Chen) whenever it is on the `semantic_vad` (ticket) path, and its stale default model moved to `qwen3.5-omni-flash-realtime`.
4. ~~`steer` / `focusDirective()` dead code~~ — removed (no callers, no tests).

### Open
5. **The screening prompt is dead weight on the production path** (kept for the fallback transports) — easy to mistake for the thing driving the check-in.
6. **`hindi`/`urdu`/`custom` have no localized check-in script** and silently fall back to English.
7. **Memory fetch is on the critical path** — awaited before the WS opens, adding a round-trip to every conversation start.
