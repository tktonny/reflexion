# Mirror Realtime Turn-taking Contract

> Phase 0 baseline: 2026-07-22
> First production candidate: native Qwen realtime WebSocket + native PCM bridge
> Scope: the audio-turn lifecycle only; language quality, tools and longitudinal analysis are later phases

## 1. Product promise

The user must be able to finish speaking, and Aria must finish playing every reply. A provider event that says generation is complete is not evidence that the device speaker has finished. Question progression, recall steering, goodbye generation, microphone reopening and transport teardown are therefore allowed only after the local playback gate has completed.

WebRTC remains an experimental transport until it exposes equivalent playout-state evidence and passes this same contract on target hardware. It must not become the product default merely because it has built-in acoustic echo cancellation.

## 2. Canonical state model

```text
idle
  -> connecting
  -> configuring
  -> listening
  -> user_speaking
  -> thinking
  -> assistant_generating
  -> assistant_playing
  -> playback_guard
  -> listening | assistant_generating | closing
  -> ended
```

`error` can be entered from any active state. `closing` stays microphone-muted through goodbye generation and playback.

## 3. Non-negotiable invariants

1. At most one provider response may be requested or active at a time.
2. Capture is muted before an explicit response request and remains muted throughout generation, playback and the acoustic tail guard.
3. `response.audio.done` means Qwen finished generating audio; it does not mean the user heard it.
4. `response.done` means Qwen completed the response; it does not reopen capture, advance the screening agenda or end the session.
5. The next action is selected exactly once, after the native playback backlog reaches the drain threshold.
6. Recall steering and goodbye generation cannot overlap the preceding spoken response.
7. A closing response must be observed and its playback must drain before the transport is torn down.
8. Fixed delays are safety timeouts only. They are never the normal proof that speech finished.
9. A playback timeout must keep capture muted and surface an error; it must never be treated as a safe drain.
10. Companion and screening share the same audio lifecycle. Only their post-playback agenda policy differs.
11. Runtime fallback is not allowed to weaken the close contract. Its player must report native playback completion before cleanup, navigation or assessment begins.
12. The process may own only one native conversation runtime. Starting a new screen or fallback must terminate the prior owner and invalidate every outstanding ASR, chat, TTS and playback continuation before the new owner opens audio.
13. Push-to-talk is a physical hold contract: press-in starts one capture, press-out submits it once. A release during asynchronous recorder preparation must cancel that continuation, and an accidental short tap must never reach ASR.
14. Four screening domains must not collapse into four questions. Delayed recall may begin only after six accepted patient answers have covered self/place/time orientation, recent narrative continuity and daily function; the recall answer is the seventh patient turn and only then may the goodbye begin.
15. After assistant playback drains, the retained transcript must remain visible as the current question while the UI follows the authoritative listening state. The mirror must show question progress and a neutral answer cue without covering the question or removing manual End Chat.
16. Companion and screening outputs are data-isolated. A companion transcript may be retained for continuity but must not produce a cognitive risk assessment. Companion auto-close requires explicit user farewell intent; a polite assistant phrase such as “Have a good day” or “Take care” is never sufficient evidence to end the session.

## 4. Initial tuning baseline

| Parameter | Phase 0 value | Reason |
|---|---:|---|
| direct WS turn mode | manual (`turn_detection: null`) | Apply the correct agenda before an explicit `response.create` |
| local VAD speech start | RMS 0.015 for 200 ms | Reject short noise while preserving prompt turn detection |
| local VAD continuation | RMS 0.008 | Keep a started utterance alive through quieter speech |
| local VAD silence | 1200 ms | Allow older adults and deliberate speakers to pause without being cut off |
| local VAD max turn | 30 s | Bound an input turn without relying on provider auto-response |
| output token ceiling | 256 | Avoid truncating otherwise valid spoken replies mid-sentence |
| playback drain threshold | 40 ms | Keep the remaining device buffer very small before progressing |
| microphone acoustic-tail guard | 1100 ms | Prevent speaker tail/room echo from becoming the next user turn |
| playback stall timeout | 25 s | Fail closed instead of reopening capture over stuck audio |
| graceful-close safety timeout | 30 s | Bound a provider/network failure without clipping the normal goodbye path |

Only one sampling-diversity control is sent to Qwen. The baseline uses `temperature`; `top_p` is omitted.

The direct WebSocket path mutes capture, commits the audio buffer, waits for input transcription, applies the next session configuration and then explicitly sends `response.create`. Relay/WebRTC callers may retain provider semantic VAD, but they are not the Phase 1 product baseline. Recall and closing use deterministic language templates rendered by Qwen TTS because a mid-session realtime instruction was not reliable enough for these safety-critical agenda transitions.

## 5. Acceptance gates

### Deterministic replay

- Twenty lifecycle traces pass with zero unexpected invariant violations.
- Replays cover normal audio, text-only output, recall, closing, manual stop, duplicated provider events, premature microphone-open attempts and playback timeout.
- Policy tests prove the recall deadline, hard turn cap and companion behavior are evaluated only after playback.

### Target-device smoke test

- 30 consecutive alternating user/assistant turns on the actual mirror hardware.
- No assistant sentence is audibly clipped.
- No question begins before the previous answer or assistant audio completes.
- No assistant audio is transcribed as a new patient turn.
- Manual end during user speech, model generation and assistant playback each produces one complete goodbye and one teardown.
- Push-to-talk press/release produces one user turn; release-before-prepare and accidental taps produce none.

### Soak test

- 30-minute session and 100-turn scripted acoustic run.
- Zero overlapping provider responses.
- Zero microphone reopen events while response/playback ownership is active.
- No unrecovered stuck-muted state; a genuine playback stall ends fail-closed with telemetry.

## 6. Required telemetry

Each lifecycle event should eventually be uploaded with `sessionId`, monotonic timestamp, turn sequence, transport, state before/after, provider response ID when available, playback backlog, drain latency and violation/error code. Raw audio is not required for operational turn telemetry.

Phase 1 exposes the current lifecycle state locally. Server ingestion and dashboards are implemented with the backend observability phase rather than blocking the device-side correctness fix.

The [MuMu acceptance run](./phase1-mumu-acceptance.md) passed the logical/device-simulator gates. Target-mirror acoustic validation remains mandatory before claiming that physical echo and audible sentence completion have passed.
