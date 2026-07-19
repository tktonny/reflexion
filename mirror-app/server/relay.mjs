// Qwen Omni Realtime relay — ported from REFLEXION realtime_service.py
// (_run_live_qwen + _relay_live_qwen_session + _build_live_session_update + _prepare_live_client_event).
// The browser/client never talks to DashScope directly; this relay holds the key,
// runs the bidirectional pump, orchestration session.update, dynamic voice switching,
// wrap-up, and the China-backup handshake retry.

import { WebSocket } from 'ws'
import { randomUUID } from 'node:crypto'

import { qwenConfig } from './qwenConfig.mjs'
import { buildLiveInstructions, closingGoodbyeSentence, flowId, promptStepCount } from './orchestrator.mjs'
import {
  detectLanguageSignal,
  languageInputValue,
  shouldRestartResponseForLanguageSwitch,
  voiceProfileForSession,
  voiceProfileFromRecentSignals,
} from './voice.mjs'

const SESSION_READY_EVENTS = new Set(['session.updated', 'session.created'])
const CLIENT_EVENT_WHITELIST = new Set([
  'input_audio_buffer.append',
  'input_audio_buffer.commit',
  'input_audio_buffer.clear',
  'input_image_buffer.append',
  'response.create',
  'response.cancel',
  'reflexion.language_hint',
  'reflexion.wrap_up',
])

function log(...args) {
  console.log('[relay]', ...args)
}

function eventId() {
  return `event_${randomUUID().replace(/-/g, '').slice(0, 12)}`
}

function buildLiveSessionUpdate(patientId, language, { voice, wrapUp = false } = {}) {
  let instructions = buildLiveInstructions(patientId, language, {})
  if (wrapUp) {
    const goodbye = closingGoodbyeSentence(language)
    instructions +=
      '\nThe live capture is ending now. In your next reply, briefly thank the patient, ' +
      'say the conversation is ending, and end with exactly this goodbye sentence: ' +
      `"${goodbye}" The goodbye must be the final sentence. Do not ask another question after that goodbye.`
  }
  return {
    event_id: eventId(),
    type: 'session.update',
    session: {
      modalities: ['text', 'audio'],
      voice: voice || voiceProfileForSession(language).voice,
      instructions,
      max_tokens: qwenConfig.maxTokens,
      temperature: qwenConfig.temperature,
      top_p: qwenConfig.topP,
      input_audio_format: 'pcm',
      output_audio_format: 'pcm',
      turn_detection: {
        type: 'server_vad',
        threshold: qwenConfig.vadThreshold,
        prefix_padding_ms: qwenConfig.vadPrefixPaddingMs,
        silence_duration_ms: qwenConfig.vadSilenceDurationMs,
        create_response: true,
        interrupt_response: false,
      },
      input_audio_transcription: { model: qwenConfig.transcriptionModel },
    },
  }
}

// Returns { event, audioAppendStarted } | { event: null, audioAppendStarted }
function prepareLiveClientEvent(event, audioAppendStarted) {
  const type = String(event?.type || '')
  if (type === 'reflexion.close') return { event, audioAppendStarted }
  if (!CLIENT_EVENT_WHITELIST.has(type)) return { event: null, audioAppendStarted }
  if (type === 'input_audio_buffer.append') return { event, audioAppendStarted: true }
  if (type === 'input_image_buffer.append' && !audioAppendStarted) return { event: null, audioAppendStarted }
  return { event, audioAppendStarted }
}

function realtimeUpstreamUrls() {
  const urls = []
  for (const c of [qwenConfig.realtimeUrl, qwenConfig.realtimeUrlChina]) {
    const s = String(c || '').trim()
    if (s && !urls.includes(s)) urls.push(s)
  }
  return urls
}

// Open one upstream ws to DashScope; resolves to a connected socket, else rejects.
// `statusCode` is attached to the error on handshake failure (for China-backup gating).
function connectUpstream(url) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url, {
      headers: { Authorization: `Bearer ${qwenConfig.apiKey}` },
      maxPayload: 0,
    })
    let settled = false
    socket.once('open', () => {
      settled = true
      resolve(socket)
    })
    socket.once('unexpected-response', (_req, res) => {
      if (settled) return
      settled = true
      const err = new Error(`Qwen realtime handshake HTTP ${res.statusCode}`)
      err.statusCode = res.statusCode
      try { socket.terminate() } catch {}
      reject(err)
    })
    socket.once('error', (err) => {
      if (settled) return
      settled = true
      reject(err)
    })
  })
}

export async function runLiveQwen(clientWs, { patientId, language }) {
  if (!qwenConfig.apiKey) {
    throw new Error('missing_qwen_api_key (set QWEN_API_KEY or DASHSCOPE_API_KEY)')
  }
  const urls = realtimeUpstreamUrls()
  const selectedVoiceProfile = voiceProfileForSession(language)
  let lastError = null

  for (let i = 0; i < urls.length; i += 1) {
    const url = `${urls[i]}?model=${qwenConfig.realtimeModel}`
    log(`opening upstream patient_id=${patientId} language=${language} url=${urls[i]} model=${qwenConfig.realtimeModel} attempt=${i + 1}/${urls.length}`)
    try {
      const upstream = await connectUpstream(url)
      log(`upstream connected url=${urls[i]}`)
      await relaySession(clientWs, upstream, { patientId, language, selectedVoiceProfile })
      return
    } catch (err) {
      lastError = err
      const status = err?.statusCode
      if (i + 1 < urls.length && (status === 401 || status === 403)) {
        log(`upstream handshake failed status=${status}, retrying on China backup ${urls[i + 1]}`)
        continue
      }
      throw err
    }
  }
  if (lastError) throw lastError
}

function relaySession(clientWs, upstream, { patientId, language, selectedVoiceProfile }) {
  return new Promise((resolve, reject) => {
    let selected = selectedVoiceProfile
    let deferredVoiceProfile = null
    let recentLanguageSignals = []
    let sessionReady = false
    let assistantResponseActive = false
    let assistantResponseDoneCount = 0
    let transcriptTurnCount = 0
    let pendingFirstResponseRestart = false
    let openingResponseRequested = false
    let audioAppendStarted = false
    let finished = false

    const sendUpstream = (event) => {
      if (!event.event_id) event.event_id = eventId()
      if (upstream.readyState === WebSocket.OPEN) upstream.send(JSON.stringify(event))
    }
    const sendClient = (payload) => {
      if (clientWs.readyState === WebSocket.OPEN) clientWs.send(JSON.stringify(payload))
    }
    const sendSessionUpdate = (profile, reason) => {
      log(`session.update flow_id=${flowId} steps=${promptStepCount} voice=${profile.voice} language=${profile.languageLabel} reason=${reason}`)
      sendUpstream(buildLiveSessionUpdate(patientId, profile.languageLabel, { voice: profile.voice }))
    }
    const applyVoiceProfile = (profile, reason) => {
      sendSessionUpdate(profile, reason)
      selected = profile
      sendClient({
        type: 'reflexion.voice.selected',
        voice: selected.voice,
        language: selected.languageLabel,
        language_key: selected.languageKey,
        language_input: languageInputValue(selected.languageKey, selected.languageLabel),
        source: selected.source,
      })
    }
    const sendWrapUp = () => {
      log(`wrap-up patient_id=${patientId} voice=${selected.voice} language=${selected.languageLabel}`)
      sendUpstream(buildLiveSessionUpdate(patientId, selected.languageLabel, { voice: selected.voice, wrapUp: true }))
      sendUpstream({ type: 'response.create' })
    }

    const cleanup = (err) => {
      if (finished) return
      finished = true
      clearInterval(pingTimer)
      try { upstream.close() } catch {}
      if (err) reject(err)
      else resolve()
    }

    // Keep the upstream alive (Python used ping_interval=20).
    const pingTimer = setInterval(() => {
      if (upstream.readyState === WebSocket.OPEN) { try { upstream.ping() } catch {} }
    }, 20000)

    // Kick off: initial session.update.
    sendSessionUpdate(selected, selected.source)

    upstream.on('message', (data) => {
      let payload
      try { payload = JSON.parse(data.toString()) } catch { return }
      const type = String(payload.type || '')

      if (type === 'conversation.item.input_audio_transcription.completed') {
        transcriptTurnCount += 1
        const signal = detectLanguageSignal(String(payload.transcript || ''))
        if (signal) { recentLanguageSignals.push(signal); recentLanguageSignals = recentLanguageSignals.slice(-3) }
        const current = selected
        const detected = voiceProfileFromRecentSignals({ languageHint: language, recentSignals: recentLanguageSignals, currentProfile: current })
        if (detected && (detected.voice !== current.voice || detected.languageLabel !== current.languageLabel)) {
          const restart = shouldRestartResponseForLanguageSwitch({
            transcriptTurnIndex: transcriptTurnCount,
            currentProfile: current,
            detectedProfile: detected,
            assistantResponseDoneCount,
          })
          log(`voice reassessment ${current.voice}->${detected.voice} ${current.languageLabel}->${detected.languageLabel}`)
          if (sessionReady) { try { applyVoiceProfile(detected, 'transcript_reassessment') } catch (e) { log('voice update failed', e?.message) } }
          else deferredVoiceProfile = detected
          if (restart) {
            if (assistantResponseActive && sessionReady) {
              pendingFirstResponseRestart = false
              assistantResponseActive = false
              sendUpstream({ type: 'response.cancel' })
              sendUpstream({ type: 'response.create' })
            } else {
              pendingFirstResponseRestart = true
            }
          }
        }
      }

      if (SESSION_READY_EVENTS.has(type)) {
        if (!sessionReady) sessionReady = true
        if (deferredVoiceProfile) {
          try { applyVoiceProfile(deferredVoiceProfile, 'transcript_reassessment') } catch (e) { log('deferred voice failed', e?.message) }
          deferredVoiceProfile = null
        } else if (!openingResponseRequested) {
          openingResponseRequested = true
          log('requesting opening response.create')
          sendUpstream({ type: 'response.create' })
        }
      }

      if (type === 'response.created') {
        assistantResponseActive = true
        if (pendingFirstResponseRestart && assistantResponseDoneCount === 0) {
          pendingFirstResponseRestart = false
          assistantResponseActive = false
          sendUpstream({ type: 'response.cancel' })
          sendUpstream({ type: 'response.create' })
        }
      } else if (type === 'response.done') {
        assistantResponseActive = false
        assistantResponseDoneCount += 1
      }

      sendClient(payload)
    })

    clientWs.on('message', (data) => {
      let event
      try { event = JSON.parse(data.toString()) } catch { return }
      const prepared = prepareLiveClientEvent(event, audioAppendStarted)
      audioAppendStarted = prepared.audioAppendStarted
      if (!prepared.event) return
      event = prepared.event
      const type = String(event.type || '')

      if (type === 'reflexion.language_hint') {
        const signal = detectLanguageSignal(String(event.text || ''))
        if (!signal) return
        recentLanguageSignals.push(signal); recentLanguageSignals = recentLanguageSignals.slice(-3)
        const detected = voiceProfileFromRecentSignals({ languageHint: language, recentSignals: recentLanguageSignals, currentProfile: selected })
        if (!detected || (detected.voice === selected.voice && detected.languageLabel === selected.languageLabel)) return
        if (sessionReady) { try { applyVoiceProfile(detected, 'browser_hint') } catch (e) { log('hint voice failed', e?.message) } }
        else deferredVoiceProfile = detected
        return
      }
      if (type === 'reflexion.wrap_up') { sendWrapUp(); return }
      if (type === 'reflexion.close') { log('client requested upstream close'); cleanup(); return }
      if (!event.event_id) event.event_id = eventId()
      sendUpstream(event)
    })

    upstream.on('close', () => { log('upstream closed'); cleanup() })
    upstream.on('error', (err) => { log('upstream error', err?.message); cleanup() })
    clientWs.on('close', () => { log('client closed'); cleanup() })
    clientWs.on('error', (err) => { log('client error', err?.message); cleanup() })
  })
}
