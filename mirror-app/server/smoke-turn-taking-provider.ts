import { WebSocket } from 'ws'

import { buildLiveSessionUpdate, realtimeWsUrl } from '../src/orchestration/realtime'

const bearer = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
if (!bearer) throw new Error('Set QWEN_API_KEY or DASHSCOPE_API_KEY before running this smoke test.')

const events: string[] = []
let audioDeltaCount = 0
let requested = false
let settled = false

const socket = new WebSocket(realtimeWsUrl(), { headers: { Authorization: `Bearer ${bearer}` }, maxPayload: 0 })
const timeout = setTimeout(() => finish(new Error('Timed out waiting for response.done.')), 30_000)

socket.on('open', () => {
  socket.send(JSON.stringify(buildLiveSessionUpdate('turn-taking-smoke', 'English', {
    voice: 'Cherry',
    languageKey: 'english',
    persona: 'companion',
  })))
})

socket.on('message', (data) => {
  let payload: Record<string, unknown>
  try { payload = JSON.parse(data.toString()) as Record<string, unknown> } catch { return }
  const type = String(payload.type || '')
  events.push(type)

  if ((type === 'session.created' || type === 'session.updated') && !requested) {
    requested = true
    socket.send(JSON.stringify({ type: 'response.create' }))
  }
  if (type === 'response.audio.delta') audioDeltaCount += 1
  if (type === 'error') {
    const providerError = payload.error as { message?: string } | undefined
    finish(new Error(providerError?.message || 'Provider returned an error event.'))
  }
  if (type === 'response.done') {
    const createdAt = events.indexOf('response.created')
    const audioAt = events.indexOf('response.audio.delta')
    const audioDoneAt = events.indexOf('response.audio.done')
    const responseDoneAt = events.indexOf('response.done')
    if (createdAt < 0 || audioAt < createdAt || audioDoneAt < audioAt || responseDoneAt < audioDoneAt) {
      finish(new Error(`Unexpected provider lifecycle: ${[...new Set(events)].join(' -> ')}`))
      return
    }
    if (audioDeltaCount === 0) {
      finish(new Error('Provider completed without audio deltas.'))
      return
    }
    finish()
  }
})

socket.on('error', (error) => finish(error))
socket.on('close', () => {
  if (!settled) finish(new Error('Provider socket closed before response.done.'))
})

function finish(error?: Error): void {
  if (settled) return
  settled = true
  clearTimeout(timeout)
  try { socket.close() } catch {}
  if (error) {
    console.error(`FAIL: ${error.message}`)
    process.exitCode = 1
    return
  }
  console.log(`PASS: Qwen lifecycle completed with ${audioDeltaCount} audio deltas.`)
  console.log(`Event order: ${[...new Set(events)].join(' -> ')}`)
}
