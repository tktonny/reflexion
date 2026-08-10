// Background-logging debug module. All dbg.patch/log/mic/startSession calls always write to
// console.log so diagnostics are available via `adb logcat` without an on-screen overlay.
// The DebugOverlay component always returns null — kept as a stub for backward compatibility.

const TAG = '[rfx-dbg]'

export type ConnState = 'idle' | 'ticket' | 'connecting' | 'open' | 'failed' | 'fallback'

export type DebugSnapshot = {
  seq: number
  pairing: { deviceId?: string; patientId?: string; region?: string; override?: string; online?: boolean }
  persona?: string
  ticket: { obtained?: boolean; region?: string; variant?: string; host?: string; model?: string; backendMs?: number }
  conn: { state: ConnState; tier?: string; openMs?: number; reason?: string }
  mic?: { peak?: number; thr?: number; muted?: boolean; armed?: boolean; barges?: number; err?: string }
  llmMedianMs?: number | null
  lines: string[]
}

let snap: DebugSnapshot = { seq: 0, pairing: {}, ticket: {}, conn: { state: 'idle' }, lines: [] }

/** hostname of a ws/wss/https URL, without leaking any query (the ticket rides in the URL/headers). */
export function hostOf(url?: string): string | undefined {
  if (!url) return undefined
  try { return new URL(url).host } catch { return url.replace(/^[a-z]+:\/\//, '').split(/[/?]/)[0] }
}

export const dbg = {
  patch(partial: Partial<Omit<DebugSnapshot, 'seq' | 'lines'>>) {
    snap = {
      ...snap, ...partial,
      pairing: { ...snap.pairing, ...(partial.pairing || {}) },
      ticket: { ...snap.ticket, ...(partial.ticket || {}) },
      conn: { ...snap.conn, ...(partial.conn || {}) },
    }
    const p = partial.pairing
    const t = partial.ticket
    const c = partial.conn
    const parts: string[] = []
    if (p) parts.push(`pair region=${p.region || '?'} dev=${(p.deviceId || '?').slice(-6)} online=${p.online ?? '?'}`)
    if (t) parts.push(`ticket ok=${t.obtained} region=${t.region} host=${t.host} model=${t.model}`)
    if (c) parts.push(`conn state=${c.state} tier=${c.tier || '-'} reason=${c.reason || '-'}`)
    if (partial.llmMedianMs != null) parts.push(`llmMedian=${partial.llmMedianMs}ms`)
    if (partial.mic) parts.push(`mic peak=${partial.mic.peak} muted=${partial.mic.muted} barges=${partial.mic.barges}`)
    if (parts.length) console.log(`${TAG} patch ${parts.join(' | ')}`)
  },
  log(line: string) {
    console.log(`${TAG} ${line}`)
  },
  /** Live mic level / barge-in state. Callers throttle this (audio chunks arrive ~10x/s). */
  mic(partial: NonNullable<DebugSnapshot['mic']>) {
    // Mic events are high-frequency; log only errors or barge-in transitions to keep logcat readable.
    if (partial.err) console.warn(`${TAG} mic err: ${partial.err}`)
    else if (partial.barges && partial.barges > (snap.mic?.barges ?? 0)) console.log(`${TAG} mic barge-in #${partial.barges}`)
    snap = { ...snap, mic: { ...snap.mic, ...partial } }
  },
  /** Reset the per-conversation fields (ticket/conn) at the start of a new session; keep pairing + log. */
  startSession(persona?: string) {
    console.log(`${TAG} session start persona=${persona || '?'}`)
    snap = { ...snap, persona, ticket: {}, conn: { state: 'idle' }, mic: { barges: 0 } }
  },
}

/** Always returns null — debug info is now routed to console.log for adb logcat access. */
export function DebugOverlay() {
  return null
}
