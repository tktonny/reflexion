import React from 'react'
import { StyleSheet, Text, View } from 'react-native'

// On-device debug HUD for field testing when logs are unreachable (the test device isn't in hand). A
// module singleton collects the live pairing / ticket / connection / latency picture; the overlay
// subscribes and renders a small top-left box. Security: it shows the region + endpoint HOST only —
// NEVER the API key or the ticket token. OFF by default — build a test APK with
// EXPO_PUBLIC_DEBUG_OVERLAY=on to show it; a normal/production build never renders it and all the
// instrumentation compiles to no-ops.

const ENABLED = process.env.EXPO_PUBLIC_DEBUG_OVERLAY === 'on'

export type ConnState = 'idle' | 'ticket' | 'connecting' | 'open' | 'failed' | 'fallback'

export type DebugSnapshot = {
  seq: number
  pairing: { deviceId?: string; patientId?: string; region?: string; override?: string; online?: boolean }
  persona?: string
  ticket: { obtained?: boolean; region?: string; variant?: string; host?: string; model?: string; backendMs?: number }
  conn: { state: ConnState; tier?: string; openMs?: number; reason?: string }
  // Live mic picture for the barge-in / echo diagnosis: peak RMS seen recently vs the interruption
  // threshold, whether capture is muted, whether talk-over is armed (Aria owns the audio), and how
  // many times barge-in has actually fired this session.
  mic?: { peak?: number; thr?: number; muted?: boolean; armed?: boolean; barges?: number; err?: string }
  llmMedianMs?: number | null
  lines: string[]
}

let snap: DebugSnapshot = { seq: 0, pairing: {}, ticket: {}, conn: { state: 'idle' }, lines: [] }
const listeners = new Set<() => void>()
function emit() { snap = { ...snap, seq: snap.seq + 1 }; listeners.forEach((listener) => listener()) }
function stamp() {
  const d = new Date()
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
}

/** hostname of a ws/wss/https URL, without leaking any query (the ticket rides in the URL/headers). */
export function hostOf(url?: string): string | undefined {
  if (!url) return undefined
  try { return new URL(url).host } catch { return url.replace(/^[a-z]+:\/\//, '').split(/[/?]/)[0] }
}

export const dbg = {
  patch(partial: Partial<Omit<DebugSnapshot, 'seq' | 'lines'>>) {
    if (!ENABLED) return
    snap = {
      ...snap, ...partial,
      pairing: { ...snap.pairing, ...(partial.pairing || {}) },
      ticket: { ...snap.ticket, ...(partial.ticket || {}) },
      conn: { ...snap.conn, ...(partial.conn || {}) },
    }
    emit()
  },
  log(line: string) {
    if (!ENABLED) return
    snap = { ...snap, lines: [`${stamp()} ${line}`, ...snap.lines].slice(0, 12) }
    emit()
  },
  /** Live mic level / barge-in state. Callers throttle this (audio chunks arrive ~10x/s). */
  mic(partial: NonNullable<DebugSnapshot['mic']>) {
    if (!ENABLED) return
    snap = { ...snap, mic: { ...snap.mic, ...partial } }
    emit()
  },
  /** Reset the per-conversation fields (ticket/conn) at the start of a new session; keep pairing + log. */
  startSession(persona?: string) {
    if (!ENABLED) return
    snap = { ...snap, persona, ticket: {}, conn: { state: 'idle' }, mic: { barges: 0 } }
    emit()
  },
  get(): DebugSnapshot { return snap },
  subscribe(cb: () => void) { listeners.add(cb); return () => { listeners.delete(cb) } },
}

function connColor(state: ConnState) {
  if (state === 'open') return { color: '#3ddc84' }
  if (state === 'fallback' || state === 'failed') return { color: '#ff6b6b' }
  return { color: '#ffd166' }
}

export function DebugOverlay() {
  if (!ENABLED) return null
  const s = React.useSyncExternalStore(dbg.subscribe, dbg.get, dbg.get)
  const openS = s.conn.openMs != null ? ` ${(s.conn.openMs / 1000).toFixed(1)}s` : ''
  return (
    <View pointerEvents="none" style={styles.box}>
      <Text style={styles.head}>DEBUG</Text>
      <Text style={styles.text}>pair …{tail(s.pairing.deviceId)} · pat …{tail(s.pairing.patientId)}{s.pairing.online === false ? ' · OFFLINE' : ''}</Text>
      <Text style={styles.text}>region {s.pairing.region || '?'}{s.pairing.override ? ` (ovr ${s.pairing.override})` : ''} · {s.persona || '-'}</Text>
      <Text style={styles.text}>ticket {s.ticket.obtained ? 'OK' : '—'} {s.ticket.region || ''}{s.ticket.variant ? `/${s.ticket.variant}` : ''}{s.ticket.backendMs != null ? ` ${s.ticket.backendMs}ms` : ''}</Text>
      <Text style={styles.text} numberOfLines={1}>host {s.ticket.host || '—'}</Text>
      <Text style={styles.text} numberOfLines={1}>model {s.ticket.model || '—'}</Text>
      <Text style={[styles.text, connColor(s.conn.state)]}>conn {s.conn.state}{openS}{s.conn.tier ? ` [${s.conn.tier}]` : ''}</Text>
      {s.conn.reason ? <Text style={[styles.text, { color: '#ff9f43' }]} numberOfLines={2}>↳ {s.conn.reason}</Text> : null}
      {s.mic?.err ? <Text style={[styles.text, { color: '#ff6b6b' }]} numberOfLines={2}>err {s.mic.err}</Text> : null}
      <Text style={styles.text}>
        mic {micBar(s.mic?.peak)} {fmt(s.mic?.peak)}/{fmt(s.mic?.thr)} {s.mic?.muted ? 'MUTED' : 'open'} {s.mic?.armed ? 'armed' : 'idle'} b{s.mic?.barges ?? 0}
      </Text>
      <Text style={styles.text}>llm median {s.llmMedianMs != null ? `${Math.round(s.llmMedianMs)}ms` : '—'}</Text>
      <View style={styles.sep} />
      {s.lines.slice(0, 6).map((line, index) => <Text key={`${line}-${index}`} style={styles.log} numberOfLines={1}>{line}</Text>)}
    </View>
  )
}

function tail(value?: string) { return value ? value.slice(-6) : '——' }
function fmt(value?: number) { return value != null ? value.toFixed(3) : '—' }
// 8-step bar for the mic peak (0..~0.12 → the useful barge-in range) so a live level is readable at a glance.
function micBar(peak?: number) {
  if (peak == null) return '········'
  const n = Math.max(0, Math.min(8, Math.round((peak / 0.12) * 8)))
  return '█'.repeat(n).padEnd(8, '·').slice(0, 8)
}

const styles = StyleSheet.create({
  box: { position: 'absolute', top: 8, left: 8, maxWidth: 268, backgroundColor: 'rgba(0,0,0,0.74)', borderRadius: 8, padding: 8, zIndex: 9999 },
  head: { color: '#8ab4f8', fontSize: 10, fontWeight: '700', marginBottom: 2, letterSpacing: 1.5 },
  text: { color: '#eee', fontSize: 10, lineHeight: 14, fontFamily: 'monospace' },
  sep: { height: 1, backgroundColor: 'rgba(255,255,255,0.2)', marginVertical: 4 },
  log: { color: '#bbb', fontSize: 9, lineHeight: 12, fontFamily: 'monospace' },
})
