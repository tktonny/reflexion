import { useEffect, useRef, useState } from 'react'
import { api, type Message, type Thread } from '../api'

export function Support() {
  const [threads, setThreads] = useState<Thread[]>([])
  const [filter, setFilter] = useState<'open' | 'closed' | 'all'>('open')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [error, setError] = useState('')

  const loadThreads = () => {
    api.threads(filter === 'all' ? undefined : filter).then(setThreads).catch((e) => setError(e.message))
  }
  useEffect(loadThreads, [filter])

  return (
    <div className="stack">
      <div className="row">
        <h1 style={{ fontSize: 26 }}>Support</h1>
        <div className="spacer" />
        <select className="select" style={{ width: 160 }} value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
          <option value="open">Open</option>
          <option value="closed">Closed</option>
          <option value="all">All</option>
        </select>
      </div>
      {error ? <p className="error">{error}</p> : null}
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 12, alignItems: 'start' }}>
        <div className="card" style={{ padding: 0, maxHeight: '72vh', overflowY: 'auto' }}>
          {threads.length === 0 ? <div className="muted" style={{ padding: 16 }}>No conversations.</div> : threads.map((thread) => (
            <button
              key={thread.threadId}
              onClick={() => setSelectedId(thread.threadId)}
              style={{
                display: 'block', width: '100%', textAlign: 'left', border: 'none', cursor: 'pointer',
                padding: 14, borderBottom: '1px solid var(--border)',
                background: selectedId === thread.threadId ? 'var(--primary-light)' : '#fff',
              }}
            >
              <div className="row" style={{ gap: 8 }}>
                <strong style={{ fontSize: 14 }}>{thread.subject}</strong>
                {thread.adminUnread ? <span style={{ width: 8, height: 8, borderRadius: 999, background: 'var(--worth-checking)' }} /> : null}
              </div>
              <div className="muted" style={{ fontSize: 12, marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {thread.lastMessagePreview}
              </div>
            </button>
          ))}
        </div>
        <Conversation threadId={selectedId} onChanged={loadThreads} />
      </div>
    </div>
  )
}

function Conversation({ threadId, onChanged }: { threadId: string | null; onChanged: () => void }) {
  const [thread, setThread] = useState<(Thread & { messages: Message[] }) | null>(null)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  const load = () => { if (threadId) api.thread(threadId).then(setThread).catch((e) => setError(e.message)) }
  useEffect(() => { setThread(null); setError(''); load() }, [threadId])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [thread?.messages.length])

  if (!threadId) return <div className="card muted" style={{ minHeight: 200 }}>Select a conversation to reply.</div>
  if (!thread) return <div className="card muted" style={{ minHeight: 200 }}>{error || 'Loading…'}</div>

  const send = async () => {
    const body = draft.trim()
    if (!body) return
    setBusy(true); setError('')
    try {
      await api.reply(thread.threadId, body)
      setDraft('')
      load(); onChanged()
    } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  const toggleStatus = async () => {
    const next = thread.status === 'open' ? 'closed' : 'open'
    try { await api.setThreadStatus(thread.threadId, next); load(); onChanged() } catch (e) { setError((e as Error).message) }
  }

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', minHeight: 200 }}>
      <div className="row">
        <div>
          <h2 style={{ fontSize: 18 }}>{thread.subject}</h2>
          <div className="muted" style={{ fontSize: 12 }}>from {thread.caregiverName || thread.caregiverUserId}</div>
        </div>
        <div className="spacer" />
        <span className={`pill ${thread.status}`}>{thread.status}</span>
        <button className="btn-secondary btn" style={{ height: 32 }} onClick={toggleStatus}>{thread.status === 'open' ? 'Close' : 'Reopen'}</button>
      </div>
      <div className="stack" style={{ margin: '16px 0', maxHeight: '46vh', overflowY: 'auto' }}>
        {thread.messages.map((message) => {
          const mine = message.authorType === 'admin'
          return (
            <div key={message.messageId} style={{ alignSelf: mine ? 'flex-end' : 'flex-start', maxWidth: '78%' }}>
              <div style={{
                background: mine ? 'var(--primary)' : 'var(--primary-light)', color: mine ? '#fff' : 'var(--text)',
                borderRadius: 12, padding: '10px 12px', fontSize: 14, whiteSpace: 'pre-wrap',
              }}>{message.body}</div>
              <div className="muted" style={{ fontSize: 11, marginTop: 2, textAlign: mine ? 'right' : 'left' }}>
                {mine ? 'You' : 'Caregiver'} · {new Date(message.createdAt).toLocaleString()}
              </div>
            </div>
          )
        })}
        <div ref={bottomRef} />
      </div>
      {error ? <p className="error">{error}</p> : null}
      <div className="row" style={{ gap: 8, alignItems: 'flex-end' }}>
        <textarea placeholder="Write a reply…" value={draft} onChange={(e) => setDraft(e.target.value)} />
        <button className="btn" onClick={send} disabled={busy || !draft.trim()}>{busy ? 'Sending…' : 'Send'}</button>
      </div>
    </div>
  )
}
