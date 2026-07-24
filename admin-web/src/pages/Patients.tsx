import { useEffect, useState } from 'react'
import { api, ApiError, type Patient } from '../api'

const LANGUAGES = ['en', 'zh-CN', 'zh-HK', 'ms-MY', 'ta-IN']

export function Patients() {
  const [patients, setPatients] = useState<Patient[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [showCreate, setShowCreate] = useState(false)

  const load = () => {
    setLoading(true)
    api.patients().then((rows) => setPatients(rows ?? [])).catch((e) => setError(e.message)).finally(() => setLoading(false))
  }
  useEffect(load, [])

  const toggleStatus = async (patient: Patient) => {
    const next = patient.status === 'active' ? 'paused' : 'active'
    try { await api.updatePatient(patient.patientId, { status: next }); load() } catch (e) { setError((e as Error).message) }
  }

  return (
    <div className="stack">
      <div className="row">
        <h1 style={{ fontSize: 26 }}>Loved ones</h1>
        <div className="spacer" />
        <button className="btn" onClick={() => setShowCreate(true)}>Add loved one</button>
      </div>
      {error ? <p className="error">{error}</p> : null}
      <div className="card" style={{ padding: 0 }}>
        <table>
          <thead>
            <tr><th>Name</th><th>Language</th><th>Timezone</th><th>Status</th><th /></tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={5} className="muted">Loading…</td></tr>
            ) : patients.length === 0 ? (
              <tr><td colSpan={5} className="muted">No loved ones yet.</td></tr>
            ) : patients.map((patient) => (
              <tr key={patient.patientId}>
                <td>{patient.displayName}</td>
                <td>{patient.preferredLanguage}</td>
                <td className="muted">{patient.timezone}</td>
                <td><span className={`pill ${patient.status === 'active' ? 'active' : 'closed'}`}>{patient.status}</span></td>
                <td style={{ textAlign: 'right' }}>
                  <button className="btn-secondary btn" style={{ height: 32 }} onClick={() => toggleStatus(patient)}>
                    {patient.status === 'active' ? 'Pause' : 'Activate'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {showCreate ? <CreatePatient onClose={() => setShowCreate(false)} onCreated={() => { setShowCreate(false); load() }} /> : null}
    </div>
  )
}

function CreatePatient({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [displayName, setDisplayName] = useState('')
  const [preferredLanguage, setPreferredLanguage] = useState('en')
  const [timezone, setTimezone] = useState(Intl.DateTimeFormat().resolvedOptions().timeZone || 'Asia/Singapore')
  const [caregiverUserId, setCaregiverUserId] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setBusy(true); setError('')
    try {
      await api.createPatient({ displayName: displayName.trim(), preferredLanguage, timezone: timezone.trim(), caregiverUserId: caregiverUserId.trim() || undefined })
      onCreated()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not create.')
    } finally { setBusy(false) }
  }

  return (
    <div className="overlay" onClick={onClose}>
      <form className="modal" onClick={(e) => e.stopPropagation()} onSubmit={submit}>
        <h2 style={{ fontSize: 20 }}>Add a loved one</h2>
        <label>Name (as they like to be called)</label>
        <input className="input" value={displayName} onChange={(e) => setDisplayName(e.target.value)} required />
        <label>Preferred language</label>
        <select className="select" value={preferredLanguage} onChange={(e) => setPreferredLanguage(e.target.value)}>
          {LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}
        </select>
        <label>Timezone (IANA)</label>
        <input className="input" value={timezone} onChange={(e) => setTimezone(e.target.value)} required />
        <label>Link to caregiver user id (optional)</label>
        <input className="input" value={caregiverUserId} onChange={(e) => setCaregiverUserId(e.target.value)} placeholder="usr_…" />
        {error ? <p className="error" style={{ marginTop: 12 }}>{error}</p> : null}
        <div className="row" style={{ marginTop: 20, justifyContent: 'flex-end' }}>
          <button type="button" className="btn-secondary btn" onClick={onClose}>Cancel</button>
          <button type="submit" className="btn" disabled={busy}>{busy ? 'Saving…' : 'Create'}</button>
        </div>
      </form>
    </div>
  )
}
