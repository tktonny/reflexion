import { useEffect, useState } from 'react'
import { api, type AdminUser } from '../api'

export function Users() {
  const [users, setUsers] = useState<AdminUser[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    api.users().then((rows) => setUsers(rows ?? [])).catch((e) => setError(e.message)).finally(() => setLoading(false))
  }, [])
  return (
    <div className="stack">
      <h1 style={{ fontSize: 26 }}>Users</h1>
      {error ? <p className="error">{error}</p> : null}
      <div className="card" style={{ padding: 0 }}>
        <table>
          <thead>
            <tr><th>Name</th><th>Email</th><th>Roles</th><th>Status</th></tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={4} className="muted">Loading…</td></tr>
            ) : users.length === 0 ? (
              <tr><td colSpan={4} className="muted">No users yet.</td></tr>
            ) : users.map((user) => (
              <tr key={user.userId}>
                <td>{user.name || '—'}</td>
                <td>{user.email || '—'}</td>
                <td>{(user.roles || []).join(', ') || 'caregiver'}</td>
                <td><span className={`pill ${user.status === 'active' ? 'active' : 'closed'}`}>{user.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
