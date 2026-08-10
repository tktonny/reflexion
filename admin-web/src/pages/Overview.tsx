import { useEffect, useState } from 'react'
import { api, type Overview as OverviewData } from '../api'

const TILES: { key: keyof OverviewData; label: string }[] = [
  { key: 'patients', label: 'Loved ones' },
  { key: 'users', label: 'Users' },
  { key: 'devices', label: 'Mirrors' },
  { key: 'openThreads', label: 'Open support' },
]

export function Overview() {
  const [data, setData] = useState<OverviewData | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api.overview().then(setData).catch((e) => setError(e.message)) }, [])
  return (
    <div className="stack">
      <h1 style={{ fontSize: 26 }}>Overview</h1>
      {error ? <p className="error">{error}</p> : null}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
        {TILES.map((tile) => (
          <div key={tile.key} className="card">
            <div className="muted" style={{ fontSize: 13 }}>{tile.label}</div>
            <div style={{ fontFamily: 'var(--font-heading)', fontSize: 34, marginTop: 8 }}>{data ? data[tile.key] : '—'}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
