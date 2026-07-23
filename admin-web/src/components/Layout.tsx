import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { getActor, logout } from '../api'

const NAV = [
  { to: '/', label: 'Overview', end: true },
  { to: '/users', label: 'Users' },
  { to: '/patients', label: 'Loved ones' },
  { to: '/support', label: 'Support' },
]

export function Layout() {
  const navigate = useNavigate()
  const actor = getActor()
  const onLogout = async () => { await logout(); navigate('/login', { replace: true }) }
  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <aside style={{ width: 220, background: '#fff', borderRight: '1px solid var(--border)', padding: 20, display: 'flex', flexDirection: 'column' }}>
        <h2 style={{ fontSize: 20, marginBottom: 24 }}>Reflexion</h2>
        <nav className="stack" style={{ gap: 4 }}>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              style={({ isActive }) => ({
                padding: '10px 12px', borderRadius: 10, fontWeight: 600, fontSize: 14,
                color: isActive ? 'var(--primary)' : 'var(--text-secondary)',
                background: isActive ? 'var(--primary-light)' : 'transparent',
              })}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="spacer" />
        <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>{actor?.email || actor?.name || 'Signed in'}</div>
        <button className="btn-secondary btn" onClick={onLogout}>Sign out</button>
      </aside>
      <main style={{ flex: 1, padding: 28, maxWidth: 1100 }}>
        <Outlet />
      </main>
    </div>
  )
}
