import { StrictMode, type ReactElement } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, Navigate, RouterProvider } from 'react-router-dom'
import './theme.css'
import { isAuthed } from './api'
import { Layout } from './components/Layout'
import { Login } from './pages/Login'
import { Overview } from './pages/Overview'
import { Users } from './pages/Users'
import { Patients } from './pages/Patients'
import { Support } from './pages/Support'

function Protected({ children }: { children: ReactElement }) {
  return isAuthed() ? children : <Navigate to="/login" replace />
}

const router = createBrowserRouter([
  { path: '/login', element: <Login /> },
  {
    path: '/',
    element: (
      <Protected>
        <Layout />
      </Protected>
    ),
    children: [
      { index: true, element: <Overview /> },
      { path: 'users', element: <Users /> },
      { path: 'patients', element: <Patients /> },
      { path: 'support', element: <Support /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
