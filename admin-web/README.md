# Reflexion Admin Web

The Admin / Onboarding portal (product doc §1.1, third component). A **Vite + React + TypeScript SPA**
that is fully decoupled from the backend (前后端分离) and talks to the existing **reflexion-server**
`/api/v1/admin` API over HTTP — reusing the same MongoDB, JWT auth, and tenant model (no separate DB).

## MVP features
- **Operator login** — `POST /api/v1/auth/sessions`; requires an `operator`/`tenant_admin` role.
- **Users** — list tenant users.
- **Loved ones** — list / create / pause-activate patients (tenant-scoped onboarding).
- **Support** — read caregiver support conversations and reply; open/close threads.
  (More features to be added later.)

## Backend it depends on
Endpoints added in `reflexion-server/src/v1/routes/admin.ts` (mounted at `/api/v1`):
`GET /admin/overview`, `GET /admin/users`, `GET /admin/patients`, `POST /admin/patients`,
`PATCH /admin/patients/:id`, `GET /admin/support/threads`, `GET /admin/support/threads/:id`,
`POST /admin/support/threads/:id/messages`, `PATCH /admin/support/threads/:id`. Caregivers post
inquiries via `POST /support/threads` + `POST /support/threads/:id/messages` (any authenticated human).

## Run (dev)
```bash
cd admin-web
npm install
# point the dev proxy at your running reflexion-server (default http://localhost:3001):
VITE_DEV_API_TARGET=http://localhost:3001 npm run dev   # serves on http://localhost:5174
```
An operator account must exist in the `users` collection with role `operator` or `tenant_admin`.

## Build (prod)
```bash
# Set the API origin the SPA should call (the deployed reflexion-server):
VITE_API_BASE_URL=https://reflexion.production.tktonny.top npm run build   # -> dist/ (static)
```
Deploy `dist/` to any static host (Vercel/Netlify/Cloudflare Pages/Render static). It is a pure SPA;
ensure the host rewrites unknown routes to `index.html` (client-side routing).

## Notes
- Tokens are stored in `localStorage`; a 401 triggers a single-flight refresh via
  `/auth/session-refreshes`, then one retry.
- Design follows the doc's Option-1 palette (sage/bronze/terracotta), Georgia headings + Inter body.
