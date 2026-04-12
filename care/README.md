# Care Surface

Caregiver and patient-facing dashboard for receiving simplified follow-up updates from the formal assessment pipeline.

## Scope In This Repo

- `frontend_app/`
  Standalone care dashboard frontend assets served by the shared backend at `/care`

## Current Data Source

The care dashboard currently reads from the clinic backend surface APIs:

- `GET /api/care/patients`
- `GET /api/care/dashboard/{patient_id}`

These endpoints intentionally expose a simplified summary instead of the full provider console output.

## Run

Start the existing backend:

`uvicorn backend.src.app.main:app --reload`

Then open:

- `http://127.0.0.1:8000/care`
