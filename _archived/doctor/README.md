# Doctor Surface

Provider-facing management console for reviewing formal assessment results, recent patient activity, and screening workload across the program.

## Scope In This Repo

- `frontend_app/`
  Standalone doctor console frontend assets served by the shared backend at `/doctor`

## Current Data Source

The doctor console currently reads from the clinic backend surface APIs:

- `GET /api/doctor/dashboard`

Those APIs aggregate persisted assessments from `data/assessments/`.

## Run

Start the existing backend:

`uvicorn backend.src.app.main:app --reload`

Then open:

- `http://127.0.0.1:8000/doctor`
