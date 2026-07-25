# Caregiver push notifications, completion alerts, Qwen daily summary & feedback

Branch `feat/caregiver-push-and-qwen-summary`. Closes the caregiver gaps vs the original
`caregiver-app-server`. All AI runs on **Qwen** (never OpenAI).

## What changed (backend `reflexion-server`)

| Capability | How it works now |
|---|---|
| **Phone push delivery** | `src/v1/notifications/push.ts` `dispatchPendingPushes` reads each notification's recipient `notification_devices` Expo tokens and POSTs to `https://exp.host/--/api/v2/push/send`. Mongo-backed pending→sent, idempotent, retryable; deactivates `DeviceNotRegistered` tokens. **Rides the existing `reflexion-worker` loop — no new process.** |
| **"Checked in today" + late follow-up** | `materializeSessionCompletionNotifications` fires `completion` on a real daily check-in (and `late_completion` if a `missed_7pm` already went out), from the outbox worker's `session.completed` handler. `completion` doubles as the "daily summary ready" signal. |
| **Missed / streak / technical alerts** | `evaluate7pm` (in `jobs/finalizeDay.ts`) now sets `push:true`. **Requires the finalizeDay process to be running (see below).** |
| **AI daily summary (Qwen + cache)** | `POST /patient-summary` uses `qwenChatCompletion` (compatible-mode chat, region-resolved, key server-side) and caches per `(patient, day)` in `daily_summaries` (invalidated when that day's transcript grows; `?refresh`/`{refresh:true}` forces). The caregiver app's day view already calls this. |
| **Feedback** | legacy `POST /feedback` `{nurseId, message}` → `feedback` collection. |

New collections: `daily_summaries`, `feedback`. New indexes: `notifications.pushState`, `daily_summaries {patientId,dateKey}` unique, `feedback {nurseId,createdAt}`.

## Environment

No **new required** env — Qwen chat reuses the existing region keys (`QWEN_CN_API_KEY` / `QWEN_CN_HTTP_BASE`, already set in prod). Optional override: `QWEN_CN_CHAT_MODEL` (default `qwen-plus`). Expo push needs no key for basic sends.

## Deploy

```bash
# on the server, in reflexion-server, after pulling this branch's build
npm ci && npm run typecheck && npm run build
npm run db:indexes          # creates daily_summaries / feedback / notifications.pushState indexes

# 1) push delivery + completion pushes + AI summary + feedback → just restart the worker + api
pm2 restart reflexion-worker reflexion-api --update-env

# 2) the 7pm missed / 3-day-streak / mirror-offline alerts need the finalize job running as its OWN
#    long-lived process (it is NOT one of the two core processes). Start it once:
RUN_FINALIZE_JOB=1 pm2 start dist/jobs/finalizeDay.js --name reflexion-finalize --update-env
pm2 save
```

`reflexion-finalize` runs a 60s idempotent supervisor loop; `evaluate7pm` and `finalizeDay` gate on each patient's local time internally, so running it every minute is safe.

## Verify

```bash
# a completed daily check-in should now create a pushed completion notification:
#   db.notifications.find({ type: 'completion' }).sort({createdAt:-1}).limit(1)
#   → pushState transitions 'pending' → 'sent' (or 'skipped' if the caregiver has no active device)
# AI summary is cached after first view:
#   db.daily_summaries.find().sort({createdAt:-1}).limit(1)   // { summary, model, version, dateKey }
# feedback:
#   curl -X POST $ORIGIN/feedback -H 'content-type: application/json' -d '{"nurseId":"<hex>","message":"hi"}'
```

A caregiver receives a phone push only if their app registered an Expo token (`POST /api/v1/notification-devices`, which the app does on sign-in) **and** they hold `monitoring:read` on the patient.
