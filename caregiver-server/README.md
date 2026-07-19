# Reflexion Caregiver App Server

Express and MongoDB backend for the Reflexion caregiver mobile app. It stores caregiver onboarding data, elderly patient profiles, mirror pairing state, and conversation metrics used by the dashboard, session history, trend, and summary screens.

## What This Service Does

- Creates caregiver accounts and elderly patient profiles during onboarding.
- Authenticates caregivers with email/password against MongoDB-stored password hashes.
- Stores notification preferences for each caregiver.
- Links and unlinks physical Reflexion mirror devices to patient profiles through short-lived pairing sessions.
- Serves dashboard data such as patient status, last spoken time, and mirror state.
- Reads conversation logs and conversation-to-patient mappings to power session history, monthly session counts, trend charts, and AI-generated daily summaries.

This server does not currently issue JWTs or session tokens. The mobile app stores the returned `nurseId`, `name`, and `email` locally and sends `nurseId` to endpoints that need caregiver context.

## Tech Stack

- Node.js with TypeScript
- Express 4
- MongoDB native driver
- dotenv for local environment loading
- OpenAI Chat Completions API for caregiver summaries

## Project Structure

```text
src/
  app.ts                                  Express app, middleware, health routes, error handlers
  index.ts                                Server entry point
  lib/
    constants.ts                          Database, collection, and timezone constants
    conversations.ts                      Helpers for patient lookup and conversation serialization
    dates.ts                              Asia/Singapore date range helpers
    env.ts                                Environment variable accessors
    mongo.ts                              MongoDB connection helper
    objectId.ts                           ObjectId normalization helpers
    password.ts                           PBKDF2 password hashing and verification
    patients.ts                           Patient validation, mirror pairing, mirror assignment helpers
    validation.ts                         Allowed enum values used by onboarding/settings
  routes/
    auth/sign-in.ts                       Caregiver sign-in
    nurse-patient-config/*.ts             Caregiver profile, notifications, patient, mirror APIs
    conversation-*.ts                     Session history and count APIs
    patient-summary.ts                    OpenAI-powered transcript summary API
    patient-trend.ts                      7-day and 30-day trend API
    router.ts                             Route mounting
```

## Setup

Install dependencies:

```bash
npm install
```

Create `.env`:

```bash
MONGODB_URI=mongodb+srv://...
OPENAI_API_KEY=sk-...
PORT=3001
HOST=0.0.0.0
```

Run locally:

```bash
npm run dev
```

Build and run production output:

```bash
npm run build
npm start
```

Typecheck only:

```bash
npm run typecheck
```

## Environment Variables

| Variable | Required | Default | Used by | Purpose |
| --- | --- | --- | --- | --- |
| `MONGODB_URI` | Yes | None | all database-backed routes | MongoDB connection string. |
| `OPENAI_API_KEY` | Yes, only for `POST /patient-summary` | None | `patient-summary.ts` | Calls OpenAI Chat Completions API. |
| `PORT` | No | `3001` | `src/index.ts` | HTTP server port. |
| `HOST` | No | `0.0.0.0` | `src/index.ts` | HTTP bind host. |

The service uses database `ref`.

## Database Collections And Schema

MongoDB schemas are not enforced by the server with a schema validator; the shapes below are the application-level documents the current code reads and writes.

### `NursePatientConfig`

Stores one caregiver account plus embedded patient profiles and caregiver notification settings.

Created by `POST /nurse-patient-config/create`; read and updated by sign-in, latest config, notification, patient append, and mirror APIs.

```ts
type NursePatientConfig = {
  _id: ObjectId
  name: string
  email: string
  passwordHash: string
  phoneNumber: string
  relationshipToElderly:
    | 'parent'
    | 'sibling'
    | 'spouse'
    | 'inlaw'
    | 'grandpa'
    | 'grandma'
    | 'other'
  pushNotificationsEnabled: boolean
  alertSensitivity:
    | 'notify_me_about_everything'
    | 'only_important_changes'
    | 'only_urgent_alerts'
  preferredDailySummaryTime: '09:00' | '19:00'
  patients: Patient[]
  createdAt: Date
  updatedAt: Date
}
```

`passwordHash` format:

```text
pbkdf2_sha256$120000$<salt hex>$<hash hex>
```

Embedded patient profile:

```ts
type Patient = {
  _id: ObjectId
  name: string
  phoneNumber?: string
  age: number
  gender: 'male' | 'female' | 'other'
  preferredLanguage: 'english' | 'mandarin' | 'other'
  usualWakeTime: string
  speechOrHearingConditions?: string
  speechSpeed?: string
  photoUrl?: string
  keyTopics: Array<'family' | 'food' | 'travel' | 'work' | 'others'>
  keyTopicsOtherText?: string

  mirrorId: ObjectId | null
  mirrorName: string
  mirrorVerified: boolean
  mirrorPairingStatus: 'paired' | 'awaiting_pairing' | 'replaced' | '' | string
  mirrorPairingCode?: string
  mirrorPairedAt?: Date | null
  deviceAuthToken?: string
  timezone: string
}
```

What it is used for:

- Account lookup and password verification by `email`.
- Dashboard and settings profile data.
- Patient list and status display.
- Embedded mirror connection state for each patient.
- Patient details passed into the OpenAI summary prompt.

### `MirrorPairingSessions`

Stores mirror pairing sessions created outside this server, usually by a mirror/device flow. The caregiver server reads pending sessions, validates the six-digit code, and marks sessions as paired.

```ts
type MirrorPairingSession = {
  _id?: ObjectId
  pairingCode: string
  status: 'pending' | 'paired' | string
  deviceId: ObjectId | string
  authToken?: string
  timezone?: string
  expiresAt: Date
  nurseId?: ObjectId
  patientId?: ObjectId
  pairedAt?: Date
  createdAt?: Date
  updatedAt?: Date
}
```

What it is used for:

- `pairingCode` is normalized to digits only and capped at six characters.
- Pending sessions are valid only when `status` is `pending` and `expiresAt` is in the future.
- `deviceId` becomes the patient `mirrorId`.
- `authToken` is copied into the patient as `deviceAuthToken` when available.
- On successful pairing, the session is updated to `status: 'paired'` with `nurseId`, `patientId`, and `pairedAt`.
- Unlinking a mirror deletes matching pairing sessions.

### `MirrorIdToNurseIdMap`

Maps a mirror device to exactly one caregiver and one patient. The explicit mirror connect route attempts to create unique indexes for `patientId` and `mirrorId`.

```ts
type MirrorIdToNurseIdMap = {
  _id?: ObjectId
  mirrorId: ObjectId
  nurseId: ObjectId
  patientId: ObjectId
  mirrorName: string
  patientName: string
  pairingCode?: string
  createdAt: Date
  updatedAt: Date
}
```

What it is used for:

- Prevents one mirror from being linked to multiple patients.
- Prevents one patient from having multiple linked mirrors.
- Lets unlink logic find all map rows tied to a patient or mirror.
- Supports mirror/device ownership lookups by external mirror-facing services.

### `ConversationIdToPatientIdMap`

Maps conversation records to the patient and caregiver they belong to. Conversation dashboard APIs generally query this collection first, then load matching `Conversation` documents.

```ts
type ConversationIdToPatientIdMap = {
  _id?: ObjectId
  conversationId: ObjectId
  patientId: ObjectId
  nurseId?: ObjectId
  createdAt: Date
  updatedAt?: Date
}
```

What it is used for:

- Finds all sessions for a patient on a Singapore-local day or month.
- Powers session history, monthly session counts, latest patient status, and trend charts.
- Acts as the source of session timestamps when the matching `Conversation` document is missing `createdAt`.

### `Conversation`

Stores metrics and transcript logs for each voice conversation.

```ts
type Conversation = {
  _id?: ObjectId
  conversationId?: ObjectId
  duration?: number
  words?: number
  exchanges?: number
  avgLatency?: number
  logs?: ConversationLog[]
  sessionStatus?: string
  createdAt?: Date
  updatedAt?: Date
}

type ConversationLog = {
  sentence?: string
  role?: string
  words?: number
  duration?: number
  wordsPerSecond?: number
}
```

What it is used for:

- Session details shown in the mobile app.
- Monthly `completedCount` when `sessionStatus === 'completed'`.
- Daily AI summary transcript generation.
- Trend duration data.
- Conversation records may be matched by either `_id` or `conversationId`.

## API Reference

All request and response bodies are JSON unless noted. The native app may call paths prefixed with `/api`, but `reflexion-native-app/src/lib/apiUrl.ts` strips `/api` before sending the request. The server routes below are the actual Express paths.

### Health

#### `GET /health`

Returns:

```json
{ "ok": true }
```

#### `GET /healthcheck`

Returns:

```json
{ "ok": true }
```

### Auth

#### `POST /auth/sign-in`

Signs in a caregiver by email and password.

Request:

```json
{
  "email": "caregiver@example.com",
  "password": "password123"
}
```

Response:

```json
{
  "nurseId": "64f...",
  "name": "Caregiver Name",
  "email": "caregiver@example.com"
}
```

Errors:

- `400` when email or password is missing.
- `401` when the account does not exist or password verification fails.

### Caregiver, Patients, And Notifications

#### `POST /nurse-patient-config/create`

Creates a caregiver account, notification settings, and one or more embedded patient profiles.

Request:

```json
{
  "account": {
    "name": "Caregiver Name",
    "email": "caregiver@example.com",
    "password": "password123",
    "phoneNumber": "+65 9000 0000",
    "relationshipToElderly": "parent"
  },
  "patients": [
    {
      "name": "Patient Name",
      "phoneNumber": "+65 8888 8888",
      "age": 78,
      "gender": "female",
      "preferredLanguage": "english",
      "usualWakeTime": "08:00",
      "speechOrHearingConditions": "Mild hearing loss",
      "photoUrl": "https://example.com/photo.jpg",
      "keyTopics": ["family", "food"],
      "keyTopicsOtherText": "",
      "mirrorName": "Living Room Mirror",
      "mirrorPairingCode": "123456",
      "timezone": "Asia/Singapore"
    }
  ],
  "notifications": {
    "pushNotificationsEnabled": true,
    "alertSensitivity": "only_important_changes",
    "preferredDailySummaryTime": "09:00"
  }
}
```

Response:

```json
{
  "insertedId": "64f...",
  "nurseId": "64f...",
  "name": "Caregiver Name",
  "email": "caregiver@example.com",
  "mirrorMapCount": 1,
  "patientCount": 1
}
```

Validation rules:

- Password must be at least eight characters.
- Email must contain `@`.
- At least one patient is required.
- Patient age must be an integer from 1 to 130.
- `gender`, `preferredLanguage`, `relationshipToElderly`, notification values, and `keyTopics` must match the allowed enum values in `src/lib/validation.ts`.
- If `keyTopics` includes `others`, `keyTopicsOtherText` is required.

Side effects:

- Hashes the password with PBKDF2 before storing.
- Resolves valid mirror pairing codes from `MirrorPairingSessions`.
- Clears any existing assignment for newly paired mirror IDs.
- Inserts paired mirrors into `MirrorIdToNurseIdMap`.
- Marks matching pairing sessions as `paired`.

#### `PATCH /nurse-patient-config/add-patients`

Adds one or more patients to an existing caregiver. If `nurseId` is omitted, the latest-created caregiver config is used.

Request:

```json
{
  "nurseId": "64f...",
  "patients": [
    {
      "name": "Patient Name",
      "age": 78,
      "gender": "female",
      "preferredLanguage": "english",
      "usualWakeTime": "08:00",
      "keyTopics": ["family"]
    }
  ]
}
```

Response:

```json
{
  "nurseId": "64f...",
  "patientCount": 1
}
```

Uses the same patient validation and mirror pairing behavior as account creation.

#### `GET /nurse-patient-config/latest?nurseId=<id>`

Returns caregiver settings and patient dashboard cards. If `nurseId` is omitted, the latest-created caregiver config is returned.

Response:

```json
{
  "nurseId": "64f...",
  "caregiverName": "Caregiver Name",
  "email": "caregiver@example.com",
  "phoneNumber": "+65 9000 0000",
  "pushNotificationsEnabled": true,
  "alertSensitivity": "only_important_changes",
  "preferredDailySummaryTime": "09:00",
  "patients": [
    {
      "id": "650...",
      "patientId": "650...",
      "name": "Patient Name",
      "phoneNumber": "+65 8888 8888",
      "age": 78,
      "preferredLanguage": "english",
      "speechSpeed": "Slow",
      "mirrorName": "Living Room Mirror",
      "photoUrl": "",
      "status": "doing_well",
      "statusLabel": "Doing well",
      "lastSpokenAt": "2026-07-13T02:10:00.000Z",
      "lastSpokenLabel": "Today, 10:10am",
      "duration": 320
    }
  ]
}
```

Status logic:

- `doing_well`: latest conversation was today in `Asia/Singapore`.
- `worth_checking`: latest conversation was one or two days ago.
- `needs_attention`: no conversation, or latest conversation was more than two days ago.

#### `PATCH /nurse-patient-config/notifications`

Updates caregiver notification preferences. If `nurseId` is omitted, the latest-created caregiver config is used.

Request:

```json
{
  "nurseId": "64f...",
  "pushNotificationsEnabled": true,
  "alertSensitivity": "only_important_changes",
  "preferredDailySummaryTime": "19:00"
}
```

Response:

```json
{
  "nurseId": "64f...",
  "pushNotificationsEnabled": true,
  "alertSensitivity": "only_important_changes",
  "preferredDailySummaryTime": "19:00"
}
```

### Mirrors

#### `GET /nurse-patient-config/mirrors?nurseId=<id>`

Returns mirror connection state for every patient under a caregiver.

Response:

```json
{
  "nurseId": "64f...",
  "patients": [
    {
      "patientId": "650...",
      "patientName": "Patient Name",
      "mirrorId": "651...",
      "mirrorName": "Living Room Mirror",
      "mirrorVerified": true,
      "mirrorPairingStatus": "paired",
      "mirrorPairingCode": "123456",
      "mirrorPairedAt": "2026-07-13T02:10:00.000Z",
      "deviceAuthTokenPresent": true,
      "timezone": "Asia/Singapore"
    }
  ]
}
```

#### `POST /nurse-patient-config/mirrors/connect`

Connects a pending mirror pairing session to an existing patient.

Request:

```json
{
  "nurseId": "64f...",
  "patientId": "650...",
  "pairingCode": "123456",
  "mirrorName": "Living Room Mirror",
  "timezone": "Asia/Singapore"
}
```

Response:

```json
{
  "success": true,
  "patientId": "650...",
  "mirrorId": "651...",
  "mirrorName": "Living Room Mirror",
  "mirrorPairingStatus": "paired",
  "mirrorPairedAt": "2026-07-13T02:10:00.000Z"
}
```

Important behavior:

- Requires valid `nurseId`, `patientId`, and pairing code.
- Rejects patients that already appear to have a mirror connection.
- Rejects mirrors already present in `MirrorIdToNurseIdMap` or embedded patient records.
- Creates unique partial indexes on `MirrorIdToNurseIdMap.patientId` and `MirrorIdToNurseIdMap.mirrorId` when possible.

#### `PATCH /nurse-patient-config/mirrors`

Currently supports only unlinking a mirror from a patient.

Request:

```json
{
  "action": "unlink",
  "nurseId": "64f...",
  "patientId": "650..."
}
```

Response:

```json
{
  "success": true,
  "patientId": "650...",
  "mirrorPairingStatus": "",
  "deletedMirrorMapCount": 1,
  "deletedPairingSessionCount": 1
}
```

Side effects:

- Clears mirror fields on the patient.
- Clears any other embedded patient assignment using the same mirror ID.
- Deletes matching rows from `MirrorIdToNurseIdMap`.
- Deletes matching rows from `MirrorPairingSessions`.

### Conversation Sessions

#### `GET /conversation-session?id=<patientId>`

Returns all sessions for the patient for the current Singapore-local day.

Response:

```json
{
  "patientName": "Patient Name",
  "patientId": "650...",
  "sessions": [
    {
      "id": "652...",
      "patientId": "650...",
      "patientName": "Patient Name",
      "duration": 320,
      "words": 850,
      "exchanges": 24,
      "avgLatency": 1.2,
      "createdAt": "2026-07-13T02:10:00.000Z",
      "updatedAt": "2026-07-13T02:16:00.000Z",
      "logs": [
        {
          "sentence": "Good morning.",
          "role": "ai",
          "words": 2,
          "duration": 1.1,
          "wordsPerSecond": 1.8
        }
      ]
    }
  ]
}
```

#### `GET /conversation-sessions-by-day?id=<patientId>&date=YYYY-MM-DD`

Returns all sessions for a patient on a specific Singapore-local date. If `date` is omitted, today is used.

Response shape is the same as `/conversation-session`, with an additional `date` field.

```json
{
  "date": "2026-07-13",
  "patientName": "Patient Name",
  "patientId": "650...",
  "sessions": []
}
```

#### `GET /conversation-session-counts?id=<patientId>&month=YYYY-MM`

Returns daily session counts for a patient across a Singapore-local month. If `month` is omitted, the current Singapore-local month is used.

Response:

```json
{
  "patientId": "650...",
  "month": "2026-07",
  "counts": {
    "1": 0,
    "2": 1
  },
  "days": [
    {
      "date": "2026-07-01",
      "day": 1,
      "count": 0,
      "completedCount": 0,
      "hasCompletedSession": false
    },
    {
      "date": "2026-07-02",
      "day": 2,
      "count": 1,
      "completedCount": 1,
      "hasCompletedSession": true
    }
  ]
}
```

`completedCount` is calculated from matching `Conversation` records where `sessionStatus === 'completed'`.

### Patient Trend

#### `GET /patient-trend?id=<patientId>&days=7`

Returns daily duration/missed-session data for the last 7 or 30 Singapore-local days.

Query parameters:

- `id`: required patient ObjectId.
- `days`: optional, must be `7` or `30`; defaults to `7`.

Response:

```json
{
  "cacheDate": "2026-07-13",
  "days": 7,
  "trend": [
    {
      "date": "2026-07-07",
      "duration": 0,
      "status": "red",
      "missed": true
    },
    {
      "date": "2026-07-13",
      "duration": 320,
      "status": "green",
      "missed": false
    }
  ]
}
```

Current status values:

- `green`: a conversation with duration greater than zero exists for that day.
- `red`: no conversation duration exists for that day.

The `TrendDay` type includes `yellow`, but current implementation does not return it.

### Patient Summary

#### `POST /patient-summary`

Generates a 2-4 sentence caregiver summary for a patient on a Singapore-local date using OpenAI.

Request:

```json
{
  "patientId": "650...",
  "date": "2026-07-13"
}
```

Response:

```json
{
  "summary": "Patient Name seemed calm and engaged today. They talked mostly about family and meals. No urgent follow-up was apparent from the transcript."
}
```

Behavior:

- Defaults `date` to today in `Asia/Singapore`.
- Loads all logs from mapped conversations for that date.
- Converts log role `ai` to `Aria`; all other roles are labeled `Patient`.
- Includes patient profile details in the model prompt for context.
- Uses model `gpt-4o-mini`.
- Returns a no-transcript message when no usable logs exist.

## APIs Used By The Mobile App

The native app currently calls these backend endpoints:

| Screen/file | Backend endpoint |
| --- | --- |
| `app/sign-in.tsx` | `POST /auth/sign-in` |
| `app/onboarding.tsx` | `POST /nurse-patient-config/create` |
| `app/onboarding.tsx` add-patient mode | `PATCH /nurse-patient-config/add-patients` |
| `app/(tabs)/index.tsx` | `GET /nurse-patient-config/latest?nurseId=<id>` |
| `app/(tabs)/settings.tsx` | `GET /nurse-patient-config/latest?nurseId=<id>` |
| `app/(tabs)/settings.tsx` | `PATCH /nurse-patient-config/notifications` |
| `app/mirror-management.tsx` | `GET /nurse-patient-config/mirrors?nurseId=<id>` |
| `app/mirror-management.tsx` | `PATCH /nurse-patient-config/mirrors` |
| `app/mirror-management/add.tsx` | `GET /nurse-patient-config/mirrors?nurseId=<id>` |
| `app/mirror-management/add.tsx` | `POST /nurse-patient-config/mirrors/connect` |
| `app/session/[id].tsx` | `GET /conversation-sessions-by-day?id=<patientId>&date=<date>` |
| `app/session-history/[id].tsx` | `GET /conversation-session-counts?id=<patientId>&month=<month>` |
| `app/session-history/[id]/[date].tsx` | `GET /conversation-sessions-by-day?id=<patientId>&date=<date>` |
| `app/profile/[id].tsx` | `POST /patient-summary` |
| `app/session-history/[id]/[date].tsx` | `POST /patient-summary` |
| `src/lib/patientTrendClient.ts` | `GET /patient-trend?id=<patientId>&days=<7-or-30>` |

The mobile app calls these as `/api/...`, then strips the `/api` prefix before building the final URL.

## Date And Timezone Behavior

The server treats day and month windows as `Asia/Singapore`.

- Dashboard status, session history, monthly counts, trends, and summaries use Singapore-local date boundaries.
- UTC MongoDB `Date` values are queried with Singapore-local start/end bounds.
- Default patient timezone is `Asia/Singapore`, unless supplied by patient input or pairing session.

## Error Shape

Most route-level errors return:

```json
{ "error": "Human readable message." }
```

Unknown routes return:

```json
{ "error": "Not found" }
```

Unhandled errors are converted by the Express error handler into the same shape with status `500`, unless the thrown error has a numeric `status`.

## Current Implementation Notes

- There is no auth middleware; routes trust supplied `nurseId` values.
- MongoDB connections are opened and closed per request through `withMongo`.
- `NursePatientConfig.patients` is embedded rather than stored in a separate `Patients` collection.
- Mirror unlinking clears multiple possible duplicate/stale assignments defensively.
- `OPENAI_API_KEY` is read only when `POST /patient-summary` is called.
- Request JSON body limit is `1mb`.
- CORS is enabled for all origins.
