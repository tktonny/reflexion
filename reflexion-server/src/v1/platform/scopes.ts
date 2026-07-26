/**
 * The scopes a care_relationship must carry for a caregiver to use the app for one family member.
 *
 * There is one list because there were three, and they drifted. A caregiver's relationship row is written by
 * three different paths — legacy sign-in (lib/legacyV1Bridge.ts), self-service onboarding
 * (routes/patients.ts POST /patients) and operator onboarding (routes/admin.ts POST /admin/patients) — and
 * each had its own literal. `session:read` was missing from two of them and `device:assign` from a third,
 * so which screens worked depended on which path had created the patient. Nothing caught it, because every
 * bridged caregiver also carried `tenant_admin`, and platform/auth.ts authorizePatient() returns before it
 * reads the relationship for a tenant admin — the scope lists were never actually exercised.
 *
 * Adding a caregiver-facing route that calls authorizePatient() with a new scope means adding it here.
 * Deliberately NOT included:
 *   - `review:read` / `review:write` — the clinical review queue, which a caregiver must never reach;
 *   - `session:write` / `reminder:respond` / `device:heartbeat` — the mirror's upload and tool paths, which
 *     are granted to device credentials (DEVICE_SCOPES in routes/devices.ts), not to humans.
 */
export const CAREGIVER_RELATIONSHIP_SCOPES = [
  'patient:read',
  'patient:write',
  'device:assign',
  'care_plan:read',
  'care_plan:write',
  'monitoring:read',
  'session:read',
] as const
