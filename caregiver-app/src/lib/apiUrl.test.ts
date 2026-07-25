import assert from 'node:assert/strict';
import test from 'node:test';

// The two URL builders are deliberately incompatible and it is a documented footgun (CLAUDE.md): the legacy
// one STRIPS a leading /api because the legacy routes are mounted bare at the server root, while the v1 one
// APPENDS /api/v1. Getting them the wrong way round produces a 404 that the app used to render as a
// caregiver-facing headline, so both directions are pinned here.
//
// Each builder reads the base URL at call time, so the env var is set before the module is imported.

const BASE = 'https://reflexion.production.tktonny.top';

test('the legacy builder strips a leading /api and normalises slashes', async () => {
  process.env.EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL = BASE;
  const { getApiUrl } = await import('./apiUrl');

  assert.equal(getApiUrl('/api/auth/sign-in'), `${BASE}/auth/sign-in`);
  assert.equal(getApiUrl('/auth/sign-in'), `${BASE}/auth/sign-in`);
  assert.equal(getApiUrl('auth/sign-in'), `${BASE}/auth/sign-in`);
  // Only a leading /api segment is removed — a path that merely starts with those letters is untouched.
  assert.equal(getApiUrl('/apiary/thing'), `${BASE}/apiary/thing`);
  assert.equal(getApiUrl('/api'), `${BASE}/`);
});

test('the legacy builder tolerates a trailing slash on the base URL', async () => {
  process.env.EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL = `${BASE}///`;
  const { getApiUrl } = await import('./apiUrl');
  assert.equal(getApiUrl('/api/patient-trend'), `${BASE}/patient-trend`);
});

test('the v1 builder appends /api/v1 and never doubles the /api segment', async () => {
  process.env.EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL = BASE;
  const { getV1Url } = await import('./apiUrl');

  assert.equal(getV1Url('/notifications'), `${BASE}/api/v1/notifications`);
  assert.equal(getV1Url('notifications'), `${BASE}/api/v1/notifications`);
  // A base that already ends in /api must not become /api/api/v1.
  process.env.EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL = `${BASE}/api`;
  const fresh = await import(`./apiUrl?doubled=${Date.now()}`);
  assert.equal(fresh.getV1Url('/notifications'), `${BASE}/api/v1/notifications`);
});

test('a build with no API origin fails loudly instead of silently calling nothing', async () => {
  // An APK shipped without this env var must not quietly succeed at doing nothing.
  delete process.env.EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL;
  const { getApiUrl } = await import(`./apiUrl?missing=${Date.now()}`);
  assert.throws(() => getApiUrl('/api/auth/sign-in'), /EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL/);
});
