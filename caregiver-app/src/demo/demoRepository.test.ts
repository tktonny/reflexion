import assert from 'node:assert/strict';
import test from 'node:test';

import { getDemoScenario, requestDemo, resetDemoData, setDemoScenario } from './demoRepository';

async function data<T>(response: Response): Promise<T> {
  const body = await response.json() as { data: T };
  return body.data;
}

test('demo repository serves the normal caregiver home resources locally', async () => {
  resetDemoData();
  const patients = await data<Array<{ patientId: string }>>(await requestDemo('/patients?limit=100'));
  const assignments = await data<{ assignments: Array<{ deviceId: string | null }> }>(await requestDemo('/device-assignments'));

  assert.equal(patients.length, 2);
  assert.equal(assignments.assignments.some((assignment) => Boolean(assignment.deviceId)), true);
});

test('demo mutations stay in the fixture repository and do not need a production session', async () => {
  resetDemoData();
  setDemoScenario({ messageDelivery: 'queued', messageInteraction: 'none' });
  const created = await data<{ body: string; state: string }>(await requestDemo('/patients/demo-margaret/family-messages', {
    method: 'POST',
    body: JSON.stringify({ body: 'Local demo message' }),
  }));
  const messages = await data<{ messages: Array<{ body: string }> }>(await requestDemo('/patients/demo-margaret/family-messages'));

  assert.equal(created.body, 'Local demo message');
  assert.equal(created.state, 'queued');
  assert.equal(messages.messages[0].body, 'Local demo message');
  assert.equal(getDemoScenario().messageDelivery, 'queued');
});

