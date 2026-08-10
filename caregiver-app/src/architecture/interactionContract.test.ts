import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';
import { interactionContracts } from './interactionContract';

test('every interaction contract has an outcome and user-visible states', () => {
  assert.ok(interactionContracts.length >= 20);
  for (const contract of interactionContracts) {
    assert.ok(contract.sourceRoute && contract.controlId && contract.label);
    assert.ok(contract.loadingState && contract.successState && contract.errorState);
    assert.ok(contract.persistenceEffect && contract.mirrorEffect);
    if (contract.actionType === 'navigate') assert.ok(contract.destinationRoute, `${contract.controlId} needs a route`);
  }
});

test('settings account flow is complete and contains no visual-only action', () => {
  const account = interactionContracts.filter((entry) => entry.sourceRoute.startsWith('/settings/account'));
  for (const label of ['Edit personal information', 'Change email', 'Change phone number', 'Change password', 'Sign-in methods', 'Sign out']) {
    assert.ok(account.some((entry) => entry.label === label), `missing ${label}`);
  }
  assert.ok(account.every((entry) => entry.actionType !== undefined));
});

test('active settings routes exist and do not retain placeholder outcomes', () => {
  const root = resolve(import.meta.dirname, '../../app');
  for (const route of ['settings/account.tsx', 'settings/notifications.tsx', 'settings/language.tsx', 'settings/feedback.tsx', 'settings/help.tsx', 'settings/household.tsx', 'settings/devices.tsx']) {
    const source = readFileSync(resolve(root, route), 'utf8');
    assert.doesNotMatch(source, /onPress=\{\(\) => undefined|ready to save when this service is connected|console\.log\(/);
  }
});
