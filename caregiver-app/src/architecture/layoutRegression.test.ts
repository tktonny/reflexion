import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import test from 'node:test';

const appRoot = resolve(import.meta.dirname, '../../app');

function routeFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? routeFiles(path) : path.endsWith('.tsx') ? [path] : [];
  });
}

function isRedirectOrChrome(source: string, file: string): boolean {
  const name = file.split('/').pop();
  return name === '_layout.tsx' || name === '+html.tsx' || (source.includes('<Redirect') && !source.includes('<ScreenLayout'));
}

test('every caregiver route uses the shared responsive screen layout', () => {
  const routes = routeFiles(appRoot).filter((file) => !isRedirectOrChrome(readFileSync(file, 'utf8'), file));
  assert.ok(routes.length >= 40, `expected the Version 4 route tree, found ${routes.length} screens`);
  for (const file of routes) {
    const source = readFileSync(file, 'utf8');
    assert.match(source, /ScreenLayout/, `${relative(appRoot, file)} must use ScreenLayout`);
    assert.doesNotMatch(source, /\bSafeAreaView\b/, `${relative(appRoot, file)} must not define a competing safe area`);
    assert.doesNotMatch(source, /numberOfLines\s*=|maxFontSizeMultiplier\s*=|onPress=\{\s*\(\)\s*=>\s*(?:undefined|console\.)/, `${relative(appRoot, file)} contains truncation or a placeholder control`);
  }
});

test('phone inputs keep country code and national number as separate shared fields', () => {
  const sources = routeFiles(appRoot).map((file) => readFileSync(file, 'utf8')).join('\n');
  assert.doesNotMatch(sources, /keyboardType=["']phone-pad["']/, 'routes must use PhoneField instead of an unsplit phone input');
  assert.match(readFileSync(resolve(import.meta.dirname, '../components/Field.tsx'), 'utf8'), /export function PhoneField/);
});

