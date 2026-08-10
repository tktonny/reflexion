#!/usr/bin/env node
// Fails if any text colour in src/theme.ts drops below WCAG AA on any surface it can appear on.
//
// This exists because the audit that found the problem was a throwaway script: the muted meta colour had
// been at 2.70:1 on white for the app's whole life, across 42 usages, and nothing would have caught it
// drifting back. Run it with `npm run check:contrast`.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const theme = readFileSync(join(root, 'src/theme.ts'), 'utf8');

function hexesIn(block) {
  return [...block.matchAll(/(\w+):\s*'(#[0-9A-Fa-f]{6})'/g)].map((m) => ({ name: m[1], hex: m[2] }));
}

function section(name) {
  const start = theme.indexOf(`${name}: {`);
  if (start === -1) return [];
  return hexesIn(theme.slice(start, theme.indexOf('},', start)));
}

const relativeLuminance = (hex) => {
  const channels = [1, 3, 5]
    .map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
};

const contrast = (a, b) => {
  const [lighter, darker] = [relativeLuminance(a), relativeLuminance(b)].sort((x, y) => y - x);
  return (lighter + 0.05) / (darker + 0.05);
};

const surfaces = section('surface');
const textTokens = section('text').filter((token) => token.name !== 'onAccent');
const accent = theme.match(/accent:\s*'(#[0-9A-Fa-f]{6})'/)?.[1];
const decorative = theme.match(/textDecorative:\s*'(#[0-9A-Fa-f]{6})'/)?.[1];
const placeholder = theme.match(/placeholder:\s*'(#[0-9A-Fa-f]{6})'/)?.[1];

const failures = [];
const check = (label, hex, threshold) => {
  for (const surface of surfaces) {
    const ratio = contrast(hex, surface.hex);
    const line = `${label} (${hex}) on surface.${surface.name} (${surface.hex}): ${ratio.toFixed(2)}:1`;
    if (ratio < threshold) failures.push(`${line} — needs ${threshold}:1`);
    else if (process.env.VERBOSE) console.log(`  ok  ${line}`);
  }
};

for (const token of textTokens) check(`text.${token.name}`, token.hex, 4.5);
if (accent) check('accent (used for link text)', accent, 4.5);
// Decorative glyphs and placeholder hints are held to the non-text 3:1 bar on purpose.
if (decorative) check('textDecorative', decorative, 3);
if (placeholder) check('placeholder', placeholder, 3);

// White on the accent fill — the primary button.
if (accent) {
  const ratio = contrast('#FFFFFF', accent);
  if (ratio < 4.5) failures.push(`text.onAccent (#FFFFFF) on accent (${accent}): ${ratio.toFixed(2)}:1 — needs 4.5:1`);
}

// The unselected tab label is text on the tab bar (a card surface).
const tabInactive = theme.match(/tabInactive:\s*'(#[0-9A-Fa-f]{6})'/)?.[1];
if (tabInactive) check('tabInactive', tabInactive, 4.5);

// The auth error box is its own little surface, so its text and border are checked against it rather than
// against the page. Its border was 1.53:1 before it became a token — a boundary nobody could see.
const errorTokens = Object.fromEntries(section('error').map((token) => [token.name, token.hex]));
if (errorTokens.surface && errorTokens.text) {
  const ratio = contrast(errorTokens.text, errorTokens.surface);
  if (ratio < 4.5) failures.push(`error.text (${errorTokens.text}) on error.surface (${errorTokens.surface}): ${ratio.toFixed(2)}:1 — needs 4.5:1`);
}
if (errorTokens.surface && errorTokens.border) {
  for (const [label, against] of [['error.surface', errorTokens.surface], ['surface.card', '#FFFFFF']]) {
    const ratio = contrast(errorTokens.border, against);
    if (ratio < 3) failures.push(`error.border (${errorTokens.border}) on ${label} (${against}): ${ratio.toFixed(2)}:1 — needs 3:1`);
  }
}

if (failures.length) {
  console.error(`\n${failures.length} contrast failure(s) in src/theme.ts:\n`);
  for (const failure of failures) console.error(`  ✗ ${failure}`);
  console.error('');
  process.exit(1);
}

console.log(`Contrast OK: ${textTokens.length + 4} text/graphic tokens against ${surfaces.length} surfaces, plus the error box and the primary button.`);
