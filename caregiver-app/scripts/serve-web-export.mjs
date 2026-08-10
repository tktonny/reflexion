import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('../dist', import.meta.url)));
const args = process.argv.slice(2);
const valueAfter = (flag, fallback) => {
  const index = args.indexOf(flag);
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback;
};
const host = valueAfter('--host', '0.0.0.0');
const port = Number(valueAfter('--port', '8091'));
const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.webmanifest': 'application/manifest+json',
};

async function fileFor(pathname) {
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return null;
  }
  if (decoded.includes('..')) return null;
  const relative = decoded.replace(/^\/+/, '');
  const direct = normalize(join(root, relative));
  const candidates = [direct];
  if (!extname(relative)) candidates.push(`${direct}.html`, join(direct, 'index.html'));
  if (relative === '') candidates.unshift(join(root, 'index.html'));
  for (const candidate of candidates) {
    if (!candidate.startsWith(root)) continue;
    try {
      const details = await stat(candidate);
      if (details.isFile()) return candidate;
    } catch {
      // Try the next clean route candidate.
    }
  }
  return null;
}

const server = createServer(async (request, response) => {
  const pathname = new URL(request.url || '/', `http://${request.headers.host || 'localhost'}`).pathname;
  const file = await fileFor(pathname);
  if (!file) {
    response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    response.end('Expo web route not found');
    return;
  }
  try {
    const contents = await readFile(file);
    response.writeHead(200, { 'cache-control': 'no-cache', 'content-type': mime[extname(file)] || 'application/octet-stream' });
    if (request.method === 'HEAD') response.end();
    else response.end(contents);
  } catch {
    response.writeHead(500, { 'content-type': 'text/plain; charset=utf-8' });
    response.end('Expo web preview could not read this asset');
  }
});

server.listen(port, host, () => {
  console.log(`Serving exported Expo caregiver app on http://${host}:${port}`);
  console.log(`Demo entry: http://localhost:${port}/demo`);
});

