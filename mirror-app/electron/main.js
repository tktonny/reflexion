// Electron main process for the Linux (Ubuntu) build of the Reflexion mirror.
//
// The mirror app is a React Native / Expo app; its production Android path uses native modules
// (expo-pcm-audio, onnxruntime wake word, direct-WS omni). Linux has no RN desktop target, so the Linux
// build is the SAME app compiled for WEB (react-native-web) and wrapped in Electron (Chromium). The
// conversation therefore runs the `relay` transport (browser <-WS-> local Node relay <-WS-> Qwen) with
// Web-Audio capture — see docs/mirror-app/linux-electron.md for what differs from Android.
//
// Two local services run inside this process:
//   1. a tiny static HTTP server for the exported SPA (dist/) — file:// breaks Expo Router history
//      routing and relative asset URLs, so we serve it over http://127.0.0.1 instead;
//   2. (optional) the Node relay (server/relay.mjs), spawned only when a Qwen key is present in the
//      environment. The key is appliance config on the device, NEVER embedded in the distributed app.

const { app, BrowserWindow, session } = require('electron')
const path = require('path')
const http = require('http')
const fs = require('fs')
const { spawn } = require('child_process')

const WEB_DIR = path.join(__dirname, '..', 'dist')
const WEB_PORT = Number(process.env.REFLEXION_MIRROR_WEB_PORT) || 8899
const RELAY_PORT = Number(process.env.REFLEXION_RELAY_PORT) || 8787

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.map': 'application/json',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif',
  '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.webp': 'image/webp',
  '.ttf': 'font/ttf', '.otf': 'font/otf', '.woff': 'font/woff', '.woff2': 'font/woff2',
  '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.bin': 'application/octet-stream',
  '.mp3': 'audio/mpeg', '.wav': 'audio/wav',
}

/** Serve the exported SPA with a history-API fallback (unknown, extension-less routes -> index.html). */
function startWebServer() {
  const server = http.createServer((req, res) => {
    let pathname = '/'
    try { pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname) } catch { /* keep / */ }
    // Prevent path traversal: resolve inside WEB_DIR only.
    let filePath = path.normalize(path.join(WEB_DIR, pathname))
    if (!filePath.startsWith(WEB_DIR)) filePath = path.join(WEB_DIR, 'index.html')
    const isFile = fs.existsSync(filePath) && fs.statSync(filePath).isFile()
    if (!isFile) filePath = path.join(WEB_DIR, 'index.html') // SPA fallback
    res.setHeader('Content-Type', MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream')
    fs.createReadStream(filePath).on('error', () => { res.statusCode = 500; res.end('read error') }).pipe(res)
  })
  server.on('error', (err) => console.error('[electron] web server error', err))
  server.listen(WEB_PORT, '127.0.0.1', () => console.log(`[electron] SPA on http://127.0.0.1:${WEB_PORT}`))
  return server
}

/** Spawn the bundled relay only if a Qwen key is configured. Without it the UI still loads; the
 *  conversation just reports the relay as unavailable (documented). Set REFLEXION_MIRROR_SKIP_RELAY=1
 *  to point at an external relay instead. */
let relayProc = null
function startRelay() {
  if (process.env.REFLEXION_MIRROR_SKIP_RELAY === '1') return
  const hasKey = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
  // server/index.mjs is the relay entry (`npm run relay`). It needs the orchestration bundle
  // (server/generated/, produced by `npm run build:orch`) to exist — see the Linux build doc.
  const relayPath = path.join(__dirname, '..', 'server', 'index.mjs')
  const orchBuilt = fs.existsSync(path.join(__dirname, '..', 'server', 'generated', 'orchestration.mjs'))
  if (!hasKey || !fs.existsSync(relayPath) || !orchBuilt) {
    console.warn('[electron] relay not started (need QWEN_API_KEY + built server/generated/) — conversation unavailable; UI still loads')
    return
  }
  relayProc = spawn(process.execPath, [relayPath], {
    // ELECTRON_RUN_AS_NODE makes the Electron binary behave as plain Node for the child.
    env: { ...process.env, ELECTRON_RUN_AS_NODE: '1', REFLEXION_RELAY_PORT: String(RELAY_PORT) },
    stdio: 'inherit',
  })
  relayProc.on('exit', (code) => console.log(`[electron] relay exited (${code})`))
}

function createWindow() {
  const win = new BrowserWindow({
    fullscreen: true,
    kiosk: process.env.REFLEXION_MIRROR_KIOSK !== '0',
    backgroundColor: '#000000',
    autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, sandbox: true },
  })
  win.loadURL(`http://127.0.0.1:${WEB_PORT}/`)
  if (process.env.REFLEXION_MIRROR_DEVTOOLS === '1') win.webContents.openDevTools({ mode: 'detach' })
  return win
}

app.whenReady().then(() => {
  // The mirror is a controlled appliance and the daily check-in needs the microphone, so auto-grant
  // media capture rather than prompting an elder for permission.
  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === 'media' || permission === 'audioCapture' || permission === 'mediaKeySystem')
  })
  startWebServer()
  startRelay()
  createWindow()
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow() })
})

app.on('window-all-closed', () => {
  if (relayProc) { try { relayProc.kill() } catch { /* ignore */ } }
  app.quit()
})
