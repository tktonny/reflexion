// Electron main process for the Linux (Ubuntu) build of the Reflexion mirror.
//
// The mirror app is a React Native / Expo app; its production Android path uses native modules
// (expo-pcm-audio, onnxruntime wake word, direct-WS omni). Linux has no RN desktop target, so the Linux
// build is the SAME app compiled for WEB (react-native-web) and wrapped in Electron (Chromium). The
// conversation runs the `relay` transport (browser <-WS-> local Node relay <-WS-> Qwen) with Web-Audio
// capture — see docs/mirror-app/linux-electron.md for what differs from Android.
//
// KEYLESS (same security model as the Android APK): NO Qwen key lives on the device. The renderer pairs
// with the backend using an embedded API URL + per-device bootstrap token, and per conversation mints a
// short-lived Qwen realtime TICKET from the backend. It hands that ticket to the local relay (first WS
// message), and the relay opens the header-authed Qwen WS with the ticket. Chromium WebSockets can't set
// the Authorization header, which is the only reason a Node relay sits in the middle at all.
//
// Two local services run inside this process:
//   1. a tiny static HTTP server for the exported SPA (dist/) — file:// breaks Expo Router history
//      routing and relative asset URLs, so we serve it over http://127.0.0.1 instead;
//   2. the Node relay (server/index.mjs) — keyless; it authenticates to Qwen with the renderer's ticket.

const { app, BrowserWindow, ipcMain, session } = require('electron')
const path = require('path')
const http = require('http')
const fs = require('fs')
const crypto = require('crypto')
const { spawn } = require('child_process')

const network = require('./network')
const { DEFAULT_API_BASE, createBackendProxy, isBackendPath, normalizeBase } = require('./apiProxy')
const { DEFAULT_PORTAL_PORT, generatePin, startSetupPortal } = require('./setupPortal')

const WEB_DIR = path.join(__dirname, '..', 'dist')
const WEB_PORT = Number(process.env.REFLEXION_MIRROR_WEB_PORT) || 8899
const RELAY_PORT = Number(process.env.REFLEXION_RELAY_PORT) || 8787
const PORTAL_PORT = Number(process.env.REFLEXION_SETUP_PORTAL_PORT) || DEFAULT_PORTAL_PORT
// How long to let NetworkManager reconnect to a known network on its own before taking over the radio for
// a setup hotspot. Grabbing the radio too early would PREVENT a mirror that was simply slow to associate
// from ever rejoining its own Wi-Fi.
const SETUP_GRACE_MS = Number(process.env.REFLEXION_SETUP_GRACE_MS) || 45_000
const SETUP_POLL_MS = 8_000

/**
 * Backend origin, resolved at RUNTIME (not baked into the SPA bundle).
 *
 * Precedence: launch env -> device config file -> production default. Runtime resolution is what lets a
 * unit be re-pointed at a different backend without re-exporting the web bundle, and it is what makes the
 * /api proxy possible at all: the main process must know the target before the renderer runs.
 */
function resolveApiBase() {
  const fromEnv = process.env.REFLEXION_API_BASE || process.env.EXPO_PUBLIC_API_BASE
  if (fromEnv && fromEnv.trim()) return normalizeBase(fromEnv)
  try {
    const configPath = path.join(app.getPath('userData'), 'device-config.json')
    if (fs.existsSync(configPath)) {
      const parsed = JSON.parse(fs.readFileSync(configPath, 'utf8'))
      if (typeof parsed?.apiBase === 'string' && parsed.apiBase.trim()) return normalizeBase(parsed.apiBase)
    }
  } catch (error) {
    console.warn('[electron] could not read device-config.json', error)
  }
  return DEFAULT_API_BASE
}

let API_BASE = DEFAULT_API_BASE
const proxyToBackend = createBackendProxy(() => API_BASE)

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
    if (isBackendPath(pathname)) { proxyToBackend(req, res, pathname); return }
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

/** Spawn the bundled relay. KEYLESS: the relay no longer needs a Qwen key — the renderer mints a
 *  short-lived ticket from the backend (device-authenticated) and hands it to the relay per session, so
 *  the relay authenticates to Qwen with that ticket (same model as the Android APK; no key on device).
 *  Set REFLEXION_MIRROR_SKIP_RELAY=1 to point at an external relay instead. */
let relayProc = null
function startRelay() {
  if (process.env.REFLEXION_MIRROR_SKIP_RELAY === '1') return
  // server/index.mjs is the relay entry (`npm run relay`). It needs the orchestration bundle
  // (server/generated/, produced by `npm run build:orch`) to exist — see the Linux build doc.
  const relayPath = path.join(__dirname, '..', 'server', 'index.mjs')
  const orchBuilt = fs.existsSync(path.join(__dirname, '..', 'server', 'generated', 'orchestration.mjs'))
  if (!fs.existsSync(relayPath) || !orchBuilt) {
    console.warn('[electron] relay not started (server/generated/ not built — run npm run build:orch); UI still loads')
    return
  }
  relayProc = spawn(process.execPath, [relayPath], {
    // ELECTRON_RUN_AS_NODE makes the Electron binary behave as plain Node for the child.
    env: { ...process.env, ELECTRON_RUN_AS_NODE: '1', REFLEXION_RELAY_PORT: String(RELAY_PORT) },
    stdio: 'inherit',
  })
  relayProc.on('exit', (code) => console.log(`[electron] relay exited (${code})`))
}

// ---------------------------------------------------------------------------
// Setup mode — how a mirror with no keyboard, no mouse and no network gets onto Wi-Fi.
//
// The unit boots into a fullscreen kiosk; the family has no input device. If the mirror is still offline
// after a grace period, it brings up its OWN hotspot and serves the setup portal on it. The mirror's SCREEN
// shows the hotspot name, passphrase, a join QR, the portal address and a 6-digit code; the caregiver does
// the typing on their PHONE. Nothing is typed on the mirror.
//
// The mirror screen is also the only place the RESULT can appear: applying a network takes the radio the
// hotspot runs on, so the phone's connection dies at that moment. Hence the renderer mirrors this state.
// ---------------------------------------------------------------------------

/** @type {{active:boolean, ssid:string, password:string, pin:string, address:string, portalUrl:string, lastResult:null|object, reason:string}} */
let setupState = {
  active: false, ssid: '', password: '', pin: '', address: '', portalUrl: '', lastResult: null, reason: '',
}
let portal = null
let setupPollTimer = null

function publishSetupState() {
  const payload = setupStateForRenderer()
  for (const win of BrowserWindow.getAllWindows()) {
    // The renderer may not have finished loading; a failed send here is not worth crashing over.
    try { win.webContents.send('reflexion:setup:state', payload) } catch { /* ignore */ }
  }
}

/** Never expose the PIN's purpose ambiguously: the renderer NEEDS it, because it is displayed on-screen. */
function setupStateForRenderer() {
  return { ...setupState }
}

async function startSetupMode(reason) {
  if (setupState.active) return
  const started = await network.hotspotStart({ ssid: buildSetupSsid() })
  if (!started.ok) {
    // No AP-capable adapter, or NetworkManager refused. Record it so the mirror screen can say why rather
    // than silently showing nothing — this is the one path where the appliance cannot self-serve.
    setupState = { ...setupState, active: false, reason: started.error || 'hotspot_failed' }
    console.warn(`[electron] setup hotspot could not start: ${started.error}`)
    publishSetupState()
    return
  }
  const address = await network.hotspotAddress()
  const pin = generatePin()
  portal = startSetupPortal({
    pin,
    port: PORTAL_PORT,
    scanWifi: () => network.wifiScan({ rescan: true }),
    status: () => network.status(),
    applyWifi: applyWifiFromPortal,
  })
  setupState = {
    active: true,
    ssid: started.ssid || '',
    password: started.passphrase || '',
    pin,
    address,
    portalUrl: address ? `http://${address}:${PORTAL_PORT}` : `http://<mirror address>:${PORTAL_PORT}`,
    lastResult: null,
    reason,
  }
  console.log(`[electron] setup mode ON (${reason}) — hotspot "${setupState.ssid}", portal ${setupState.portalUrl}`)
  publishSetupState()
}

async function stopSetupMode() {
  if (portal) { await portal.close().catch(() => undefined); portal = null }
  if (setupState.active) await network.hotspotStop().catch(() => undefined)
  setupState = { ...setupState, active: false, ssid: '', password: '', pin: '', portalUrl: '' }
  publishSetupState()
}

/**
 * Apply the network the caregiver chose on their phone.
 *
 * Ordering matters: the hotspot must come DOWN first, because it holds the only Wi-Fi radio and a join
 * cannot happen while it is up. If the join then fails (a mistyped password is the common case), setup mode
 * is brought straight back so the caregiver can retry — otherwise one typo would strand the unit with no
 * hotspot, no network, and no input device, which is worse than where it started.
 */
async function applyWifiFromPortal({ ssid, password }) {
  console.log(`[electron] applying Wi-Fi "${ssid}" from setup portal`) // never log the password
  if (portal) { await portal.close().catch(() => undefined); portal = null }
  await network.hotspotStop().catch(() => undefined)
  setupState = { ...setupState, active: false }
  publishSetupState()

  const result = await network.wifiConnect({ ssid, password })
  if (result.ok) {
    setupState = { ...setupState, active: false, lastResult: { ok: true, ssid }, reason: '' }
    publishSetupState()
    console.log(`[electron] setup mode OFF — joined "${ssid}"`)
    return { ok: true }
  }
  setupState = { ...setupState, lastResult: { ok: false, ssid, error: result.error || '' } }
  await startSetupMode('retry_after_failed_join')
  setupState = { ...setupState, lastResult: { ok: false, ssid, error: result.error || '' } }
  publishSetupState()
  return { ok: false, error: result.error }
}

/**
 * Watch connectivity for the whole life of the process: start setup mode when the mirror is offline and
 * cannot fix itself, and tear it down the moment it has real internet. It keeps working after boot too —
 * a router replaced a year later strands the unit exactly the same way.
 */
function startConnectivityWatch() {
  let offlineSince = Date.now()
  const tick = async () => {
    const status = await network.status().catch(() => null)
    if (!status) return
    // 'full' is NetworkManager's own verdict for a real route out; a captive portal ('portal') or a link
    // with no internet ('limited') must NOT count as connected, or the mirror sits there unable to reach
    // the backend with no setup path offered.
    const online = status.connectivity === 'full'
    if (online) {
      offlineSince = 0
      if (setupState.active) await stopSetupMode()
      return
    }
    if (setupState.active) return // already offering setup
    if (!offlineSince) offlineSince = Date.now()
    if (Date.now() - offlineSince < SETUP_GRACE_MS) return
    if (!status.capabilities?.hotspot) return // nothing we can do without an AP-capable radio
    await startSetupMode(status.connectivity === 'portal' ? 'captive_portal' : 'offline')
  }
  void tick()
  setupPollTimer = setInterval(() => { void tick() }, SETUP_POLL_MS)
}

function buildSetupSsid() {
  // A stable, per-unit-ish name so an installer with two mirrors in one room can tell them apart.
  const suffix = crypto.createHash('sha256').update(String(app.getPath('userData'))).digest('hex').slice(0, 4)
  return `Reflexion-Setup-${suffix.toUpperCase()}`
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

/**
 * Expose the privileged network operations to the renderer over IPC.
 *
 * The renderer is sandboxed and context-isolated, so this is the only path by which the setup screen can
 * join a Wi-Fi network, start the mirror's hotspot, or tether to a phone over Bluetooth. Each handler is
 * an explicit, named operation — no "run this command" escape hatch — and every argument is validated
 * inside network.js before it reaches NetworkManager or BlueZ.
 */
function registerNetworkIpc() {
  const handlers = {
    'reflexion:network:capabilities': () => network.capabilities(),
    'reflexion:network:status': () => network.status(),
    'reflexion:wifi:scan': (options) => network.wifiScan(options || {}),
    'reflexion:wifi:connect': (options) => network.wifiConnect(options || {}),
    'reflexion:wifi:connect-saved': (options) => network.wifiConnectSaved(options || {}),
    'reflexion:wifi:forget': (options) => network.wifiForget(options || {}),
    'reflexion:wifi:radio': (options) => network.wifiSetRadio(Boolean(options?.enabled)),
    'reflexion:hotspot:start': (options) => network.hotspotStart(options || {}),
    'reflexion:hotspot:stop': () => network.hotspotStop(),
    'reflexion:bluetooth:status': () => network.bluetoothStatus(),
    'reflexion:bluetooth:power': (options) => network.bluetoothSetPower(Boolean(options?.enabled)),
    'reflexion:bluetooth:scan': (options) => network.bluetoothScan(options || {}),
    'reflexion:bluetooth:pair': (options) => network.bluetoothPair(options || {}),
    'reflexion:bluetooth:tether': (options) => network.bluetoothTether(options || {}),
    'reflexion:bluetooth:disconnect': (options) => network.bluetoothDisconnect(options || {}),
    // Setup mode: the renderer displays the hotspot name, passphrase, portal address and PIN on the
    // mirror's screen, because that screen is the only output a keyboard-less unit has.
    'reflexion:setup:state': () => setupStateForRenderer(),
    'reflexion:setup:start': () => startSetupMode('requested').then(setupStateForRenderer),
    'reflexion:setup:stop': () => stopSetupMode().then(() => setupStateForRenderer()),
  }
  for (const [channel, handler] of Object.entries(handlers)) {
    ipcMain.handle(channel, async (_event, options) => {
      try {
        return await handler(options)
      } catch (error) {
        // A rejected invoke surfaces in the renderer as an opaque Error; a shaped failure keeps the setup
        // screen able to show the installer something specific.
        console.error(`[electron] ${channel} threw`, error)
        return { ok: false, error: error instanceof Error ? error.message : 'Network operation failed.' }
      }
    })
  }
}

app.whenReady().then(() => {
  API_BASE = resolveApiBase()
  console.log(`[electron] backend proxied at http://127.0.0.1:${WEB_PORT}/api -> ${API_BASE}`)
  registerNetworkIpc()
  // The mirror is a controlled appliance and the daily check-in needs the microphone, so auto-grant
  // media capture rather than prompting an elder for permission.
  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === 'media' || permission === 'audioCapture' || permission === 'mediaKeySystem')
  })
  startWebServer()
  startRelay()
  createWindow()
  // Setup mode is what makes an unattended, input-less unit recoverable: if it cannot reach the network on
  // its own, it offers its own hotspot + phone portal instead of sitting on an error screen forever.
  startConnectivityWatch()
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow() })
})

app.on('window-all-closed', () => {
  if (relayProc) { try { relayProc.kill() } catch { /* ignore */ } }
  if (setupPollTimer) clearInterval(setupPollTimer)
  // Leaving a hotspot broadcasting after the app exits would keep the radio off the home Wi-Fi.
  void stopSetupMode().catch(() => undefined).finally(() => app.quit())
})
