// Phone-driven Wi-Fi setup for a mirror with NO usable input device.
//
// THE PROBLEM: an Ubuntu mirror unit is a wall-mounted display. It boots straight into the kiosk app, and
// the end user has no mouse and no keyboard (a technician can attach one over USB; a family cannot). So a
// unit delivered to a home with unconfigured Wi-Fi had no path to the network at all — and therefore no
// path to the backend, which is what "it says it cannot connect" actually was.
//
// WHY NOT VOICE: the mirror's voice pipeline is Qwen realtime — a CLOUD service. Using voice to configure
// the network that the cloud connection depends on is circular: with no internet there is no ASR, so there
// is no voice. Network setup has to work fully offline, which rules voice out as the mechanism.
//
// THE APPROACH: the mirror brings up its OWN Wi-Fi hotspot and serves this small page on it. The caregiver
// joins that hotspot from a phone (the mirror shows the name, passphrase and a scannable QR), opens the
// page, and picks the home Wi-Fi using the PHONE's keyboard. Nothing is typed on the mirror. It needs no
// app install, works identically on iOS and Android, and requires no internet.
//
// SECURITY: this server binds to all interfaces, so it is deliberately minimal and tightly scoped.
//   - It serves ONLY this page and three endpoints. It does NOT serve the mirror SPA and does NOT expose
//     the backend proxy — a LAN neighbour must never be able to borrow the device's credentials.
//   - It runs ONLY while setup mode is active, and is torn down the moment the mirror is online.
//   - Applying a network requires a 6-digit PIN shown on the MIRROR'S SCREEN, so being in radio range is
//     not enough — you must be able to see the device. Attempts are rate-limited, because 6 digits is
//     brute-forceable in seconds otherwise.
//   - Wi-Fi passwords arrive in a POST body over plain HTTP. That link is the mirror's own WPA2-encrypted
//     hotspot, and the alternative (a self-signed cert) trades it for a browser warning an installer would
//     have to click through anyway. Passwords are never logged and never echoed back.

const http = require('http')
const crypto = require('crypto')

const DEFAULT_PORTAL_PORT = 8900
const MAX_PIN_ATTEMPTS = 10
const MAX_BODY_BYTES = 8 * 1024

/** Six digits, uniformly distributed (Math.random would bias and is not for access control). */
function generatePin() {
  return String(crypto.randomInt(0, 1_000_000)).padStart(6, '0')
}

function timingSafeEqual(a, b) {
  const left = Buffer.from(String(a))
  const right = Buffer.from(String(b))
  if (left.length !== right.length) return false
  return crypto.timingSafeEqual(left, right)
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    req.on('data', (chunk) => {
      size += chunk.length
      // A setup POST is a few hundred bytes; anything larger is not a real client.
      if (size > MAX_BODY_BYTES) { reject(new Error('body_too_large')); req.destroy(); return }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

function json(res, statusCode, payload) {
  const body = JSON.stringify(payload)
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  })
  res.end(body)
}

/**
 * Start the setup portal.
 *
 * @param {object} deps
 * @param {() => Promise<{ok:boolean, networks?:Array, error?:string}>} deps.scanWifi
 * @param {(options:{ssid:string,password:string}) => Promise<{ok:boolean,error?:string}>} deps.applyWifi
 *   Called AFTER the response is sent — applying takes the radio, which drops this very connection.
 * @param {() => Promise<object>} deps.status
 * @param {string} deps.pin the code shown on the mirror's screen
 * @param {number} [deps.port]
 */
function startSetupPortal({ scanWifi, applyWifi, status, pin, port = DEFAULT_PORTAL_PORT }) {
  let attemptsRemaining = MAX_PIN_ATTEMPTS
  /** Last outcome, so the page can report a failure that happened after the hotspot came back. */
  let lastResult = null

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host || 'mirror'}`)
    const route = `${req.method} ${url.pathname}`

    try {
      // Captive-portal probes (iOS /hotspot-detect.html, Android /generate_204) land here too; answering
      // them with the page is what makes a phone pop the "Sign in to network" sheet automatically.
      if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/index.html'
        || url.pathname === '/hotspot-detect.html' || url.pathname === '/generate_204'
        || url.pathname === '/ncsi.txt' || url.pathname === '/connecttest.txt')) {
        const html = renderPortalPage()
        res.writeHead(200, {
          'Content-Type': 'text/html; charset=utf-8',
          'Cache-Control': 'no-store',
          'X-Content-Type-Options': 'nosniff',
          // The page is fully self-contained; forbid any external load so it works with zero internet.
          'Content-Security-Policy': "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; form-action 'self'; connect-src 'self'",
        })
        res.end(html)
        return
      }

      if (req.method === 'GET' && url.pathname === '/api/networks') {
        const result = await scanWifi()
        // SSIDs only — never any stored secret.
        const networks = (result.networks || []).map((entry) => ({
          ssid: entry.ssid, signal: entry.signal, secured: entry.secured, saved: entry.saved, band: entry.band,
        }))
        json(res, result.ok ? 200 : 503, { ok: result.ok, networks, error: result.error })
        return
      }

      if (req.method === 'GET' && url.pathname === '/api/status') {
        const current = await status()
        json(res, 200, {
          ok: true,
          online: Boolean(current?.online),
          connectivity: current?.connectivity || 'unknown',
          activeConnection: current?.activeConnection?.name || '',
          lastResult,
        })
        return
      }

      if (req.method === 'POST' && url.pathname === '/api/connect') {
        if (attemptsRemaining <= 0) {
          json(res, 429, { ok: false, error: 'Too many incorrect codes. Restart the mirror to try again.' })
          return
        }
        let payload
        try {
          payload = JSON.parse(await readBody(req))
        } catch {
          json(res, 400, { ok: false, error: 'Could not read that request.' })
          return
        }
        const ssid = typeof payload?.ssid === 'string' ? payload.ssid : ''
        const password = typeof payload?.password === 'string' ? payload.password : ''
        const providedPin = typeof payload?.pin === 'string' ? payload.pin : ''

        if (!timingSafeEqual(providedPin, pin)) {
          attemptsRemaining -= 1
          json(res, 403, {
            ok: false,
            error: `That code is not right. ${attemptsRemaining} ${attemptsRemaining === 1 ? 'try' : 'tries'} left.`,
          })
          return
        }
        if (!ssid.trim()) {
          json(res, 400, { ok: false, error: 'Choose a Wi-Fi network first.' })
          return
        }
        // A correct PIN resets the budget: the caregiver is legitimate, and a later typo in a Wi-Fi
        // password must not lock them out of retrying.
        attemptsRemaining = MAX_PIN_ATTEMPTS
        lastResult = null

        // RESPOND FIRST. Joining a network takes the radio the hotspot is running on, so this very TCP
        // connection dies as soon as the switch happens — a phone waiting on the response would only ever
        // see a network error and could not tell success from failure. The mirror's screen reports the
        // outcome instead, which is the one display that survives the switch.
        json(res, 202, {
          ok: true,
          applying: true,
          message: 'Connecting the mirror. This setup network will disappear — watch the mirror screen for the result.',
        })

        // Let the response actually flush before the radio goes away.
        setTimeout(() => {
          void applyWifi({ ssid, password })
            .then((result) => { lastResult = { ok: result.ok, error: result.error || '', ssid } })
            .catch((error) => { lastResult = { ok: false, error: error?.message || 'Connection failed.', ssid } })
        }, 1200)
        return
      }

      json(res, 404, { ok: false, error: 'Not found' })
    } catch (error) {
      console.error(`[portal] ${route} failed`, error)
      if (!res.headersSent) json(res, 500, { ok: false, error: 'The mirror could not complete that.' })
    }
  })

  server.on('error', (error) => console.error('[portal] server error', error))
  // 0.0.0.0 is required: the phone reaches this over the hotspot, not over loopback. Scope is limited by
  // this server exposing nothing but the routes above, and by it running only during setup mode.
  server.listen(port, '0.0.0.0', () => console.log(`[portal] Wi-Fi setup portal listening on :${port}`))

  return {
    port,
    /** Bound address — the real port when `port: 0` was requested (tests use an ephemeral port). */
    address: () => server.address(),
    close: () => new Promise((resolve) => server.close(resolve)),
    getLastResult: () => lastResult,
  }
}

/**
 * The whole phone-facing UI: one self-contained HTML document.
 *
 * No build step, no bundler, no external asset — it must render on a phone that has NO internet (its only
 * network is the mirror's hotspot), so every byte is inline. Deliberately plain and large-touch-target;
 * the person holding the phone is a caregiver or installer, often in a hurry, in someone's home.
 */
function renderPortalPage() {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Connect the Reflexion mirror</title>
<style>
  :root { --ink:#282828; --muted:#686868; --cream:#FFF9F1; --sand:#F6F1E8; --line:rgba(118,94,62,0.18);
          --gold:#B98954; --sage:#637B5F; --coral:#C9786E; }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  body { margin:0; background:var(--cream); color:var(--ink); font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         padding:24px 18px 48px; max-width:520px; margin-inline:auto; }
  .eyebrow { color:var(--gold); font-size:12px; letter-spacing:2px; font-weight:600; }
  h1 { font-size:26px; line-height:1.25; margin:6px 0 4px; }
  p.lede { color:var(--muted); margin:0 0 20px; }
  .card { background:var(--sand); border:1px solid var(--line); border-radius:16px; padding:16px; margin-bottom:16px; }
  label { display:block; font-weight:600; font-size:14px; margin-bottom:6px; }
  select, input { width:100%; font-size:17px; padding:14px; border:1px solid var(--line); border-radius:12px;
                  background:#fff; color:var(--ink); margin-bottom:14px; }
  button { width:100%; font-size:17px; font-weight:600; padding:16px; border:0; border-radius:24px;
           background:var(--ink); color:var(--cream); }
  button[disabled] { opacity:.45; }
  button.secondary { background:#fff; color:var(--ink); border:1px solid var(--line); margin-top:10px; font-weight:500; }
  .row { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }
  .msg { border-radius:12px; padding:13px 14px; margin-bottom:14px; font-size:15px; display:none; }
  .msg.err { background:rgba(201,120,110,.13); border:1px solid var(--coral); color:#8d3f37; display:block; }
  .msg.ok  { background:rgba(171,197,161,.18); border:1px solid var(--sage); color:#3f5a3b; display:block; }
  .msg.info{ background:#fff; border:1px solid var(--line); color:var(--muted); display:block; }
  .hint { color:var(--muted); font-size:13px; margin:-6px 0 14px; }
  .pin { letter-spacing:8px; font-size:22px; text-align:center; font-variant-numeric:tabular-nums; }
  .sig { color:var(--muted); font-size:13px; }
</style>
</head>
<body>
  <div class="eyebrow">REFLEXION MIRROR</div>
  <h1>Connect the mirror to Wi-Fi</h1>
  <p class="lede">Pick the home Wi-Fi below and enter its password. Then type the 6-digit code shown on the mirror.</p>

  <div id="msg" class="msg"></div>

  <div class="card">
    <div class="row">
      <label for="ssid" style="margin:0">Wi-Fi network</label>
      <button class="secondary" id="rescan" style="width:auto; padding:8px 14px; margin:0; font-size:14px;">Refresh</button>
    </div>
    <select id="ssid"><option value="">Looking for networks…</option></select>

    <label for="password">Wi-Fi password</label>
    <input id="password" type="password" autocomplete="off" autocapitalize="off" spellcheck="false" placeholder="Password for that network">
    <div class="hint">Leave empty for an open network.</div>

    <label for="pin">Code shown on the mirror</label>
    <input id="pin" class="pin" type="text" inputmode="numeric" autocomplete="off" maxlength="6" placeholder="000000">

    <button id="submit">Connect the mirror</button>
  </div>

<script>
(function () {
  var el = function (id) { return document.getElementById(id) }
  var msg = el('msg')
  function show(kind, text) { msg.className = 'msg ' + kind; msg.textContent = text }
  function clear() { msg.className = 'msg'; msg.textContent = '' }

  function loadNetworks() {
    var select = el('ssid')
    select.innerHTML = '<option value="">Looking for networks…</option>'
    fetch('/api/networks').then(function (r) { return r.json() }).then(function (data) {
      if (!data.ok) { show('err', data.error || 'Could not look for networks.'); return }
      var list = data.networks || []
      if (!list.length) { select.innerHTML = '<option value="">No networks found — tap Refresh</option>'; return }
      select.innerHTML = '<option value="">Choose a network…</option>'
      list.forEach(function (n) {
        var option = document.createElement('option')
        option.value = n.ssid
        var bits = []
        if (n.saved) bits.push('saved')
        if (!n.secured) bits.push('open')
        if (n.band) bits.push(n.band)
        option.textContent = n.ssid + (bits.length ? '  (' + bits.join(', ') + ')' : '')
        option.dataset.secured = n.secured ? '1' : ''
        select.appendChild(option)
      })
    }).catch(function () { show('err', 'Lost contact with the mirror. Make sure the phone is still on the mirror’s setup network.') })
  }

  el('rescan').addEventListener('click', function (event) { event.preventDefault(); clear(); loadNetworks() })

  el('submit').addEventListener('click', function () {
    var ssid = el('ssid').value
    var password = el('password').value
    var pin = el('pin').value.trim()
    if (!ssid) { show('err', 'Choose a Wi-Fi network first.'); return }
    if (pin.length !== 6) { show('err', 'Enter the 6-digit code shown on the mirror.'); return }
    el('submit').disabled = true
    show('info', 'Sending to the mirror…')
    fetch('/api/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ssid: ssid, password: password, pin: pin })
    }).then(function (r) { return r.json().then(function (b) { return { status: r.status, body: b } }) })
      .then(function (result) {
      if (!result.body.ok) {
        show('err', result.body.error || 'That did not work.')
        el('submit').disabled = false
        return
      }
      // Expected: this setup network is about to disappear, so there is nothing more to wait for here.
      show('ok', result.body.message || 'Connecting the mirror — watch the mirror screen.')
      el('submit').textContent = 'Sent — check the mirror'
    }).catch(function () {
      // A dropped request right after submitting usually means the switch already began.
      show('ok', 'The mirror is switching networks. Watch the mirror screen for the result.')
    })
  })

  loadNetworks()
})()
</script>
</body>
</html>`
}

module.exports = { DEFAULT_PORTAL_PORT, generatePin, renderPortalPage, startSetupPortal }
