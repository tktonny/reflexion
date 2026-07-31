// Main-process network control for the Linux (Ubuntu) mirror appliance.
//
// WHY THIS EXISTS: an Ubuntu mirror unit is delivered to a home with no network configured, and the
// renderer is a sandboxed Chromium page — it cannot join a Wi-Fi network, start a hotspot, or pair a
// phone over Bluetooth. So the first thing an installer needs to do is the one thing the app could not
// do, and the unit just sat on "unable to reach the Reflexion service" forever. Everything here is the
// privileged half of that: the renderer asks over IPC, this module drives NetworkManager / BlueZ.
//
// SAFETY: every call goes through execFile with an ARGUMENT ARRAY — never a shell string — because SSIDs
// and Wi-Fi passwords are attacker-adjacent free text typed on a touch keyboard (`"; rm -rf ~` is a
// legal Wi-Fi password). No value below is ever interpolated into a shell.
//
// Secrets: Wi-Fi passwords are passed as argv to nmcli and never logged. `logArgs()` redacts them before
// anything reaches stdout, because the Electron log on a mirror is not a secret store.

const { execFile } = require('child_process')

const NMCLI = 'nmcli'
const BLUETOOTHCTL = 'bluetoothctl'
const DEFAULT_TIMEOUT_MS = 20_000
// Joining a network (DHCP + DNS) is genuinely slow on a weak home router; nmcli's own default wait is 90s.
const CONNECT_TIMEOUT_MS = 75_000

/** Arguments we must never print. Everything after these nmcli keywords is a secret. */
const SECRET_AFTER = new Set(['password', 'wifi-sec.psk', '802-11-wireless-security.psk'])

function logArgs(args) {
  const out = []
  let redactNext = false
  for (const arg of args) {
    if (redactNext) { out.push('***'); redactNext = false; continue }
    out.push(arg)
    if (SECRET_AFTER.has(arg)) redactNext = true
  }
  return out.join(' ')
}

function run(command, args, { timeout = DEFAULT_TIMEOUT_MS } = {}) {
  return new Promise((resolve) => {
    execFile(command, args, { timeout, maxBuffer: 4 * 1024 * 1024 }, (error, stdout, stderr) => {
      const failed = Boolean(error)
      if (failed) console.warn(`[network] ${command} ${logArgs(args)} -> ${String(stderr || error.message).trim()}`)
      resolve({
        ok: !failed,
        stdout: String(stdout || ''),
        stderr: String(stderr || (error ? error.message : '')),
        // ENOENT means the tool is not installed at all — a different problem from a failed command.
        missing: Boolean(error && error.code === 'ENOENT'),
      })
    })
  })
}

/** nmcli -t output is colon-separated with `\:` escaping inside fields (SSIDs contain colons). */
function splitTerse(line) {
  const fields = []
  let current = ''
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index]
    if (char === '\\' && line[index + 1] === ':') { current += ':'; index += 1; continue }
    if (char === '\\' && line[index + 1] === '\\') { current += '\\'; index += 1; continue }
    if (char === ':') { fields.push(current); current = ''; continue }
    current += char
  }
  fields.push(current)
  return fields
}

function lines(stdout) {
  return stdout.split('\n').map((line) => line.trim()).filter(Boolean)
}

/** Turn a failed nmcli/bluetoothctl run into something an installer standing at the mirror can act on. */
function friendlyError(result, fallback) {
  if (result.missing) return 'This mirror is missing its network tools (NetworkManager / BlueZ).'
  const text = (result.stderr || '').toLowerCase()
  if (text.includes('not authorized') || text.includes('permission denied')) {
    return 'This mirror is not allowed to change network settings. An installer needs to grant NetworkManager permission.'
  }
  if (text.includes('secrets were required') || text.includes('no key available') || text.includes('802-1x') || text.includes('invalid password')) {
    return 'That password was not accepted. Please check it and try again.'
  }
  if (text.includes('timeout') || text.includes('timed out')) return 'The connection attempt timed out. Move closer to the router and try again.'
  if (text.includes('no network with ssid')) return 'That network is no longer in range.'
  return fallback
}

// ---------------------------------------------------------------------------
// Capability probe
// ---------------------------------------------------------------------------

let capabilitiesCache = null

async function capabilities() {
  if (capabilitiesCache) return capabilitiesCache
  const [nm, bt] = await Promise.all([
    run(NMCLI, ['--version'], { timeout: 5_000 }),
    run(BLUETOOTHCTL, ['--version'], { timeout: 5_000 }),
  ])
  const wifiDevices = nm.ok ? await run(NMCLI, ['-t', '-f', 'DEVICE,TYPE', 'device', 'status']) : null
  const hasWifiHardware = Boolean(wifiDevices?.ok && lines(wifiDevices.stdout).some((line) => splitTerse(line)[1] === 'wifi'))
  capabilitiesCache = {
    platform: 'linux-electron',
    wifi: nm.ok && hasWifiHardware,
    // A hotspot needs the same radio as Wi-Fi. `nmcli device wifi hotspot` fails on adapters without AP
    // mode, but that only becomes knowable when it is attempted, so treat it as available with Wi-Fi.
    hotspot: nm.ok && hasWifiHardware,
    ethernet: nm.ok,
    bluetooth: nm.ok && bt.ok,
    // Bluetooth tethering (PAN) needs BOTH BlueZ (pair) and NetworkManager (bring up the panu link).
    bluetoothTethering: nm.ok && bt.ok,
  }
  return capabilitiesCache
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async function status() {
  const caps = await capabilities()
  const [general, devices, radio] = await Promise.all([
    run(NMCLI, ['-t', '-f', 'STATE,CONNECTIVITY', 'general', 'status']),
    run(NMCLI, ['-t', '-f', 'DEVICE,TYPE,STATE,CONNECTION', 'device', 'status']),
    run(NMCLI, ['-t', '-f', 'WIFI', 'radio']),
  ])
  const [state = '', connectivity = ''] = general.ok ? splitTerse(lines(general.stdout)[0] || '') : []
  const interfaces = devices.ok
    ? lines(devices.stdout).map((line) => {
        const [device, type, deviceState, connection] = splitTerse(line)
        return { device, type, state: deviceState, connection: connection === '--' ? '' : connection }
      }).filter((entry) => entry.type !== 'loopback')
    : []
  const active = interfaces.find((entry) => entry.state === 'connected' && entry.type !== 'loopback')
  return {
    capabilities: caps,
    // NetworkManager's own verdict: 'full' means real internet, 'portal'/'limited' means a link but no
    // route out — the distinction the mirror's "cannot reach the service" screen could never make.
    connectivity: connectivity || 'unknown',
    online: connectivity === 'full',
    managerState: state || 'unknown',
    wifiRadio: radio.ok ? lines(radio.stdout)[0] === 'enabled' : false,
    activeConnection: active ? { type: active.type, name: active.connection, device: active.device } : null,
    interfaces,
    hotspot: await hotspotStatus(),
  }
}

// ---------------------------------------------------------------------------
// Wi-Fi
// ---------------------------------------------------------------------------

async function wifiSetRadio(enabled) {
  const result = await run(NMCLI, ['radio', 'wifi', enabled ? 'on' : 'off'])
  if (!result.ok) return { ok: false, error: friendlyError(result, 'Could not switch the Wi-Fi radio.') }
  return { ok: true }
}

async function wifiScan({ rescan = true } = {}) {
  const caps = await capabilities()
  if (!caps.wifi) return { ok: false, error: 'This mirror has no Wi-Fi adapter.', networks: [] }
  const radio = await run(NMCLI, ['-t', '-f', 'WIFI', 'radio'])
  if (radio.ok && lines(radio.stdout)[0] === 'disabled') {
    const enabled = await wifiSetRadio(true)
    if (!enabled.ok) return { ...enabled, networks: [] }
  }
  const [list, saved] = await Promise.all([
    // --rescan yes forces a fresh sweep: the cached list is stale on a unit that was just moved into a
    // home, which is exactly when this screen is used.
    run(NMCLI, ['-t', '-f', 'IN-USE,SSID,SIGNAL,SECURITY,FREQ', 'device', 'wifi', 'list', '--rescan', rescan ? 'yes' : 'no'], { timeout: 45_000 }),
    run(NMCLI, ['-t', '-f', 'NAME,TYPE', 'connection', 'show']),
  ])
  if (!list.ok) return { ok: false, error: friendlyError(list, 'Could not look for Wi-Fi networks.'), networks: [] }
  const savedNames = new Set(saved.ok
    ? lines(saved.stdout).map(splitTerse).filter(([, type]) => type === '802-11-wireless').map(([name]) => name)
    : [])

  const bySsid = new Map()
  for (const line of lines(list.stdout)) {
    const [inUse, ssid, signal, security, freq] = splitTerse(line)
    if (!ssid) continue // hidden networks cannot be joined from a list, so don't show empty rows
    const entry = {
      ssid,
      signal: Number(signal) || 0,
      secured: Boolean(security && security !== '--'),
      security: security === '--' ? '' : security || '',
      connected: inUse === '*',
      saved: savedNames.has(ssid),
      band: Number(freq) >= 5000 ? '5 GHz' : '2.4 GHz',
    }
    // The same SSID appears once per BSSID/band; keep the strongest so the list reads like a phone's.
    const existing = bySsid.get(ssid)
    if (!existing || entry.signal > existing.signal || entry.connected) bySsid.set(ssid, { ...entry, connected: entry.connected || existing?.connected || false })
  }
  const networks = [...bySsid.values()].sort((a, b) => (Number(b.connected) - Number(a.connected)) || (b.signal - a.signal))
  return { ok: true, networks }
}

async function wifiConnect({ ssid, password = '', hidden = false }) {
  if (typeof ssid !== 'string' || !ssid.trim()) return { ok: false, error: 'Choose a network first.' }
  const caps = await capabilities()
  if (!caps.wifi) return { ok: false, error: 'This mirror has no Wi-Fi adapter.' }

  // A previously saved network with a NEW password would silently reconnect with the stale secret and
  // fail, so an explicit password always wins: drop the saved profile and join fresh.
  if (password) await run(NMCLI, ['connection', 'delete', 'id', ssid], { timeout: 10_000 })

  const args = ['device', 'wifi', 'connect', ssid]
  if (password) args.push('password', password)
  if (hidden) args.push('hidden', 'yes')
  const result = await run(NMCLI, args, { timeout: CONNECT_TIMEOUT_MS })
  if (!result.ok) {
    // Don't leave a half-created profile behind that would auto-retry a wrong password on every boot.
    if (password) await run(NMCLI, ['connection', 'delete', 'id', ssid], { timeout: 10_000 })
    return { ok: false, error: friendlyError(result, `Could not join ${ssid}.`) }
  }
  return { ok: true, status: await status() }
}

/** Rejoin an already-saved network (no password needed — NetworkManager still holds the secret). */
async function wifiConnectSaved({ ssid }) {
  if (typeof ssid !== 'string' || !ssid.trim()) return { ok: false, error: 'Choose a network first.' }
  const result = await run(NMCLI, ['connection', 'up', 'id', ssid], { timeout: CONNECT_TIMEOUT_MS })
  if (!result.ok) return { ok: false, error: friendlyError(result, `Could not reconnect to ${ssid}.`) }
  return { ok: true, status: await status() }
}

async function wifiForget({ ssid }) {
  if (typeof ssid !== 'string' || !ssid.trim()) return { ok: false, error: 'Choose a network first.' }
  const result = await run(NMCLI, ['connection', 'delete', 'id', ssid])
  if (!result.ok) return { ok: false, error: friendlyError(result, `Could not forget ${ssid}.`) }
  return { ok: true, status: await status() }
}

// ---------------------------------------------------------------------------
// Hotspot
//
// This is the mirror BROADCASTING its own access point. It is a setup aid, not an internet source: an
// installer joins it from a phone or laptop to reach the unit when there is no shared network yet. It
// takes over the Wi-Fi radio, so a mirror that is online over Wi-Fi will go offline while it is on —
// the UI says so before starting it.
// ---------------------------------------------------------------------------

const HOTSPOT_CONNECTION = 'reflexion-mirror-hotspot'

async function hotspotStatus() {
  const active = await run(NMCLI, ['-t', '-f', 'NAME,TYPE,DEVICE', 'connection', 'show', '--active'])
  if (!active.ok) return { active: false, ssid: '', password: '' }
  const row = lines(active.stdout).map(splitTerse).find(([name]) => name === HOTSPOT_CONNECTION)
  if (!row) return { active: false, ssid: '', password: '' }
  const [ssid, psk] = await Promise.all([
    run(NMCLI, ['-g', '802-11-wireless.ssid', 'connection', 'show', HOTSPOT_CONNECTION]),
    // -s is required for nmcli to print secrets at all.
    run(NMCLI, ['-s', '-g', '802-11-wireless-security.psk', 'connection', 'show', HOTSPOT_CONNECTION]),
  ])
  return {
    active: true,
    ssid: ssid.ok ? lines(ssid.stdout)[0] || '' : '',
    password: psk.ok ? lines(psk.stdout)[0] || '' : '',
    device: row[2] || '',
  }
}

async function hotspotStart({ ssid = '', password = '' } = {}) {
  const caps = await capabilities()
  if (!caps.hotspot) return { ok: false, error: 'This mirror has no Wi-Fi adapter, so it cannot create a hotspot.' }
  const finalSsid = (ssid || 'Reflexion-Mirror').slice(0, 32)
  // WPA2 requires >= 8 characters; a short password fails with an unhelpful nmcli error.
  const finalPassword = password && password.length >= 8 ? password : randomPassphrase()

  await run(NMCLI, ['connection', 'delete', 'id', HOTSPOT_CONNECTION], { timeout: 10_000 })
  const created = await run(NMCLI, [
    'device', 'wifi', 'hotspot',
    'con-name', HOTSPOT_CONNECTION,
    'ssid', finalSsid,
    'password', finalPassword,
  ], { timeout: 40_000 })
  if (!created.ok) return { ok: false, error: friendlyError(created, 'Could not start the hotspot. This mirror’s Wi-Fi adapter may not support it.') }
  return { ok: true, hotspot: await hotspotStatus(), password: finalPassword, ssid: finalSsid }
}

async function hotspotStop() {
  const down = await run(NMCLI, ['connection', 'down', 'id', HOTSPOT_CONNECTION], { timeout: 20_000 })
  await run(NMCLI, ['connection', 'delete', 'id', HOTSPOT_CONNECTION], { timeout: 10_000 })
  if (!down.ok && !down.stderr.toLowerCase().includes('not an active')) {
    return { ok: false, error: friendlyError(down, 'Could not stop the hotspot.') }
  }
  // Dropping the AP frees the radio; ask NetworkManager to fall back to a known network immediately
  // rather than waiting for its own retry timer.
  await run(NMCLI, ['device', 'connect', await firstWifiDevice()], { timeout: CONNECT_TIMEOUT_MS })
  return { ok: true, status: await status() }
}

async function firstWifiDevice() {
  const devices = await run(NMCLI, ['-t', '-f', 'DEVICE,TYPE', 'device', 'status'])
  if (!devices.ok) return 'wlan0'
  const row = lines(devices.stdout).map(splitTerse).find(([, type]) => type === 'wifi')
  return row?.[0] || 'wlan0'
}

/** Readable-out-loud passphrase: an installer reads this off the mirror and types it into a phone. */
function randomPassphrase() {
  const alphabet = 'abcdefghjkmnpqrstuvwxyz23456789' // no look-alikes (0/o, 1/l/i)
  const bytes = require('crypto').randomBytes(10)
  return [...bytes].map((byte) => alphabet[byte % alphabet.length]).join('')
}

// ---------------------------------------------------------------------------
// Bluetooth
//
// Two separate jobs, both asked for by installers:
//   1. make the mirror pairable so a phone can find it (bluetoothctl / BlueZ);
//   2. tether — take internet FROM a paired phone over Bluetooth PAN (NetworkManager `bt-type panu`),
//      which is the fallback when a home has no usable Wi-Fi at all.
// ---------------------------------------------------------------------------

async function bluetoothSetPower(enabled) {
  const power = await run(BLUETOOTHCTL, ['power', enabled ? 'on' : 'off'], { timeout: 10_000 })
  if (!power.ok) return { ok: false, error: friendlyError(power, 'Could not switch Bluetooth.') }
  if (enabled) {
    // Discoverable + pairable is what makes the mirror show up in a phone's Bluetooth list at all.
    await run(BLUETOOTHCTL, ['pairable', 'on'], { timeout: 10_000 })
    await run(BLUETOOTHCTL, ['discoverable', 'on'], { timeout: 10_000 })
  }
  return { ok: true }
}

async function bluetoothStatus() {
  const caps = await capabilities()
  if (!caps.bluetooth) return { available: false, powered: false, discoverable: false, name: '', devices: [] }
  const show = await run(BLUETOOTHCTL, ['show'], { timeout: 10_000 })
  const text = show.ok ? show.stdout : ''
  const field = (label) => new RegExp(`^\\s*${label}:\\s*(.+)$`, 'm').exec(text)?.[1]?.trim() || ''
  return {
    available: true,
    powered: field('Powered') === 'yes',
    discoverable: field('Discoverable') === 'yes',
    pairable: field('Pairable') === 'yes',
    name: field('Alias') || field('Name'),
    devices: await bluetoothDevices(),
  }
}

async function bluetoothDevices() {
  const [known, connected, tethers] = await Promise.all([
    run(BLUETOOTHCTL, ['devices'], { timeout: 10_000 }),
    run(BLUETOOTHCTL, ['devices', 'Connected'], { timeout: 10_000 }),
    run(NMCLI, ['-t', '-f', 'NAME,TYPE', 'connection', 'show', '--active']),
  ])
  const parse = (result) => (result.ok ? lines(result.stdout) : [])
    .map((line) => /^Device\s+([0-9A-F:]{17})\s+(.*)$/i.exec(line))
    .filter(Boolean)
    .map((match) => ({ address: match[1].toUpperCase(), name: match[2].trim() }))
  const connectedSet = new Set(parse(connected).map((device) => device.address))
  const activeTether = new Set((tethers.ok ? lines(tethers.stdout) : [])
    .map(splitTerse).filter(([, type]) => type === 'bluetooth').map(([name]) => name))
  return parse(known).map((device) => ({
    ...device,
    connected: connectedSet.has(device.address),
    tethering: activeTether.has(tetherConnectionName(device.address)),
  }))
}

/** Discovery has to be time-boxed: `scan on` never returns, so run it under --timeout and then read. */
async function bluetoothScan({ seconds = 12 } = {}) {
  const caps = await capabilities()
  if (!caps.bluetooth) return { ok: false, error: 'This mirror has no Bluetooth adapter.', devices: [] }
  const powered = await bluetoothSetPower(true)
  if (!powered.ok) return { ...powered, devices: [] }
  const bounded = Math.min(Math.max(Number(seconds) || 12, 3), 30)
  await run(BLUETOOTHCTL, ['--timeout', String(bounded), 'scan', 'on'], { timeout: (bounded + 10) * 1000 })
  return { ok: true, devices: await bluetoothDevices() }
}

const MAC_PATTERN = /^[0-9A-F]{2}(:[0-9A-F]{2}){5}$/i

function tetherConnectionName(address) {
  return `reflexion-bt-${address.replace(/:/g, '').toLowerCase()}`
}

async function bluetoothPair({ address }) {
  if (typeof address !== 'string' || !MAC_PATTERN.test(address)) return { ok: false, error: 'That is not a valid Bluetooth address.' }
  const powered = await bluetoothSetPower(true)
  if (!powered.ok) return powered
  const paired = await run(BLUETOOTHCTL, ['pair', address], { timeout: 45_000 })
  // Already-paired is success, not failure — an installer who taps twice should not see an error.
  const alreadyPaired = paired.stdout.toLowerCase().includes('already') || paired.stderr.toLowerCase().includes('already')
  if (!paired.ok && !alreadyPaired) {
    return {
      ok: false,
      error: friendlyError(paired, 'Pairing did not complete. Accept the pairing request on the phone, then try again.'),
    }
  }
  await run(BLUETOOTHCTL, ['trust', address], { timeout: 10_000 })
  await run(BLUETOOTHCTL, ['connect', address], { timeout: 30_000 })
  return { ok: true, bluetooth: await bluetoothStatus() }
}

/**
 * Take internet from a paired phone over Bluetooth PAN.
 * The phone must have Bluetooth tethering switched ON in its own settings — nothing on this side can
 * enable that, so a failure here is reported as "turn on Bluetooth tethering on the phone".
 */
async function bluetoothTether({ address }) {
  if (typeof address !== 'string' || !MAC_PATTERN.test(address)) return { ok: false, error: 'That is not a valid Bluetooth address.' }
  const caps = await capabilities()
  if (!caps.bluetoothTethering) return { ok: false, error: 'This mirror cannot use Bluetooth internet sharing.' }
  const name = tetherConnectionName(address)

  await run(BLUETOOTHCTL, ['connect', address], { timeout: 30_000 })
  const existing = await run(NMCLI, ['-t', '-f', 'NAME', 'connection', 'show'])
  const alreadyDefined = existing.ok && lines(existing.stdout).map(splitTerse).some(([entry]) => entry === name)
  if (!alreadyDefined) {
    const added = await run(NMCLI, [
      'connection', 'add',
      'type', 'bluetooth',
      'bt-type', 'panu',
      'con-name', name,
      'bdaddr', address.toUpperCase(),
    ], { timeout: 20_000 })
    if (!added.ok) return { ok: false, error: friendlyError(added, 'Could not set up Bluetooth internet sharing.') }
  }
  const up = await run(NMCLI, ['connection', 'up', name], { timeout: CONNECT_TIMEOUT_MS })
  if (!up.ok) {
    return {
      ok: false,
      error: friendlyError(up, 'The phone did not share its internet. Turn on Bluetooth tethering on the phone, then try again.'),
    }
  }
  return { ok: true, status: await status() }
}

async function bluetoothDisconnect({ address }) {
  if (typeof address !== 'string' || !MAC_PATTERN.test(address)) return { ok: false, error: 'That is not a valid Bluetooth address.' }
  await run(NMCLI, ['connection', 'down', tetherConnectionName(address)], { timeout: 20_000 })
  await run(BLUETOOTHCTL, ['disconnect', address], { timeout: 20_000 })
  return { ok: true, status: await status() }
}

module.exports = {
  capabilities,
  status,
  // Exported for electron/network.test.js: the terse-output parser and the error translator are pure and
  // are where a silent bug hides whole Wi-Fi networks from the list.
  __testables: { splitTerse, friendlyError, tetherConnectionName, randomPassphrase },
  wifiScan,
  wifiConnect,
  wifiConnectSaved,
  wifiForget,
  wifiSetRadio,
  hotspotStart,
  hotspotStop,
  hotspotStatus,
  bluetoothStatus,
  bluetoothScan,
  bluetoothSetPower,
  bluetoothPair,
  bluetoothTether,
  bluetoothDisconnect,
}
