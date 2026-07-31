import { Platform } from 'react-native'
import * as Linking from 'expo-linking'

/**
 * Network setup for the mirror — the step BEFORE the app can do anything.
 *
 * A mirror is delivered to a home with no network configured. Until this existed, the boot flow could
 * only report the consequence ("unable to reach the Reflexion service") and the installer had no way to
 * fix it from the app: on the Linux appliance the screen is a kiosk with no desktop to escape to, and on
 * Android the settings app is behind a locked-down launcher.
 *
 * Two very different capabilities per platform, deliberately behind one interface:
 *
 * - **Linux (Electron)** — REAL control. The main process drives NetworkManager and BlueZ (see
 *   electron/network.js), so the app itself can scan and join Wi-Fi, start the mirror's own hotspot, and
 *   tether to a phone over Bluetooth PAN. This is the appliance path.
 * - **Android** — the OS does not let an app join a network or enable tethering on its behalf (it has
 *   been privileged since Android 10). The honest capability there is to OPEN the right system settings
 *   panel, so `openSystemSettings` is the whole Android surface and the UI adapts to it.
 */

export type NetworkCapabilities = {
  platform: 'linux-electron' | 'android' | 'unsupported'
  /** The app can scan for and join Wi-Fi networks itself. */
  wifi: boolean
  /** The app can make the mirror broadcast its own access point. */
  hotspot: boolean
  ethernet: boolean
  bluetooth: boolean
  /** The app can take internet FROM a paired phone over Bluetooth PAN. */
  bluetoothTethering: boolean
  /** True when the app can only hand off to the OS settings app (Android). */
  settingsOnly: boolean
}

export type WifiNetwork = {
  ssid: string
  /** 0-100. */
  signal: number
  secured: boolean
  security: string
  connected: boolean
  /** Already known to the device — rejoining needs no password. */
  saved: boolean
  band: string
}

export type NetworkInterface = { device: string; type: string; state: string; connection: string }

export type HotspotState = { active: boolean; ssid: string; password: string; device?: string }

export type NetworkStatus = {
  capabilities: NetworkCapabilities
  /** NetworkManager's verdict: 'full' | 'portal' | 'limited' | 'none' | 'unknown'. */
  connectivity: string
  /** True only for real internet — a Wi-Fi link with no route out is NOT online. */
  online: boolean
  managerState: string
  wifiRadio: boolean
  activeConnection: { type: string; name: string; device: string } | null
  interfaces: NetworkInterface[]
  hotspot: HotspotState
}

export type BluetoothDevice = { address: string; name: string; connected: boolean; tethering: boolean }

export type BluetoothStatus = {
  available: boolean
  powered: boolean
  discoverable: boolean
  pairable?: boolean
  name: string
  devices: BluetoothDevice[]
}

export type NetworkResult<T = unknown> = { ok: boolean; error?: string } & Partial<T>

/**
 * Setup mode: the mirror is broadcasting its own hotspot and serving a phone-facing Wi-Fi setup page.
 *
 * This exists because the Ubuntu unit has no keyboard and no mouse — the family cannot type on it. When the
 * mirror cannot reach the network it puts these details on its SCREEN, and the caregiver does the typing on
 * their phone. The screen is also the only place the RESULT can be shown: applying a network takes the radio
 * the hotspot runs on, so the phone's connection to the mirror dies at that exact moment.
 */
export type SetupModeState = {
  active: boolean
  /** Hotspot name to join. */
  ssid: string
  /** Hotspot passphrase, shown on the mirror so it can be typed into a phone. */
  password: string
  /** 6-digit code the portal requires — proves the person can actually see this mirror. */
  pin: string
  address: string
  portalUrl: string
  /** Outcome of the last apply, so a failed join can be reported after the hotspot returns. */
  lastResult: { ok: boolean; ssid: string; error?: string } | null
  reason: string
}

type NetworkBridge = {
  capabilities(): Promise<Omit<NetworkCapabilities, 'settingsOnly'>>
  status(): Promise<NetworkStatus>
  wifiScan(options?: { rescan?: boolean }): Promise<NetworkResult<{ networks: WifiNetwork[] }>>
  wifiConnect(options: { ssid: string; password?: string; hidden?: boolean }): Promise<NetworkResult<{ status: NetworkStatus }>>
  wifiConnectSaved(options: { ssid: string }): Promise<NetworkResult<{ status: NetworkStatus }>>
  wifiForget(options: { ssid: string }): Promise<NetworkResult<{ status: NetworkStatus }>>
  wifiSetRadio(options: { enabled: boolean }): Promise<NetworkResult>
  hotspotStart(options?: { ssid?: string; passphrase?: string }): Promise<NetworkResult<{ hotspot: HotspotState; ssid: string; passphrase: string }>>
  hotspotStop(): Promise<NetworkResult<{ status: NetworkStatus }>>
  bluetoothStatus(): Promise<BluetoothStatus>
  bluetoothSetPower(options: { enabled: boolean }): Promise<NetworkResult>
  bluetoothScan(options?: { seconds?: number }): Promise<NetworkResult<{ devices: BluetoothDevice[] }>>
  bluetoothPair(options: { address: string }): Promise<NetworkResult<{ bluetooth: BluetoothStatus }>>
  bluetoothTether(options: { address: string }): Promise<NetworkResult<{ status: NetworkStatus }>>
  bluetoothDisconnect(options: { address: string }): Promise<NetworkResult<{ status: NetworkStatus }>>
  setupState(): Promise<SetupModeState>
  setupStart(): Promise<SetupModeState>
  setupStop(): Promise<SetupModeState>
  onSetupState(listener: (state: SetupModeState) => void): () => void
}

function bridge(): NetworkBridge | null {
  if (Platform.OS !== 'web' || typeof window === 'undefined') return null
  return (window as unknown as { reflexionNetwork?: NetworkBridge }).reflexionNetwork ?? null
}

export function networkSetupAvailable() {
  return Boolean(bridge()) || Platform.OS === 'android'
}

const ANDROID_CAPABILITIES: NetworkCapabilities = {
  platform: 'android',
  wifi: false,
  hotspot: false,
  ethernet: false,
  bluetooth: false,
  bluetoothTethering: false,
  settingsOnly: true,
}

const UNSUPPORTED_CAPABILITIES: NetworkCapabilities = {
  platform: 'unsupported',
  wifi: false,
  hotspot: false,
  ethernet: false,
  bluetooth: false,
  bluetoothTethering: false,
  settingsOnly: false,
}

export async function getNetworkCapabilities(): Promise<NetworkCapabilities> {
  const api = bridge()
  if (!api) return Platform.OS === 'android' ? ANDROID_CAPABILITIES : UNSUPPORTED_CAPABILITIES
  try {
    return { ...(await api.capabilities()), settingsOnly: false }
  } catch {
    return UNSUPPORTED_CAPABILITIES
  }
}

export async function getNetworkStatus(): Promise<NetworkStatus | null> {
  const api = bridge()
  if (!api) return null
  try {
    return await api.status()
  } catch {
    return null
  }
}

/** Every mutating call funnels through here so a dead bridge never throws into a render. */
async function call<T>(operation: (api: NetworkBridge) => Promise<NetworkResult<T>>, unavailable: string): Promise<NetworkResult<T>> {
  // A failure carries no payload fields, so it satisfies the caller's `Partial<T>` vacuously — but TS
  // cannot see that for an unresolved T, hence the cast on the two failure paths only.
  const failure = (error: string) => ({ ok: false, error } as NetworkResult<T>)
  const api = bridge()
  if (!api) return failure(unavailable)
  try {
    return await operation(api)
  } catch (error) {
    return failure(error instanceof Error ? error.message : 'That did not work. Please try again.')
  }
}

const NO_WIFI = 'This mirror cannot change Wi-Fi settings from the app.'
const NO_BLUETOOTH = 'This mirror cannot change Bluetooth settings from the app.'

export function scanWifi(options: { rescan?: boolean } = {}) {
  return call((api) => api.wifiScan(options), NO_WIFI)
}

export function connectWifi(options: { ssid: string; password?: string; hidden?: boolean }) {
  return call((api) => api.wifiConnect(options), NO_WIFI)
}

export function connectSavedWifi(ssid: string) {
  return call((api) => api.wifiConnectSaved({ ssid }), NO_WIFI)
}

export function forgetWifi(ssid: string) {
  return call((api) => api.wifiForget({ ssid }), NO_WIFI)
}

export function setWifiRadio(enabled: boolean) {
  return call((api) => api.wifiSetRadio({ enabled }), NO_WIFI)
}

export function startHotspot(options: { ssid?: string; passphrase?: string } = {}) {
  return call((api) => api.hotspotStart(options), 'This mirror cannot create a hotspot.')
}

export function stopHotspot() {
  return call((api) => api.hotspotStop(), 'This mirror cannot create a hotspot.')
}

export async function getBluetoothStatus(): Promise<BluetoothStatus | null> {
  const api = bridge()
  if (!api) return null
  try {
    return await api.bluetoothStatus()
  } catch {
    return null
  }
}

export function setBluetoothPower(enabled: boolean) {
  return call((api) => api.bluetoothSetPower({ enabled }), NO_BLUETOOTH)
}

export function scanBluetooth(seconds?: number) {
  return call((api) => api.bluetoothScan({ seconds }), NO_BLUETOOTH)
}

export function pairBluetooth(address: string) {
  return call((api) => api.bluetoothPair({ address }), NO_BLUETOOTH)
}

/** Take internet from an already-paired phone. The phone must have Bluetooth tethering switched on. */
export function tetherBluetooth(address: string) {
  return call((api) => api.bluetoothTether({ address }), NO_BLUETOOTH)
}

export function disconnectBluetooth(address: string) {
  return call((api) => api.bluetoothDisconnect({ address }), NO_BLUETOOTH)
}

export async function getSetupModeState(): Promise<SetupModeState | null> {
  const api = bridge()
  if (!api?.setupState) return null
  try {
    return await api.setupState()
  } catch {
    return null
  }
}

/** Bring up the setup hotspot + phone portal on demand (the watcher also does this automatically). */
export function startSetupMode() {
  return call((api) => api.setupStart().then((state) => ({ ok: true, state })), 'This mirror cannot start setup mode.')
}

export function stopSetupMode() {
  return call((api) => api.setupStop().then((state) => ({ ok: true, state })), 'This mirror cannot start setup mode.')
}

/** Subscribe to setup-state pushes. Returns an unsubscribe function (a no-op off-Electron). */
export function subscribeSetupMode(listener: (state: SetupModeState) => void) {
  const api = bridge()
  if (!api?.onSetupState) return () => {}
  try {
    return api.onSetupState(listener)
  } catch {
    return () => {}
  }
}

export type SystemSettingsPanel = 'wifi' | 'hotspot' | 'bluetooth'

// Android cannot let an app join a network or enable tethering itself, so the honest action is to open the
// OS panel. `Linking.sendIntent` avoids adding expo-intent-launcher just for three intents.
const ANDROID_SETTINGS_INTENT: Record<SystemSettingsPanel, string> = {
  wifi: 'android.settings.WIFI_SETTINGS',
  hotspot: 'android.settings.TETHER_SETTINGS',
  bluetooth: 'android.settings.BLUETOOTH_SETTINGS',
}

export async function openSystemSettings(panel: SystemSettingsPanel): Promise<NetworkResult> {
  if (Platform.OS !== 'android') return { ok: false, error: 'System settings can only be opened on the Android mirror.' }
  try {
    await Linking.sendIntent(ANDROID_SETTINGS_INTENT[panel])
    return { ok: true }
  } catch {
    try {
      // TETHER_SETTINGS is missing on some vendor ROMs; the general settings app always exists.
      await Linking.sendIntent('android.settings.SETTINGS')
      return { ok: true }
    } catch {
      return { ok: false, error: 'Could not open the system settings on this mirror.' }
    }
  }
}
