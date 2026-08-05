import { Ionicons } from '@expo/vector-icons'
import { router } from 'expo-router'
import { useCallback, useEffect, useRef, useState } from 'react'
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

import { OnScreenKeyboard } from '../src/components/mirror/OnScreenKeyboard'
import {
  connectSavedWifi,
  connectWifi,
  disconnectBluetooth,
  forgetWifi,
  getBluetoothStatus,
  getNetworkCapabilities,
  getNetworkStatus,
  openSystemSettings,
  pairBluetooth,
  scanBluetooth,
  scanWifi,
  setBluetoothPower,
  startHotspot,
  stopHotspot,
  tetherBluetooth,
  type BluetoothDevice,
  type BluetoothStatus,
  type HotspotState,
  type NetworkCapabilities,
  type NetworkStatus,
  type WifiNetwork,
} from '../src/native/networkSetup'
import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'

/**
 * "Connect this mirror" — the setup step that has to happen before anything else can work.
 *
 * A mirror unit is installed in a home with no network configured, and until this screen existed the app
 * could only state the consequence ("unable to reach the Reflexion service") with no way to fix it: the
 * Linux appliance runs fullscreen kiosk with no desktop to escape to. Three ways in, because homes differ:
 *
 *   Wi-Fi      — join the home router, or a phone's personal hotspot. The normal path.
 *   Bluetooth  — take internet from a paired phone over Bluetooth PAN, when there is no usable Wi-Fi.
 *   Hotspot    — the mirror broadcasts its OWN network so an installer can reach the unit. NOT internet;
 *                the screen says so, because switching it on takes the Wi-Fi radio and drops the mirror
 *                offline, which would otherwise look like a new fault.
 *
 * This is an installer/caregiver surface, not an elder one: it is denser than the check-in screens and
 * uses plain network words (SSID-level detail stays out, but "Wi-Fi" and "Bluetooth" are named). It stays
 * reachable from the boot screen and from settings, and never blocks a check-in.
 */

type Tab = 'wifi' | 'bluetooth' | 'hotspot'

export default function NetworkSetupScreen() {
  const [capabilities, setCapabilities] = useState<NetworkCapabilities | null>(null)
  const [status, setStatus] = useState<NetworkStatus | null>(null)
  const [tab, setTab] = useState<Tab>('wifi')
  const [busy, setBusy] = useState('')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const [networks, setNetworks] = useState<WifiNetwork[]>([])
  const [selected, setSelected] = useState<WifiNetwork | null>(null)
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)

  const [bluetooth, setBluetooth] = useState<BluetoothStatus | null>(null)

  // Guard every async completion: an installer can leave this screen mid-scan (a Wi-Fi scan is ~30s).
  const mounted = useRef(true)
  useEffect(() => () => { mounted.current = false }, [])

  const refreshStatus = useCallback(async () => {
    const [nextStatus, nextBluetooth] = await Promise.all([getNetworkStatus(), getBluetoothStatus()])
    if (!mounted.current) return
    if (nextStatus) setStatus(nextStatus)
    if (nextBluetooth) setBluetooth(nextBluetooth)
  }, [])

  useEffect(() => {
    void (async () => {
      const caps = await getNetworkCapabilities()
      if (!mounted.current) return
      setCapabilities(caps)
      if (caps.settingsOnly) return
      await refreshStatus()
      if (caps.wifi) await runWifiScan({ rescan: false })
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // While this screen is open the mirror is actively being connected, so poll: the moment NetworkManager
  // reports real internet, the screen can say so without the installer guessing.
  useEffect(() => {
    if (capabilities?.settingsOnly || !capabilities) return
    const interval = setInterval(() => { if (!busy) void refreshStatus() }, 5000)
    return () => clearInterval(interval)
  }, [busy, capabilities, refreshStatus])

  async function withBusy(key: string, operation: () => Promise<{ ok: boolean; error?: string }>, success: string) {
    setBusy(key)
    setError('')
    setMessage('')
    const result = await operation()
    if (!mounted.current) return result
    setBusy('')
    if (result.ok) setMessage(success)
    else setError(result.error || 'That did not work. Please try again.')
    await refreshStatus()
    return result
  }

  async function runWifiScan({ rescan = true } = {}) {
    setBusy('scan')
    setError('')
    const result = await scanWifi({ rescan })
    if (!mounted.current) return
    setBusy('')
    if (result.ok) setNetworks(result.networks || [])
    else setError(result.error || 'Could not look for Wi-Fi networks.')
  }

  async function joinSelected() {
    if (!selected) return
    const target = selected
    const result = await withBusy('connect', () => (
      target.secured ? connectWifi({ ssid: target.ssid, password }) : connectWifi({ ssid: target.ssid })
    ), `Connected to ${target.ssid}.`)
    if (result.ok && mounted.current) {
      setSelected(null)
      setPassword('')
      await runWifiScan({ rescan: false })
    }
  }

  async function tapNetwork(network: WifiNetwork) {
    setError('')
    setMessage('')
    if (network.connected) return
    // A saved network already holds its secret — asking for the password again would be theatre.
    if (network.saved && !network.secured) { await withBusy('connect', () => connectSavedWifi(network.ssid), `Connected to ${network.ssid}.`); return }
    if (network.saved) {
      const result = await withBusy('connect', () => connectSavedWifi(network.ssid), `Connected to ${network.ssid}.`)
      // The saved password can be stale (router changed); fall through to asking for a new one.
      if (!result.ok && mounted.current) { setSelected(network); setPassword(''); setError('That saved password no longer works. Please enter the Wi-Fi password.') }
      return
    }
    if (!network.secured) { await withBusy('connect', () => connectWifi({ ssid: network.ssid }), `Connected to ${network.ssid}.`); return }
    setSelected(network)
    setPassword('')
    setShowPassword(false)
  }

  if (!capabilities) return <Shell><CenterNote>Checking this mirror’s network options…</CenterNote></Shell>

  if (capabilities.platform === 'unsupported') {
    return (
      <Shell>
        <CenterNote>
          Network setup is not available in this build. Connect the mirror using the device’s own settings,
          then return to this screen.
        </CenterNote>
      </Shell>
    )
  }

  // Android: the OS reserves joining networks and tethering for privileged apps, so the honest offer is to
  // open the right settings panel rather than to pretend the app can do it.
  if (capabilities.settingsOnly) {
    return (
      <Shell>
        <Header online={false} status={null} />
        <Text style={styles.sectionBody}>
          This mirror uses the Android settings app to connect. Open the panel you need, connect, then come
          back — the mirror will pick up the connection on its own.
        </Text>
        <View style={styles.settingsList}>
          <SettingsRow icon="wifi" label="Wi-Fi networks" hint="Join the home Wi-Fi or a phone’s hotspot" onPress={() => void openSystemSettings('wifi')} />
          <SettingsRow icon="phone-portrait-outline" label="Hotspot and tethering" hint="Share a connection with this mirror" onPress={() => void openSystemSettings('hotspot')} />
          <SettingsRow icon="bluetooth" label="Bluetooth" hint="Pair a phone for Bluetooth internet sharing" onPress={() => void openSystemSettings('bluetooth')} />
        </View>
        <DoneButton />
      </Shell>
    )
  }

  const online = Boolean(status?.online)

  return (
    <Shell>
      <Header online={online} status={status} />

      <View style={styles.tabs}>
        {([
          { key: 'wifi', label: 'Wi-Fi', icon: 'wifi' as const, enabled: capabilities.wifi },
          { key: 'bluetooth', label: 'Bluetooth', icon: 'bluetooth' as const, enabled: capabilities.bluetooth },
          { key: 'hotspot', label: 'Mirror hotspot', icon: 'radio-outline' as const, enabled: capabilities.hotspot },
        ] satisfies { key: Tab; label: string; icon: 'wifi' | 'bluetooth' | 'radio-outline'; enabled: boolean }[])
          .filter((entry) => entry.enabled)
          .map((entry) => (
            <Pressable
              key={entry.key}
              onPress={() => { setTab(entry.key); setError(''); setMessage('') }}
              style={[styles.tab, tab === entry.key && styles.tabActive]}
            >
              <Ionicons name={entry.icon} size={17} color={tab === entry.key ? c.cream : c.text} />
              <Text style={[styles.tabLabel, tab === entry.key && styles.tabLabelActive]}>{entry.label}</Text>
            </Pressable>
          ))}
      </View>

      {message ? <Text style={styles.successText}>{message}</Text> : null}
      {error ? <Text style={styles.errorText}>{error}</Text> : null}

      {tab === 'wifi' ? (
        <WifiPanel
          busy={busy}
          networks={networks}
          onForget={(ssid) => void withBusy('forget', () => forgetWifi(ssid), `Forgot ${ssid}.`).then(() => runWifiScan({ rescan: false }))}
          onJoin={joinSelected}
          onPasswordBackspace={() => setPassword((value) => value.slice(0, -1))}
          onPasswordKey={(key) => setPassword((value) => value + key)}
          onRescan={() => void runWifiScan()}
          onSelect={(network) => void tapNetwork(network)}
          onCancelSelected={() => { setSelected(null); setPassword('') }}
          password={password}
          selected={selected}
          showPassword={showPassword}
          onToggleShowPassword={() => setShowPassword((value) => !value)}
        />
      ) : null}

      {tab === 'bluetooth' ? (
        <BluetoothPanel
          bluetooth={bluetooth}
          busy={busy}
          canTether={capabilities.bluetoothTethering}
          onDisconnect={(device) => void withBusy(`bt-${device.address}`, () => disconnectBluetooth(device.address), `Disconnected ${device.name || device.address}.`)}
          onPair={(device) => void withBusy(`bt-${device.address}`, () => pairBluetooth(device.address), `Paired with ${device.name || device.address}.`)}
          onPower={(enabled) => void withBusy('bt-power', () => setBluetoothPower(enabled), enabled ? 'Bluetooth is on.' : 'Bluetooth is off.')}
          onScan={() => void withBusy('bt-scan', async () => {
            const result = await scanBluetooth()
            if (result.ok && mounted.current) setBluetooth((current) => (current ? { ...current, devices: result.devices || [] } : current))
            return result
          }, 'Finished looking for nearby devices.')}
          onTether={(device) => void withBusy(`bt-${device.address}`, () => tetherBluetooth(device.address), `Using ${device.name || device.address} for internet.`)}
        />
      ) : null}

      {tab === 'hotspot' ? (
        <HotspotPanel
          busy={busy}
          hotspot={status?.hotspot ?? null}
          onStart={() => void withBusy('hotspot', () => startHotspot(), 'The mirror’s hotspot is on.')}
          onStop={() => void withBusy('hotspot', () => stopHotspot(), 'The mirror’s hotspot is off.')}
          wifiIsUplink={status?.activeConnection?.type === 'wifi'}
        />
      ) : null}

      <DoneButton online={online} />
    </Shell>
  )
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="always">
        {children}
      </ScrollView>
    </SafeAreaView>
  )
}

function CenterNote({ children }: { children: React.ReactNode }) {
  return (
    <View style={styles.centerNote}>
      <Text style={styles.sectionBody}>{children}</Text>
      <DoneButton />
    </View>
  )
}

function Header({ online, status }: { online: boolean; status: NetworkStatus | null }) {
  // 'portal' / 'limited' is the case that most needed naming: the mirror IS on a network but has no route
  // out (captive portal, or a hotspot with no data), which used to be indistinguishable from a dead server.
  const captive = status?.connectivity === 'portal'
  const limited = status?.connectivity === 'limited'
  const detail = online
    ? status?.activeConnection?.name
      ? `Connected through ${status.activeConnection.name}`
      : 'Connected to the internet'
    : captive
      ? 'This network needs sign-in in a browser before it allows internet access.'
      : limited
        ? 'This mirror is on a network, but that network has no internet.'
        : 'This mirror is not connected to the internet yet.'
  return (
    <View style={styles.header}>
      <Text style={styles.eyebrow}>MIRROR SETUP</Text>
      <Text style={styles.title}>Connect this mirror</Text>
      <View style={[styles.statusPill, online ? styles.statusPillOnline : styles.statusPillOffline]}>
        <Ionicons name={online ? 'checkmark-circle' : 'alert-circle-outline'} size={16} color={online ? c.sageDeep : c.coral} />
        <Text style={[styles.statusPillText, { color: online ? c.sageDeep : c.coral }]}>{detail}</Text>
      </View>
    </View>
  )
}

function DoneButton({ online }: { online?: boolean } = {}) {
  return (
    <Pressable onPress={() => router.replace('/')} style={styles.doneButton}>
      <Text style={styles.doneText}>{online ? 'Done — continue' : 'Back to the mirror'}</Text>
    </Pressable>
  )
}

function SettingsRow({ icon, label, hint, onPress }: { icon: 'wifi' | 'bluetooth' | 'phone-portrait-outline'; label: string; hint: string; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={styles.row}>
      <Ionicons name={icon} size={22} color={c.text} />
      <View style={styles.rowBody}>
        <Text style={styles.rowLabel}>{label}</Text>
        <Text style={styles.rowHint}>{hint}</Text>
      </View>
      <Ionicons name="chevron-forward" size={20} color={c.textSecondary} />
    </Pressable>
  )
}

function WifiPanel({
  busy, networks, onCancelSelected, onForget, onJoin, onPasswordBackspace, onPasswordKey, onRescan, onSelect,
  onToggleShowPassword, password, selected, showPassword,
}: {
  busy: string
  networks: WifiNetwork[]
  onCancelSelected: () => void
  onForget: (ssid: string) => void
  onJoin: () => void
  onPasswordBackspace: () => void
  onPasswordKey: (key: string) => void
  onRescan: () => void
  onSelect: (network: WifiNetwork) => void
  onToggleShowPassword: () => void
  password: string
  selected: WifiNetwork | null
  showPassword: boolean
}) {
  if (selected) {
    return (
      <View style={styles.panel}>
        <Text style={styles.sectionTitle}>{selected.ssid}</Text>
        <Text style={styles.sectionBody}>Enter the Wi-Fi password, then tap Connect.</Text>
        <View style={styles.passwordField}>
          <Text style={styles.passwordValue}>
            {password ? (showPassword ? password : '•'.repeat(password.length)) : 'Password'}
          </Text>
          <Pressable onPress={onToggleShowPassword} style={styles.eyeButton}>
            <Ionicons name={showPassword ? 'eye-off-outline' : 'eye-outline'} size={20} color={c.textSecondary} />
          </Pressable>
        </View>
        {busy === 'connect' ? (
          <View style={styles.inlineBusy}>
            <ActivityIndicator color={c.goldDeep} />
            <Text style={styles.rowHint}>Joining {selected.ssid}…</Text>
          </View>
        ) : (
          <OnScreenKeyboard
            onBackspace={onPasswordBackspace}
            onKey={onPasswordKey}
            onSubmit={onJoin}
            submitDisabled={password.length < 8 && selected.secured}
            submitLabel="Connect"
          />
        )}
        <Pressable onPress={onCancelSelected} style={styles.linkButton}>
          <Text style={styles.linkText}>Choose a different network</Text>
        </Pressable>
      </View>
    )
  }

  return (
    <View style={styles.panel}>
      <View style={styles.panelHead}>
        <Text style={styles.sectionTitle}>Wi-Fi networks</Text>
        <Pressable disabled={Boolean(busy)} onPress={onRescan} style={styles.smallButton}>
          {busy === 'scan' ? <ActivityIndicator color={c.text} size="small" /> : <Text style={styles.smallButtonText}>Search again</Text>}
        </Pressable>
      </View>
      <Text style={styles.sectionBody}>
        Pick the home Wi-Fi, or switch on a phone’s personal hotspot and pick that.
      </Text>
      {busy === 'scan' && !networks.length ? <View style={styles.inlineBusy}><ActivityIndicator color={c.goldDeep} /><Text style={styles.rowHint}>Looking for networks…</Text></View> : null}
      {!busy && !networks.length ? <Text style={styles.rowHint}>No networks found yet. Tap “Search again”.</Text> : null}
      {networks.map((network) => (
        <Pressable
          key={network.ssid}
          disabled={Boolean(busy)}
          onPress={() => onSelect(network)}
          style={[styles.row, network.connected && styles.rowActive]}
        >
          <Ionicons name={signalIcon(network.signal)} size={22} color={network.connected ? c.sageDeep : c.text} />
          <View style={styles.rowBody}>
            <Text style={styles.rowLabel}>{network.ssid}</Text>
            <Text style={styles.rowHint}>
              {[
                network.connected ? 'Connected' : network.saved ? 'Saved' : null,
                network.secured ? 'Password needed' : 'Open network',
                network.band,
              ].filter(Boolean).join(' · ')}
            </Text>
          </View>
          {busy === 'connect' ? null : network.saved && !network.connected ? (
            <Pressable onPress={() => onForget(network.ssid)} style={styles.forgetButton}>
              <Text style={styles.forgetText}>Forget</Text>
            </Pressable>
          ) : network.connected ? (
            <Ionicons name="checkmark-circle" size={22} color={c.sageDeep} />
          ) : (
            <Ionicons name={network.secured ? 'lock-closed-outline' : 'chevron-forward'} size={18} color={c.textSecondary} />
          )}
        </Pressable>
      ))}
    </View>
  )
}

function BluetoothPanel({
  bluetooth, busy, canTether, onDisconnect, onPair, onPower, onScan, onTether,
}: {
  bluetooth: BluetoothStatus | null
  busy: string
  canTether: boolean
  onDisconnect: (device: BluetoothDevice) => void
  onPair: (device: BluetoothDevice) => void
  onPower: (enabled: boolean) => void
  onScan: () => void
  onTether: (device: BluetoothDevice) => void
}) {
  const powered = Boolean(bluetooth?.powered)
  return (
    <View style={styles.panel}>
      <View style={styles.panelHead}>
        <Text style={styles.sectionTitle}>Bluetooth</Text>
        <Pressable disabled={busy === 'bt-power'} onPress={() => onPower(!powered)} style={[styles.smallButton, powered && styles.smallButtonActive]}>
          {busy === 'bt-power' ? <ActivityIndicator color={c.text} size="small" /> : <Text style={styles.smallButtonText}>{powered ? 'Turn off' : 'Turn on'}</Text>}
        </Pressable>
      </View>
      <Text style={styles.sectionBody}>
        {canTether
          ? 'Use a phone’s Bluetooth internet sharing when there is no Wi-Fi. Pair the phone here, switch on “Bluetooth tethering” on the phone, then tap “Use for internet”.'
          : 'Pair a phone with this mirror. This build cannot take internet over Bluetooth.'}
      </Text>
      {powered && bluetooth?.name ? (
        <Text style={styles.rowHint}>
          On a phone, this mirror appears as “{bluetooth.name}”{bluetooth.discoverable ? ' and is visible now' : ''}.
        </Text>
      ) : null}

      <Pressable disabled={Boolean(busy) || !powered} onPress={onScan} style={[styles.wideButton, !powered && styles.wideButtonDisabled]}>
        {busy === 'bt-scan' ? <ActivityIndicator color={c.cream} size="small" /> : <Text style={styles.wideButtonText}>Look for nearby devices</Text>}
      </Pressable>
      {busy === 'bt-scan' ? <Text style={styles.rowHint}>Searching for about 12 seconds — keep the phone’s Bluetooth screen open.</Text> : null}

      {(bluetooth?.devices ?? []).map((device) => (
        <View key={device.address} style={[styles.row, device.tethering && styles.rowActive]}>
          <Ionicons name="phone-portrait-outline" size={22} color={device.tethering ? c.sageDeep : c.text} />
          <View style={styles.rowBody}>
            <Text style={styles.rowLabel}>{device.name || device.address}</Text>
            <Text style={styles.rowHint}>
              {device.tethering ? 'Sharing internet with this mirror' : device.connected ? 'Paired and connected' : 'Paired'}
            </Text>
          </View>
          {busy === `bt-${device.address}` ? (
            <ActivityIndicator color={c.goldDeep} size="small" />
          ) : device.tethering ? (
            <Pressable onPress={() => onDisconnect(device)} style={styles.forgetButton}>
              <Text style={styles.forgetText}>Stop</Text>
            </Pressable>
          ) : (
            <View style={styles.rowActions}>
              {!device.connected ? (
                <Pressable onPress={() => onPair(device)} style={styles.forgetButton}>
                  <Text style={styles.forgetText}>Pair</Text>
                </Pressable>
              ) : null}
              {canTether ? (
                <Pressable onPress={() => onTether(device)} style={styles.smallButton}>
                  <Text style={styles.smallButtonText}>Use for internet</Text>
                </Pressable>
              ) : null}
            </View>
          )}
        </View>
      ))}
      {powered && !(bluetooth?.devices ?? []).length ? (
        <Text style={styles.rowHint}>No devices yet. Open the phone’s Bluetooth screen, then tap “Look for nearby devices”.</Text>
      ) : null}
    </View>
  )
}

function HotspotPanel({
  busy, hotspot, onStart, onStop, wifiIsUplink,
}: {
  busy: string
  hotspot: HotspotState | null
  onStart: () => void
  onStop: () => void
  wifiIsUplink: boolean
}) {
  const active = Boolean(hotspot?.active)
  return (
    <View style={styles.panel}>
      <Text style={styles.sectionTitle}>Mirror hotspot</Text>
      <Text style={styles.sectionBody}>
        This makes the mirror broadcast its own Wi-Fi network so a phone or laptop can connect directly to
        it. It does not give the mirror internet.
      </Text>
      {/* Said BEFORE the tap, not after: taking the radio drops the mirror offline, and an installer who
          was not warned reads that as a new fault. */}
      {!active && wifiIsUplink ? (
        <Text style={styles.warnText}>
          This mirror is currently online over Wi-Fi. Starting the hotspot uses the same radio, so the
          mirror will go offline until you stop it.
        </Text>
      ) : null}
      {active && hotspot ? (
        <View style={styles.hotspotCard}>
          <Text style={styles.hotspotLabel}>NETWORK NAME</Text>
          <Text style={styles.hotspotValue}>{hotspot.ssid}</Text>
          <Text style={styles.hotspotLabel}>PASSWORD</Text>
          <Text style={styles.hotspotValue}>{hotspot.password || '—'}</Text>
        </View>
      ) : null}
      <Pressable disabled={busy === 'hotspot'} onPress={active ? onStop : onStart} style={styles.wideButton}>
        {busy === 'hotspot'
          ? <ActivityIndicator color={c.cream} size="small" />
          : <Text style={styles.wideButtonText}>{active ? 'Stop the hotspot' : 'Start the hotspot'}</Text>}
      </Pressable>
    </View>
  )
}

function signalIcon(signal: number): 'wifi' | 'wifi-outline' {
  return signal >= 55 ? 'wifi' : 'wifi-outline'
}

const styles = StyleSheet.create({
  safeArea: { backgroundColor: c.cream, flex: 1 },
  scroll: { alignItems: 'stretch', gap: 14, paddingBottom: 40, paddingHorizontal: 26, paddingTop: 26 },
  centerNote: { alignItems: 'center', gap: 18, paddingTop: 60 },
  header: { gap: 8 },
  eyebrow: { color: c.goldDeep, fontFamily: f.bodyMedium, fontSize: 12, letterSpacing: 2.2 },
  title: { color: c.text, fontFamily: f.display, fontSize: 32, lineHeight: 40 },
  statusPill: { alignItems: 'center', alignSelf: 'flex-start', borderRadius: 20, borderWidth: 1, flexDirection: 'row', gap: 8, paddingHorizontal: 14, paddingVertical: 8 },
  statusPillOnline: { backgroundColor: 'rgba(171,197,161,0.18)', borderColor: c.sage },
  statusPillOffline: { backgroundColor: 'rgba(201,120,110,0.12)', borderColor: c.coral },
  statusPillText: { flexShrink: 1, fontFamily: f.bodyMedium, fontSize: 14, lineHeight: 20 },
  tabs: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  tab: { alignItems: 'center', backgroundColor: c.white, borderColor: c.lineWarm, borderRadius: 22, borderWidth: 1, flexDirection: 'row', gap: 7, paddingHorizontal: 15, paddingVertical: 10 },
  tabActive: { backgroundColor: c.text, borderColor: c.text },
  tabLabel: { color: c.text, fontFamily: f.bodyMedium, fontSize: 14 },
  tabLabelActive: { color: c.cream },
  panel: { backgroundColor: c.sand, borderColor: c.lineWarm, borderRadius: 20, borderWidth: 1, gap: 10, padding: 18 },
  panelHead: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  sectionTitle: { color: c.text, fontFamily: f.display, fontSize: 21 },
  sectionBody: { color: c.textSecondary, fontFamily: f.body, fontSize: 15, lineHeight: 22 },
  settingsList: { gap: 10 },
  row: { alignItems: 'center', backgroundColor: c.white, borderColor: c.lineWarm, borderRadius: 14, borderWidth: 1, flexDirection: 'row', gap: 13, paddingHorizontal: 15, paddingVertical: 14 },
  rowActive: { backgroundColor: 'rgba(171,197,161,0.16)', borderColor: c.sage },
  rowBody: { flex: 1, gap: 2 },
  rowActions: { alignItems: 'center', flexDirection: 'row', gap: 8 },
  rowLabel: { color: c.text, fontFamily: f.bodyMedium, fontSize: 16 },
  rowHint: { color: c.textSecondary, fontFamily: f.body, fontSize: 13, lineHeight: 19 },
  smallButton: { backgroundColor: c.white, borderColor: c.lineWarm, borderRadius: 16, borderWidth: 1, minWidth: 84, paddingHorizontal: 13, paddingVertical: 9 },
  smallButtonActive: { backgroundColor: c.beige },
  smallButtonText: { color: c.text, fontFamily: f.bodyMedium, fontSize: 13, textAlign: 'center' },
  wideButton: { alignItems: 'center', backgroundColor: c.text, borderRadius: 24, marginTop: 4, paddingVertical: 14 },
  wideButtonDisabled: { opacity: 0.4 },
  wideButtonText: { color: c.cream, fontFamily: f.bodyMedium, fontSize: 15 },
  forgetButton: { borderColor: c.lineWarm, borderRadius: 14, borderWidth: 1, paddingHorizontal: 12, paddingVertical: 7 },
  forgetText: { color: c.textSecondary, fontFamily: f.body, fontSize: 13 },
  passwordField: { alignItems: 'center', backgroundColor: c.white, borderColor: c.lineWarm, borderRadius: 14, borderWidth: 1, flexDirection: 'row', minHeight: 54, paddingHorizontal: 16 },
  passwordValue: { color: c.text, flex: 1, fontFamily: f.bodyMedium, fontSize: 20, letterSpacing: 1.5 },
  eyeButton: { padding: 8 },
  inlineBusy: { alignItems: 'center', flexDirection: 'row', gap: 12, paddingVertical: 14 },
  linkButton: { alignSelf: 'center', padding: 8 },
  linkText: { color: c.textSecondary, fontFamily: f.body, fontSize: 14, textDecorationLine: 'underline' },
  hotspotCard: { backgroundColor: c.white, borderColor: c.lineWarm, borderRadius: 16, borderWidth: 1, gap: 4, padding: 16 },
  hotspotLabel: { color: c.goldDeep, fontFamily: f.bodyMedium, fontSize: 11, letterSpacing: 1.6, marginTop: 6 },
  hotspotValue: { color: c.text, fontFamily: f.display, fontSize: 24 },
  successText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 14, lineHeight: 20 },
  errorText: { color: c.coral, fontFamily: f.body, fontSize: 14, lineHeight: 20 },
  warnText: { color: c.amber, fontFamily: f.bodyMedium, fontSize: 14, lineHeight: 21 },
  doneButton: { alignItems: 'center', alignSelf: 'center', backgroundColor: c.beige, borderRadius: 24, marginTop: 8, paddingHorizontal: 28, paddingVertical: 14 },
  doneText: { color: c.text, fontFamily: f.bodyMedium, fontSize: 15 },
})
