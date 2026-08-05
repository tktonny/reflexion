import { Ionicons } from '@expo/vector-icons'
import { useMemo } from 'react'
import { StyleSheet, Text, View } from 'react-native'
import qrcode from 'qrcode-generator'

import type { SetupModeState } from '../../native/networkSetup'
import { mirrorColors as c, mirrorFonts as f } from '../../theme/mirrorTheme'

/**
 * What the mirror shows when it cannot reach the network and has no way to be typed on.
 *
 * The Ubuntu unit is a wall-mounted display: it auto-starts the app on boot, and the family has no keyboard
 * and no mouse (a technician can attach one over USB; an end user cannot). So the mirror broadcasts its own
 * hotspot, serves a setup page on it, and puts THESE instructions on the glass — the caregiver joins with a
 * phone and does all the typing there.
 *
 * Two QR codes, because they remove the two things most likely to be mistyped:
 *   1. a `WIFI:` code a phone camera turns into a one-tap join (no passphrase typing at all);
 *   2. the portal URL, so nobody has to key in an IP address and port.
 * The passphrase and address are still printed underneath, because QR scanning fails often enough on a
 * reflective mirror surface that it cannot be the only route.
 *
 * The PIN is deliberately on-screen only: it is what proves the person configuring the mirror can actually
 * see it, so being within Wi-Fi range of the hotspot is not enough on its own.
 */
export function PhoneSetupInstructions({ state }: { state: SetupModeState }) {
  const failed = state.lastResult && !state.lastResult.ok ? state.lastResult : null
  return (
    <View style={styles.wrap}>
      {failed ? (
        <View style={styles.failBanner}>
          <Ionicons name="alert-circle-outline" size={18} color={c.coral} />
          <Text style={styles.failText}>
            {failed.error || `Could not join ${failed.ssid}.`} Try again with the phone.
          </Text>
        </View>
      ) : null}

      <Text style={styles.eyebrow}>SET UP WITH A PHONE</Text>
      <Text style={styles.title}>Connect this mirror using your phone</Text>

      <View style={styles.steps}>
        <Step number="1" label="Join this Wi-Fi network from your phone">
          <View style={styles.qrRowWrap}>
            <QrBlock value={wifiJoinPayload(state)} caption="Scan to join" />
            <View style={styles.credentials}>
              <Text style={styles.fieldLabel}>NETWORK</Text>
              <Text style={styles.fieldValue}>{state.ssid || '—'}</Text>
              <Text style={styles.fieldLabel}>PASSWORD</Text>
              <Text style={styles.fieldValueMono}>{state.password || '—'}</Text>
            </View>
          </View>
        </Step>

        <Step number="2" label="Open the setup page">
          <View style={styles.qrRowWrap}>
            <QrBlock value={state.portalUrl} caption="Scan to open" />
            <View style={styles.credentials}>
              <Text style={styles.fieldLabel}>OR TYPE THIS ADDRESS</Text>
              <Text style={styles.fieldValueMono}>{state.portalUrl || '—'}</Text>
              <Text style={styles.hint}>Most phones open it on their own once joined.</Text>
            </View>
          </View>
        </Step>

        <Step number="3" label="Choose the home Wi-Fi and enter this code">
          <Text style={styles.pin}>{formatPin(state.pin)}</Text>
          <Text style={styles.hint}>
            This setup network disappears while the mirror connects — the result appears here.
          </Text>
        </Step>
      </View>
    </View>
  )
}

function Step({ number, label, children }: { number: string; label: string; children: React.ReactNode }) {
  return (
    <View style={styles.step}>
      <View style={styles.stepHead}>
        <View style={styles.stepNumber}><Text style={styles.stepNumberText}>{number}</Text></View>
        <Text style={styles.stepLabel}>{label}</Text>
      </View>
      <View style={styles.stepBody}>{children}</View>
    </View>
  )
}

/**
 * The standard Wi-Fi QR payload every phone camera understands. Special characters in an SSID or passphrase
 * must be escaped or the phone parses the payload as multiple fields and the join silently fails.
 */
export function wifiJoinPayload({ ssid, password }: { ssid: string; password: string }) {
  const escape = (value: string) => value.replace(/([\\;,:"])/g, '\\$1')
  if (!ssid) return 'WIFI:;'
  if (!password) return `WIFI:T:nopass;S:${escape(ssid)};;`
  return `WIFI:T:WPA;S:${escape(ssid)};P:${escape(password)};;`
}

export function formatPin(pin: string) {
  const digits = (pin || '').replace(/\D/g, '')
  if (digits.length !== 6) return digits || '—'
  return `${digits.slice(0, 3)} ${digits.slice(3)}`
}

function QrBlock({ value, caption }: { value: string; caption: string }) {
  const grid = useMemo(() => {
    if (!value) return null
    const qr = qrcode(0, 'M')
    qr.addData(value)
    qr.make()
    const count = qr.getModuleCount()
    const rows: boolean[][] = []
    for (let row = 0; row < count; row += 1) {
      const cells: boolean[] = []
      for (let column = 0; column < count; column += 1) cells.push(qr.isDark(row, column))
      rows.push(cells)
    }
    return { rows, cell: Math.max(3, Math.floor(132 / count)) }
  }, [value])
  if (!grid) return null
  return (
    <View style={styles.qrBlock}>
      <View style={styles.qr}>
        {grid.rows.map((row, rowIndex) => (
          <View key={rowIndex} style={styles.qrRow}>
            {row.map((dark, columnIndex) => (
              <View key={columnIndex} style={{ backgroundColor: dark ? c.ink : 'transparent', height: grid.cell, width: grid.cell }} />
            ))}
          </View>
        ))}
      </View>
      <Text style={styles.qrCaption}>{caption}</Text>
    </View>
  )
}

const styles = StyleSheet.create({
  wrap: { alignItems: 'stretch', gap: 12, maxWidth: 720, width: '100%' },
  eyebrow: { color: c.goldDeep, fontFamily: f.bodyMedium, fontSize: 12, letterSpacing: 2.2, textAlign: 'center' },
  title: { color: c.text, fontFamily: f.display, fontSize: 29, lineHeight: 37, marginBottom: 4, textAlign: 'center' },
  steps: { gap: 10 },
  step: { backgroundColor: c.sand, borderColor: c.lineWarm, borderRadius: 16, borderWidth: 1, padding: 14 },
  stepHead: { alignItems: 'center', flexDirection: 'row', gap: 10 },
  stepNumber: { alignItems: 'center', backgroundColor: c.text, borderRadius: 13, height: 26, justifyContent: 'center', width: 26 },
  stepNumberText: { color: c.cream, fontFamily: f.bodyMedium, fontSize: 14 },
  stepLabel: { color: c.text, flex: 1, fontFamily: f.bodyMedium, fontSize: 17 },
  stepBody: { paddingLeft: 36, paddingTop: 10 },
  qrRowWrap: { alignItems: 'center', flexDirection: 'row', gap: 18 },
  qrBlock: { alignItems: 'center', gap: 5 },
  qr: { backgroundColor: c.white, borderColor: c.lineWarm, borderRadius: 10, borderWidth: 1, padding: 8 },
  qrRow: { flexDirection: 'row' },
  qrCaption: { color: c.textSecondary, fontFamily: f.body, fontSize: 11 },
  credentials: { flex: 1, gap: 1 },
  fieldLabel: { color: c.goldDeep, fontFamily: f.bodyMedium, fontSize: 10, letterSpacing: 1.4, marginTop: 7 },
  fieldValue: { color: c.text, fontFamily: f.display, fontSize: 21 },
  fieldValueMono: { color: c.text, fontFamily: f.bodyMedium, fontSize: 19, letterSpacing: 0.6 },
  hint: { color: c.textSecondary, fontFamily: f.body, fontSize: 12, lineHeight: 18, marginTop: 5 },
  pin: { color: c.text, fontFamily: f.display, fontSize: 40, fontVariant: ['tabular-nums'], letterSpacing: 6, lineHeight: 50 },
  failBanner: {
    alignItems: 'center', backgroundColor: 'rgba(201,120,110,0.12)', borderColor: c.coral, borderRadius: 12,
    borderWidth: 1, flexDirection: 'row', gap: 9, paddingHorizontal: 13, paddingVertical: 10,
  },
  failText: { color: c.coral, flex: 1, fontFamily: f.bodyMedium, fontSize: 14, lineHeight: 20 },
})
