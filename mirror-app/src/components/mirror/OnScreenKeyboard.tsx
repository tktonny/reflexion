import { Ionicons } from '@expo/vector-icons'
import { useState } from 'react'
import { Pressable, StyleSheet, Text, View } from 'react-native'

import { mirrorColors as c, mirrorFonts as f } from '../../theme/mirrorTheme'

/**
 * Touch keyboard for the mirror.
 *
 * The mirror is a wall-mounted kiosk: fullscreen, no physical keyboard, and on the Linux appliance no
 * desktop environment to summon one from. A Wi-Fi password therefore cannot be typed at all without this
 * — which is why the network setup screen owns its own keyboard rather than relying on a `TextInput`
 * bringing up a system IME.
 *
 * Keys are large because the person using it is standing at arm's length from the glass.
 */

const ROWS_LOWER = [
  ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
  ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
  ['z', 'x', 'c', 'v', 'b', 'n', 'm'],
]

const ROWS_UPPER = ROWS_LOWER.map((row) => row.map((key) => key.toUpperCase()))

const ROWS_SYMBOL = [
  ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
  ['-', '_', '.', ',', '@', '#', '$', '&', '*'],
  ['!', '?', '/', ':', ';', '+', '=', '%', '~'],
]

type Layout = 'lower' | 'upper' | 'symbol'

export function OnScreenKeyboard({
  onKey,
  onBackspace,
  onSubmit,
  submitLabel = 'Connect',
  submitDisabled = false,
}: {
  onKey: (key: string) => void
  onBackspace: () => void
  onSubmit: () => void
  submitLabel?: string
  submitDisabled?: boolean
}) {
  const [layout, setLayout] = useState<Layout>('lower')
  const rows = layout === 'lower' ? ROWS_LOWER : layout === 'upper' ? ROWS_UPPER : ROWS_SYMBOL

  const press = (key: string) => {
    onKey(key)
    // One-shot shift, like a phone keyboard — otherwise every password becomes SHOUTED.
    if (layout === 'upper') setLayout('lower')
  }

  return (
    <View style={styles.keyboard}>
      {rows.map((row, rowIndex) => (
        <View key={rowIndex} style={styles.row}>
          {row.map((key) => (
            <Pressable key={key} onPress={() => press(key)} style={({ pressed }) => [styles.key, pressed && styles.keyPressed]}>
              <Text style={styles.keyLabel}>{key}</Text>
            </Pressable>
          ))}
        </View>
      ))}
      <View style={styles.row}>
        <Pressable
          onPress={() => setLayout(layout === 'upper' ? 'lower' : 'upper')}
          style={({ pressed }) => [styles.key, styles.keyWide, layout === 'upper' && styles.keyActive, pressed && styles.keyPressed]}
        >
          <Ionicons name={layout === 'upper' ? 'arrow-up' : 'arrow-up-outline'} size={20} color={c.text} />
        </Pressable>
        <Pressable
          onPress={() => setLayout(layout === 'symbol' ? 'lower' : 'symbol')}
          style={({ pressed }) => [styles.key, styles.keyWide, layout === 'symbol' && styles.keyActive, pressed && styles.keyPressed]}
        >
          <Text style={styles.keyLabelSmall}>{layout === 'symbol' ? 'abc' : '?123'}</Text>
        </Pressable>
        <Pressable onPress={() => onKey(' ')} style={({ pressed }) => [styles.key, styles.keySpace, pressed && styles.keyPressed]}>
          <Text style={styles.keyLabelSmall}>space</Text>
        </Pressable>
        <Pressable onPress={onBackspace} style={({ pressed }) => [styles.key, styles.keyWide, pressed && styles.keyPressed]}>
          <Ionicons name="backspace-outline" size={20} color={c.text} />
        </Pressable>
        <Pressable
          disabled={submitDisabled}
          onPress={onSubmit}
          style={({ pressed }) => [styles.key, styles.keySubmit, submitDisabled && styles.keyDisabled, pressed && styles.keyPressed]}
        >
          <Text style={styles.keySubmitLabel}>{submitLabel}</Text>
        </Pressable>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  keyboard: { gap: 8, marginTop: 16 },
  row: { flexDirection: 'row', gap: 8, justifyContent: 'center' },
  key: {
    alignItems: 'center',
    backgroundColor: c.white,
    borderColor: c.lineWarm,
    borderRadius: 12,
    borderWidth: 1,
    justifyContent: 'center',
    minWidth: 46,
    paddingHorizontal: 10,
    paddingVertical: 14,
  },
  keyPressed: { backgroundColor: c.gold },
  keyActive: { backgroundColor: c.beige },
  keyDisabled: { opacity: 0.4 },
  keyWide: { minWidth: 62 },
  keySpace: { flexGrow: 1, minWidth: 120 },
  keySubmit: { backgroundColor: c.text, borderColor: c.text, minWidth: 108 },
  keyLabel: { color: c.text, fontFamily: f.bodyMedium, fontSize: 19 },
  keyLabelSmall: { color: c.text, fontFamily: f.body, fontSize: 14 },
  keySubmitLabel: { color: c.cream, fontFamily: f.bodyMedium, fontSize: 15 },
})
