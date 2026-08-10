import { StyleSheet, Text, View } from 'react-native'

export type MirrorIconName = string

type Props = {
  name: MirrorIconName
  size?: number
  color?: string
}

/**
 * Small cross-platform line icons for the web demo and final Mirror surfaces.
 * This intentionally does not depend on an icon font, which can render missing
 * glyphs as square placeholders while a web bundle is loading.
 */
export function MirrorIcon({ name, size = 24, color = '#35564d' }: Props) {
  const stroke = Math.max(1.5, size / 10)
  const common = { height: size, width: size }

  if (name.includes('chevron') || name === 'arrow-forward') {
    return <View style={[styles.icon, common]}><View style={[styles.chevron, { borderColor: color, borderBottomWidth: stroke, borderRightWidth: stroke, height: size * 0.32, width: size * 0.32 }]} /></View>
  }
  if (name === 'close' || name === 'close-circle') {
    return <View style={[styles.icon, common, name === 'close-circle' && { borderColor: color, borderRadius: size, borderWidth: stroke }]}><View style={[styles.crossLine, { backgroundColor: color, height: stroke, transform: [{ rotate: '45deg' }], width: size * 0.56 }]} /><View style={[styles.crossLine, { backgroundColor: color, height: stroke, transform: [{ rotate: '-45deg' }], width: size * 0.56 }]} /></View>
  }
  if (name === 'menu') {
    return <View style={[styles.icon, common, { gap: size * 0.16, justifyContent: 'center' }]}>{[0, 1, 2].map((line) => <View key={line} style={{ backgroundColor: color, height: stroke, width: size * 0.68 }} />)}</View>
  }
  if (name.includes('mic')) return <MicrophoneIcon size={size} color={color} outline={name.includes('outline')} off={name.includes('off')} />
  if (name.includes('wifi')) return <WifiIcon size={size} color={color} stroke={stroke} />
  if (name.includes('sunny')) return <SunCloudIcon size={size} color={color} stroke={stroke} />
  if (name.includes('medical')) return <MedicalIcon size={size} color={color} stroke={stroke} />
  if (name.includes('mail') || name.includes('message')) return <MailIcon size={size} color={color} stroke={stroke} open={name.includes('open')} />
  if (name.includes('shield')) return <ShieldIcon size={size} color={color} stroke={stroke} />
  if (name.includes('checkmark')) return <CheckIcon size={size} color={color} stroke={stroke} />
  if (name.includes('close-circle')) return <CheckIcon size={size} color={color} stroke={stroke} close />
  if (name.includes('help-circle')) return <QuestionIcon size={size} color={color} stroke={stroke} />
  if (name.includes('refresh')) return <RefreshIcon size={size} color={color} stroke={stroke} />
  if (name.includes('stop-circle') || name === 'stop') return <StopIcon size={size} color={color} stroke={stroke} />
  if (name.includes('play')) return <PlayIcon size={size} color={color} />
  if (name.includes('volume')) return <SpeakerIcon size={size} color={color} />
  if (name.includes('globe')) return <GlobeIcon size={size} color={color} stroke={stroke} />
  if (name.includes('cloud')) return <CloudIcon size={size} color={color} stroke={stroke} />
  if (name.includes('calendar')) return <CalendarIcon size={size} color={color} stroke={stroke} />
  if (name.includes('clipboard')) return <ClipboardIcon size={size} color={color} stroke={stroke} />
  if (name.includes('people')) return <PeopleIcon size={size} color={color} />
  if (name.includes('flask')) return <FlaskIcon size={size} color={color} stroke={stroke} />
  if (name.includes('qr-code')) return <QrIcon size={size} color={color} stroke={stroke} />
  if (name.includes('time')) return <ClockIcon size={size} color={color} stroke={stroke} />

  return <View style={[styles.fallback, common, { borderColor: color, borderRadius: size, borderWidth: stroke }]}><View style={{ backgroundColor: color, borderRadius: size, height: stroke, width: size * 0.38 }} /></View>
}

function MicrophoneIcon({ size, color, outline, off }: { size: number; color: string; outline: boolean; off: boolean }) {
  const stroke = Math.max(1.5, size / 10)
  return <View style={[styles.icon, { height: size, width: size }]}>
    <View style={{ borderColor: color, borderRadius: size * 0.22, borderWidth: outline ? stroke : 0, backgroundColor: outline ? 'transparent' : color, height: size * 0.48, width: size * 0.25 }} />
    <View style={{ borderBottomColor: color, borderBottomLeftRadius: size, borderBottomRightRadius: size, borderBottomWidth: stroke, borderLeftColor: color, borderLeftWidth: stroke, borderRightColor: color, borderRightWidth: stroke, bottom: size * 0.2, height: size * 0.28, position: 'absolute', width: size * 0.48 }} />
    <View style={{ backgroundColor: color, bottom: size * 0.11, height: size * 0.16, position: 'absolute', width: stroke }} />
    <View style={{ backgroundColor: color, bottom: size * 0.07, height: stroke, position: 'absolute', width: size * 0.42 }} />
    {off ? <View style={{ backgroundColor: color, height: stroke, position: 'absolute', transform: [{ rotate: '-45deg' }], width: size * 0.88 }} /> : null}
  </View>
}

function WifiIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { height: size, width: size }]}>
    <View style={{ borderColor: color, borderRadius: size, borderTopWidth: stroke, height: size * 0.62, position: 'absolute', top: size * 0.18, width: size * 0.84 }} />
    <View style={{ borderColor: color, borderRadius: size, borderTopWidth: stroke, height: size * 0.38, position: 'absolute', top: size * 0.31, width: size * 0.54 }} />
    <View style={{ backgroundColor: color, borderRadius: size, bottom: size * 0.12, height: size * 0.13, position: 'absolute', width: size * 0.13 }} />
  </View>
}

function SunCloudIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { height: size, width: size }]}>
    <View style={{ backgroundColor: '#d18a4c', borderRadius: size, height: size * 0.29, position: 'absolute', right: size * 0.12, top: size * 0.08, width: size * 0.29 }} />
    <View style={{ backgroundColor: color, borderRadius: size * 0.18, bottom: size * 0.16, height: size * 0.29, left: size * 0.13, position: 'absolute', width: size * 0.62 }} />
    <View style={{ backgroundColor: color, borderRadius: size, height: size * 0.38, left: size * 0.28, position: 'absolute', top: size * 0.33, width: size * 0.38 }} />
    <View style={{ borderColor: color, borderTopWidth: stroke, bottom: size * 0.16, position: 'absolute', width: size * 0.7 }} />
  </View>
}

function MedicalIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { borderColor: color, borderRadius: size * 0.22, borderWidth: stroke, height: size, width: size }]}>
    <View style={{ backgroundColor: color, height: size * 0.14, position: 'absolute', width: size * 0.54 }} />
    <View style={{ backgroundColor: color, height: size * 0.54, position: 'absolute', width: size * 0.14 }} />
  </View>
}

function MailIcon({ size, color, stroke, open }: { size: number; color: string; stroke: number; open: boolean }) {
  return <View style={[styles.icon, { borderColor: color, borderWidth: stroke, height: size * 0.67, marginTop: size * 0.16, width: size * 0.88 }]}>
    <View style={{ borderColor: color, borderBottomWidth: stroke, height: size * 0.34, position: 'absolute', transform: [{ rotate: '28deg' }], width: size * 0.56 }} />
    <View style={{ borderColor: color, borderBottomWidth: stroke, height: size * 0.34, position: 'absolute', right: 0, transform: [{ rotate: '-28deg' }], width: size * 0.56 }} />
    {open ? <View style={{ backgroundColor: color, height: stroke, position: 'absolute', top: size * 0.22, width: size * 0.48 }} /> : null}
  </View>
}

function ShieldIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { backgroundColor: `${color}22`, borderColor: color, borderBottomLeftRadius: size * 0.35, borderBottomRightRadius: size * 0.35, borderTopLeftRadius: size * 0.16, borderTopRightRadius: size * 0.16, borderWidth: stroke, height: size * 0.9, width: size * 0.76 }]}><View style={{ backgroundColor: color, height: stroke, position: 'absolute', transform: [{ rotate: '45deg' }], width: size * 0.28 }} /><View style={{ backgroundColor: color, height: stroke, position: 'absolute', transform: [{ rotate: '-45deg' }], width: size * 0.52 }} /></View>
}

function CheckIcon({ size, color, stroke, close = false }: { size: number; color: string; stroke: number; close?: boolean }) {
  return <View style={[styles.icon, { borderColor: color, borderRadius: size, borderWidth: stroke, height: size, width: size }]}>
    <View style={[styles.crossLine, { backgroundColor: color, height: stroke, transform: [{ rotate: close ? '45deg' : '45deg' }], width: size * (close ? 0.52 : 0.28), left: close ? size * 0.24 : size * 0.23, top: close ? size * 0.46 : size * 0.53 }]} />
    <View style={[styles.crossLine, { backgroundColor: color, height: stroke, transform: [{ rotate: close ? '-45deg' : '-45deg' }], width: size * (close ? 0.52 : 0.48), left: close ? size * 0.24 : size * 0.36, top: close ? size * 0.46 : size * 0.43 }]} />
  </View>
}

function QuestionIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { borderColor: color, borderRadius: size, borderWidth: stroke, height: size, width: size }]}><Text style={{ color, fontSize: size * 0.66, fontWeight: '600', lineHeight: size * 0.78 }}>?</Text></View>
}

function RefreshIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { borderColor: color, borderRadius: size, borderTopWidth: stroke, borderWidth: stroke, height: size * 0.72, transform: [{ rotate: '-28deg' }], width: size * 0.72 }]}><View style={{ borderBottomColor: color, borderBottomWidth: stroke, borderRightColor: color, borderRightWidth: stroke, bottom: -size * 0.1, height: size * 0.2, position: 'absolute', right: size * 0.02, transform: [{ rotate: '-18deg' }], width: size * 0.2 }} /></View>
}

function StopIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { backgroundColor: color, borderColor: color, borderRadius: size, borderWidth: stroke, height: size, width: size }]}><View style={{ backgroundColor: color === '#ffffff' ? '#35564d' : '#ffffff', borderRadius: size * 0.08, height: size * 0.3, width: size * 0.3 }} /></View>
}

function PlayIcon({ size, color }: { size: number; color: string }) {
  return <View style={[styles.icon, { height: size, width: size }]}><View style={{ borderBottomColor: 'transparent', borderBottomWidth: size * 0.25, borderLeftColor: color, borderLeftWidth: size * 0.42, borderTopColor: 'transparent', borderTopWidth: size * 0.25, marginLeft: size * 0.1 }} /></View>
}

function SpeakerIcon({ size, color }: { size: number; color: string }) {
  return <View style={[styles.icon, { height: size, width: size }]}><View style={{ backgroundColor: color, height: size * 0.28, width: size * 0.24 }} /><View style={{ backgroundColor: color, height: size * 0.62, position: 'absolute', transform: [{ skewY: '-28deg' }], width: size * 0.38 }} /><View style={{ borderColor: color, borderRadius: size, borderRightWidth: Math.max(1.5, size / 10), height: size * 0.58, position: 'absolute', right: size * 0.06, width: size * 0.4 }} /></View>
}

function GlobeIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { borderColor: color, borderRadius: size, borderWidth: stroke, height: size, width: size }]}><View style={{ borderColor: color, borderRadius: size, borderWidth: stroke / 2, height: size * 0.82, width: size * 0.38 }} /><View style={{ borderColor: color, borderTopWidth: stroke / 2, position: 'absolute', width: size * 0.8 }} /></View>
}

function CloudIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { height: size, width: size }]}><View style={{ backgroundColor: `${color}33`, borderColor: color, borderRadius: size * 0.2, borderWidth: stroke, bottom: size * 0.2, height: size * 0.35, left: size * 0.08, position: 'absolute', width: size * 0.76 }} /><View style={{ backgroundColor: `${color}33`, borderColor: color, borderRadius: size, borderWidth: stroke, height: size * 0.45, left: size * 0.26, position: 'absolute', top: size * 0.25, width: size * 0.45 }} /></View>
}

function CalendarIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { borderColor: color, borderRadius: size * 0.12, borderWidth: stroke, height: size * 0.82, width: size * 0.82 }]}><View style={{ backgroundColor: color, height: stroke, position: 'absolute', top: size * 0.23, width: size * 0.82 }} /><View style={{ borderColor: color, borderLeftWidth: stroke, borderTopWidth: stroke, height: size * 0.2, position: 'absolute', right: size * 0.2, top: -size * 0.1, width: size * 0.2 }} /></View>
}

function ClipboardIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { borderColor: color, borderRadius: size * 0.1, borderWidth: stroke, height: size * 0.86, width: size * 0.68 }]}><View style={{ backgroundColor: color, borderRadius: size * 0.08, height: size * 0.16, position: 'absolute', top: -size * 0.08, width: size * 0.35 }} />{[0, 1, 2].map((row) => <View key={row} style={{ backgroundColor: color, height: stroke, marginTop: row === 0 ? size * 0.28 : size * 0.12, width: size * 0.4 }} />)}</View>
}

function PeopleIcon({ size, color }: { size: number; color: string }) {
  return <View style={[styles.icon, { height: size, width: size }]}><View style={{ backgroundColor: color, borderRadius: size, height: size * 0.23, left: size * 0.16, position: 'absolute', top: size * 0.16, width: size * 0.23 }} /><View style={{ backgroundColor: color, borderRadius: size * 0.2, bottom: size * 0.14, height: size * 0.34, left: size * 0.08, position: 'absolute', width: size * 0.4 }} /><View style={{ backgroundColor: color, borderRadius: size, height: size * 0.2, position: 'absolute', right: size * 0.16, top: size * 0.22, width: size * 0.2 }} /><View style={{ backgroundColor: color, borderRadius: size * 0.2, bottom: size * 0.14, height: size * 0.3, position: 'absolute', right: size * 0.08, width: size * 0.36 }} /></View>
}

function FlaskIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { height: size, width: size }]}><View style={{ borderColor: color, borderBottomLeftRadius: size * 0.2, borderBottomRightRadius: size * 0.2, borderLeftWidth: stroke, borderRightWidth: stroke, borderTopWidth: stroke, height: size * 0.62, marginTop: size * 0.22, transform: [{ rotate: '0deg' }], width: size * 0.48 }} /><View style={{ backgroundColor: color, height: stroke, position: 'absolute', top: size * 0.18, width: size * 0.36 }} /></View>
}

function QrIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { flexDirection: 'row', flexWrap: 'wrap', gap: size * 0.08, height: size, padding: size * 0.12, width: size }]}>{[0, 1, 2, 3].map((item) => <View key={item} style={{ borderColor: color, borderWidth: stroke, height: size * 0.27, width: size * 0.27 }} />)}</View>
}

function ClockIcon({ size, color, stroke }: { size: number; color: string; stroke: number }) {
  return <View style={[styles.icon, { borderColor: color, borderRadius: size, borderWidth: stroke, height: size, width: size }]}><View style={{ backgroundColor: color, height: size * 0.28, position: 'absolute', top: size * 0.2, width: stroke }} /><View style={{ backgroundColor: color, height: stroke, left: size * 0.5, position: 'absolute', top: size * 0.47, transform: [{ rotate: '32deg' }], width: size * 0.23 }} /></View>
}

const styles = StyleSheet.create({
  icon: { alignItems: 'center', justifyContent: 'center', position: 'relative' },
  fallback: { alignItems: 'center', justifyContent: 'center' },
  chevron: { transform: [{ rotate: '-45deg' }] },
  crossLine: { position: 'absolute' },
})
