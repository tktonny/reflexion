import { Feather } from '@expo/vector-icons';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { colors, fontFamily } from '../theme';

/** Small native shapes keep the reference-image character without adding an image or native dependency. */
export function MirrorIllustration({ size = 112, showBear = true }: { size?: number; showBear?: boolean }) {
  const mirrorWidth = size * 0.47;
  return <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={[styles.illustration, { height: size, width: size * 1.15 }]}>
    <View style={[styles.mirror, { borderRadius: mirrorWidth, height: size * 0.78, left: size * 0.15, top: size * 0.02, width: mirrorWidth }]}><View style={styles.mirrorShine} /><View style={styles.mirrorMark}><Feather color={colors.textDecorative} name="feather" size={size * 0.16} /></View></View>
    <View style={[styles.mirrorBase, { height: size * 0.07, left: size * 0.12, top: size * 0.79, width: size * 0.55 }]} />
    {showBear ? <Text style={[styles.bear, { fontSize: size * 0.31, left: size * 0.62, top: size * 0.52 }]}>🧸</Text> : null}
    <Feather color="#B7C8B0" name="feather" size={size * 0.35} style={[styles.leaf, { left: size * 0.78, top: size * 0.06 }]} />
  </View>;
}

export function BearIllustration({ size = 92 }: { size?: number }) {
  return <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={[styles.bearOnly, { height: size, width: size }]}><Text style={{ fontSize: size * 0.62 }}>🧸</Text><Feather color="#B7C8B0" name="feather" size={size * 0.34} style={styles.bearLeaf} /></View>;
}

export function BotanicalSprig({ size = 64 }: { size?: number }) {
  return <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={[styles.sprig, { height: size, width: size * 0.7 }]}><View style={styles.stem} /><Feather color="#A8BEA2" name="feather" size={size * 0.32} style={[styles.sprigLeaf, { left: size * 0.18, top: size * 0.28, transform: [{ rotate: '-38deg' }] }]} /><Feather color="#B7C8B0" name="feather" size={size * 0.28} style={[styles.sprigLeaf, { left: size * 0.36, top: size * 0.04, transform: [{ rotate: '20deg' }] }]} /><Feather color="#C5D2BE" name="feather" size={size * 0.25} style={[styles.sprigLeaf, { left: size * 0.04, top: size * 0.54, transform: [{ rotate: '-50deg' }] }]} /></View>;
}

const styles = StyleSheet.create({
  illustration: { position: 'relative' },
  mirror: { backgroundColor: '#E9F0EE', borderColor: '#C7A477', borderWidth: 2, overflow: 'hidden', position: 'absolute' },
  mirrorShine: { backgroundColor: 'rgba(255,255,255,0.75)', borderRadius: 99, height: '70%', left: '18%', position: 'absolute', top: '12%', transform: [{ rotate: '18deg' }], width: '30%' },
  mirrorMark: { alignItems: 'center', bottom: '10%', left: 0, position: 'absolute', right: 0 },
  mirrorBase: { backgroundColor: '#C7A477', borderRadius: 99, position: 'absolute' },
  bear: { fontFamily: fontFamily.regular, position: 'absolute' },
  leaf: { opacity: 0.82, position: 'absolute', transform: [{ rotate: '20deg' }] },
  bearOnly: { alignItems: 'center', justifyContent: 'center', position: 'relative' },
  bearLeaf: { bottom: 2, opacity: 0.7, position: 'absolute', right: 0, transform: [{ rotate: '30deg' }] },
  sprig: { position: 'relative' },
  stem: { backgroundColor: '#A8BEA2', height: '94%', left: '47%', position: 'absolute', top: '3%', transform: [{ rotate: '25deg' }], width: 1 },
  sprigLeaf: { opacity: 0.75, position: 'absolute' },
});
