import { Linking, Pressable, StyleSheet, Text, View } from 'react-native'

import { MirrorCard, MirrorPage, OutlineButton, PageHeading, PrimaryButton } from './MirrorChrome'
import { MirrorIcon } from './MirrorIcon'
import { mirrorColors as c, mirrorFonts as f } from '../../theme/mirrorTheme'

export function WifiSetupView({ error, onRetry }: { error?: string; onRetry?: () => void }) {
  return (
    <MirrorPage headerStatus="Mirror setup" onHelp={() => undefined}>
      <PageHeading title="Connect to Wi-Fi" subtitle="Choose a network to continue setting up your mirror." />
      <MirrorCard accent>
        <View style={styles.networkHeader}>
          <View style={styles.networkIcon}><MirrorIcon name="wifi" size={30} color={c.sageDeep} /></View>
          <View style={styles.networkCopy}>
            <Text style={styles.networkTitle}>Wi-Fi setup</Text>
            <Text style={styles.networkBody}>Wi-Fi networks and passwords are managed by Android on this mirror.</Text>
          </View>
        </View>
        {error ? <Text style={styles.error}>{error}</Text> : null}
        <PrimaryButton label="Open Wi-Fi settings" icon="settings-outline" onPress={() => { void Linking.openSettings() }} />
        <OutlineButton label="Try again" icon="refresh-outline" onPress={onRetry} />
        <Text style={styles.note}>After connecting, return to Reflexion and the readiness checks will continue automatically.</Text>
      </MirrorCard>
      <View style={styles.helpRow}>
        <MirrorIcon name="help-circle-outline" size={24} color={c.sageDeep} />
        <Text style={styles.helpText}>Choose another network in Android Wi-Fi settings if the current network is unavailable.</Text>
      </View>
    </MirrorPage>
  )
}

const styles = StyleSheet.create({
  networkHeader: { alignItems: 'center', flexDirection: 'row', gap: 18, marginBottom: 20 },
  networkIcon: { alignItems: 'center', backgroundColor: c.beige, borderRadius: 34, height: 68, justifyContent: 'center', width: 68 },
  networkCopy: { flex: 1 },
  networkTitle: { color: c.text, fontFamily: f.display, fontSize: 30 },
  networkBody: { color: c.textSecondary, fontFamily: f.body, fontSize: 17, lineHeight: 24, marginTop: 5 },
  error: { color: c.coral, fontFamily: f.bodyMedium, fontSize: 16, marginBottom: 15 },
  note: { color: c.textSecondary, fontFamily: f.body, fontSize: 15, lineHeight: 22, marginTop: 18, textAlign: 'center' },
  helpRow: { alignItems: 'center', flexDirection: 'row', gap: 10, marginTop: 22, maxWidth: 760, paddingHorizontal: 10 },
  helpText: { color: c.textSecondary, flex: 1, fontFamily: f.body, fontSize: 15, lineHeight: 21 },
})
