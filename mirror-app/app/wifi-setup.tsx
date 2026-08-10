import { router } from 'expo-router'

import { WifiSetupView } from '../src/components/mirror/WifiSetupView'

export default function WifiSetupScreen() {
  return <WifiSetupView onRetry={() => router.replace('/')} />
}
