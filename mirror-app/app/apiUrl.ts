import Constants from 'expo-constants'
import { Platform } from 'react-native'

export function getApiUrl(path: string) {
  if (Platform.OS === 'web' && typeof window !== 'undefined') {
    return path
  }

  const hostUri =
    Constants.expoConfig?.hostUri ||
    Constants.expoGoConfig?.debuggerHost ||
    Constants.manifest2?.extra?.expoGo?.debuggerHost

  if (!hostUri) return path

  const host = hostUri.split(':').slice(0, 2).join(':')
  return `http://${host}${path}`
}
