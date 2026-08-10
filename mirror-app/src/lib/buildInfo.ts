import Constants from 'expo-constants'
import * as Updates from 'expo-updates'

import { getApiHostname } from '../config/apiUrl'
import { DEMO_BUILD_ENABLED } from '../demo/demoConfig'

export type SafeBuildInfo = {
  appVersion: string
  buildNumber: string
  updateId: string
  runtimeVersion: string
  environment: string
  backendHostname: string
}

export function getMirrorBuildInfo(): SafeBuildInfo {
  const config = Constants.expoConfig
  return {
    appVersion: config?.version || 'unknown',
    buildNumber: String(config?.android?.versionCode || config?.ios?.buildNumber || 'unknown'),
    updateId: Updates.updateId || 'embedded',
    runtimeVersion: String(config?.runtimeVersion || 'unknown'),
    environment: DEMO_BUILD_ENABLED ? 'demo' : process.env.EXPO_PUBLIC_ENVIRONMENT || (__DEV__ ? 'development' : 'production'),
    backendHostname: getApiHostname(),
  }
}

export function logMirrorBuildInfo() {
  console.info(`[rfx-build] ${JSON.stringify(getMirrorBuildInfo())}`)
}
