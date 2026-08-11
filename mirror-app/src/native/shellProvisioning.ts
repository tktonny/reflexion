import { Platform } from 'react-native'

/**
 * The Linux (Electron) shell's provisioning surface.
 *
 * On Android the bootstrap token is either typed in on the device screen (app/test-device.tsx) or, for a
 * one-off build, inlined from `EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN`. Neither fits the Ubuntu appliance: it
 * has no keyboard, and inlining an identity into the AppImage would make every unit built from it claim the
 * same device — a bootstrap token is device-bound, so they would knock each other's pairing over.
 *
 * So the Electron shell reads the token from `<userData>/device-config.json` at runtime and hands it over
 * this bridge. Returns null everywhere else, which is what keeps the Android path unchanged.
 */

type ProvisioningBridge = {
  bootstrapToken(): Promise<{ token: string | null; configPath: string }>
}

function bridge(): ProvisioningBridge | null {
  if (Platform.OS !== 'web' || typeof window === 'undefined') return null
  return (window as unknown as { reflexionProvisioning?: ProvisioningBridge }).reflexionProvisioning ?? null
}

/** True when running inside the Electron shell, i.e. when a token can come from a file at all. */
export function shellProvisioningAvailable(): boolean {
  return Boolean(bridge())
}

/**
 * The bootstrap token the operator dropped into the config file, or null. Never throws — an unprovisioned
 * unit is a normal state (network setup and the self-check still run; only pairing needs a token), and a
 * failure here must not stop the app from booting.
 */
export async function readShellBootstrapToken(): Promise<string | null> {
  const api = bridge()
  if (!api) return null
  try {
    const result = await api.bootstrapToken()
    const token = result?.token?.trim()
    return token ? token : null
  } catch {
    return null
  }
}

/** Where an operator should put the token on this unit. Shown on the provisioning screen. */
export async function shellDeviceConfigPath(): Promise<string | null> {
  const api = bridge()
  if (!api) return null
  try {
    return (await api.bootstrapToken())?.configPath ?? null
  } catch {
    return null
  }
}
