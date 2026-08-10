/**
 * Demo builds are intentionally opt-in at bundle time. The normal production APK does not contain
 * the entry point, so a production credential or route cannot be switched into demo mode at runtime.
 */
export const DEMO_BUILD_ENABLED = process.env.EXPO_PUBLIC_ENABLE_DEMO_MODE === 'true'

export const DEMO_STORAGE_NAMESPACE = 'reflexion:demo:'

export function isDemoRoute(value: unknown): boolean {
  return DEMO_BUILD_ENABLED && (value === '1' || value === 'true' || value === true)
}
