// Pure capability → conversation-version decision table. No react-native / no env imports, so it
// runs headlessly (see server/smoke-hwcheck.mjs) and is the single place the routing rule lives.
export type RecommendedMode = 'relay' | 'http' | 'ws' | 'none'
export type Capabilities = { micOk: boolean; relayOk: boolean; turnOk: boolean; rtOk: boolean }

export function recommendMode(platform: string, caps: Capabilities): { mode: RecommendedMode; reason: string } {
  const { micOk, relayOk, turnOk, rtOk } = caps
  if (platform === 'web') {
    if (relayOk && micOk) return { mode: 'relay', reason: 'web + 中继可达 + 麦克风就绪 → 实时(v1)' }
    if (turnOk) return { mode: 'http', reason: 'web 无中继 → 回合制(v2,直连)' }
    return { mode: 'none', reason: 'web 但麦克风/音频不可用' }
  }
  if (rtOk && micOk) return { mode: 'ws', reason: '原生 + PCM 流 + 麦克风 → 直连实时(v3)' }
  if (turnOk && micOk) return { mode: 'http', reason: '原生 + expo-audio + 麦克风 → 回合制(v2)' }
  if (relayOk && micOk) return { mode: 'relay', reason: '原生回退中继(v1)' }
  return { mode: 'none', reason: '麦克风或音频能力不足' }
}
