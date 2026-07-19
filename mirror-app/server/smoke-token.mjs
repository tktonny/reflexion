// Verify DashScope short-lived token minting (the token-mint endpoint for v3/prod).
// Report flagged this as "strong inference, not confirmed" — confirm live.
// Run: node --env-file=.env server/smoke-token.mjs

const KEY = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
const H = { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' }
const mask = (s) => (s ? `${String(s).slice(0, 6)}…${String(s).slice(-4)} (len ${String(s).length})` : s)

// Documented forms vary; try the api-key token endpoint on the China host.
const candidates = [
  { name: 'GET tokens?expire_in_seconds', url: 'https://dashscope.aliyuncs.com/api/v1/tokens?expire_in_seconds=1800', method: 'GET' },
  { name: 'POST tokens?expire_in_seconds', url: 'https://dashscope.aliyuncs.com/api/v1/tokens?expire_in_seconds=1800', method: 'POST' },
  { name: 'POST apiKeys token', url: 'https://dashscope.aliyuncs.com/api/v1/apiKeys?action=create_temporary_key&expire_in_seconds=1800', method: 'POST' },
]

for (const c of candidates) {
  console.log(`\n=== ${c.name} (${c.method}) ===`)
  try {
    const res = await fetch(c.url, { method: c.method, headers: H, body: c.method === 'POST' ? JSON.stringify({}) : undefined })
    const text = await res.text()
    console.log('status', res.status)
    let token = null
    try { const j = JSON.parse(text); token = j?.data?.api_key || j?.api_key || j?.data?.token || j?.token; console.log('body keys:', Object.keys(j), 'token:', mask(token)) }
    catch { console.log('body:', text.slice(0, 240)) }
    if (token) { console.log('>>> MINTED TOKEN OK via', c.name); process.env.__MINTED = token; break }
  } catch (e) { console.error('error', e?.message) }
}
console.log('\n=== token smoke done ===')
