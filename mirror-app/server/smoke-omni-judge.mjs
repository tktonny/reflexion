// Does qwen3-omni-flash produce a text judgment via /compatible-mode (non-streamed)? If not, the
// client's assessViaOmni will throw and fall back to vl+plus — this decides the OMNI_JUDGMENT default.
import { qwenConfig } from './qwenConfig.mjs'
const KEY = qwenConfig.apiKey
const SYS = 'You are a clinical screening assistant. Return STRICT JSON only: {"risk_score":0..1,"risk_tier":"low|medium|high","screening_classification":"healthy|needs_observation|dementia","summary":"...","findings":[],"evidence_for_risk":[],"evidence_against_risk":[]}'
const USER = 'Transcript:\nAria: How are you today?\nPatient: I am well, I had breakfast with my daughter and we talked about the garden.'
async function tryCall(body, label) {
  try {
    const res = await fetch('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions', {
      method: 'POST', headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    })
    const j = await res.json()
    const txt = j?.choices?.[0]?.message?.content
    console.log(`${label}: HTTP ${res.status} ${res.ok ? 'OK' : 'ERR'} ${res.ok ? '| text len '+String(txt||'').length : '| '+JSON.stringify(j).slice(0,160)}`)
    if (res.ok && txt) console.log('   sample:', String(txt).replace(/\s+/g,' ').slice(0,140))
  } catch (e) { console.log(`${label}: threw ${e.message}`) }
}
await tryCall({ model:'qwen3-omni-flash', messages:[{role:'system',content:SYS},{role:'user',content:USER}], max_tokens:600, temperature:0.2 }, 'omni non-stream, no modalities')
await tryCall({ model:'qwen3-omni-flash', modalities:['text'], messages:[{role:'system',content:SYS},{role:'user',content:USER}], max_tokens:600, temperature:0.2 }, 'omni non-stream, modalities[text]')
