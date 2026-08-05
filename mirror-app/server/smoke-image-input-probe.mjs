// 探测 qwen omni realtime 是否接受图像输入(摄像头可行性的决定性测试)。
// modalities 不接受 'video'(已实测),所以问题是:图像能否作为 INPUT 进入会话?
import { WebSocket } from 'ws'
import { readFileSync } from 'node:fs'
import { qwenConfig } from './qwenConfig.mjs'

const KEY = process.env.REPRO_KEY || qwenConfig.apiKey
const HOST = process.env.REPRO_HOST || qwenConfig.realtimeUrlChina
const MODEL = process.env.REPRO_MODEL || 'qwen3.5-omni-flash-realtime'
const IMG = readFileSync('/tmp/probe.jpg').toString('base64')

async function mint() {
  const r = await fetch('https://dashscope.aliyuncs.com/api/v1/tokens?expire_in_seconds=1800',
    { method: 'POST', headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' }, body: '{}' })
  return (await r.json())?.token
}

// 两种可能的图像输入方式都试
const WAYS = {
  'input_image_buffer.append (relay 白名单里的那个)': (ws) =>
    ws.send(JSON.stringify({ type: 'input_image_buffer.append', image: IMG })),
  'conversation.item.create 带 input_image': (ws) =>
    ws.send(JSON.stringify({ type: 'conversation.item.create', item: { type: 'message', role: 'user',
      content: [{ type: 'input_image', image: IMG }, { type: 'input_text', text: 'What do you see?' }] } })),
}

for (const [label, sendImage] of Object.entries(WAYS)) {
  const token = await mint()
  const result = await new Promise((resolve) => {
    const ws = new WebSocket(`${HOST}?model=${MODEL}`, { headers: { Authorization: `Bearer ${token}` }, maxPayload: 0 })
    let done = false
    const finish = (r) => { if (done) return; done = true; clearTimeout(t); try { ws.close() } catch {}; resolve(r) }
    const t = setTimeout(() => finish('无错误(6s 内接受)'), 6000)
    ws.on('open', () => ws.send(JSON.stringify({ type: 'session.update', session: {
      modalities: ['text', 'audio'], voice: 'Serena', instructions: 'You can see images.',
      input_audio_format: 'pcm', output_audio_format: 'pcm',
    } })))
    ws.on('message', (d) => {
      let m; try { m = JSON.parse(d.toString()) } catch { return }
      if (m.type === 'session.updated') { sendImage(ws); setTimeout(() => ws.send(JSON.stringify({ type: 'response.create' })), 200) }
      if (m.type === 'error') finish(`ERROR ${m.error?.code || '?'}: ${String(m.error?.message).slice(0, 150)}`)
      if (m.type === 'response.audio_transcript.delta' || m.type === 'response.text.delta') finish(`回复了: ${String(m.delta).slice(0,60)}`)
    })
    ws.on('error', (e) => finish(`ws-error ${e?.message}`))
  })
  console.log(`  ${label}\n    → ${result}\n`)
}
process.exit(0)
