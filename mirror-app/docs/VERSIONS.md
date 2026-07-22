# 对话版本(一套代码库,开关切换)

镜子端的对话链路有多种实现,靠 `EXPO_PUBLIC_CONVERSATION_MODE` 切换,各版对 UI 暴露同一接口(`ConversationApi`),`conversation.tsx` / `realtime-test.tsx` 经 `useConversation` 选择器零改动复用。命名按**传输方式**(见 `VERSION_LABELS`)。

| 版本 | mode | 传输 | 回声处理 | 平台 | 验证 |
|---|---|---|---|---|---|
| relay-v0.0.0 | `relay` | 客户端 ↔ Node 中继 ↔ Qwen 实时 WS | 浏览器 AEC(web) | web + 原生 | `server/smoke.mjs` ✅ |
| http-v0.0.0 | `http` | 设备直调 ASR→chat→TTS(HTTPS,回合制) | 半双工(回合制天然无回声) | web + 原生* | `smoke-turnbased/turnloop.mjs` ✅ |
| **websocket-v0.0.0** | `ws` | 原生 RN WebSocket 直连 Qwen 实时(header 鉴权) | 半双工静音 + 转写级回声抑制(软件,模拟器仍会漏) | 仅原生(web 回退 relay) | `smoke-direct-ws.mjs` ✅ |
| **webrtc-v0.0.0** | `webrtc` | 原生 WebRTC 直连 Qwen(SDP 握手 + `oai-events` DataChannel) | **内置硬件级 AEC + 降噪(回声正解)** | 仅原生(web 回退 relay) | 真机验收 |

\* v2 原生音频用 expo-audio;websocket-v0.0.0 原生流式音频由本地模块 `modules/expo-pcm-audio` 实现。

**websocket vs webrtc**:两者都直连 Qwen 实时、编排逻辑(开场/四阶段/回忆floor/收尾)完全共用。区别只在**音频传输层**——WebSocket 需客户端自己处理回声(无 AEC,模拟器上 Aria 会听到自己→自问自答/答两次);**WebRTC 的音频走 RTP,libwebrtc 自带回声消除,从根上消除自问自答**。WebRTC 端点需 workspace 专属域名:设 `EXPO_PUBLIC_QWEN_WORKSPACE_ID`(+可选 `EXPO_PUBLIC_QWEN_WEBRTC_REGION`)或整串 `EXPO_PUBLIC_QWEN_WEBRTC_URL`。

## 切换与运行

编辑 `.env` 的 `EXPO_PUBLIC_CONVERSATION_MODE`。本地模式(http/ws)需要客户端可见 key:`EXPO_PUBLIC_QWEN_API_KEY`(仅 kiosk/demo);生产走 `/api/qwen-token` 短期 token。

```bash
cd REFLEXION/mirror-app

# 版本一(relay):
#   .env: EXPO_PUBLIC_CONVERSATION_MODE=relay
npm run relay        # 终端A,中继 :8787(读服务端 QWEN_API_KEY)
npm run web          # 终端B → http://localhost:8081/realtime-test → Start

# 版本二(http,回合制,浏览器可完整跑):
#   .env: EXPO_PUBLIC_CONVERSATION_MODE=http  (EXPO_PUBLIC_QWEN_API_KEY 已写入)
npm run web          # /realtime-test → Start(Aria 开场)→ 🎤 开始说话 → 发送 → 循环

# 版本三(ws,原生直连,需真机 dev build):
#   .env: EXPO_PUBLIC_CONVERSATION_MODE=ws
#   token 端点由 Expo 服务端 +api.ts 提供(npm run web 时可用),或部署后端
#   原生 PCM 音频已由 modules/expo-pcm-audio 提供(autolink);dev build 到真机即可
#   → npx expo run:android  (详见 docs/ANDROID_BUILD.md 方式 D + 验收清单)
```

## 无需真机就能跑的验证(已全部 PASS)
```bash
node --env-file=.env server/smoke.mjs            # v1: 中继→Qwen
node --env-file=.env server/smoke-turnbased.mjs  # v2: chat/tts/asr 三段
node --env-file=.env server/smoke-turnloop.mjs   # v2: 4 阶段编排大脑
node --env-file=.env server/smoke-token.mjs      # token 签发
node --env-file=.env server/smoke-direct-ws.mjs  # v3: 临时token→直连WS→编排→audio
```

## 关键文件
- 开关/配置:`src/config/conversationMode.ts`,选择器 `src/hooks/useConversation.ts`
- 共享编排(TS 移植):`src/orchestration/{orchestrator,voice,realtime}.ts` + `conversationFlow.json`
- v1:`src/hooks/useQwenRealtimeConversation.ts` + `server/*.mjs`
- v2:`src/hooks/useTurnBasedConversation.ts` + `src/api/qwenClient.ts`(HTTP 形状已实测)
- v3:`src/hooks/useDirectRealtimeConversation.ts` + `src/native/pcmAudio.ts`(音频桥)+ `app/api/qwen-token+api.ts`

## 已验证的关键事实(见 docs/ON_DEVICE_LLM.md)
- key 为**中国区** → 所有 HTTP/WS 走 `dashscope.aliyuncs.com`。
- **临时 token 可用于实时 WS 握手**(v3 据此避免长期 key 落端)。
- 浏览器 WebSocket 无法设鉴权头 → **web 的 ws 模式必然回退 relay**。
- v2 HTTP:chat=`qwen-plus`、tts=`qwen-tts`、asr=`qwen3-asr-flash`(compatible-mode + multimodal-generation)。

## 尚未接入(设备侧,唯一剩余)
- **v3 原生 PCM 音频**:`src/native/pcmAudio.ts` 是接口 + 抛错占位。需装原生模块(如 `react-native-audio-api` 或 `@fugood/react-native-audio-pcm-stream` + 原生 AudioTrack 播放)并实现 `createPcmAudioBridge`,再 EAS dev build 到真机。WS/编排逻辑已完整并验证,仅缺这一层。
- **v2 原生录音/播放**:现为 web-only;原生需 `expo-audio` 录/放(回合制,较简单)。
- 待实测语种:`qwen3-asr-flash` 对 Tamil / 闽南语覆盖需真机验证。
