# 本地对话(设备直连 LLM)可行性分析 — 已验证

> 由工作流 `on-device-llm-feasibility` 产出(5 项查证 → 3 项对抗复核 → 合成)。
> 关键鉴权/平台事实为高置信度并经对抗复核;回合制 HTTP 接口细节为中置信度,标注了需真机实测的未决项。
> 与之对应:版本一 = 后端中继(已实现,见 `QWEN_RELAY.md`);版本二 = 本地(本文分析的两种形态)。

## 1. 直接结论

**能，但要分形态且有一个硬前提。** 对话编排逻辑(`orchestrator.mjs` 那套 4 段隐藏计划 + 语言/语音选择)本身是纯字符串拼装,完全可以移到 RN 客户端跑,不需要中继;真正决定"中继能不能彻底去掉"的是两点:(a) 目标平台——**原生 Android 可以直连,Expo Web/kiosk 网页永远不能直连实时 WS**(浏览器 WebSocket 无法设置 `Authorization` 头,已核实);(b) 密钥托管——**把长期 DashScope key 塞进分发的 APK 是可提取的真实风险**。因此:demo/自持 kiosk 可以做到"零后端";量产则只能把中继**缩成一个无状态的临时 token 签发端点**,无法做到既去掉后端又保证密钥安全。

## 2. 两种形态

### Flavor A — 直连实时 WebSocket
设备直接开 `wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model=qwen3-omni-flash-realtime`,自己在 handshake 上挂 `Authorization: Bearer <key>`,并把中继现有的实时编排(server_vad、动态语音、`response.cancel`+重启、wrap-up、401/403 中国端点回退)全部搬到客户端。
- **原生 Android:可行(鉴权层面)。** RN 的 WebSocket 支持三参构造 `new WebSocket(url, protocols, { headers })`,Android 端 OkHttp 逐个 `addHeader`,可复刻中继 `relay.mjs:97-98`。
- **Web/kiosk 网页:不可行。** 浏览器 WebSocket 只接受 `url` 和 `protocols`;DashScope 不接受 query key、无 subprotocol 后门(与 OpenAI 的 `openai-insecure-api-key` 子协议逃生口不同)。
- **额外硬门槛:** 当前 hook 硬阻断 `Platform.OS !== 'web'`,整条是浏览器 Web Audio;RN 原生跑不了。`expo-audio` 的 `useAudioStream` 能做原生 PCM 采集,但**没有**原始 PCM 分片无缝播放 API——需自研或第三方 PCM 流式播放器。

### Flavor B — 回合制 HTTP
录一段 → base64 → ASR(`qwen3-asr-flash`,OpenAI 兼容端点)→ 追加进 `messages[]` → Qwen chat(system 放编排 prompt)→ TTS(`qwen3-tts-flash`,返回 24kHz 音频 URL)→ 播放。全程带 `Authorization: Bearer` 头的 `fetch` POST。
- **原生 Android:可行,无需自定义原生模块**(`expo-audio` 录/放 + `expo-file-system` 读 base64)。
- **Web/kiosk:可行(已实测 CORS)。** compatible-mode 与 multimodal-generation 端点 OPTIONS 预检 200,`allow-headers: authorization,content-type`。`fetch` 任何平台都能设 header——所以"浏览器不能设鉴权"只卡 Flavor A。
- **代价:** 三段串行往返,端到端 ~**1.5–4s/回合**(估算,须实测);丢掉 server-VAD、barge-in、话中换音、cancel/重启、wrap-up 注入。

## 3. 决策矩阵

| 方案 | 原生 Android | Web/kiosk | 延迟/体验 | 密钥安全 | 编排保真 | 工作量 |
|---|---|---|---|---|---|---|
| **保留中继(现状)** | 是 | 是(Web 唯一实时路径) | 最优,全双工可打断 | 最好(key 只在服务端) | 100% 已实现 | 0 |
| **A 直连实时 WS** | 是 | **否**(浏览器限制) | 与中继相当 | 差(key 落 APK)/临时 token 改善 | 高保真需重写 | 高 |
| **B 回合制 HTTP** | 是,无需原生模块 | 是(CORS 已放行) | +1.5–4s,无打断 | 差(同 A;web 更明文)/临时 token 改善 | 静态 prompt 全保,丢实时行为 | 中 |

## 4. 密钥安全(诚实说清)
- 落设备=可提取(APK 即 ZIP,JADX 反编译即得;混淆只抬成本不修复;TruffleHog 已有 DashScope key 检测器)。kiosk 网页最弱(JS 明文 + CORS 全开)。
- blast radius=整个账号(DashScope 限流账号级,无 per-key 上限;omni-flash 约 60 RPM/100k TPM 共享)。
- 自持 kiosk/demo:嵌 key 勉强可接受,前提=专用 Custom-scope key 限该模型 + IP 白名单 + 用新加坡区(美国弗吉尼亚不支持白名单/自定义 scope)+ 监控轮换。
- 量产/分发 APK:任何混淆下都不要发长期 key。
- 临时 key(`POST .../api/v1/tokens?expire_in_seconds=`,默认 60s、最长 1800s,继承父 key 权限、不可提前吊销)能把后端缩成无状态签发端点,但**签发仍需长期 key 在服务端** → 分发 App 去不掉最小后端。

## 5. 推荐
- 编排本身移到设备(纯逻辑,应该做)。
- **Demo/自持 kiosk → Flavor B**:真正零后端,web+原生通吃,无需原生模块,代价是延迟与无打断(单人问诊通常可接受)。
- **量产 Android → Flavor A + 极小 token 端点**:拿回全双工/低延迟,长期 key 不落端(设备用 1800s 临时 token 开 WS);代价是自研 PCM 流式播放 + 搬实时编排。
- **Web/kiosk 要实时:中继去不掉**(结构性限制)。

**须先实测的未决项:** ① `qwen3-asr-flash` 未明确列 Tamil / 闽南语,须验证;② 临时 key 能否用于实时 WS 握手未官方确认;③ SDK 56 `expo-audio` 录音稳定性;④ 端点 host 可能按 workspace/region 变化;⑤ Web `MediaRecorder` 出 webm/opus,可能需端上转 WAV 才能喂 ASR。

## 6. 仓库具体改法

**需移到 RN 客户端(TS):** `server/orchestrator.mjs`(纯字符串,`conversationFlow.json` 作静态资源打包)、`server/voice.mjs`(纯函数)。Flavor A 还需在客户端复刻 `relay.mjs` 的实时编排(`buildLiveSessionUpdate`/server_vad/动态换音+cancel 重启/wrap-up/401-403 回退);Flavor B 不需要这些。

**`server/` 去留:** Flavor B(kiosk)可删 `relay.mjs`;Flavor A/保护 key 则把中继降级为 `app/api/qwen-token+api.ts`(服务端用长期 key 调 `/api/v1/tokens` 签发临时 key)。`app/api/*+api.ts` 配对/持久化路由保留不动。

**hook:** Flavor B 新写 `useTurnBasedConversation.ts`(expo-audio 录/放 + 三段 fetch + `buildLiveInstructions` 拼 system);Flavor A 去 web 门禁、原生分支 `new WebSocket(url,[],{headers})` 直连 + `useAudioStream` 采集 + PCM 流式播放,**按 `Platform.OS` 分叉**(web 继续走中继)。

**文件清单:** 新增 `src/orchestration/{orchestrator,voice}.ts`、`src/prompts/conversationFlow.json`、`src/hooks/useTurnBasedConversation.ts`、(量产)`app/api/qwen-token+api.ts`;修改 `useQwenRealtimeConversation.ts`、`qwenConfig.mjs`、`server/index.mjs`;视形态删 `server/relay.mjs` 等;保留配对/Mongo 路由。
