# 真机/真人测试指南(Mac + Android)

测试屏 `/realtime-test` 现在一屏包含:**实时字幕(模型反馈)+ 4 阶段对话推进(你的 prompt)+ 结束后的结构化判断结果**(风险分/分级/分类 + 证据)。判断由服务端 `/api/assess` 调 Qwen 产出(已实测:健康对话→low/healthy,受损对话→high/dementia,有区分度)。

> `.env` 里切 `EXPO_PUBLIC_CONVERSATION_MODE`:`relay`=v1实时 / `http`=v2回合制 / `ws`=v3原生直连。

---

## A. 在 Mac 上测(今天即可,已端到端验证)

**A1. v1 实时(最佳体验,推荐先测这个)**
```bash
cd REFLEXION/mirror-app
# .env: EXPO_PUBLIC_CONVERSATION_MODE=relay
npm run relay        # 终端A:Qwen 中继 :8787
npm run web          # 终端B:Expo web :8081
# 浏览器打开 http://localhost:8081/realtime-test
# 点“开始对话”→ 允许麦克风 → 说话(Aria 先开场,按 4 阶段推进)→ 点“结束并评估”→ 看判断结果
```
localhost 是"安全上下文",浏览器麦克风可用。

**A2. v2 回合制(按键说话)**
```bash
# .env: EXPO_PUBLIC_CONVERSATION_MODE=http
npm run web          # 无需 relay
# /realtime-test → 开始对话 → 🎤开始说话 → 发送 → 循环 → 结束并评估
```

---

## B. 在 Android 上测

### ⚠️ 关键限制
手机 Chrome 打开 **LAN 明文 http**(如 `http://192.168.x.x:8081`)时,浏览器**禁用麦克风**(非安全上下文)。所以 Android 上有两条真正可行的路:

### B1. 今天就能测(无需打包)—— https 隧道 + v2
```bash
# .env: EXPO_PUBLIC_CONVERSATION_MODE=http
npm i -g @expo/ngrok        # 首次
npx expo start --tunnel     # 生成一个 https://xxxx.exp.direct 隧道
# 手机 Chrome 打开该 https 地址 + /realtime-test
# → https 满足麦克风安全上下文;v2 是纯 HTTPS(录音→ASR→chat→TTS + /api/assess),无需 relay
# → 真实语音交互 + 判断结果都在手机上跑
# 可在 Chrome“添加到主屏幕”当 PWA,体验接近已安装
```
为什么用 v2:v2 全程普通 HTTPS,隧道一给 https 就通;v1 实时还需把 :8787 的 wss 也暴露给手机,较麻烦。

### B2. 安装 APK(v2 原生音频已接入 · 需你的 Expo 账号云构建)
v2 原生录放已用 `expo-audio` 接好(录 m4a → base64 → Qwen ASR → chat → 播 TTS);判断在无后端的独立 APK 上会自动走**客户端直连兜底**,所以 APK 自包含可跑。`app.json`(android.package/权限/expo-audio 插件)与 `eas.json`(preview=apk, mode=http)已就绪。
```bash
cd REFLEXION/mirror-app
npx eas-cli login
npx eas-cli init          # 首次:写入 projectId 到 app.json
# 把 Qwen key 作为 preview 构建期变量(会内联进 APK,仅自测用):
npx eas-cli env:create --name EXPO_PUBLIC_QWEN_API_KEY --value "<你的 DASHSCOPE key>" --environment preview --visibility sensitive
npx eas-cli build -p android --profile preview   # 云构建 → 下载 APK 安装
```
装好后打开 App:首屏是配对屏(需 MongoDB,可忽略)→ 点最下方**「开始每日检查(演示)」**进入测试屏 → 「开始对话」→ 🎤 开始说话 / 发送(v2 按键说话)→ 「结束并评估」看判断卡片。

> ⚠️ v2 原生音频**尚未真机验证**(本环境无安卓设备):代码已 typecheck 通过,但 `expo-audio` 录音格式 / ASR 对 m4a 的接受度可能要微调。装上后若录音或 ASR 报错,把报错发我即修。
> **v3 原生实时**(直连 WS 全双工)仍需流式 PCM 模块,较难,后续(见 `docs/VERSIONS.md`)。

---

## 你要重点看什么
1. **实时反馈**:Aria 是否即时出声/出字幕(v1/v3 流式;v2 每回合 1.5–4s)。
2. **按 prompt 推进**:是否走 问候→近况→日常功能→回忆 四阶段、每次一个问题、简短、不诊断。
3. **判断是否准确**:结束后的卡片——分类(healthy/needs_observation/dementia)、风险分、证据是否合理。可点"重新评估这段对话"复评。

## C. 硬件自检(启动自动运行,无需真机)
每次启动 app(`app/_layout`)都会自动跑一次硬件自检并打日志;也可在 `/hardware-check` 屏查看,或从
`/realtime-test` 顶部“🔧 硬件自检”进入。检查项:网络、后端/中继、麦克风、扬声器、摄像头、
回合制音频(v2)、实时音频(v3 原生流式),并据此**推荐可用的对话版本**(relay/http/ws/none)。
- **web(Mac 上现在就能看真实结果)**:麦克风/网络/摄像头走浏览器 API,relay 走 `/health` 探测。
- **原生(真机才有真实结果)**:麦克风/摄像头走系统权限,v3 需要原生 PCM 模块(现为未接入→标红)。
- v3 就绪门:接入 `src/native/pcmAudio.ts` 后把 `HAS_NATIVE_PCM_STREAM` 置 true,自检即转绿并推荐 ws。

## 已验证(无需真机的冒烟)
```bash
node --env-file=.env server/smoke.mjs            # v1 relay→Qwen
node --env-file=.env server/smoke-turnbased.mjs  # v2 chat/tts/asr
node --env-file=.env server/smoke-turnloop.mjs   # v2 4阶段编排
node --env-file=.env server/smoke-direct-ws.mjs  # v3 token→直连WS
node --env-file=.env server/smoke-assess.mjs     # 判断结果(健康 vs 受损)
node server/smoke-hwcheck.mjs                     # 硬件自检决策表(7 组合全 PASS)
node --env-file=.env server/smoke-vision.mjs      # 多模态视觉筛查(qwen-vl-max 收图+返回完整判断)
```

## 视频输入(多模态认知筛查)
每日检查在语音之外还接入了**摄像头**:界面显示实时镜面预览(`src/components/MirrorCameraPanel.tsx`),
会话进行中每 ~8 秒采样一帧(前置、低分辨率),结束评估时连同转写一起送 **qwen-vl-max**,判断卡片
多出「视觉观察」一栏(参与度/情绪/警觉,**非**依据外貌诊断;分类仍以对话为准)。
- web:`npm run web` → `/realtime-test` → 允许摄像头 → 预览出现、结束后卡片含视觉观察。
- 原生:同一套 `CameraView`,dev build/APK 上自动生效(`app.json` 已含 CAMERA 权限)。
- 帧数上限 6(客户端 + 服务端各自 cap),控制体积与成本。
