# Android 安装 / 打包指南(mirror device app)

本仓库已 **build-ready**:`app.json` 有 `android.package = com.reflexion.mirror` + `versionCode`,
并提供 `eas.json`(preview 出 APK)。因本机无 Android SDK 且 JDK 为 23(RN 0.85 需 JDK 17),
**本地 Gradle 打包不可行**,故走 EAS 云构建。

---

## ⚠️ 先读:当前 Android 原生能力边界

镜子端由三块组成,APK 里**只打包了 UI**,另两块要可达:
1. **Qwen 语音中继**(`server/*.mjs`)—— 独立 Node 进程,不在 APK 内。
2. **配对/持久化 API**(`app/api/**+api.ts`)—— 这是 Expo Router **服务端**路由(`web.output:"server"`);
   原生 APK **不自带**这个服务端,需部署或指向 dev server。
3. **实时音频**:
   - v1 中继 / v2 回合制在 web 用 Web Audio(已验证);v2 原生用 expo-audio。
   - **v3 原生直连**的流式音频现由本地 Expo 模块 `modules/expo-pcm-audio` 提供
     (Android:AudioRecord 16k 采集 + AudioTrack 24k 播放 + AEC/NS/AGC 回声抑制;iOS:AVAudioEngine)。
     **代码已完整**,通过 `modules/` 自动 autolink 进原生工程;但**只能在 dev build/真机上验证**
     (Expo Go / web 加载不到该原生模块,`requireOptionalNativeModule` 返回 null → 自动回退到清晰报错的 stub)。

**结论:**
- **今天就能在 Android 上完整跑通语音的最快形式 = 用设备浏览器(Chrome)以 kiosk 打开 Web 版**(见下"方式 C")。
- **EAS/本地 dev build** 现在会把 `expo-pcm-audio` 一并编入,可在真机上验证 **v3 原生直连语音**(见下"方式 D — v3 原生 dev build")。
- 纯 **preview APK** 也能安装、走开机自检/配对 UI;要 v3 语音请用 dev build 或 preview(两者都含该原生模块)。

---

## 方式 A — EAS 云构建 APK(需要一个免费 Expo 账号)

```bash
cd REFLEXION/mirror-app

# 1) 装 CLI(或全程用 npx eas-cli)
npm i -g eas-cli            # 可选

# 2) 登录你的 Expo 账号(交互式,只有你能做)
npx eas-cli login

# 3) 初始化项目(写入 app.json 的 extra.eas.projectId)
npx eas-cli init

# 4) 把中继地址指到你电脑的局域网 IP(编辑 eas.json 的 preview.env.EXPO_PUBLIC_RELAY_WS_URL,
#    例如 ws://192.168.1.50:8787;查 IP:  ipconfig getifaddr en0 )

# 5) 出 APK(云端构建,完成后给下载链接)
npx eas-cli build -p android --profile preview
```
下载 APK 后:手机开启"未知来源安装",或 `adb install app.apk`。

## 方式 B — 本地 Gradle APK(需自备 Android SDK + JDK 17)
```bash
# 装 JDK 17 与 Android SDK 后:
cd REFLEXION/mirror-app
npx expo prebuild -p android         # 生成 android/ 原生工程
cd android && ./gradlew assembleRelease
# 产物: android/app/build/outputs/apk/release/app-release.apk
```

## 方式 C — Web / Chrome kiosk(推荐用于智能镜子,今天即可完整运行含语音)
智能镜子本质是一块屏,用设备浏览器全屏打开 Web 版最省事,且**语音在 Web 路径已验证可用**。
```bash
# 主机上(同一局域网):
cd REFLEXION/mirror-app
QWEN_API_KEY=...  npm run relay       # 终端 A:中继 :8787
npm run web                            # 终端 B:Web 应用 :8081(或部署静态导出)
# 镜子设备 Chrome 打开  http://<主机IP>:8081/realtime-test  → 允许麦克风 → 讲话
# 生产可用 Chrome 的"添加到主屏幕"(PWA)或 kiosk/全屏 App 包装。
```
> 局域网 http 下 getUserMedia 可能受"安全上下文"限制;`localhost` 可用,跨机建议 https 或在 Chrome 里
> 将该来源加入 `chrome://flags/#unsafely-treat-insecure-origin-as-secure` 白名单(仅测试)。

## 方式 D — v3 原生 dev build(真机验证流式实时语音)
`modules/expo-pcm-audio` 是**本地 Expo 原生模块**,`expo prebuild` / `expo run:android` / EAS 会自动 autolink,
无需 `npm install`。它实现 `src/native/pcmAudio.ts` 依赖的 PcmAudioBridge。

```bash
cd REFLEXION/mirror-app

# 1) 选 v3 模式 + 指向 token 端点(生产走短期 token;kiosk 自测可开客户端 key)
#    .env: EXPO_PUBLIC_CONVERSATION_MODE=ws
#    (原生直连需要一个能签发 /api/qwen-token 的可达后端;或临时 EXPO_PUBLIC_ALLOW_INSECURE_CLIENT_KEY=true)

# 2) 连真机(USB 调试)后,一条命令生成原生工程 + 编入 expo-pcm-audio + 安装运行:
npx expo run:android            # 需本地 Android SDK + JDK 17
#    或云端:npx eas-cli build -p android --profile development  然后装到真机

# 3) 装好后在设备上打开 → 开机会自动跑硬件自检(见 /hardware-check):
#    "实时音频(v3 原生流式)" 应从红变绿(原生 PCM 模块已加载),推荐版本应变成 WS。
```

### 真机验收清单(v3)
- [ ] `/hardware-check`:麦克风 ✓、实时音频(v3)✓、推荐版本 = WS。
- [ ] `/realtime-test` 选到 `ws`:点开始 → 授权麦克风 → Aria 先开场(流式出声,延迟应 < ~1s)。
- [ ] 讲话时 Aria 能听懂并按 4 阶段推进(编排已用 `smoke-direct-ws` 验证);对方说话时不自我回声(AEC + 半双工静音生效)。
- [ ] 结束 → 生成判断卡片 → 写入护理端(与 v1/v2 相同的 `/api/conversations` 保存路径)。
- [ ] 观察 logcat `pcm-capture`/`pcm-playback` 线程无异常;结束后无残留(stop 释放 AudioRecord/AudioTrack)。

> 若自检里 v3 仍为红:多半是跑在 Expo Go(加载不到原生模块)或没 prebuild。用 `expo run:android` 出的是含原生模块的 dev build。

---

## 让 APK 真正可用的后续项(按序)
1. ~~原生音频~~ ✅ 已完成:`modules/expo-pcm-audio`(AudioRecord 16k / AudioTrack 24k + AEC/NS/AGC)。**待真机验收**(方式 D)。
2. **部署后端**:把 `app/api/**+api.ts`(配对/Mongo)与 `server/`(Qwen 中继)部署到常驻云 Node 服务,
   APK 用固定 https/wss 地址访问(`EXPO_PUBLIC_RELAY_WS_URL` + `getApiUrl` 基址)。
3. 之后 production `app-bundle`(.aab)上架或内部分发。

版本:`app.json` `android.versionCode` 每次发布 +1。
