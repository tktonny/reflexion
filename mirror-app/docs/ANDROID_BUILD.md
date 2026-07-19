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
3. **实时音频**目前是 **Web-only**(`useQwenRealtimeConversation` 在原生上会抛"web only")。
   原生要能语音,需先补 `MirrorAudio` 原生模块(见蓝图 M0/M2)。

**结论:**
- **今天就能在 Android 上完整跑通语音的最快形式 = 用设备浏览器(Chrome)以 kiosk 打开 Web 版**(见下"方式 C")。
- **EAS APK** 现在能安装、走开机自检/配对 UI,但语音 check-in 要等原生音频模块;适合先验证安装与界面。

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

---

## 让 APK 真正可用的后续项(按序)
1. **原生音频 `MirrorAudio`**(AudioRecord 16k / AudioTrack 24k + AEC/NS)→ 语音在原生可用。
2. **部署后端**:把 `app/api/**+api.ts`(配对/Mongo)与 `server/`(Qwen 中继)部署到常驻云 Node 服务,
   APK 用固定 https/wss 地址访问(`EXPO_PUBLIC_RELAY_WS_URL` + `getApiUrl` 基址)。
3. 之后 production `app-bundle`(.aab)上架或内部分发。

版本:`app.json` `android.versionCode` 每次发布 +1。
