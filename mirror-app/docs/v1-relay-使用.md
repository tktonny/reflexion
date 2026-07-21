# 版本 1（relay）· 网页版 · 安装与使用

**一句话**：在电脑上跑一个中转服务器 + 用 **Chrome** 打开网页，全双工实时和 Aria 对话。这是**唯一的网页版**，也是 v1 唯一能跑的形态。

## 为什么 v1 是网页、需要服务器

- v1 的音频引擎是**基于浏览器的**（`useQwenRealtimeConversation`），在原生设备上会直接抛错，所以**不能打成安卓 APK**。
- 浏览器无法给 WebSocket 加鉴权请求头，Qwen 实时接口会拒绝（401），所以必须有一个**中转服务器**（relay）在中间补上 `Authorization` 头。
- 结论：v1 = 桌面 Chrome + 本机 relay。真机场景请改用 **v3**（同样是全双工，但不需要服务器）。

---

## 一、准备（一次性）

1. 安装 Node.js（18+）。
2. 进项目：
   ```bash
   cd /Users/macbookair/Documents/Cloud/REFLEXION/mirror-app
   npm install
   ```
3. 配置 `.env`（项目根，已 gitignore）。v1 需要：
   ```
   EXPO_PUBLIC_CONVERSATION_MODE=relay
   EXPO_PUBLIC_RELAY_WS_URL=ws://localhost:8787
   QWEN_API_KEY=sk-...              # relay 服务器用它连 Qwen
   REFLEXION_RELAY_PORT=8787        # 可选，默认 8787
   ```
   > 注意 `EXPO_PUBLIC_CONVERSATION_MODE` 影响**网页构建**，改了要重启 `npm run web`。

---

## 二、启动（开两个终端）

**终端 1 — 中转服务器：**
```bash
npm run relay
```
它会先编译 orchestration，再启动 `server/index.mjs`（监听 8787）。看到监听日志即就绪。

**终端 2 — 网页：**
```bash
npm run web
```
Expo 会打印一个本地地址（如 `http://localhost:8081`）。

---

## 三、使用

1. 用 **Chrome** 打开上面的地址。
2. 页面走到配对页后，点金色按钮 **「开始每日检查(演示)→」**（进入 `/realtime-test`）。
   （也可直接访问 `http://localhost:8081/realtime-test`。）
3. 点 **「开始对话」**，浏览器弹出麦克风授权 → **允许**。
4. Aria 会**先开场**，然后**直接自然对话即可**——v1 是全双工流式，不用按任何键，你说话它就在听，边说边回。
5. 按 Aria 引导走完 4 个阶段。**Aria 说出告别语后会自动结束并生成判断**；也可随时点 **「结束并评估」**。
6. 页面下方出现判断卡片（分级/风险分/证据/视觉观察）。

> 视觉观察：网页端也会通过摄像头采样面部帧做多模态筛查（需允许摄像头权限）。

---

## 四、排错

| 现象 | 原因 / 处理 |
|------|-------------|
| 一直"连接中"或报鉴权错误 | relay 没起来，或 `.env` 里 `QWEN_API_KEY` 无效。确认终端 1 在跑、key 有效。 |
| 页面报"web only / 原生不支持" | 你在原生环境打开了 v1。v1 只支持 Chrome；真机请用 v3。 |
| 麦克风无反应 | Chrome 地址栏检查麦克风权限；`localhost` 才被视为安全上下文（非 localhost 需 HTTPS）。 |
| 换了 mode 不生效 | `EXPO_PUBLIC_*` 是构建期注入，改 `.env` 后要重启 `npm run web`。 |
