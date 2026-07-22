# webrtc-v0.0.0 使用说明

原生 **WebRTC** 直连 Qwen-Omni 实时。相对 `websocket-v0.0.0`,音频走 RTP media track,**libwebrtc 自带硬件级回声消除 + 降噪**——从根上消除镜子"扬声器→麦克风"的自问自答/答两次。编排逻辑(开场、四阶段、回忆 floor、自然收尾)与 websocket 版完全共用。

## 开启

`.env`:
```
EXPO_PUBLIC_CONVERSATION_MODE=webrtc
# WebRTC 端点需要 workspace 专属域名(通用 dashscope 域名只支持 WebSocket):
EXPO_PUBLIC_QWEN_WORKSPACE_ID=<你的 DashScope WorkspaceId>
EXPO_PUBLIC_QWEN_WEBRTC_REGION=cn-beijing      # 或 ap-southeast-1(新加坡)
# 或直接给整串 URL(优先级最高):
# EXPO_PUBLIC_QWEN_WEBRTC_URL=https://<ws>.<region>.maas.aliyuncs.com/api/v1/webrtc/realtime
```

> ⚠️ 没有 `WORKSPACE_ID`(或整串 URL)时,连接会失败并提示"未配置 WebRTC workspace 域名"。这是当前唯一的外部依赖——把你的 WorkspaceId 填进去即可。

## 连接原理(实现细节)

1. `RTCPeerConnection` + `getUserMedia({audio})` 本地麦克风轨 + `oai-events` DataChannel。
2. `createOffer` → 等 ICE 收集完成 → 把 offer SDP `POST` 到端点(`Content-Type: application/sdp`,`Authorization: Bearer <token>`)。
3. 用返回的 answer SDP `setRemoteDescription`,连接建立。
4. **音频**:上行=本地麦克风轨(libwebrtc 做 AEC/NS/AGC);下行=Aria 的 RTP 轨,自动播放。
5. **事件**:`session.update`、转写、`response.*` 生命周期全走 DataChannel。
6. 只支持 server_vad / `semantic_vad`(我们用 semantic_vad),不支持手动模式——本来就要全自动。

## 与 websocket-v0.0.0 的差异

| | websocket-v0.0.0 | webrtc-v0.0.0 |
|---|---|---|
| 音频传输 | RN WebSocket + 本地 PCM 模块 | WebRTC RTP media track |
| 回声消除 | 无(半双工静音 + 软件转写抑制,模拟器仍漏) | **内置硬件级 AEC + 降噪** |
| 自问自答/答两次 | 无 AEC 时会 | 不会(根治) |
| 依赖 | expo-pcm-audio | react-native-webrtc(原生) + workspace 域名 |
| 编排/收尾逻辑 | 共用 | 共用 |

## 构建

```bash
cd REFLEXION/mirror-app
EXPO_PUBLIC_CONVERSATION_MODE=webrtc npx expo prebuild --platform android   # 应用 @config-plugins/react-native-webrtc
# 确认 android/gradle/wrapper/gradle-wrapper.properties = gradle-8.14.3(RN 0.85 要求;9.x 会让 onnxruntime 编译失败)
cd android && ./gradlew assembleRelease -PreactNativeArchitectures=arm64-v8a
```
产物:`dist-apks/reflexion-webrtc-v0.0.0.apk`。

## 验收(真机)

- 回声:免手对话,Aria **不应**接自己的话/答两次(硬件 AEC)。
- 四阶段推进、第 3 轮回忆、5 轮内自然告别 —— 与 websocket 版一致。
- 若连接失败:多半是 WorkspaceId/region 没配对,查 logcat 里的 `webrtc_sdp_<状态码>`。
