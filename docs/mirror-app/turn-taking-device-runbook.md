# Turn-taking 真机验收手册

> 对应 Phase 1。MuMu 可关闭逻辑、Android 生命周期和长时间稳定性 gate；目标 Android 镜面硬件仍必须验证扬声器回声、物理播放结束和麦克风重新开放时机。已完成的模拟器结果见 [Phase 1 MuMu 验收报告](./phase1-mumu-acceptance.md)。

## 1. 当前候选构建

- transport：`ws`（Native Qwen WebSocket + `expo-pcm-audio`）
- model：`qwen3.5-omni-flash-realtime`
- 本地构建命令：`cd mirror-app/android && ./gradlew :app:assembleDebug`
- APK：`mirror-app/android/app/build/outputs/apk/debug/app-debug.apk`
- 安装：`adb install -r mirror-app/android/app/build/outputs/apk/debug/app-debug.apk`

安装前确认 `.env` 中 `EXPO_PUBLIC_CONVERSATION_MODE=ws`。测试凭据只能用于受控开发设备；生产版本必须改为统一后端签发的短期 token。

## 2. 采集日志

```bash
adb logcat -c
adb logcat -v threadtime | rg "turn-taking|ExpoPcmAudio|ReactNativeJS"
```

测试页会显示当前 lifecycle state。日志只包含事件、状态、mic ownership 和播放 backlog 时间，不包含患者正文或原始音频。

每个正常回合应出现：

```text
user_speech_started
user_speech_stopped
response_requested
response_created
audio_delta
response_done
playback_drained
mic_reopened
```

`response_done` 与 `mic_reopened` 之间必须有 `playback_drained`，并且实际扬声器声音已经结束。

direct WS 使用本地能量 VAD 和 Qwen manual turn：检测到 1200 ms 持续静音后先 mute，再发送 `input_audio_buffer.commit`；收到 transcription 并完成 `session.updated` 后才发送 `response.create`。不要把 provider semantic VAD 的自动 response 当作此路径的验收基线。

## 3. 30 回合 smoke matrix

| 回合 | 场景 | 期望 |
|---:|---|---|
| 1–5 | 普通短问答 | 每次只生成一条回复；无自问自答 |
| 6–10 | 用户句中停顿 0.5–1.0 秒 | 不在停顿处抢答；完整收到用户句子 |
| 11–15 | 要求 2–3 句较长回答 | Aria 每句完整播完；无 token 截断 |
| 16–20 | 电视/环境人声与正常距离 | 非目标声音不形成连续虚假回合；出现问题时保持 fail-closed |
| 21–25 | Screening 第 6–7 回合 | 前 6 个回答完整覆盖定向、近期叙事和日常功能；第 6 个回答播放完成后才发 recall；第 7 个回答后才告别 |
| 26–30 | 中文、英文及一次语言切换 | voice/language 更新不产生重叠 response |

每回合记录：是否听完、是否抢话、是否回声自触发、`response.done → playback_drained` 毫秒数、`playback_drained → mic_reopened` 毫秒数和任何 violation。

## 4. 手动结束矩阵

分别在以下时刻点击“结束并评估”：

1. 正在 listening、用户尚未说话；
2. 用户正在说话；
3. `assistant_generating`；
4. `assistant_playing`；
5. `playback_guard`。

每种情况必须只有一条完整 goodbye。当前普通/assistant 回合若已开始，先播完它，再生成 goodbye；goodbye 播完后才 teardown 和评估。不得出现固定 3 秒截断或零 backlog 误判提前退出。

## 5. Soak

- 30 分钟自然会话；
- 100 回合录音/真人交替测试；
- 锁屏、前后台切换和一次 Wi-Fi 短断；
- 统计重叠 response、截断、虚假 user turn、stuck-muted、playback timeout 和 provider error。

通过标准：重叠 response、截断和提前开麦均为 0；任何真实播放卡死必须保持麦克风关闭并产生清晰错误，不得静默恢复为 listening。

## 6. WebRTC 准入

WebRTC 当前不是 Phase 1 默认路径。只有它能提供或可靠推导等价的本地 playout-complete 信号，并在同一 30 回合/30 分钟/100 回合矩阵中达到相同结果，才允许替换 WebSocket 基线。
