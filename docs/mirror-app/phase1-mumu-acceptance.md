# Phase 1 MuMu 验收报告

> 验收日期：2026-07-22
> 结论：MuMu 逻辑与 Android 设备模拟门通过；目标镜面声学门待执行
> 范围：native Qwen realtime WebSocket 主路径的 turn-taking，不包含 Phase 5 唤醒词产品验收

## 1. 环境与方法

| 项目 | 值 |
|---|---|
| 模拟器 | MuMu，Android 12，model `2312DRA50C` |
| 分辨率 | 1440 × 2560 |
| App | `com.reflexion.mirror` |
| Transport / model | native WebSocket / `qwen3.5-omni-flash-realtime` |
| 输入 | 开发模式专用 PCM16 注入器，16 kHz；不经过扬声器 |
| 输出 | native PCM 播放队列照常运行，但 macOS 系统输出音量固定为 0 |
| 网络 | 验收结束时飞行模式为 0，连接状态为 `VALIDATED` |

静音注入器只在 `__DEV__` 且配置 `EXPO_PUBLIC_PHASE1_AUDIO_INJECTOR_URL` 时启用，不会进入默认或 production 路径。它复用真实的本地 VAD、Qwen WebSocket、native 播放 backlog 和 turn 状态机；没有跳过响应生命周期。

## 2. 验收结果

| Gate | 结果 | 证据 |
|---|---|---|
| Screening 正确流程 | 通过 | 7 个 user turns；称呼、地点、时间、近期事件、叙事顺序、日常功能依次完成；第 6 个回答后才提出 recall，第 7 个回答后只执行完整 goodbye；第 4 问不再提前结束 |
| 连续 30 回合 smoke | 通过 | 30/30；重叠 response 0，提前开麦 0，状态机 violation 0 |
| 100 回合 soak | 通过 | 100/100；连续 29 分 31.8 秒完成，再保持 listening 60.0 秒，总连续运行超过 30 分钟 |
| Soak 播放闸门 | 通过 | `response.done → playback_drained` 最小 3900 ms、最大 9052 ms、平均 6238 ms |
| 手动结束矩阵 | 通过 | listening、user_speaking、assistant_generating、assistant_playing、playback_guard 五种状态均只产生一条 goodbye，播放 drain 后 teardown |
| 中英文切换 | 通过 | 中文输入得到中文回答；随后英文输入得到英文回答；两回合均 drain 后才重新开麦 |
| 前后台恢复 | 通过 | HOME 3 秒后恢复 Activity，下一回合完成并重新进入 listening |
| 单一会话 owner | 通过 | 修复 Expo Router 多页面同时启动 native capture 后，只保留一个 direct conversation owner |
| Fail-closed | 通过 | provider error 或 playback stall 路径保持 mic muted，不把异常当作安全 drain |
| TypeScript | 通过 | `npm run typecheck` |
| 确定性测试 | 通过 | `npm run test:turn-taking`：16/16；包含 7 问 screening 顺序、companion 结束/评估隔离、native fallback 播放完成、单一 runtime owner、旧播放取消、PTT 录音准备竞态与短按丢弃 |
| Provider lifecycle smoke | 通过 | `npm run smoke:turn-taking-provider`：收到完整 audio deltas 并到达 `response.done` |
| Android debug build | 通过 | `./gradlew :app:assembleDebug`，443 tasks，BUILD SUCCESSFUL |
| 降级路径 graceful close | 通过 | MuMu 强制 `http-v0.0.0`；结束发生在 opening 播放期间，先显示 wrapping-up 并等待当前播放，再执行三句固定收尾，收到 native `didJustFinish` 后才显示完成并保存 |
| 多页面/旧进程隔离 | 通过 | 第一条 HTTP opening 播放中保留原 Router 页面并启动第二个对话页面；runtime 记录 `superseding old -> new`，旧播放立即 abort，第二页只有一份 opening，结束后唯一 owner 正常 `released` |
| 手动按住说话 | 通过 | direct WS：按下进入 `user_speaking`，松手只 commit 一次；native HTTP：短于 250 ms 的触摸停止并丢弃，5 秒长按显示“正在听 · 松开发送”，松手只转写一次并恢复可录音；助手说话/处理中按钮禁用 |
| 问题与回答引导 | 通过 | 开场先说明“没有标准答案、慢慢说”；每题含中性思考范围。MuMu 在 opening `playback_drained → mic_reopened` 后持续显示完整问题、`第 1 / 7 题` 和“慢慢说 / 请按自己的想法回答”，视觉采样窗不遮挡问题；End Chat 仍可用 |
| 日常助手隔离 | 通过 | 日常助手显示“结束对话”而不是“结束并评估”；结束后保留普通会话记录但不调用认知评估、不生成风险分数或“重新评估”入口。普通礼貌回复不再触发自动结束，只有用户明确告别或手动结束才关闭 |

100 回合 soak 汇总：

```text
turns=100
elapsed_ms=1771784
post_soak_idle_ms=60011
drain_min_ms=3900
drain_max_ms=9052
drain_avg_ms=6238
overlap=0
early_mic=0
violations=0
```

## 3. 本轮发现并修复的问题

1. Qwen semantic VAD 自动创建 response，导致 recall/closing 的 session update 与 response 创建竞态。direct WS 改为 `turn_detection: null`，由本地 VAD 显式执行 mute、commit、等待 transcript、session update、`response.create`。
2. 自适应噪声阈值在长时间 idle 后漂移，普通语音可能无法触发。改为 native NS/AGC 后的固定双阈值：开始 RMS 0.015、持续 RMS 0.008、200 ms 起说、1200 ms 静音结束、30 秒单 turn 上限。
3. Qwen 在同一 realtime session 中没有稳定遵循动态 recall/closing 指令。关键流程改用固定文本模板和 Qwen TTS，仍经过同一个 native playback drain gate。
4. `response.done` 曾可能被误当作扬声器播放完成。现在只有 native backlog 达到 40 ms 阈值并完成 1100 ms acoustic-tail guard 才能重新开麦或推进 agenda。
5. Expo Router 保留页面可产生重复 realtime session。增加进程级 direct-session owner 与启动锁。
6. MuMu 新架构缺少可用 ONNX native module 时，静态导入会导致页面崩溃。改为能力检测后的 lazy load，保留 tap-to-start fallback；正式 wake word 仍属于 Phase 5。
7. v3 自动降级到 native HTTP 后，`player.play()` 曾被当作播放完成，`finalize` 会立即 remove player，导致结束生硬并截断告别。现在等待 expo-audio `playbackStatusUpdate.didJustFinish`；手动结束固定执行“收尾提示 → 完成当前播放 → goodbye → native playback complete → 保存”，并加入 consumer-facing closing guide。
8. Router stack 中的旧页面或未完成的 ASR/Chat/TTS Promise 可在新会话开始后重新创建播放器。现在 direct WS 与 native HTTP 共用进程级 conversation lease；新 owner 同步终止旧 owner，递增 lifecycle epoch、abort 旧播放，旧异步结果在每个 await 边界失效，且旧 cleanup 不能释放新的 owner。
9. “按住说话”原来实际使用 tap-to-toggle，且用户在 native recorder 尚未 prepare 完成前松手后，旧异步任务仍会开始录音。现在 UI 使用 `onPressIn`/`onPressOut`，generation-guarded PTT 状态机取消迟到的 prepare；短于 250 ms 的误触被停止并丢弃，录音文件通过 Expo 56 兼容的 filesystem legacy API 读取。
10. 四个采集域曾被 turn counter 错误压缩成约四个问题：第 3 个回答后强制 recall，第 4 个回答后告别。现在协议要求 recall 前至少 6 个有效患者回答；前六问由版本化确定性模板按“称呼 → 地点 → 时间 → 近期事件 → 叙事顺序 → 日常功能”推进，第 7 问执行同会话 recall，回答后才告别。MuMu 完整 7 回合到 `#65 finished -> ended`，无 lifecycle violation。
11. Assistant transcript 原本会在播放完成后继续把界面锁在 response 状态，真实 turn state 已进入 listening 却没有回答引导。现在 `statusKind` 是页面状态权威，保留的 transcript 只作为当前问题展示；listening 页面显示本地化问题进度、回答提示和 End Chat。MuMu 手动结束到 `#16 finished -> ended`，runtime 正常释放。
12. 日常助手原本和认知检查共用无条件 assessment，空对话也会产生 `HEALTHY / 风险 0%`；同时只凭 assistant 回复中的 `Have a good day` 或 `Take care` 就可能自动结束。现在 assessment 只接受含患者回答的 screening persona；companion 在 direct WS、native HTTP fallback 和 web relay 中只响应明确的用户告别意图或手动结束。MuMu 结束后无风险卡、无重新评估入口，唯一 runtime 正常释放。

## 4. 已知边界与后续项

- MuMu 无法证明实体镜面的扬声器回声、AEC、房间声学、真实可闻句尾完整性和物理麦克风重新开放时机。因此本报告关闭的是本地 MuMu 验收门，不替代目标镜面 30 回合声学 smoke。
- 完成 100 回合后，同一条超过 30 分钟的 Qwen session 在第 101 次请求返回 provider `InvalidParameter`；全新 session 可处理相同输入。客户端正确 fail-closed。Phase 1.1 应在约 80 回合或服务端时限前做无感 session rotation。
- MuMu 的飞行模式或虚拟 Wi-Fi 关闭命令会令 ADB shell 自身卡死，无法形成只影响 App 数据面的可信短断网证据；两次尝试均已恢复模拟器、飞行模式和网络。网络恢复应在目标硬件用物理 Wi-Fi 开关或独立网络代理复测。
- 当前 MuMu 环境不能验收正式 wake-word 的误唤醒、漏唤醒、距离、电视声和方言指标；这些属于 Phase 5。
- 会话保存到本地无效 backend 地址以及 Expo Router 对 `apiUrl.ts` 的默认导出警告不影响 turn-taking gate，但应在 Phase 3 后端统一时清理。

## 5. 构建产物

- APK：`mirror-app/android/app/build/outputs/apk/debug/app-debug.apk`
- 大小：410,538,504 bytes
- SHA-256：`244fd0273c71b6a30fefc407b979d83221e96a7f29dc2b1c4d7625724ece03d1`

本地 MuMu 验收结论：**通过**。Phase 1 在路线图中保留“实体镜面声学门待完成”的状态，完成该门后才可标记为面向硬件的完整 Phase 1。
