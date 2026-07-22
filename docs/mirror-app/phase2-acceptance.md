# Phase 2 Mirror 消费者前端验收

日期：2026-07-22
权威需求源：`docs/June-Aug 2026, Reflexion Tech Document.docx`（106 页、7 张内嵌设计图，已完整读取与渲染）
目标设备：10.1 英寸智能镜 / Android 14 / 1080 × 1920 纵向 / 4 GB RAM / 32 GB ROM

## 1. 本轮重做结论

旧版黑色测试仪表盘视觉不再作为 Phase 2 基线。生产界面已按文档的 warm cream、sand、beige、soft gold、sage 设计语言重新实现，并以语音优先、非触摸依赖、远距离可读为约束。

- Home、Boot、Pairing、会话、结束、离线和错误页使用同一套消费者设计系统。
- 主界面不显示 dementia、cognitive、risk、红黄绿状态或诊断分数。
- 日常使用不依赖点击或长按；由唤醒词、计划任务、caregiver trigger 或服务端 session trigger 启动。
- 麦克风开启时始终显示 `MICROPHONE ACTIVE`，不进行静默录音。
- 生产会话不自动启动摄像头或采样人脸。相机能力保留给管理员诊断或未来独立、明确同意的流程。
- Aria 使用消费者级头像与平静、非医疗化语言；结尾包含完整告别、保存反馈和回到 Home 的引导。

## 2. 状态必须由真实系统事件驱动

`conversation.tsx` 将真实 pairing、network、provider 和 turn-taking 状态映射到视觉状态。生产页没有用固定 timer 伪造 Listening → Thinking → Speaking 流程。

| 产品状态 | 生产状态来源 | 用户可见反馈 |
| --- | --- | --- |
| Home / ambient | 已配对、无 active session、无故障 | 问候、姓名、日期、时间、`Say “Hello Aria”`；仅显示有真实数据的天气/提醒 widget |
| Connecting | 已收到启动意图，provider/session 尚未 ready | `Getting everything ready…` |
| Listening | session active 且 turn owner 为 user | `MICROPHONE ACTIVE`、`I’m listening…`、实时音量反馈 |
| User heard | VAD confirmed speech / user speaking / transcript 更新 | `I can hear you` 与最新用户 transcript |
| Thinking | user turn 已 commit，provider processing | `Just a moment…`，麦克风不显示为活动 |
| Aria speaking | native playback owner / assistant audio delta | Aria 文本、波形、`You can speak at any time to interrupt me` |
| Closing | graceful-close state，模型已开始完整告别 | 完整告别文案，不在固定问题数后强制结束 |
| Saving | session complete 请求与本地持久化正在执行 | `Your check-in is being saved` |
| Session complete | `saveCheckin` 已 synced 或 durable queued | 完成确认、caregiver 更新结果、自动返回 Home 提示 |
| Offline | pairing verification、heartbeat 或 provider network failure | 明确离线、已记录数据不会丢失、恢复后上传与重试入口；不承诺离线运行 LLM |
| Microphone error | permission、native recorder 或 audio-start failure | 麦克风修复说明与重试入口 |
| Service error | 非网络、非麦克风的 provider/service failure | 服务暂不可用与重试入口 |
| Pairing | Pairing v2 返回的真实 code、QR、expiry 与 polling | 大号 6 位 code、QR、caregiver app 操作说明 |
| Admin diagnostics | 实际 hardware report、backend、pending outbox 与 app/device 信息 | 受限设备区；不暴露患者诊断内容 |

`/visual-acceptance?state=...` 是 `__DEV__` 专用渲染夹具，只把同一生产组件置于指定输入状态，以便截图比较；release 访问该路由会返回 Home。唯一的时间行为是 Session 已真实完成后的 9 秒 Home 回退，它不参与会话状态模拟。

## 3. 视觉基线

| Token | 值 | 用途 |
| --- | --- | --- |
| Cream | `#FFF9F1` | 主背景 |
| Sand | `#F6F1E8` | 大面积柔和背景形状 |
| Beige | `#EDE5D6` | 卡片、分隔与次级表面 |
| Soft gold | `#E7CFA6` | 品牌与轻提示 |
| Deep gold | `#B98954` | 活动状态、按钮和强调 |
| Sage | `#ABC5A1` | 已听见、在线和安全反馈 |
| Primary text | `#282828` | 高对比正文 |
| Secondary text | `#686868` | 辅助说明 |

排版和布局原则：

- 1080 × 1920 纵向为当前基准，不再以 1440 × 2560 模拟器默认尺寸代替硬件规格。
- 主状态文字保持短句、大字号和宽行距；关键反馈在 1–2 米距离仍能辨认。
- 不使用霓虹、数据仪表盘、医疗告警墙或需要近距离阅读的小按钮。
- 系统状态栏、导航栏、错误页、配对页和管理员页全部使用浅色视觉，避免从消费者界面突然跳回黑色开发风格。

## 4. MuMu 验收记录

MuMu 模拟器已通过 `wm size 1080x1920` 设置为目标 viewport，debug 包使用 `com.reflexion.mirror.debug`，可与正式包 `com.reflexion.mirror` 并存。验收期间未启动语音 session，因此没有触发扬声器播放；正式包未硬编码静音，也没有禁用麦克风或相机硬件能力。

已逐页冷启动并截图的状态：

- Boot
- Pairing
- Home / ambient
- Connecting
- Listening
- User heard
- Thinking
- Aria speaking
- Closing
- Saving
- Offline
- Microphone error
- Service error
- Session complete
- Admin diagnostics

截图位于 `docs/mirror-app/phase2-acceptance-assets/`，全部为 1080 × 1920，且不包含 Metro bundling、warning toast 或 debugger 遮罩。

关键样例：

- `phase2-ambient.png`
- `phase2-pairing.png`
- `phase2-heard.png`
- `phase2-speaking.png`
- `phase2-offline.png`
- `phase2-session-complete.png`
- `phase2-admin-diagnostics.png`

## 5. 验收边界

模拟器可以验收布局、状态映射、导航、权限错误和独立 APK 启动，但不能代替以下物理镜验收：

- 10.1 英寸实际可视区域和镜面镀层裁切；
- 1–2 米观看距离和老人视力；
- 白天、室内暖光、逆光和夜间亮度；
- 四麦阵列回声、真实扬声器音量和 barge-in；
- 真实老人音量、语速、停顿与口音。

完整的断网本地对话执行仍属于 Phase 7。当前 Phase 2 已实现离线状态、已采集数据的 durable queue 语义与恢复入口，但不能把尚未实现的离线 LLM check-in 当作已验收能力。

## 6. 自动化与正式包

2026-07-22 回归结果：

- Mirror `npm run typecheck`：通过；
- Mirror `npm run test:turn-taking`：19/19 通过，包含 21 组 deterministic lifecycle replay 以及 Daily Conversation v2 的条件阶段/信号映射；行覆盖率 93.15%；
- Caregiver Server `npm run typecheck`：通过；
- Caregiver Server `npm test`：16/16 通过；
- Android `assembleRelease`：通过；
- APK Signature Scheme v2：通过，证书 `CN=Reflexion Mirror, OU=Mobile, O=Reflexion, L=Singapore, C=SG`；
- MuMu 独立安装与冷启动：通过，不依赖 Metro；
- release bundle 的统一 API domain：`https://reflexion-caregiver-app-server.vercel.app`；
- release JS bundle 未包含本地 Qwen、DashScope、MongoDB secret 值。

产物：

- `mirror-app/android/app/build/outputs/apk/release/app-release-arm64-v8a.apk`：RK3576/ARM64 镜面设备推荐包，约 95 MB；
- `mirror-app/android/app/build/outputs/apk/release/app-release-universal.apk`：ARM64、ARMv7、x86、x86_64 通用包，约 278 MB；
- `mirror-app/android/app/build/outputs/apk/release/app-release.apk`：与 universal 相同的 Gradle 默认产物。

通用 release 没有嵌入共享 bootstrap token。安装后如设备尚未由后端 provision，会安全地显示 `Pairing is unavailable`；正式交付前需使用 `caregiver-server` 的 device provisioning 流程为每台镜子发放唯一 bootstrap credential。
