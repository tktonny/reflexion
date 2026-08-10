# 对话架构对比:v1 中继 / v2 回合制HTTP / v3 原生直连 + 其他选择

> 由工作流 conversation-arch-tradeoffs 产出(6 维度评估 → 6 类替代方案 → 4 项对抗复核 → 合成)。以本项目已实测事实为地基。

## 1. 一页结论

- **V1 中继（relay 模式）**：唯一在 Web-kiosk 与原生端"今天就能跑通"的流式实时体验——服务端持有密钥、集中编排、China-endpoint 自动回退，代价是必须常驻一个有状态 Node 进程。
- **V2 回合制 HTTP（http 模式）**：最简单、最省钱、最可靠、最好维护的无状态方案，但按当前实现是"按键说话"+ 串行 ASR→chat→整段 TTS，有 1.5-4s 空白，实时感最差。
- **V3 原生直连 WS（ws 模式）**：延迟天花板最低（少一跳中继），编排已移到端上，但端上 PCM 音频桥是一个会抛异常的桩，**今天在真机上没有可用的语音 UX**，且 Web 完全不支持（回退到 relay）。

**默认推荐**：以当前可交付状态论，**V1 中继是唯一能同时覆盖 Chrome-kiosk 与原生、且已端到端验证的实时方案，应作为默认基线**。V3 是未来"最佳手感"的目标形态，但要等原生 PCM 桥 + 自定义 EAS dev build 补齐后才成立；V2 作为弱网/降级兜底与最省运维的量产候选保留。

## 2. 维度 × 版本评分矩阵

| 工程维度 | V1 中继 | V2 回合制 HTTP | V3 原生直连 WS |
|---|---|---|---|
| **延迟 / 会话 UX** | **strong** — 流式音频增量 + 服务端 VAD 免按键，Web+原生均已验证 | **weak** — 按键说话、串行管线、1.5-4s 空白，无首字节音频 | **adequate** — 延迟天花板最低，但端上音频是抛异常桩、Web 不支持，实际手感今天不存在 |
| **安全 / 隐私** | **strong** — 密钥仅在服务端，唯一有事件白名单与单点轮换/熔断 | **weak** — 长期账号密钥被打进 APK/Web 包（.env 今天就在发），可提取、不可选择性吊销 | **adequate** — 正常路径用短时 token 不落端，但有静默回退到内置密钥、mint 端点无鉴权、token 全域且不可提前吊销 |
| **成本 / 扩展** | **weak** — 唯一常驻非无服务进程，用最贵的实时 omni-flash，账号级配额下并发最少 | **strong** — 无中继、无服务器可无状态部署，负载分摊到三个更便宜模型池，并发余量最大 | **adequate** — 基础设施像 V2（仅无状态 mint），但沿用 V1 的实时模型成本与账号池上限 |
| **可靠 / 离线** | **adequate** — 唯一有 China-endpoint 回退 + 20s 心跳 + 降级事件，但双腿中继是 SPOF、无会话中重连、回退只在握手期生效 | **strong** — 无状态独立回合，掉线只影响下一回合；TTS 失败非致命等降级已真正实现（原生尚未接） | **weak** — 端到端在真机上跑不通（PCM 桩抛异常）、无重连、无地域回退、token 生命周期脆弱 |
| **可维护 / 可测** | **adequate** — 编排集中权威、保真度最高，但跨双运行时面积最大、relaySession 巨型有状态闭包难测 | **strong** — 线性回合、纯函数编排可无音频冒烟回归、单一部署单元、最少实时细节会漂移 | **adequate** — 实时实现最薄，但编排移端后保真漂移风险最高（debounce/restart 逻辑已移植却未被调用），端到端不可测 |
| **部署 / 运维** | **adequate** — 唯一实时 Web 路径、密钥托管与服务端热修最优，但要常驻有状态 WS 进程（不可无服务器化） | **strong** — 托管最轻、Web 今天可用、原生只需标准 expo-audio（非 V3 那种自建 PCM 模块） | **weak** — 端上负担最重且未建（自建 PCM + 自定义 dev build），原生-only 使得 Web 仍被迫跑 relay |

## 3. 三版逐个优缺点（分场景）

### V1 中继（relay）

**优势**
- *延迟/UX*：全双工传输 + 逐增量播放（response.audio.delta），端点检测（阈值 0.1 / 900ms 静音）实现"只管说、停顿即到端"，最适合墙面镜子的免按键交互。是**唯一在 Web 与原生都验证过端到端音频**的版本。
- *安全*：长期密钥仅在 `server/relay.mjs` 注入 upstream 头；客户端 WS 不带任何凭据。唯一有服务端事件白名单（`CLIENT_EVENT_WHITELIST`），唯一有单点密钥轮换/熔断的真正 kill switch。
- *可靠*：唯一有 China-endpoint 自动回退（401/403 时改连 China 备用主机）、20s upstream 心跳、结构化 `reflexion.session.degraded` 降级信号。
- *运维/上线*：编排、提示词、模型 id、地域回退全在服务端，热修一次全体 kiosk 下次连线生效，无需过审/OTA；Web-kiosk 无需 APK 即可跑实时。

**劣势**
- *成本/扩展*：唯一常驻、非无服务器进程（双 WS + 心跳 + ffmpeg/worker），资源随并发线性增长；用最贵的实时 omni-flash，账号级 ~60 RPM/100k TPM 下并发最少；水平扩展需要 WS 感知的粘性负载均衡。
- *可靠短板*：双腿中继是 SPOF 且设备无法绕过；**无客户端重连/会话恢复**——`onclose` 直接 cleanup 显示"会话结束"，任何 Wi-Fi 抖动都会终止会话逼老人手动重启；地域回退只在初次握手触发，会话中掉线是终局。
- *安全短板*（均为服务端可修）：`index.mjs` 默认明文 HTTP、`.env.example` 默认 `ws://`；WS 升级无鉴权，任何能到 :8787 的人都能冒用 patient_id 跑会话。
- *会话性质*：`interrupt_response:false` + 播放期抑制采集，**没有真正 barge-in**（这是 V1/V3 共同的、为回避回声而做的取舍）。

### V2 回合制 HTTP（http）

**优势**
- *成本/扩展*：**最省**。无中继、后端仅无状态 `+api.ts`（token/配对/持久化）可无服务器部署，设备增长基础设施成本持平；负载分摊到 qwen3-asr-flash / qwen-plus / qwen-tts 三个更便宜、文本 TPM 余量更大的池；无连续音频计费，单会话 token 成本最低。
- *可靠*：**最可靠**。每回合是独立短 HTTPS 请求，无长连接可掉；掉线只延后下一回合、内存历史保留可重试。逐组件降级已真正实现（TTS 失败非致命、空 ASR 提示"没听清"）。HTTPS 比 WS 101 升级更能穿透门户/代理。
- *可维护*：**最好维护**。线性 record→ASR→chat→TTS→play（273 行），无 WS 状态机；纯函数编排可用 `smoke-turnloop.mjs` 无麦无音频回归；单一部署单元。
- *运维*：Web-kiosk 今天端到端可用（CORS 已验证），原生只需标准 expo-audio 录放，无需 V3 的自建 PCM 模块。

**劣势**
- *延迟/UX*：**最差**。按键说话（每回合点开始 + 点发送）、串行 ASR→qwen-plus→整段 WAV 合成、1.5-4s 空白、无首字节音频，像对讲机而非对话——对老人是重负担。
- *安全*：按当前实现把长期账号密钥（`EXPO_PUBLIC_QWEN_API_KEY`）内联进包，APK=ZIP、Web JS 明文，可提取；账号级限额 + 不可选择性吊销 → 一把泄露密钥可耗尽/刷爆整个账号，**这是当前活跃暴露（.env 今天就在发真 key）**。必须换成 token-mint 端点。
- *成本细节*：每回合三次往返消耗更多 RPM；每回合重发含系统提示词的完整历史，长对话输入 token 成本随轮数增长。
- *可靠细节*：无 China 地域回退、无重试/退避（单个 429/5xx 直接中断回合）；原生音频尚未接线（当前仅 Web 成立）。

### V3 原生直连 WS（ws）

**优势**
- *延迟*：**天花板最低**——与 V1 相同的实时传输与调参（同一 `buildLiveSessionUpdate`、服务端 VAD、流式增量），但设备直连 Qwen，去掉中继一跳；音频路径上无常驻服务器，会话中无物可增加 RTT 或宕机。
- *安全*：正常路径用 `POST /api/qwen-token` 铸造短时（1800s）token，长期密钥不落端；token 时间盒化，泄露自动过期。
- *成本/运维*：基础设施与 V2 同轻（仅无状态 mint 端点），编排在端上、会话开始后不依赖常驻服务。

**劣势**
- *致命现状*：`src/native/pcmAudio.ts` 的 `createPcmAudioBridge()` **是抛异常的桩**，连接即报错——16k 采集 / 24k 无缝播放这些决定实时手感的东西**都未实现、未验证**。**今天真机上没有可用语音 UX**，仅传输/token/编排被无头冒烟验证。
- *平台限制*：**原生-only**。Web/Chrome-kiosk（既定部署目标之一）无法设 WS 鉴权头，静默回退到 relay——V3 在 Web 上零收益，混合车队里你得同时跑 relay。
- *安全短板*：`bearer = token || QWEN.apiKey` 静默回退到内置密钥（.env 今天就在发）；mint 端点无鉴权（等于一台"token 自动贩卖机"）；token 继承全域、不可提前吊销；端上自建整段 `session.update`，无服务端白名单，被篡改的客户端可任意改 instructions/模型。
- *可靠/可维护*：无重连、无地域回退移植；已移植的多信号动态嗓音/首响重启逻辑（`voiceProfileFromRecentSignals`/`shouldRestartResponseForLanguageSwitch`）在 `src/` 中**确认未被调用**，端上实际行为与 V1/Python 参考**已发生漂移**；需自定义 EAS dev build，Expo Go 不可能。
- *会话性质*：同 V1，`interrupt_response:false` + 播放期静音采集，**同样没有真正 barge-in**。

> 校正说明：不能把 V1 与 V3 说成"等价"。二者只是设计配料相同（同一 session.update，V3 是 relay 的端上移植），但 V1 是 strong、V3 是 adequate；V3 延迟天花板更低却**尚未兑现**，而 V1 有已验证的可用体验。就真机今天的实际体验而言，V3 因 PCM 桩报错，甚至**不如** V2 能跑。此外三版**都没有** barge-in——它不是区分 V2 的因素。

## 4. 其他架构选择

以下均评为 viable-with-work，逐个说明工作机制、优缺、基于**已验证事实**的可行性，并标注被哪些已验证事实堵死。

### A. V4 网关 — 注入鉴权的实时 WS 透传（换头反向代理）· 工作量 M
- **怎么工作**：客户端（浏览器或原生）用可由浏览器设置的 subprotocol/query 打开 `wss://gw/realtime`；网关（`server/index.mjs` 精简分叉，长驻 Node）在 upgrade 时校验配对 token，只在 **upstream 一跳**注入 `Authorization: Bearer`（即 relay 已验证的机制），双向按不透明 buffer 转发、**不做每帧解析、不做编排**。全部编排（session.update、VAD、动态嗓音、wrap-up）移到端上、复用已移植的 `src/orchestration/realtime.ts` + 现有浏览器 Web Audio 管线。
- **优点**：不发密钥即解锁**浏览器实时**（利用浏览器 header 限制只卡"浏览器→Qwen 直连"、不卡"浏览器→网关"这一已验证事实）；Web 与原生共用一套端上编排大脑；因复用现有 Web Audio，**不被原生 PCM 桩阻塞、今天可建**；每连接 CPU 比 relay 更轻。
- **缺点/校正**：**"薄/无状态"是误导**——它仍持有每会话两条持久 WS（至 maxSessionSeconds），必须是与 relay 相同的长驻有状态进程，**不能做成 `+api.ts`（Expo Router 路由是 Request→Response，无法 101 升级——已验证事实堵死）**；运维负担与 relay 相同，仅每帧计算更轻。与 V1 同为 2 跳，延迟不优于 relay、劣于 V3 直连。纯透传丢掉 relay 的 `CLIENT_EVENT_WHITELIST` 等护栏，**必须重新加客户端鉴权 + 最小帧白名单**，否则被篡改客户端可任意改 session 并刷爆账号级配额。对原生几乎无价值（原生已 1 跳直连）。
- **被已验证事实卡点**：`+api.ts` 无法 WS 101 升级 → 必须长驻进程；账号级配额无每-key 上限 → 必须自建限流。
- **适用**：只为 **Web 实时**且拒绝在 kiosk 页内嵌密钥时。原生仍用 V3。

### B. V4 — HTTP 上的流式半双工（SSE chat + 逐句流水线 TTS + 端上 VAD）· 工作量 M
- **怎么工作**：保留 V2 的完全无状态形态，但把串行管线改成重叠流水线——麦克风常开，JS/WASM VAD 检测句末自动触发回合（免点按）；端点音频→一次 `qwenASR`；chat 用 `stream:true` 读 SSE 增量；按标点切句、逐句并行 `qwenTTS`、经现有 `playbackCursorRef` 无缝游标背靠背排播，第一句在后续句仍在生成时就开始播。**全程不用 WebSocket**——SSE/fetch 流是普通 HTTP，从而**绕开"浏览器 WS 不能设头"这一堵死 Web 实时的事实**。
- **优点**：真无状态/可无服务器（静态页 + 一个无状态 token 路由）；**是唯一能给 Web 带来近实时手感的路径**（V3 直连 WS 恰因该事实是原生-only）；首字节音频降到约 1.2-1.8s、回复连续，明显优于 V2 的 1.5-4s；大部分是增量改动（复用 V2 的 Web Audio、`encodeWavBase64`、`orchestrator.ts`）；端上 VAD 免点按，契合老人常开镜子。
- **缺点/校正**：**非真实时、半双工**。barge-in 需播放期常开麦 + 浏览器 AEC，共享扬声器/麦的镜子上 AEC 质量是成败关键（可行-需工作）。DashScope 的**流式 ASR/TTS 是 WS-only + header 鉴权 → 浏览器内被堵死**，故 ASR 只能整段、TTS 只是逐句近似流式。逐句 TTS 使每回合 HTTP 请求翻到 2-4 次，压向账号级 60 RPM。原生路径同样被 PCM 桩 + RN 端 SSE(ReadableStream) 不稳堵住 → **Web 立即见效、原生是另一笔大工程**。密钥暴露同 V2，须用 token mint。
- **被已验证事实卡点**：流式 ASR/TTS 的 WS+header 鉴权在浏览器被堵；原生 PCM 桩未建。
- **适用**：Web/Chrome-kiosk 想要近对话手感又拒绝跑常驻 relay 时——**对 V2 而言 Web 上性价比最高的升级**。

### C. V4 本地 — 端上/边缘推理（无云 LLM，mode 'local'）· 工作量 XL
- **怎么工作**：编排本就在端上且模型无关（`orchestrator/voice.ts` 纯字符串组装），V2 已把三次云调用隔离在 `qwenClient.ts`（keyed on `QWEN.base`）。V4 只把这三次换成本地引擎（OpenAI 兼容端点）。两种形态：(A) **边缘盒**——LAN 上跑 llama.cpp/whisper.cpp/Piper，设 `EXPO_PUBLIC_QWEN_BASE=http://<lan-ip>`，镜子代码几乎不动、连今天的 Web-kiosk 也能用；(B) **纯镜上**——打包量化模型 + 原生 RN 推理模块 + expo-audio，需自定义 EAS dev build。
- **优点**：**绕开几乎所有云侧已验证约束**——无 dashscope/China 主机依赖、无账号级限额、无长期密钥可嵌/可泄；隐私/数据驻留强（临床音频不出设备/LAN）；边缘盒近乎"改个 base URL"的 V2；纯镜上给真离线；编排逐字复用。
- **缺点/校正**：**校正**——"on-device/edge 都属未来/实验"这一说法**过强**。**纯镜上 6 语言临床级确属 R&D/未来**：能装进镜子级硬件(1.5-3B, RPi5/RK3588 类)的小模型，恰在本产品最需要的地方弱——**粤语/闽南语/泰米尔语（已确认正是 `voice.ts` 里被点名的三种风险语言）ASR+TTS 覆盖与多规则多语指令遵循**，临床筛查信号须整套重新基准化，不能假设 Qwen 已验证行为可迁移；且只回合制、每回合可能数秒、当前最佳 Android 形态（Chrome-kiosk）根本跑不了本地推理；原生音频层仍是抛异常桩这一前置也未建。**但边缘盒（英/普通话，GPU LAN 盒）是"现在可行"的路径**——不过它跑在独立 LAN 迷你 PC 上、**不是"典型 Android 镜子硬件"**、仅 2 语言，且 HTTPS Web-kiosk 会遇**混合内容拦截**（https 页不能 fetch 明文 http 盒），需盒上可信证书/http 或原生 kiosk。
- **被已验证事实卡点**：镜子级硬件小模型质量（尤其三种目标方言）+ 原生 PCM 桩 + Chrome-kiosk 无法本地推理。
- **适用**：隐私优先/弱连接、想摆脱共享账号配额者。**先上边缘盒变体（英/普通话）作为可行-现在，纯镜上 6 语言临床级作为 R&D 轨**。

### D. V4 WebRTC 代理桥 — 自托管 SFU(LiveKit/Pipecat 式) 前置 Qwen WS · 工作量 XL
- **怎么工作**：Qwen Omni 无 WebRTC 端点，故 WebRTC 终结在**你自建**基础设施：镜子发布 Opus 音轨到 China 同区（阿里云 ECS）的自托管 SFU/agent worker，worker 解码为 PCM16@16k、跑现有 `orchestrator.mjs/voice.mjs`、开与 relay 相同的 upstream Qwen WS，再把 response 音频转 Opus 播回。等于**只把 V1 的客户端腿从裸 PCM/WS 换成 WebRTC 传输**，服务端 Qwen 腿不变。
- **优点**：稳健房间音频（Opus + FEC + NetEQ 抖动缓冲 + 硬件 AEC/NS/AGC）远比裸 PCM over TCP 抗弱网；AEC 实质改善 barge-in（阻止镜子自己的 TTS 再触发 VAD）；`react-native-webrtc` 一模块补齐**原生麦/扬/AEC**，正好填 `pcmAudio.ts` 的空白；密钥仍在服务端；Web 与原生统一传输。
- **缺点/校正**：**不移除任何 Qwen 约束**（仍 header-only WS、仍 China-only、仍单账号 60 RPM/100k TPM）。**China-region 抹掉托管云价值**：LiveKit/Pipecat Cloud 在美/全球，会让媒体绕错大陆，必须在阿里云自托管 SFU+TURN，丢掉大部分 SaaS 便利。全-Qwen 使 SDK 的 STT/TTS/LLM 插件生态白费（须手写 Qwen 桥）。比 relay 更重更贵（SFU+TURN+转码，常驻），对单用户日检过度工程。**仍需自定义 EAS dev build**（`react-native-webrtc` 同样 Expo Go 不可能）。
- **被已验证事实卡点**：Qwen 无 WebRTC 端点、China-region 单主机 → 必须 in-region 自托管；未改变账号级配额。
- **适用**：仅当弱网音频稳健性/barge-in-AEC 被证明是真实地场问题时。若只求低延迟或补原生音频，**过度工程**（V1 已全域可用、V3 已给原生低延迟、原生音频缺口用 PCM 模块更便宜）。

### E. V4 会话 Provider Mesh（relay 侧）· 工作量 L
- **怎么工作**：抽象边界在 **relay**，非客户端。客户端 `ConversationApi` 不动。relay 内引入 `ConversationProvider` 接口（实时-双工 / 回合制两档），因 Qwen Omni 实时协议本就是 OpenAI-Realtime 形状，内部归一事件总线≈OpenAI-Realtime 形状，Qwen 适配器与 OpenAI-Realtime 适配器近乎恒等。按 healthcheck+fallback_rank 选最高健康 provider，可在**同一客户端会话内**从实时静默降级到回合制；用已有 `conversation_provider` 字段做 A/B。端上 http/ws 模式**不入 mesh**。
- **优点**：客户端零改动；直击最强已验证风险——账号级配额耗尽 + 单区依赖（加入 China DashScope vs 全球 OpenAI 等独立故障域）；N 个 provider 密钥都留服务端（端上做会把可提取密钥问题乘以 provider 数）；复用既有 provider-neutral 提示词 + 平台已有的 health+rank mesh；Qwen/OpenAI-Realtime 适配器首版近乎恒等；Web 实时本就只能在 relay，故 mesh 放这里对 Web 零额外成本。
- **缺点/校正**：单 relay 主机须同时出口到 China DashScope 与全球 OpenAI/Gemini → 须选 HK/全球区（真实托管约束）；Gemini Live 是真不同协议（须真适配器，列为拉伸项）；实时故障切换非无缝（换 provider = 新 upstream WS + 重初始化，干净切点在回合边界或降级到回合制）；端上模式无法安全入 mesh；跨 provider 临床可比性需额外验证；给"绝不能崩"的常驻进程加了归一总线 + 路由状态。
- **被已验证事实卡点**：账号级无每-key 上限（这正是它想缓解的）；Web 实时 header 限制（反而利好——mesh 只能在 relay）。
- **适用**：生产云部署、需要抗 DashScope 配额/区域故障或想 A/B provider。单机 demo 过度。

### F. V4 "Conductor" 自适应传输路由 · 工作量 L
- **怎么工作**：不用构建期 `EXPO_PUBLIC_CONVERSATION_MODE`，而在 `startConversation()`（会话边界，非每渲染）由四输入解析出一个传输：Platform.OS、部署信任标志(kiosk/distributed)、网络探测、能力探测(PCM 是否已实现)。原生阶梯：好网+PCM 已接+有凭据→V3；分发构建 mint 失败→V1；弱/计费网→V2；PCM 桩仍抛→V1。Web 阶梯确定：实时永远 relay，弱网→V2。因三 hook 已返回同一 `ConversationApi`，安全实现是把 `transportKey` 入 React state 并 remount 会话子树（`key=`），保证每传输一个 hook 实例；运行期回退在下个会话/重连重跑阶梯，绝不在回合中。
- **优点**：几乎白拿（统一 API + 选择器已存在，路由是薄决策表）；一个构建服务全部形态；优雅降级（mint 失败/弱网/PCM 未接都降到能跑的传输）；信任自适应本就是既有模式（V3 已 mint→回退）；已验证事实让整支分支确定（Web 实时唯一正解是 relay）。
- **缺点/校正**：**Hooks 规则陷阱**——现选择器故意是构建期常量；运行期路由只在会话边界解析 + remount 才合法。**临床保真漂移**：V1/V3 有服务端 VAD/低延迟、V2 没有且更慢，患者跨天/跨网被弹到不同传输会得到**实质不同的会话，混淆认知风险信号**——对筛查工具这**恰恰可能适得其反**，倾向锁定单一传输。两套编排(.mjs vs .ts)须锁步一致。测试/观测矩阵爆炸(平台×3 传输×信任×网络 + 转移，各需实机 China-Qwen 验证)。**不缓解**账号级配额上限。前置未建：无网络检测(NetInfo/navigator.connection 均缺)、PCM 桩仍抛 → 完整原生实时分支今天不可达。
- **被已验证事实卡点**：PCM 桩未建、网络检测缺失使原生实时/弱网分支不可达；账号级配额不受益。
- **适用**：**混合车队**（部分自持 kiosk、部分分发）跑不稳的老人家庭 Wi-Fi、愿投入跨传输行为契约 + 实机测试矩阵时。**同质车队（全自持 Android kiosk、好网）属过度工程，直接钉 V3 或 relay**。

## 5. 按场景推荐

- **Demo / 自持 kiosk（Web 或单机、好网）**：**V1 中继**。今天就能在 Chrome-kiosk 跑通流式实时、免按键，密钥安全托管、可服务端热修。若坚持不跑常驻进程且能接受近实时半双工，选**选择 B（HTTP 流式半双工）**作为 Web-only 轻量替代。
- **分发量产（把 APK/Web 发给不受控设备）**：**V2 + 强制 token-mint 端点**（复用 V3 已建的 `/api/qwen-token`，去掉 `EXPO_PUBLIC_QWEN_API_KEY` 内联）。最省运维、最可靠、原生只需标准 expo-audio。**绝不可**按当前 V2/V3 的内置密钥/静默回退形态分发（账号级配额 + 不可吊销 = 全账号被刷爆风险）。
- **多租户云（需抗配额/区域故障、要 A/B provider）**：**V1 中继 + 选择 E（Provider Mesh，relay 侧）**。客户端零改动即获多故障域与 provider A/B；relay 主机须部署在能同时出口 China DashScope 与全球 provider 的区域(HK/全球)。
- **离线 / 弱网**：分两层。弱网但仍联网——**V2**（无状态独立回合最抗掉线，逐组件降级已实现）；进一步近实时用**选择 B**。真离线/隐私优先——**选择 C 边缘盒变体**（英/普通话，LAN GPU 盒，≈改 base URL），注意 HTTPS 混合内容拦截需盒上证书；**纯镜上 6 语言临床级列为 R&D，不作近期量产依赖**。注意：现有离线会话队列(`conversationQueue.ts`)是休眠设施(`queuePendingConversation` 零调用、无重连自动 flush)，三版今天都不具备自动离线持久/回放，任何版本要对话都需在线。

## 6. 推荐终态与演进路径

**终态判断**：不存在"一版通吃"。生产终态应是**按部署面钉定主传输 + 服务端集中编排 + 端上薄客户端**，而非无条件上自适应路由（对同质车队过度工程，且对临床筛查有会话一致性风险）。

**阶段路线**：

1. **立刻堵安全洞（最高优先，与架构无关）**
   - 从 `.env` / 构建移除 `EXPO_PUBLIC_QWEN_API_KEY` 内联；所有端上路径改走 `/api/qwen-token`。
   - 给 `/api/qwen-token` 与 `conversations` `+api.ts` 加鉴权/授权（现均无鉴权，等于 token 贩卖机 + 未鉴权 PHI 写库）。
   - 去掉 V3 的 `token || QWEN.apiKey` 静默回退；relay 前置 TLS 并给 WS 升级加鉴权（现默认明文 + patient_id 仅为 query）。

2. **稳固 V1 为生产基线（Web + 原生今天唯一可交付实时）**
   - 补客户端重连/会话恢复（现 `onclose` 直接终止，老人被迫手动重启）。
   - 把 China-endpoint 回退从"仅初次握手"扩到会话中掉线；把 dev-facing 错误文案换成患者友好提示，补上承诺的引导降级兜底。

3. **收敛"两套编排大脑"以保保真（跨切面工程债）**
   - 现状：`server/*.mjs` 与 `src/orchestration/*.ts` 是手工同步孪生，两份 `conversationFlow.json`，orchestrator 的 fallback 规则块已在 .mjs/.ts 间分叉；`voiceProfileFromRecentSignals`/`shouldRestartResponseForLanguageSwitch` 在 `src/` 中**已移植却零调用**，V3 动态嗓音行为**已偏离** V1/Python 参考。
   - 措施：确立**单一权威编排源**（建议 TS），用构建步生成 .mjs 或让 relay 直接消费 TS 编译产物，消除双源漂移；`buildLiveSessionUpdate` 与 REALTIME 常量去重（现 relay.mjs 与 realtime.ts 各一份，且 realtime.ts 硬编码、不随服务端 env 可调）。

4. **补齐 V3 的端上 PCM 桥 + EAS dev build（解锁原生低延迟终态）**
   - 实现 `src/native/pcmAudio.ts`（16k 采集 / 24k 无缝播放，react-native-audio-api 或 @fugood/react-native-audio-pcm-stream）+ 自定义 EAS dev build。
   - 完成后**原生钉定 V3**（少一跳、无音频路径常驻服务器）；**Web 仍走 V1 或选择 A 网关**（浏览器 header 限制堵死 Web 直连——这是无法绕过的已验证事实）。

5. **Web 实时的取舍：V1 relay vs 选择 A 网关**
   - 若想让 Web 与原生共用**同一端上编排大脑**（退掉服务端 orchestrator），上**选择 A 网关**——但须清醒：它仍是**常驻有状态进程**（`+api.ts` 无法 101 升级），且必须**重新加客户端鉴权 + 帧白名单**补回 relay 丢掉的护栏。若更看重服务端集中编排/热修与已验证成熟度，**保留 V1 relay**。

6. **测试矩阵与可观测（贯穿全程）**
   - 现仅有 `tsc --noEmit` + 依赖真 China key 的 live 冒烟，无离线 CI 单测层。建立：编排纯函数的离线单测（orchestrator/voice 可无 key 测）、每传输的行为契约测试、以及对"哪台设备实际跑了哪个传输/provider"的遥测。
   - 若未来确需**选择 F 自适应路由**（仅混合车队 + 弱网场景），前置必须先建：网络检测(NetInfo/navigator.connection，现缺)、PCM 桥(现桩)、跨传输行为契约——且路由只在会话边界解析 + remount 子树，绝不在回合中切换以免破坏 Hooks 规则与临床会话一致性。

7. **配额与多租户（生产规模化）**
   - 账号级 ~60 RPM/100k TPM 是所有云路径的硬墙、无每-key 上限——任何"分发 token"方案都须在 mint 端点做每设备/每会话限流。规模化抗配额/区域故障时再引入**选择 E（relay 侧 Provider Mesh）**，因客户端零改动且 Web 实时本就只能在 relay 落地。

**一句话终态**：安全先补齐 → V1 作 Web 与当前默认 → 收敛单一编排源保真 → 补 PCM 桥后原生钉 V3 → Web 按需在 V1 relay 与选择 A 网关间取舍 → 规模化再叠加 Provider Mesh 与 token 网关限流；自适应路由与纯镜上本地推理仅在其特定场景（混合弱网车队 / 隐私离线）作为可选演进，不作生产必经路径。