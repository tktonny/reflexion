# Mirror App 现状与缺口分析

> 代码基线：2026-07-22 工作区
> 评级：P0 = 阻止生产/受控试点；P1 = MVP 核心闭环；P2 = 质量、扩展或维护性

## 1. 总体成熟度

| 能力域 | 当前判断 | 说明 |
|---|---|---|
| Mirror UI 与语音原型 | Phase 2 前端已重做，语音稳定性继续验收 | 有 warm consumer 生产页、Qwen realtime、回合制 fallback；摄像头抽帧只保留在开发研究页，不进入生产日常会话 |
| 唤醒词 | 部分完成 | 原生 openWakeWord 已接入，但模型是 “hey jarvis”；Web 监听 “Hello” |
| 日常 Agent | 原型 | 有 companion prompt，无真实工具与计划服务，部分 transport 不传 persona |
| Daily check-in | 原型 | 有 4 域流程、强制 recall 和结束后 LLM 评估，但不是受控纵向方案 |
| Caregiver 配对 | 基本可演示 | 6 位码端到端存在，QR 只显示不扫描，鉴权不足 |
| 数据上传 | 部分完成 | 会话与 assessment 写 Mongo；失败进入 AsyncStorage 队列，仅手动 flush |
| caregiver 数据消费 | 部分完成 | 可查会话与完成率；风险字段未用于趋势/提醒 |
| 纵向分析 | 分裂实现 | Python 平台有文件式实现，caregiver Mongo 链路只看完成/缺席 |
| 安全/隐私/审计 | 不满足试点门槛 | caregiver 无 token、设备 auth 默认关、consent/audit 缺失 |
| 测试与发布 | 不满足试点门槛 | Caregiver API 已部署且健康；Mirror typecheck/smoke 可用，但缺少单元、E2E、CI 和 mirror 生产 API 配置 |

## 2. 已完成能力及代码证据

### 2.1 配对

- Mirror 生成 device ID、请求 15 分钟配对码并轮询状态：`mirror-app/app/index.tsx:146-305`、`mirror-app/app/api/mirror-pairing/request-code+api.ts`。
- Caregiver 可输入 6 位码并把 mirror 绑定到 patient：`caregiver-app/app/mirror-management/add.tsx:84-117`、`reflexion-server/src/routes/nurse-patient-config/mirrors/connect/index.ts`。
- 设备状态同时核验 map、pairing session 和 patient config：`mirror-app/app/api/mirror-pairing/device-status+api.ts:34-97`。
- 双端均有解绑路径。

### 2.2 Qwen 对话

- Native direct WS 使用 `qwen3.5-omni-flash-realtime`、本地能量 VAD、manual Qwen turn 和原生 PCM；relay/WebRTC 仍可使用 provider semantic VAD：`mirror-app/src/config/conversationMode.ts:28-47`、`mirror-app/src/orchestration/realtime.ts`、`mirror-app/src/hooks/useDirectRealtimeConversation.ts`。
- Web Node relay 和 native/web turn-based fallback 已存在：`mirror-app/server/relay.mjs`、`mirror-app/src/hooks/useConversation.ts`。
- Android 原生 PCM 模块包含 capture、playback、AEC/NS/AGC 尝试和播放积压：`mirror-app/modules/expo-pcm-audio/android/.../ExpoPcmAudioModule.kt`。

### 2.3 两种 persona 与 check-in

- 生产 conversation 页面有 companion 默认入口和 screening 单独入口：`mirror-app/app/conversation.tsx:116-294`。
- Screening flow 覆盖定向、近期叙事、日常功能和回忆：`mirror-app/src/orchestration/conversationFlow.json`。
- Direct WS 和 native HTTP 实现 screening-only recall floor：`mirror-app/src/orchestration/orchestrator.ts:38-78`。

### 2.4 评估和保存

- `MirrorCameraPanel` 的每 8 秒抽帧逻辑仍保留给开发研究页 `realtime-test.tsx`，但已从生产 `conversation.tsx` 移除；当前 MVP 日常会话不自动申请相机权限、不采样人脸。
- 会话结束后进行 transcript + visual observation 的两阶段评估，也提供 omni-first 路径：`mirror-app/src/api/assess.ts`、`mirror-app/app/api/assess+api.ts`。
- Assessment 写入 `Conversation.assessment/riskScore/riskTier/screeningClassification`：`mirror-app/app/api/conversations+api.ts:137-187`。
- 保存失败进入本地队列，支持幂等 `clientSessionId`：`mirror-app/src/api/saveCheckin.ts`、`mirror-app/src/storage/conversationQueue.ts`。

### 2.5 Caregiver 查询

- Caregiver App 的 `getApiUrl()` 已读取 `EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL` 并去掉请求路径中的 `/api` 前缀：`caregiver-app/src/lib/apiUrl.ts`。
- 当前部署 `https://reflexion-caregiver-app-server.vercel.app/health` 已于 2026-07-22 验证为 HTTP 200、`{"ok":true}`。
- Caregiver 服务可查当天会话、按天会话、月度完成数、7/30 天趋势和摘要。
- `patientTrend` 与 daily cron 支持每日完成/缺席状态：`reflexion-server/src/lib/statusEngine.ts`、`reflexion-server/src/routes/patient-trend.ts`。

### 2.6 Python 平台能力

- `_archived` 有 batch video、identity profile/link、feature snapshot、longitudinal profile 和 caregiver/provider surface service。
- `_archived/schemas` 已有 session、feature、identity 和 home risk schema，可作为统一契约的起点。

## 3. P0 缺口

### P0-01 Caregiver API 没有真正认证与对象授权

**证据**

- 登录只返回 `nurseId/name/email`，不签发 token：`reflexion-server/src/routes/auth/sign-in.ts:20-44`。
- App 把该对象保存到 localStorage/普通文件：`caregiver-app/src/lib/authSession.ts`。
- Router 没有 auth middleware：`reflexion-server/src/routes/router.ts`。
- 大量接口直接接受 query/body 中的 `nurseId` 或 `patientId`。

**影响**

知道或猜到 ObjectId 的调用者可能读取或修改不属于自己的患者资料、会话、配对和通知设置。

**所需修复**

- 引入真实 access/refresh token 或托管身份系统；
- 从 token subject 推导 caregiver 身份；
- 每个 patient/device 操作执行 relationship check；
- 增加授权集成测试。

### P0-02 设备和评估接口默认未强制鉴权

**证据**

- `REFLEXION_ENFORCE_DEVICE_AUTH` 和 `RELAY_ENFORCE_AUTH` 默认关闭：`mirror-app/.env.example`、`mirror-app/app/api/qwen-token+api.ts:18-24`、`mirror-app/server/index.mjs:47-78`。
- `/api/assess` 没有设备鉴权：`mirror-app/app/api/assess+api.ts:28-37`。
- 配对申请无速率限制，配对码使用 `Math.random()`：`mirror-app/app/api/mirror-pairing/request-code+api.ts:94-101`。

**影响**

可滥用 Qwen 费用、伪造或读取设备链路、暴力尝试配对码，并扩大健康数据入口攻击面。

### P0-03 当前“单次诊断”输出缺少临床和数据安全门控

**证据**

- Prompt 直接输出 `healthy | needs_observation | dementia`：`mirror-app/src/api/assess.ts:12-33`。
- LLM JSON 仅 `JSON.parse`，无 schema/value validation：`mirror-app/src/api/assess.ts:100-103, 124-128`。
- 结果没有 consent、protocol、prompt/model/schema 版本、QC usability、identity verdict 或 reviewer state。
- `finalize` 对 companion 和 screening 都执行同一个 screening assessment：`mirror-app/app/conversation.tsx:241-268`。

**影响**

普通天气聊天也可能被当作结构化筛查；低覆盖会话可能产生确定性疾病标签；无法审计模型漂移或重算。

**所需修复**

- 立即停止 patient/caregiver 侧疾病标签；
- 分离 companion observation 与 screening assessment；
- 先完成 consent、identity、QC、schema、version 和 reviewer 状态；
- 将单日输出改为 observation/insufficient-data，将风险建立在纵向基线之上。

### P0-04 缺少同意、目的限制、撤回与完整审计

**证据**

- Mirror 配对与会话开始没有读取 consent registry。
- 摄像头采样开始后把 base64 图像用于模型调用；omni-first 默认开启且先走 client-direct：`mirror-app/src/config/conversationMode.ts:19-22`、`mirror-app/src/api/assess.ts:134-146`。
- Mongo Conversation 没有 consent/purpose/retention/audit 引用。

**影响**

无法证明 patient 对 companion、screening、媒体、第三方处理和研究用途分别授权，也无法可靠执行撤回和删除。

### P0-05 Mirror 生产 Android 与后台接入没有闭环

**证据**

- 默认 mode 是 `relay`：`mirror-app/src/config/conversationMode.ts:11-12`。
- Relay hook 明确 native 不可用：`mirror-app/src/hooks/useQwenRealtimeConversation.ts:251-255`。
- `eas.json` 只有 preview 固定 `http`；production 没有 API base、mode 或功能开关。
- Native 无 `EXPO_PUBLIC_API_BASE` 时开机绕过配对，直接进入 `/realtime-test`：`mirror-app/app/index.tsx:63-69`。
- Caregiver backend 虽已部署，但不包含 mirror 需要的 `request-code/device-status/qwen-token/conversations/assess` 路由；且 caregiver App 会剥离 `/api` 前缀，mirror App 不会。

**影响**

正式包可能进入不可用或 demo 路径。把 `EXPO_PUBLIC_API_BASE` 直接设成现有 caregiver URL 也会因路由契约不匹配而失败；需要先把 mirror 路由迁入该服务或放到统一网关后。

### P0-06 日常核心工具完全缺失

**证据**

- Companion prompt 明确写“NO live weather data”：`mirror-app/src/orchestration/orchestrator.ts:113-131`。
- Realtime `session.update` 没有 tools/tool_choice：`mirror-app/src/orchestration/realtime.ts:45-67`。
- 仓库没有 medication plan、reminder occurrence、weather provider 或 calendar tool 实现。

**影响**

天气、吃药安排和提醒只是口头聊天，不能可靠查询、调度、确认或同步 caregiver。

### P0-07 没有安全事件与人工升级流程

当前 prompt、工具和 caregiver 服务都没有跌倒、严重不适、迷路、自伤、药物误服等安全事件的分类、确认、联系顺序、失败处理和审计。产品不能声称可处理紧急情况。

## 4. P1 缺口

### P1-01 Persona 在 transport 间行为不一致

- Direct WS 和 native HTTP 支持 persona。
- Web turn-based `useTurnBasedConversation` 的 Options 没有 persona，并总是调用 `buildLiveInstructions(..., {})`。
- Relay `useQwenRealtimeConversation` 不接受 persona，server relay 也总是生成默认 screening prompt。

结果：UI 显示“日常助手”时，relay/web HTTP 仍可能执行认知 check-in。

### P1-02 真实 patient context 没有进入 Agent

- 生产页调用 `useConversation({ language, persona })`，没有传 `patientId`：`mirror-app/app/conversation.tsx:122-134`。
- 所有 hook 因此回退为 `demo-patient`。
- Orchestrator 已支持 `patientName/memory`，但调用方没有传入；实际只在 UI 显示姓名。
- Relay hook 用 `useState(options.language)` 固化初始语言，配对资料异步加载后可能不更新。

### P1-03 “上传诊断数据”没有形成消费闭环

- Risk 字段已经落 Mongo。
- `reflexion-server/src/lib/statusEngine.ts:27-48` 只根据 completed session 输出 red/green。
- `patientTrend` 展示的是时长/缺席，不是 baseline 或认知变化。
- caregiver UI 多个详情/趋势页面仍有 “not ready” placeholder，alerts 页面是 coming soon。

### P1-04 离线能力低于 UI 承诺

- Offline 首页写“仍可完成 check-in”，但只有静态页面，没有开始按钮：`mirror-app/app/index.tsx:364-379`。
- 会话本身需要云端 Qwen。
- 队列只保存结束后的 Conversation payload；没有保存待评估媒体或后续 assessment job。
- 自动重连没有调用 `flushPendingConversations`；只有 settings 手动 “Upload logs”。
- reset pairing 会清除 pending conversations，存在未上传数据丢失风险：`mirror-app/app/settings.tsx:118-155`。

### P1-05 没有完整原始会话与统一 SessionRecord

- Mirror 只保留 transcript 和最多 6 张抽帧，没有完整音频/视频 artifact。
- `conversationFlow.json` 和旧文档宣称“full recorded session uploaded”，与 mobile 实现不符。
- 保存字段的单句 duration/WPS、speech seconds、latency 全部为 0；`networkStatus` 总是 online，`technicalError` 总是 false，appVersion 写死 `0.0.1`：`mirror-app/src/api/saveCheckin.ts:28-72`。

### P1-06 Python 纵向能力未接入产品链路

`_archived` 已实现 identity、feature snapshot 和 longitudinal 文件，但：

- Mirror 保存到 Mongo `Conversation`；
- Python 平台保存本地 JSON/media；
- caregiver 只读 Mongo；
- 没有 event/job/API 把一个 mirror session 转为 Python pipeline 的正式输入；
- `_archived` 的纵向 feature 中仍出现 `mock-*-v1` modelVersion。

### P1-07 Caregiver 通知和提醒未实现

- `expo-notifications` 是依赖，但没有 push token 注册和发送逻辑。
- Alerts tab 指向 “Coming soon” 页面：`caregiver-app/app/notifications.tsx`。
- `pushNotificationsEnabled/alertSensitivity` 只是配置字段。
- `dailySummaryCron.ts` 是常驻 `setInterval` worker，仓库没有 Vercel Cron 配置；API 健康不代表该后台任务正在 serverless 部署中运行，需要单独配置并验证。

### P1-08 配对 QR 流程未完成

- Mirror 显示真实 QR。
- Caregiver 只支持输入码；帮助文字明确写 scanner support 尚未启用：`caregiver-app/app/mirror-management/add.tsx:182-188`。

### P1-09 正式唤醒词和运行指标未完成

- APK 内置模型是 openWakeWord “hey jarvis”，并非产品目标词：`mirror-app/docs/WAKEWORD.md`。
- Web 监听任意转录中的 “hello”，不是专用 wake word。
- 没有 false accept/hour、false reject、噪声、距离、方言和电视干扰测试。

### P1-10 缺少自动化质量门槛

- Mirror TypeScript typecheck 通过，hardware decision smoke 通过。
- Caregiver server 当前依赖未安装，typecheck 无法运行。
- Mirror/caregiver 未发现 unit/integration/E2E tests 和 CI。
- 多数 Qwen smoke 需要外部 key，未形成可重复的 provider mock contract test。

## 5. P2 缺口与技术债

| ID | 缺口 | 证据/影响 |
|---|---|---|
| P2-01 | 模型与配置漂移 | Direct 默认 qwen3.5 realtime；relay 默认旧 `qwen3-omni-flash-realtime`，tokens/VAD/voice 也不同 |
| P2-02 | 重复 prompt/source | `src/prompts/*.ts` 与 orchestration flow 并存，实际路径不统一；generated relay bundle 易陈旧 |
| P2-03 | 文档陈旧 | Root README 仍指向不存在的 `platform/`；README 还写 OpenAI Realtime；部分离线说明与当前代码相反 |
| P2-04 | 时区硬编码 | Caregiver 日界线与 trend 使用 Asia/Singapore，而 patient 已有 timezone 字段 |
| P2-05 | token 存储不安全 | Mirror 已安装 SecureStore 但 auth token 仍放 AsyncStorage；caregiver session 放 localStorage/普通文件 |
| P2-06 | 配对和 DB 写入非事务 | config、map、pairing session 多步更新，部分失败会形成不一致 |
| P2-07 | Mongo 连接模式重复 | Expo API 多数请求新建/关闭 MongoClient；仅 deviceAuth 使用池 |
| P2-08 | 错误状态不透明 | 结束页总说会通知 caregiver；用户不知道是已保存、排队还是 assessment 失败 |
| P2-09 | 多 caregiver 不支持 | Mock FAQ 也明确标为 future；数据模型以 nurse config 内嵌 patients 为中心 |
| P2-10 | 可观测性不足 | 没有统一 trace、session state、tool/assessment/upload 指标或隐私安全日志策略 |

## 6. 需求覆盖矩阵

| 用户目标 | 当前覆盖 | 结论 |
|---|---|---|
| Qwen3.5 Omni realtime 对话 | Native direct WS 有，relay 配置存在漂移 | 部分完成 |
| 日常自然对话 | Companion prompt 与 UI 有 | 部分完成 |
| 真实天气 | 无 provider/tool | 缺失 |
| 吃药安排 | 只有问句和口头提醒，没有结构化 plan/scheduler | 缺失 |
| 日程/提醒 | 无 tool、scheduler、occurrence 和通知闭环 | 缺失 |
| 每日认知采集 | 有协议化 conversation flow | 原型完成 |
| 从日常对话积累认知信号 | 所有结束会话会被同一 prompt 评估，但缺少有效性边界 | 实现方式不安全 |
| 纵向痴呆问题分析 | Python 有分离原型，产品 Mongo 链路无 baseline | 缺失核心闭环 |
| 唤醒词启动 | “hey jarvis”/“hello” 可演示 | 部分完成 |
| Caregiver 配对 | 6 位码可演示，QR scan 缺失 | 基本完成 |
| 上传诊断数据 | Risk 字段可写入，离线/版本/QC/消费不完整 | 部分完成 |
| Caregiver 查看诊断趋势 | 只看完成率/时长 | 缺失 |
| 安全与隐私 | 有少量 device auth 开关和 key relay | 不满足试点 |

## 7. 立即决策

在继续堆功能前需要冻结以下决策：

1. 正式产品术语使用“认知状态监测/变化筛查”，不是“每日诊断”。
2. Companion 是否允许提取被动认知特征；若允许，单独定义 consent purpose、最小字段和验证方案。
3. 首发语言建议只冻结普通话和英语，其他语言保持实验开关。
4. 是否保存完整音频/视频；必须明确默认值、保留时间、第三方处理和撤回流程。
5. Provider reviewer 是首发必需还是由受控研究人员承担；不能没有复核主体却输出疾病级风险。
6. 选择唯一生产后台。建议正式 API + Realtime Gateway + 异步 clinical pipeline，不继续让三套后端各自成为事实来源。
