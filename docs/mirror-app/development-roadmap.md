# Mirror App 开发路线图

> 起始日期：2026-07-22
> 目标：12 周内从功能原型推进到可进行小规模家庭技术试用的候选版本
> 前提：不把技术试用包装为临床诊断
> 架构依赖：实施顺序以 [Platform v2 API 与领域架构](../architecture/platform-v2-api-and-domain-architecture.md)、[Platform v2 数据库设计](../architecture/platform-v2-database-design.md) 和 [纵向监控与向量异常检测](../architecture/longitudinal-vector-anomaly-design.md) 为技术基线。

## 1. 优先级规则

- **P0**：数据越权、密钥泄露、错误临床结论、生产包不可用或核心数据丢失。
- **P1**：用户核心旅程无法闭环，或 caregiver 看不到可行动结果。
- **P2**：性能、语言、运维、扩展和长期维护性。

先关闭 P0，再增加天气、药物等表层能力。任何包含健康数据的家庭试用都不能绕过认证、同意和数据治理。

## 2. 执行主线：Phase 0–7

这条主线确定产品实施顺序；后面的 Sprint 计划说明跨工作流依赖。当前唯一第一优先级是 Phase 1 的稳定 turn-taking，在真机退出标准通过前，不把 WebRTC 或视觉改版切为默认产品路径。

| Phase | 主题 | 核心交付 | 退出标准 | 状态 |
|---|---|---|---|---|
| 0 | 设计与测量基线 | PRD、目标架构、API/数据库边界、turn-taking contract、事件回放基线 | 需求可追踪；20 条确定性音频生命周期回放通过 | 已完成首版 |
| 1 | LLM / Realtime 核心 | 单一 turn 状态机、VAD、播放完成闸门、正确问题顺序、完整告别、降级与本地可观测状态 | 真机连续 30 回合无抢话/截断；30 分钟与 100 回合 soak 无重叠响应 | **MuMu 门通过；实体镜面声学门待完成** |
| 2 | Consumer Mirror 前端 | 按需求图重做空闲、唤醒、倾听、思考、说话、提醒、异常和结束画面；适老可读性、动效与隐私状态 | 目标镜面尺寸视觉评审通过；状态与语音生命周期一致 | 未开始 |
| 3 | 统一后端与数据底座 | 将 Mirror 路由迁入已部署的 reflexion-server/统一 API 域；统一身份、设备凭据、Session API、数据库和异步处理 | Mirror/Caregiver 使用同一认证和 API 域；密钥不落客户端；幂等 session 上传 | 未开始 |
| 4 | 配对与 caregiver 闭环 | 事务化配对、设备状态、可靠 outbox、patient 状态、caregiver 查看、提醒/通知等非对话核心能力 | 配对—对话—上传—caregiver 查看端到端通过；断网重启不丢数据 | 未开始 |
| 5 | 唤醒词与设备运行时 | 产品唤醒词、误/漏唤醒测试、kiosk、崩溃/网络恢复、OTA 与设备健康 | 真机噪声/电视/距离基准达标；一周设备 soak | 未开始 |
| 6 | Agent 工具与联网能力 | 服务端 allowlist tool calling、天气、日程、用药 occurrence、提醒、受控联网搜索和安全升级 | 工具 schema/权限/超时/幂等/确认测试通过；模型不能自由执行副作用 | 未开始 |
| 7 | 长期监控与向量异常 | QC/identity gate、版本化 feature、双 baseline、shadow embedding、多分量异常、persistence、reviewer 闭环 | 不足数据不出风险；异常可复现；研究向量不直接覆盖 caregiver 产品状态 | 未开始 |

Phase 2–7 可以做不影响主链的设计准备，但在 Phase 1 真机验收前不分散 realtime 修复资源。Phase 3 的认证、同意和数据治理是任何家庭试点的发布门槛，即使它在体验实现顺序上排在前端之后。

Phase 1 的 [MuMu 验收](./phase1-mumu-acceptance.md) 已完成：30/30 smoke、100/100 soak、连续运行超过 30 分钟，重叠 response、提前开麦和状态机 violation 均为 0。模拟器无法验证实体扬声器回声与房间声学，因此目标镜面声学 smoke 仍是转入硬件完成态前的最后 gate。

## 3. 12 周计划

### Sprint 0：第 1–2 周 — 冻结边界并止血

交付：

- 冻结 intended use、companion/screening 数据用途和患者可见措辞；
- 关闭 caregiver IDOR：access token、auth middleware、patient relationship authorization；
- 设备 auth 在所有生产接口默认强制开启；
- `/assess`、token mint、pairing、upload 增加 auth、rate limit、size/schema validation；
- 停止 client-direct omni assessment 作为生产默认；
- 以已部署的 reflexion-server 为起点迁入 mirror API，或建立统一 API Gateway；统一 `/api` 前缀和认证契约；
- production EAS profile 固定新的 mirror `EXPO_PUBLIC_API_BASE`、`ws` mode 和安全 feature flags；
- 移除生产启动时自动进入 demo/realtime-test；
- 新建 CI：mirror typecheck、caregiver typecheck、Python tests、lint、secret scan、contract tests；
- 清理 README 中错误的 OpenAI/platform 路径与离线承诺。

退出标准：

- 未登录用户不能读取或修改任何 patient 数据；
- 被撤销 device 不能 mint token、连接 realtime 或上传；
- 生产 APK 在无本地 `.env` 时完成配对、对话和上传；
- caregiver 与 mirror 客户端都通过同一受控 API 契约工作，不再依赖 Expo Router 生产 API；
- 单次结果不再向 patient/caregiver 输出 `dementia`。

### Sprint 1：第 3–4 周 — 统一 Session 与可靠上传

交付：

- 定义 `SessionRecord v1`、`SessionObservation v1`、`ToolEvent v1`；
- 建立 v2 normalized collections、核心索引、outbox 和 legacy ID 对照，不删除旧集合；
- 引入 Session Service 和明确生命周期；
- Mirror 使用真实 patientId、patientName、language、deviceId、sessionId；
- 修复 relay/web HTTP 的 persona 传递；
- 设备使用 SecureStore 保存凭据；
- 实现加密 durable outbox、自动重试、指数退避、幂等上传；
- 区分 `assessment_pending/failed/completed/excluded`；
- reset pairing 不静默删除未上传健康数据；
- 真实采集 turn timing、网络、fallback、app/model/prompt 版本。

退出标准：

- 杀进程、断网和重启后 session 不丢失；
- 相同 session 重传不会重复；
- companion 与 screening 在 WS/HTTP/relay contract tests 中一致。

### Sprint 2：第 5–6 周 — 日常 Agent 工具 MVP

交付：

- Agent Tool Service 和 allowlist schema；
- 天气 provider adapter、位置/城市配置、缓存和来源时间；
- `CarePlan/MedicationPlan/ReminderOccurrence` 数据模型与 caregiver 配置 API；
- Mirror 到点提醒、taken/skipped/unsure/later/no-response 回执；
- caregiver 待办列表和用药确认历史；
- 工具副作用确认、幂等、权限、超时和失败口语；
- prompt injection 与越权工具测试。

退出标准：

- “今天天气怎样”返回真实、带时间和位置的结果；
- caregiver 配置的药物计划可准时提醒并回传状态；
- Agent 无法自行创建药名/剂量或修改 care plan。

### Sprint 3：第 7–8 周 — Screening 数据管线

交付：

- 将 post-session assessment 移出设备同步结束流程，改为异步 job；
- 接入 consent/purpose/version；
- 实现音频、语言、任务覆盖、身份和混杂 QC gate；
- LLM 输出 JSON schema/value validation；
- 将 `_archived` feature/identity/longitudinal 能力接到统一 session store；
- 移除 `mock-*` feature version 或明确仅用于 dev；
- 禁止 synthetic trend 进入产品读模型；数据不足时返回 `baseline_building/insufficient_data`；
- 保留 immutable raw result + revision，不覆盖历史；
- 建立 reviewer case API 和最小 provider/research review 页面。

退出标准：

- 无同意或 QC 不足时绝不生成确定性风险；
- 每个结果可追到 session、artifact、protocol、prompt、model、schema 和 code version；
- reviewer 可以接受、驳回、要求重测并记录原因。

### Sprint 4：第 9–10 周 — Caregiver 闭环与通知

交付：

- 7/30 天页面从“完成时长”升级为依从性 + baseline coverage + provider-approved 状态；
- 实现需求文档定义的 M1–M5、14 天且至少 7 次的运营 baseline、7pm 检查、midnight finalization、heartbeat 技术故障隔离和告警去重；
- 会话详情显示 processing/QC/repeat-needed，不显示未经批准的疾病标签；
- push token 注册、通知服务、已读、偏好和 deep link；
- missed check-in、连续未确认用药、设备离线和 review-needed 通知；
- 完成 caregiver QR scanner；
- 数据访问、导出、更正、撤回入口。

退出标准：

- 一个 mirror event 能可靠形成 caregiver 可处理任务；
- 通知内容不泄露锁屏敏感信息；
- caregiver 只能访问授权患者。

### Sprint 5：第 11–12 周 — 设备与试用准备

交付：

- 训练/替换正式 “Hey Aria” 或选定目标词模型；
- 真机噪声、距离、电视、多人、方言、误唤醒和漏唤醒测试；
- kiosk、崩溃恢复、屏幕常亮、网络恢复、OTA/回滚；
- 端到端观测 dashboard 和告警；
- 数据备份恢复、撤回删除和 incident tabletop test；
- 设备/家庭安装指南、support runbook、pilot readiness checklist；
- 5–10 台内部/友好家庭技术试用，明确不输出临床诊断。

退出标准：

- 完成一周 soak test；
- P0 缺陷为 0，关键 P1 有明确豁免；
- 上传、通知、设备在线和工具 SLO 达到 PRD 目标；
- 每台设备可远程确认版本、健康和撤销状态。

## 4. 12 周之后

### Phase A：受控可用性与采集质量研究（约 2–3 个月）

- 老年用户可用性、唤醒与对话完成率；
- 音频/语言/身份/QC 失败模式；
- reminder adherence 与 caregiver 负担；
- 同意理解、退出和隐私体验；
- 冻结 protocol 和数据字典。

### Phase B：纵向研究（至少 3–12 个月）

- 28 天以上个人基线；
- 90 天变化观察；
- 与 clinician、MoCA/等价量表和随访结局对照；
- 测量误报/漏报、alert burden、lead time、校准和亚组性能；
- 冻结阈值前不做商业诊断宣称。
- 先验证结构化 robust baseline 与 persistence，再以 shadow 模式评估同构 acoustic/semantic embedding；向量分量通过锁定评估后才加入 ensemble。

### Phase C：临床与监管准备

- 设计控制、需求追踪、风险管理、模型注册与变更控制；
- 锁定数据集和外部验证；
- 网络安全、隐私、供应商和事件处理证据；
- 临床评估与适用监管路径确认。

## 5. 工程 Backlog

### P0

- AUTH-001 Caregiver access/refresh token
- AUTH-002 Patient relationship authorization middleware
- DEV-001 Device credential rotation and revocation
- API-001 Schema/size/rate-limit layer
- BUILD-001 Reproducible production Android profile
- DATA-001 Consent registry and purpose enforcement
- DATA-002 SessionRecord + idempotent ingestion
- COG-001 Remove single-session patient-facing disease classification
- COG-002 QC/identity/version gate
- TOOL-001 Weather and medication tool framework
- TEST-001 CI and auth/data-loss contract tests

### P1

- AGENT-001 Persona parity across transports
- AGENT-002 Real patient context and memory provenance
- SYNC-001 Encrypted durable outbox and background sync
- MEDIA-001 Artifact upload decision and implementation
- LONG-001 Integrate feature snapshots and baseline engine
- REVIEW-001 Reviewer queue and disposition
- CARE-001 Risk-safe trend/summary UI
- PUSH-001 Push notification pipeline
- PAIR-001 QR scanner and transactional pairing
- WAKE-001 Product wake word and benchmark
- OBS-001 Privacy-safe telemetry and fleet health

### P2

- I18N-001 Language qualification and feature flags
- ARCH-001 Remove duplicate prompt/config implementations
- DATA-003 Patient-specific timezone everywhere
- CARE-002 Multiple caregiver roles and invites
- OPS-001 OTA canary, rollback and remote diagnostics
- DOC-001 Automated API/schema/docs generation

## 6. 测试策略

| 层级 | 必须覆盖 |
|---|---|
| Unit | prompt policy、tool validation、reminder recurrence、risk/QC rules、serialization |
| Contract | Qwen event adapter、tool schema、SessionRecord、Mongo/Python adapters |
| Integration | pairing、authz、session lifecycle、offline retry、notification、withdrawal |
| E2E | Mirror → Qwen → upload → processing → caregiver task；断网和 token revoke 场景 |
| Device | AEC、VAD、唤醒词、kiosk、长稳、相机、重启、网络切换 |
| Safety | 药物变更拒绝、紧急语句、prompt injection、越权和错误实时数据 |
| Clinical/ML | locked dataset、patient-level split、校准、亚组、QC gating、版本回归 |

CI 不调用真实生产患者数据。外部模型测试使用脱敏固定样本和可重复的 mock/recording；在线 smoke 与离线 contract test 分开。

## 7. 团队分工建议

| 工作流 | 主要负责人 |
|---|---|
| Mirror runtime / audio / wake | Mobile + embedded engineer |
| API auth / session / tools | Backend engineer |
| Caregiver UX / notification | Mobile/full-stack engineer |
| QC / feature / longitudinal | ML + backend engineer |
| Protocol / review / safety | Clinical lead + product |
| Consent / privacy / quality | Privacy/regulatory + engineering |
| Device fleet / observability | Platform/DevOps |

## 8. 每周项目指标

- P0/P1 defect burn-down；
- pairing、wake、session completion、fallback 和 crash-free；
- tool success/timeout/denied；
- outbox backlog 和 upload age；
- QC inclusion/repeat/exclusion；
- baseline coverage；
- review queue age；
- caregiver notification delivery/action；
- 未授权访问和敏感日志检查结果。

临床性能指标与工程指标分开汇报，避免把产品稳定性误称为诊断有效性。
