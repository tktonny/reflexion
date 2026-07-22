# 完整需求文档与 Platform v2 对齐说明

> 需求源：[June-Aug 2026, Reflexion Tech Document](https://docs.google.com/document/d/1eAGXsTYmgAUYaYR-JhhNBaWkw3fopzMIodCL3CH1IgE/edit)
> 本地权威副本：`docs/June-Aug 2026, Reflexion Tech Document.docx`
> 本轮读取范围：完整 106 页正文与 7 张内嵌设计图，不再只依赖 Google Docs 单一 tab 或局部文本导出
> 读取 revision：`ALtnJHzbc9274UHyAb01w1RxBAr4MSTAwEzx5lk8lEXjrG6PD_vxmCap82NtBPDeZS8IbTiX0OMGC7QPBYy17A7LASyXDyEiF9a0YId8iNk`
> 对齐日期：2026-07-22
> 重点 sections：`Layer 1: short term`、`Design of the app`、`Updated metrics`、`Reflexion Signal-to-Status Algorithm`、`MIRROR APK`

Mirror 前端硬件基线采用文档明确的 10.1 英寸、1080 × 1920 纵向、Android 14、4 GB RAM、32 GB ROM；交互按 no-touch assumption 设计，触摸只可作为管理员/开发 fallback，不能成为老人完成主流程的前提。

## 1. 需求权威关系

- `Updated metrics`：caregiver status、baseline、M1–M13、通知和显示规则的产品需求源。
- `Reflexion Signal-to-Status Algorithm`：Mirror → Backend → Caregiver 的开发交接和确定性状态规则。
- `MIRROR APK`：设备端体验、采集字段、错误处理、隐私、MVP 范围和验收标准。
- Platform v2 文档负责把这些需求映射到统一 API、数据库、异步处理、版本和权限；不得反向扩大 MVP 产品承诺。

如果不同 tab 的阈值或状态规则冲突，不能在代码中自行选一个。将冲突放入 versioned `rule_registry` 的产品决策清单，由产品 owner 明确生效版本。

## 2. 已完全对齐的原则

| Google 需求 | Platform v2 落点 |
|---|---|
| Mirror 只运行对话和上传原始信号 | `session`、`event-batches`、artifact 和 heartbeat API |
| Mirror 不计算或显示绿/黄/红 | `status-engine` 是唯一计算者；Caregiver 只读取结果 |
| Backend 保存 session、计算 baseline/ratio/status | `sessions`、`operational_baselines`、`daily_statuses` |
| Caregiver 不独立计算状态 | `GET /patients/{patientId}/status` 返回权威 read model |
| 产品是 daily reassurance，不是诊断 | 运营状态与高级 longitudinal review state 分离 |
| 单日异常不告警，关注多日 pattern | persistence rule 和 notification suppression |
| baseline 完成前显示 Establishing | `baselineState=establishing`，不伪造 green/amber/red trend |
| 设备离线不能解释为用户未参与 | 独立 heartbeat，状态 job 先判断 technical state |
| Away day 不计 missed streak | `away_periods`，当天 status 可以是 `null` |
| 阈值不硬编码 APK | `rule_registry` + backend status engine |
| 后端不可用时本地保存、稍后上传 | encrypted durable outbox + idempotent ingestion |
| 不静默录音、明确 mic-active | Mirror privacy/consent requirements |

## 3. MVP 数据契约

Mirror 每个 session 至少上传：

```text
sessionId
patient/device identity（由设备凭据和 assignment 服务端解析）
startedAt / endedAt / timezone
userSpeechSeconds / ariaSpeechSeconds / totalSessionSeconds
userTurnCount / ariaTurnCount / repromptCount
per-turn prompt-end and user-speech-start timestamps
wordCount / transcriptAvailable（存在可靠 ASR 时）
raw mood/medication response（仅在需求允许且 ASR 可用时）
technicalError / technicalErrorType
appVersion
sessionState: completed | incomplete | technical_error
```

设备 heartbeat 独立于 session，默认每 5–15 分钟发送：

```text
deviceId（从 device token 解析并与 path 校验）
recordedAt
appVersion
networkStatus
micStatus
speakerStatus（建议补充）
backendReachable
```

客户端 body 中的 `userId/deviceId` 不能作为授权事实；服务端必须从 token、tenant 和 active assignment 推导后再与 payload 校验。

## 4. Baseline 冲突的架构处理

Google 需求明确 MVP 运营 baseline：

- rolling 14 calendar days；
- minimum 7 completed sessions；
- 每 7 天更新；
- EWMA 初始 `alpha=0.1`；
- 完成前显示 Establishing。

此前研究 PRD 定义 12 high-quality sessions / 28 days / 3 weeks。两者用途不同：

| Baseline | 用途 | 可见输出 |
|---|---|---|
| `reassurance_mvp` | M1–M5 engagement/routine status | Establishing、Doing Well、Worth Checking、Needs Attention |
| `longitudinal_research` | QC 合格的 speech/task/embedding 个人变化 | insufficient_data、stable、watch、review_recommended |

两种 baseline 使用不同 collection/type、revision 和 rule version。MVP 运营 baseline 不能被称为临床认知 baseline，高级研究结果也不能未经 policy 直接覆盖 caregiver 红黄绿。

## 5. MVP 与后续研究范围

### 当前 MVP

- M1 completed session today
- M2 missed-day streak
- M3 time vs usual window
- M4 user speech/session duration
- M5 weekly engagement
- session incomplete/abandonment 基础状态
- heartbeat、technical error、away day、manual flag
- deterministic status、primary reason、notification dedupe

### 后续或研究模式

- word count、speech rate、response latency 的更深变化规则
- lexical diversity、disfluency、language coherence
- acoustic/semantic embeddings
- camera/face identity 或行为研究
- dementia score、clinical diagnosis 和疾病结论

向量化长期监控保留在架构中，但应以 shadow/research pipeline 实施，不扩大当前 MVP caregiver 文案。

## 6. 新增 API 需求

- `POST /api/v1/devices/{deviceId}/heartbeats`
- `GET /api/v1/patients/{patientId}/status`
- `POST /api/v1/patients/{patientId}/away-periods`
- `POST /api/v1/patients/{patientId}/manual-flags`

现有 session/event/artifact API 覆盖 session start/end、raw signals、offline replay 和 processing state。

## 7. 新增数据库需求

- `operational_baselines`
- `daily_statuses`
- `away_periods`
- `manual_flags`
- `notification_suppressions`
- `device_telemetry` time-series heartbeat

`daily_statuses` 必须保存 `ruleVersion`、primary/secondary reasons、technical state、local date/timezone 和 finalization time，以便复现 7pm/midnight job 的决定。

## 8. 尚需产品确认的规则

1. `Updated metrics` 中“一项 metric 最多 48 小时一个 alert”与 acknowledged alert 的 24 小时 suppression 如何叠加。
2. Establishing 期间技术故障和连续未互动是只显示运营提示，还是允许独立设备告警；不能把它混成行为异常。
3. 7pm 和 midnight 的具体时区、daylight-saving、迟到 session 和离线上传回补规则。
4. Red 自动恢复到 Green 是否需要缓冲/恢复窗口，避免状态来回跳变。
5. `mood/medication_response_text` 的保留、分类可靠性、用户同意和 caregiver 展示范围。
6. M6–M13 的准确 phase、启用顺序和阈值 owner。

这些项进入 `rule_registry` 前应有明确 decision record 和测试样例，不能以散落常量实现。

## 9. 对当前代码的直接影响

- 当前 `DailyPatientStatus` 只含完成计数和简单红黄绿，不足以保存 baseline、技术状态、原因、规则版本和 finalization trace。
- 当前 caregiver 登录无 access token，不能安全开放 status、away 和 manual flag API。
- 当前设备映射和 token 嵌入 patient，无法安全验证 heartbeat 的设备身份。
- 当前 Conversation/Map 可提供部分 M1–M5 输入，但缺少统一 session state、per-turn timing、technical events 和幂等 event ingestion。
- 当前 mock embedding/synthetic trend 与 MVP 需求无关，也不能进入高级 longitudinal evidence。
