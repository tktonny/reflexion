# 对话结果如何和后台 + caregiver 端打通

镜子端一次 check-in 的结果通过 `POST /api/conversations`(Expo Router 服务端路由)写入 MongoDB `ref`,reflexion-server / caregiver-app 直接读同一个库。这份文档说明**输出什么、caregiver 怎么消费、我们做了什么打通**。

## 1. 输出:两条文档

| 集合 | 内容 |
|---|---|
| `Conversation` | 一次会话的指标 + 逐句 `logs`(+ 我们新增的判断字段) |
| `ConversationIdToPatientIdMap` | `{ conversationId, nurseId, patientId, createdAt }` 索引行 |

`Conversation` 字段:`nurseId`/`patientId`(ObjectId,必填)、`clientSessionId`(幂等)、`deviceId`、`startedAt`/`endedAt`/`createdAt`、`sessionStatus`(completed/incomplete)、`duration`/`words`/`exchanges`/`avgLatency`、`userTurnCount`/`aiTurnCount`/`userSpeechSeconds`/`ariaSpeechSeconds`、`language`/`appVersion`/`networkStatus`/`technicalError`、`logs[]`(`{sentence, role:'Patient'|'AI', words, duration, wordsPerSecond}`)。

**我们新增(超出原 caregiver 基础 schema 的判断字段)**:`assessment`(完整结构化判断)+ 便于查询的 `riskScore`/`riskTier`/`screeningClassification`。

## 2. caregiver 端怎么消费(已核实其源码)

- **join**:按 `patientId`+`createdAt` 查 `ConversationIdToPatientIdMap` → 拿 `conversationId` → `{_id:$in}` 或 `{conversationId:$in}` 取 `Conversation`。我们写库时 `map.conversationId = Conversation._id`,join 成立。
- **状态/趋势/月计数**:只用 `duration` + `sessionStatus==='completed'` + `createdAt`。
- **会话回放**:全套指标 + `logs`。
- **AI 每日摘要**(`/patient-summary`,gpt-4o-mini):从 `logs` 拼转写,`normalizeRole = role.toLowerCase()==='ai'?'Aria':'Patient'`。我们写 `'AI'`,lowercase 后匹配,✅ 兼容。

→ **base 对话数据打通零改 caregiver**(`conversations+api.ts` 本就是 caregiver 对齐的契约)。

## 3. 之前的 gap(已修)

三个对话 hook(v1/v2/v3)之前**都没调用 `saveConversation`** → 对话结果根本没写库 → caregiver 看不到。上游是在被删掉的 OpenAI hook 里保存的。

## 4. 我们做了什么

- **`src/api/saveCheckin.ts`**:`buildCheckinPayload({messages,startedAt,endedAt,nurseId,patientId,deviceId,language,assessment})` 从 `messages` 构建 logs + 指标(`sessionStatus` 启发式:患者轮次 ≥3 记 completed;逐句 duration/wordsPerSecond 暂为 0——v2/v3 未像上游 WebRTC 那样测每句语音时长),`saveCheckin()` 调 `saveConversation` 并在失败时进离线队列。
- **`app/api/conversations+api.ts`**:扩展为持久化 `assessment` + 顶层 `riskScore`/`riskTier`/`screeningClassification`(caregiver 暂不展示,先落库;后续扩展)。
- **`app/realtime-test.tsx`**:点「结束并评估」时 → 停止 → 评估(拿判断)→ `resolveOwnerIds()`(已配对用真 ID,否则用 DEMO_IDS)→ `saveCheckin`;屏上显示"✓ 已保存到后台"或"未入库(离线队列)"。
- **`resolveOwnerIds`**:真实配对(`getStoredConversationOwnerIds`)优先,否则回退到 `DEMO_IDS`(`src/config/conversationMode.ts`,可用 `EXPO_PUBLIC_DEMO_*` 覆盖)。
- **`server/seed-demo-patient.mjs`**(`npm run seed`):把 demo nurse/patient(固定 ObjectId)upsert 进 `NursePatientConfig`,让 caregiver app 能显示。

## 5. 端到端跑通(需要你的 MongoDB Atlas)

本机没有 `MONGODB_URI`,所以实时入库/回读要在你的 Atlas 上验:
```bash
cd REFLEXION/mirror-app
# 1) .env 里填 MONGODB_URI=你的 ref 库连接串
npm run seed                 # 建 demo nurse+patient(固定 ObjectId)
# 2) 跑一次对话(Mac web,relay 或 http 均可),点「结束并评估」
#    → 屏上应显示 “✓ 已保存到后台”
npm run relay & npm run web  # /realtime-test → 开始 → 对话 → 结束并评估
# 3) 验证:在 Atlas 查 Conversation / ConversationIdToPatientIdMap(patientId=65f0…b2),
#    或用 reflexion-server 读该 patient 的 session/trend/summary。
```
已验证(无需 DB):`/api/conversations` 路由能加载并解析含 `assessment` 的 payload(无 URI 时返回 `missing_mongodb_uri`);typecheck 全绿。

## 6. 判断结果的落地选择(现选 a)

- **(a) 挂在 Conversation 上(已实现)**:`assessment` + `riskScore/riskTier/screeningClassification`。零改 caregiver、先持久化。
- (b) 独立 `ClinicAssessment` 集合(对齐蓝图临床层)——更规范,需扩展 caregiver 展示。
- (c) 复用 caregiver 自带 gpt-4o-mini 摘要(只要 logs 写好就有)——但那是它的摘要,非我们的结构化筛查。

后续要让 caregiver **显示**我们的判断,需扩展 caregiver-app/server 去读 `riskTier`/`screeningClassification`(数据已在库)。
