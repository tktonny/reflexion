# Legacy API ↔ v1 平台 对应关系 & 迁移文档(交接用)

> 目标:**caregiver app 几乎不改**。做法是——把 app 已经在用的 legacy API(`/auth/sign-in`、`/nurse-patient-config/*`、`/conversation-*`、`/patient-*`)保留**接口形状不变**,但服务端内部**改成读写 v1 数据模型**;并把现有 `NursePatientConfig` 数据**一次性迁移**到 v1 的规范化集合。镜子端已经全面 v1(bootstrap→配对→device 凭证→Qwen 短票);本文档让 caregiver 端和镜子端**落到同一套 v1 数据**上。
>
> 方案 = **(b) 一次性迁移 + legacy 路由适配 v1**。

---

## 0. 背景:当前是"两套数据模型并存"

| | Legacy(caregiver app 现在用) | v1(镜子 + 新平台用) |
|---|---|---|
| 护士账号 | `NursePatientConfig`(1 文档 = 1 护士,`_id`=nurseId,内嵌 `patients[]`) | `tenants` + `users`(+ `auth_sessions`) |
| 患者 | `NursePatientConfig.patients[]`(内嵌,`_id`=patientId) | `patients`(独立) + `care_relationships`(护士↔患者 + scopes) |
| 配对 | `MirrorPairingSessions`(6 位码明文/authToken) | `device_pairings`(codeHash=hmac) + `device_assignments` + `device_credentials` |
| 设备归属 | `MirrorIdToNurseIdMap` | `device_assignments`(tenantId+deviceId+patientId) |
| 对话/会话 | `Conversation` + `ConversationIdToPatientIdMap` | `sessions` + `session_events` + `transcript_turns` + `artifacts` |
| 每日状态 | `DailyPatientStatus` | `daily_statuses` |
| 鉴权 | **无 token**:sign-in 返回 `{nurseId,name,email}`,后续请求在 body/query 里带 nurseId,服务器信任 | **JWT**:human/device/bootstrap 三类 token(HS256),`requireActor` 中间件 + DB 会话校验 |
| id 形态 | Mongo `ObjectId`(24 hex) | 字符串 `usr_…`/`pat_…`/`ten_…`/`dev_…`(`newId(prefix)`) |

两套现在**都落在 `reflexion_production` 库**(已通过 `MONGODB_DB` 统一)。

---

## 1. ID 策略(最关键,决定 app 是否要改)

caregiver app 把 `nurseId`/`patientId` 当 **24 位 ObjectId hex** 校验(如 `session/[id].tsx` 用 `/^[0-9a-f]{24}$/`)。v1 默认 id 带前缀(`pat_…`),app 会拒绝。

**规则:迁移时,v1 记录的 `_id` 直接沿用 legacy 的 ObjectId 十六进制字符串**(v1 `_id` 本就是任意字符串):
- v1 `users._id`   = 旧 `NursePatientConfig._id`.toHexString()  → 即旧 nurseId
- v1 `patients._id` = 旧内嵌 patient `_id`.toHexString()          → 即旧 patientId
- v1 `tenants._id`  = 每个护士一个租户:`ten_<旧 nurseId hex>`(内部用,不回传 app)
- v1 `devices._id`  = 保持镜子 provision 时的 `dev_…`(不变)

这样 legacy API 回传的 id 和 app 预期完全一致,**app 零改动**;新由 v1 API 创建的记录仍用 `pat_…` 前缀(id 是不透明字符串,混用无碍)。

> 反向兼容:适配层收到 app 传来的 24-hex `nurseId`,直接当 v1 `users._id` 用;传来的 `patientId` 直接当 v1 `patients._id` 用。无需额外映射表。

---

## 2. 数据迁移映射(`NursePatientConfig` → v1)

一次性脚本 `caregiver-server/src/scripts/migrateLegacyToV1.ts`(建议 `npm run migrate:legacy-v1`),幂等(按 `_id` upsert)。逐护士文档:

### 2.1 护士 → tenant + user
| v1 `tenants` | 来源 |
|---|---|
| `_id` | `ten_<nurseHex>` |
| `name` | `config.name` + " tenant" |
| `status` | `'active'` |
| `createdAt/updatedAt` | 沿用 / now |

| v1 `users` | 来源(`NursePatientConfig`) |
|---|---|
| `_id` | `nurse._id.toHexString()`(= nurseId) |
| `tenantId` | `ten_<nurseHex>` |
| `name` / `email` / `passwordHash` | 直接搬(**passwordHash 同为 `pbkdf2_sha256$…` 格式,v1 `verifyPassword` 兼容,免重置**) |
| `phoneNumber` | 搬 |
| `roles` | `['caregiver','tenant_admin']`(tenant_admin 便于 authorizePatient 通过) |
| `scopes` | `[]`(护士的患者级权限走 care_relationships) |
| `status` | `'active'` |

> 通知偏好 `pushNotificationsEnabled/alertSensitivity/preferredDailySummaryTime` 迁到 user 上(或 `notification_preferences`),`/notifications` 适配层读写它。

### 2.2 每个内嵌 patient → patient + care_relationship（+ 已配对则 device/assignment）
| v1 `patients` | 来源(`patient`) |
|---|---|
| `_id` | `patient._id.toHexString()`(= patientId) |
| `tenantId` | `ten_<nurseHex>` |
| `displayName` | `patient.name` |
| `preferredLanguage` | `patient.preferredLanguage`(english/mandarin/other) |
| `timezone` | `patient.timezone || 'Asia/Singapore'` |
| `ageBand` | 由 `patient.age` 归桶(可空) |
| `status` / `version` | `'active'` / `1` |

| v1 `care_relationships` | 来源 |
|---|---|
| `_id` | `rel_<patientHex>` |
| `tenantId`/`patientId`/`userId` | 租户 / patientId / nurseId |
| `relationshipType` | `patient.relationshipToElderly` 或 `'caregiver'` |
| `scopes` | `['patient:read','patient:write','device:assign','care_plan:read','care_plan:write','monitoring:read']` |
| `status`/`validFrom`/`validTo` | `'active'` / now / null |

若 `patient.mirrorId`(已配对旧镜子):写 v1 `device_assignments`(tenantId+deviceId=mirrorId+patientId, status active)+ 视需要在 `devices` 补一条。**注意**:旧 `deviceAuthToken` 是 legacy relay 用的,不是 v1 device 凭证;旧镜子若要用 v1 需重新 bootstrap 配对(生产上镜子已切 v1)。

### 2.3 其它集合
- `Conversation` + `ConversationIdToPatientIdMap` → 保留读(见 §3 会话适配)或迁 `sessions`;**建议先只读兼容,不迁历史对话**(数据量大、结构差异大)。
- `DailyPatientStatus` → `daily_statuses`(字段基本一致:patientId/date/status/missed/…)。
- `MirrorPairingSessions` → 不迁(legacy 配对已废,镜子走 v1 `device_pairings`)。

---

## 3. Legacy 端点 → v1 对应表(适配层)

所有 legacy 路由**保持请求/响应形状**,内部改成操作 v1 集合。鉴权:legacy 无 token,适配层用请求里的 `nurseId`(= v1 userId)+ `patientId`(= v1 patientId)直接定位 v1 记录(信任模型不变;真正收紧留待 app 迁 v1 token)。

| Legacy 端点(app 调用) | 现在(legacy 实现) | 改为(v1 backing) |
|---|---|---|
| `POST /auth/sign-in` `{email,password}`→`{nurseId,name,email}` | 查 `NursePatientConfig` | 查 v1 `users`(email),`verifyPassword`;返回 `nurseId=user._id` |
| `POST /nurse-patient-config/create` | 建 `NursePatientConfig` + 内嵌 patients | 建 v1 `tenants`+`users`+`patients`+`care_relationships`(id 用新 ObjectId hex,保持返回形状 `{nurseId,patientCount,…}`) |
| `PATCH /nurse-patient-config/add-patients` | push patients[] | 建 v1 `patients`+`care_relationships`(同租户) |
| `GET /nurse-patient-config/latest?nurseId=` | 读 config + 拼 dashboard | ✅**混合已实现**:护士/患者档案仍读 legacy `NursePatientConfig`(档案未全迁 v1),但每个患者的 `lastSpokenAt`/`duration`/`status` 改由最近一条 v1 `sessions` 派生(`getLatestV1SessionByPatientIds`)。lastSpoken=`localCompletedAt`??`createdAt`;duration=`durationMs/1000`;状态仍用 `getMissedDays` 算 |
| `PATCH /nurse-patient-config/notifications` | 改 config 通知字段 | 改 v1 user 通知偏好 |
| `GET /nurse-patient-config/mirrors?nurseId=` | 读 patients[] 的 mirror 字段 | 读 v1 `device_assignments`+`devices`,拼成 `{patients[]{mirrorId,mirrorName,mirrorVerified,mirrorPairingStatus,mirrorPairingCode,mirrorPairedAt,deviceAuthTokenPresent,timezone}}` |
| `PATCH /nurse-patient-config/mirrors` `{action:'unlink'}` | 清 mirror 字段 + 删 map | v1 `device_assignments` + `device_credentials` 置 revoked |
| **`POST /nurse-patient-config/mirrors/connect`** `{nurseId,patientId,pairingCode,mirrorName,timezone}` | 查 `MirrorPairingSessions` | **认领 v1`device_pairings`**:`codeHash=hmac(code)` 查 pending→建 `device_assignments`(tenant+device+patientId)+写 exchange ticket(hash/digest/cipher,5min)+ pairing→paired。镜子随后 `device-credentials/exchange` 兑换。**这是解你 484476 失败的核心改动** |
| `GET /conversation-session?id=` / `-counts` / `-by-day` | 读 `Conversation`+`Map` | ✅**已实现**(`lib/v1Conversations.ts`):读 v1 `sessions`+`transcript_turns`,`serializeV1Session` 拼旧形状(duration 秒/words/exchanges/avgLatency=0/logs)。按天/月分桶仍用 `Asia/Singapore`(与 app 一致) |
| `POST /patient-summary` `{patientId,date?}` | OpenAI 读 logs | ✅**已实现**:读该日 v1 session 的 `transcript_turns`(按 session.createdAt→sequence 排序),role 标注 Aria/Patient,喂原 OpenAI 逻辑生成 `{summary}` |
| `GET /patient-trend?id=&days=` / `POST /patient-trend/daily-status` | 读 legacy `DailyPatientStatus` | ✅**已实现(改为实时派生)**:**v1 无 `daily_statuses` 集合**,改为每天实时从 v1 `sessions` 算(`getV1DailyStats`):当天有完成会话→`green`,否则`red`+`missed`;duration 取当天最近完成会话。`daily-status` POST 改为 recompute-and-return(不落库) |
| `POST /api/qwen-token` | 旧镜子取 token | 废弃(镜子走 v1 `sessions/:id/realtime-tickets`) |

对应的 v1 端点(供参考,app 不直接调):`/api/v1/auth/sessions`、`/api/v1/patients`、`/api/v1/device-pairings|device-pairing-claims|device-credentials/exchange`、`/api/v1/devices/:id/*`、`/api/v1/sessions/*`。

---

## 4. 配对桥接细节(§3 connect 的实现要点,#1)

镜子(bootstrap token）已 `POST /api/v1/device-pairings` 生成码,存 `device_pairings`:`{pairingId, deviceId, codeHash=hmac(code,PAIRING_PEPPER), codeHint, state:'pending', failedAttempts, expiresAt(10min)}`。

legacy `mirrors/connect` 认领时(等价于 v1 `POST /device-pairing-claims`,但用 legacy 身份):
1. `code = normalize(pairingCode)`;`codeHash = hmac(code, PAIRING_PEPPER)`。
2. 查 `device_pairings {codeHash, state:'pending', expiresAt>now, failedAttempts<5}`;查不到 → 记 `failedAttempts++` + 400 `PAIRING_CODE_INVALID`(app 显示"not valid or has expired")。
3. 事务:标记该 device / patient 现有 active `device_assignments` 为 `replaced`;插入新 assignment `{_id:asg_…, tenantId(=ten_<nurseHex>), deviceId(=pairing.deviceId), patientId(=v1 patientId), status:'active', assignedAt}`。
4. CAS 把 pairing 翻 `paired` + 写三份 exchange ticket:`exchangeTicketHash=hashSecret(t)`、`exchangeTicketDigest=sha256(t)`、`exchangeTicketCipher=sealSecret(t)`、`exchangeTicketExpiresAt=now+5min`、`exchangeConsumedAt:null`、`tenantId`、`patientId`、`patientDisplayName`。
5. 同步更新 legacy `NursePatientConfig.patients.$`(mirrorId/mirrorPairingStatus='paired'/…)让 app 的镜子列表/仪表盘仍显示已配对(过渡期双写)。
6. 返回旧形状 `{success, patientId, mirrorId, mirrorName, mirrorPairingStatus:'paired', mirrorPairedAt}`。

镜子 poll `GET /api/v1/device-pairings/:id` → 见 paired + 拿到明文 ticket → `POST /device-credentials/exchange` → 得 device access(15min)+refresh(90d)。

> 依赖:`PAIRING_PEPPER`、`JWT_SECRET`、`CREDENTIAL_ENCRYPTION_KEY` 三个 ≥32 字符的服务端密钥必须配置(v1 `requireServerSecret` 强制)。

---

## 5. 鉴权模型桥接

- app 现状:sign-in 后只存 `{nurseId,name,email}`,**不带 token**;后续 legacy 请求把 nurseId/patientId 放 body/query。
- 适配层:保持"信任 nurseId"模型,用 nurseId 定位 v1 user、patientId 定位 v1 patient。**不引入 token(app 不改)**。
- 安全说明(交接给同事):这是**过渡期**兼容,信任模型偏弱;正式收紧需 app 迁到 v1 JWT(`/api/v1/auth/sessions` + Authorization header),届时可关掉 legacy 适配(`ENABLE_LEGACY_API=false`)。legacy 路由 Deprecation/Sunset:2026-12-31。

---

## 6. 迁移 & 上线步骤(runbook)

1. 配好 3 个密钥(`JWT_SECRET`/`PAIRING_PEPPER`/`CREDENTIAL_ENCRYPTION_KEY`),确保 `MONGODB_DB=reflexion_production`。
2. `npm run build && npm run migrate:legacy-v1`(幂等;打印每护士迁移计数)。
3. 灰度:先只改 `mirrors/connect`(#1,解配对)→ 验证镜子端 v1 全链路。
4. 再逐个改只读端点(latest/mirrors/conversation-*/patient-*)→ 双跑校验(v1 结果 vs legacy 结果一致)。
5. 改写端点(create/add-patients/notifications/sign-in)→ 全量切 v1。
6. 过渡期 connect **双写** legacy `NursePatientConfig`(让未迁的旧读路径不崩),全部切完后移除双写。
7. 回滚:适配层按路由开关(env)可逐个回退到 legacy 实现;数据迁移只增不删,可重跑。

---

## 7. 待确认 / 风险(给同事)

### 7.0 当前实现状态(2026-07)

已切 v1 的两簇(镜子端已用 v1,这些是"让 caregiver 看得到镜子数据"的必要改动):

- **#1 配对**:`mirrors/connect` 认领 v1 `device_pairings`(见 §4)。已线上验证:配对+对话打通。
- **#2 会话簇(conversation cluster)**:`conversation-session`/`-counts`/`-by-day`、`patient-summary`、`patient-trend`(GET+`daily-status`)、以及 `latest` 的会话派生字段——全部改读 v1 `sessions`/`transcript_turns`(`src/lib/v1Conversations.ts`)。**这是本次唯一"镜子数据在旧模型里读不到"的坏点,已修**。

**仍留在 legacy(工作正常,故意不动)= 身份/设备簇**:`sign-in`、`create`、`add-patients`、`notifications`、`mirrors`(GET/unlink)、以及 `latest` 的档案字段,仍读写 `NursePatientConfig`。原因:①它们在 legacy 数据上正常工作;②`sign-in`↔`create` 强耦合(要迁必须一起迁,否则新注册账号在 v1 `users` 里不存在 → 登录失败)。迁移这簇是**后续独立阶段**,需一次性:`create`/`sign-in`/`password-reset` 全部对齐 v1 `users`(注意 v1 sign-in 查 `emailNormalized`,迁移脚本必须写 `emailNormalized=email.toLowerCase()`,否则密码正确也 401)。

> 注:`#4` 的 `password-reset.ts` 写的是 v1 `users.passwordHash`,而当前 `sign-in` 读 legacy——因忘记密码邮件目前 dormant(未配 Postmark)故无影响;身份簇迁 v1 时二者自然对齐。

### 7.1 会话簇的策略取舍(实现时锁定的口径,可按 UI 复核)

- **哪些算"一次对话"**:`session.type ∈ {daily_checkin, companion}`,排除 `device_test`(硬件自检)。**companion 必须计入**——镜子待机点击进入的是 companion 会话。
- **"完成"判定**:`localCompletedAt` 存在(镜子调过 `.../complete`)或 state ∈ {ingesting,processing,completed,excluded,review_pending}。**不能只认 state==='completed'**:刚结束的 check-in 停在 `ingesting`,要等异步 pipeline(本部署可能没跑)才升 `completed`。
- **duration 单位=秒**(app 用 `${m}m ${s}s` 格式)← `acquisition.durationMs/1000`。
- **avgLatency 无 v1 来源**:镜子给每个 turn 打同一个时间戳、不记延迟,故恒为 `0`;逐句 `log.duration` 用总时长平均分摊(仅为让回放时间轴推进,非真实每句耗时)。
- **words**:按语言分词——CJK(普通话/粤语/日/韩)按字符数计,拉丁按空白切分(`countWords`)。
- **trend 颜色**:v1 无 green/red/missed 三态字段,按 legacy `statusEngine` 口径映射(当天有完成会话=green,否则 red+missed);v1 的 `monitoringWindows.status`/`anomalyScores` 等更细的分级本次未用,后续要做红/黄告警可接。

### 7.2 其它

- **历史 `Conversation` 是否迁 `sessions`**:建议先只读兼容,不迁历史(结构差异大)。
- **一租户/护士 vs 共享租户**:本文档按"每护士一租户";若要机构多护士共患者,需改 care_relationships 建模。
- **`ageBand` 归桶规则**待定(旧是精确 age)。
- **旧已配对镜子**:legacy `deviceAuthToken`≠v1 device 凭证;旧镜子接 v1 需重新 bootstrap。生产镜子已切 v1,影响面小。
- **passwordHash 兼容**:legacy 与 v1 都用 `pbkdf2_sha256$…`(见 `lib/password.ts` vs `v1/platform/crypto.ts` 的 verify),迁移**免重置密码**——上线前用一个测试账号验证 sign-in 通过。

---

## 8. 相关文件索引(给同事)

- 迁移脚本:`src/scripts/migrateLegacyToV1.ts`(`npm run migrate:legacy-v1`,幂等)
- **会话簇 v1 读适配(#2)**:`src/lib/v1Conversations.ts`(sessions/transcript_turns → legacy 形状 + 分词 + 状态映射)
- 配对桥接(#1):`src/lib/legacyV1Bridge.ts`(`claimV1Pairing` 等)
- Legacy 路由:`src/routes/**`(会话簇已切 v1;身份/设备簇仍 legacy)
- v1 数据模型:`src/v1/platform/collections.ts`、`src/v1/routes/{identity,patients,devices,sessions}.ts`
- 配对认领参考实现:`src/v1/routes/devices.ts`(`device-pairing-claims`、`createCredentialFromExchange`)
- 加解密/哈希:`src/v1/platform/crypto.ts`(`hmac`/`hashSecret`/`sealSecret`)、`src/v1/platform/tokens.ts`
- 镜子端配对客户端:`mirror-app/src/orchestration/deviceBootstrap.ts`、`src/storage/deviceCredentials.ts`、`app/index.tsx`、`app/test-device.tsx`
