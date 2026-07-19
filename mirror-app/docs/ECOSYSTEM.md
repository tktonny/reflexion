# Reflexion 生态架构(4 个 repo + REFLEXION)

> 本文件所在的 **reflexion-mobile-mirror-interface-app** 是"镜子设备端 App",是我们当前重点开发的产品。
> 本文描述它与其余 3 个 repo 以及 REFLEXION 研究平台如何协作,共享同一个 MongoDB Atlas 数据库 `ref`。

## 1. 全景图

```
   ┌──────────────────────────┐         6-digit code / QR          ┌──────────────────────────┐
   │  MIRROR device app        │  ──── MirrorPairingSessions ────▶  │  CAREGIVER app            │
   │  reflexion-mobile-mirror  │      (mirror creates session)      │  reflexion-native-app     │
   │  = OUR PRODUCT            │◀── MirrorIdToNurseIdMap (paired) ── │  (nurse enters code)      │
   │  Qwen voice check-in      │                                    └────────────┬─────────────┘
   │  writes Conversation      │                                                 │ nurseId (no JWT)
   └────────────┬─────────────┘                                                 ▼
                │                                          ┌──────────────────────────────────┐
                │  both read/write ────────────────────▶  │  CAREGIVER SERVER (Express)        │
                ▼                                          │  reflexion-caregiver-app-server    │
        ┌───────────────────────────┐                     │  pairing/connect · dashboards ·    │
        │   MongoDB Atlas  DB 'ref'  │◀────────────────────│  OpenAI gpt-4o-mini summaries      │
        │  NursePatientConfig ·      │                     └──────────────────────────────────┘
        │  MirrorPairingSessions ·   │
        │  MirrorIdToNurseIdMap ·    │       REFLEXION (Python clinic platform) = the "AI brain"
        │  Conversation(+IdMap)      │       research/assessment; source of the Qwen realtime impl
        └───────────────────────────┘
```

## 2. 各组件职责

| 组件 | 技术栈 | 角色 | LLM |
|---|---|---|---|
| **reflexion-mobile-mirror-interface-app**(本仓库) | Expo/RN + Node 中继 | **镜子设备端**:显示 6 位配对码、创建 `MirrorPairingSessions`、跑语音 check-in、写 `Conversation` | 语音对话 = **Qwen Omni Realtime**(经我们移植的 Node 中继;见 `QWEN_RELAY.md`) |
| reflexion-native-app | Expo/RN | **照护者手机 App**:登录、引导、仪表盘、镜子管理(输入 6 位码配对)、会话回放/历史/趋势、AI 每日摘要 | 无(摘要由服务端出;内置 chatbot 是 Wizard-of-Oz 假客服) |
| reflexion-caregiver-app-server | Express + MongoDB(TS) | **照护者后端**:完成配对、账号/引导、仪表盘/趋势 API、每日摘要定时任务 | 摘要 = **OpenAI `gpt-4o-mini`**(第 2 个可换 Qwen 的点) |
| REFLEXION | Python FastAPI | **AI 大脑 / 研究平台**:多模态评估、身份/纵向、以及我们移植过来的 Qwen 实时实现的来源 | Qwen / Gemini / OpenAI provider mesh |

要点:
- 无 JWT。照护者身份用 `nurseId` 传递。
- 照护者 App 通过 `EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL` 访问照护者服务端。
- 时区全程 `Asia/Singapore`。

## 3. 共享数据库 `ref`(所有端读写同一库)

| 集合 | 关键字段 | 谁写 / 谁读 |
|---|---|---|
| `NursePatientConfig` | `_id`(=nurseId)、`email`、`passwordHash`(PBKDF2)、通知设置、**`patients[]`**(内嵌);每个 patient:`_id`(**规范患者主键**)、`name`、`age`、`preferredLanguage`、`mirrorId`、`mirrorName`、`mirrorVerified`、`mirrorPairingStatus`、`mirrorPairingCode`、`deviceAuthToken`、`timezone` | 照护者服务端写(create/connect/notifications);镜子端 `verify-mirror`、`status` 读/更新 |
| `MirrorPairingSessions` | `pairingCode`(6 位)、`status`(pending/paired)、`deviceId`(=未来的 mirrorId)、`authToken`、`expiresAt`、`nurseId?`、`patientId?`、`pairedAt?` | **镜子端创建**(pending);照护者服务端校验码后置 paired |
| `MirrorIdToNurseIdMap` | `mirrorId`、`nurseId`、`patientId`、`mirrorName`、`patientName`(mirrorId/patientId 唯一) | 照护者服务端在配对成功时写;镜子端 `device-status` 免码重连时读 |
| `Conversation` | `duration`、`words`、`exchanges`、`avgLatency`、`logs[]`(`sentence/role/words/duration/wordsPerSecond`)、`sessionStatus` | **镜子端写**;照护者端读(会话历史/趋势/摘要) |
| `ConversationIdToPatientIdMap` | `conversationId`、`patientId`、`nurseId?` | 镜子端写;照护者端先查此表再取 `Conversation` |

> 规范患者主键 = `NursePatientConfig.patients[]._id`(ObjectId)。这与我们蓝图里"统一 Mongo 主键"的决策一致。
> 临床类集合(`assessments` / `identity_profiles` / `longitudinal_*`)在当前生态里**尚不存在**,是我们蓝图要新增的。

## 4. 配对握手(跨代码库,分两半)

```
镜子端(本仓库)                          照护者 App + 服务端
────────────────                        ────────────────────
1. POST /api/mirror-pairing/request-code
   → 生成 6 位码 + authToken,
     写 MirrorPairingSessions(status=pending, deviceId)
     屏幕显示 6 位码(+二维码占位,QR 尚未实装)

2. 每 3.5s 轮询
   POST /api/mirror-pairing/status ────────▶  3. 护士在 native-app 输入 6 位码
                                                POST /nurse-patient-config/mirrors/connect
                                                → 校验 pending 码,写:
                                                  · patient.mirrorId = deviceId
                                                  · patient.deviceAuthToken
                                                  · MirrorIdToNurseIdMap(mirror↔nurse↔patient)
                                                  · MirrorPairingSessions.status = paired
4. status 返回 paired + NursePatientConfig
   → 镜子端缓存(AsyncStorage)并进入 check-in

(重连:镜子端 POST /api/mirror-pairing/device-status
  经 MirrorIdToNurseIdMap 免码反查已配对状态)
```

关键:配对码由**镜子端产生**,由**照护者端消费**,两半操作**同一批 `ref` 集合**,schema 必须一致(现已核对一致)。

## 5. LLM 部分(两处,均在向 Qwen 收敛)

1. **镜子语音 check-in**(本仓库,已完成并验证):OpenAI Realtime(WebRTC)→ **Qwen Omni Realtime**,经我们移植的 Node 中继(`server/*.mjs`)。见 `QWEN_RELAY.md`。
2. **照护者每日/患者摘要**(caregiver-app-server):当前 **OpenAI `gpt-4o-mini`**。第 2 个可换 Qwen 的点(REFLEXION 已有 Qwen chat 路径可复用)。

## 6. 重点:完成 MIRROR device app(本仓库)

### 已完成
- ✅ 复用 mobile-mirror 骨架:开机自检、6 位码/二维码配对屏、AsyncStorage 缓存、离线对话队列、设置/管理/test-device。
- ✅ **LLM 层换成 Qwen**:删除 OpenAI WebRTC 层;新增 `server/`(Node 中继:http+ws upgrade、`_relay_live_qwen_session` 1:1 端口、session.update、server-VAD、动态换声 Cherry/Roy/Kiki、wrap-up、401/403 中国备份重试)+ 客户端 `useQwenRealtimeConversation`(Web 音频 16k 采/24k 放,半双工)+ 验证屏 `app/realtime-test.tsx`。
- ✅ 已用 `node server/smoke.mjs` 实测:经中继连通 Qwen,流出音频增量,中国备份重试生效。

### 待完成(建议顺序)
1. **持久化 `Conversation`**:让 Qwen check-in 结束时写 `Conversation` + `ConversationIdToPatientIdMap`(镜子端 `conversations+api.ts` 已指向同集合),这样一次真实签到即可出现在照护者仪表盘。
2. **接入真实配对 → 对话闭环**:配对成功后带真实 `patientId`/`preferredLanguage` 进入 check-in;把 `patient.preferredLanguage` 映射到中继的 `language`。
3. **身份/记忆入提示词**:把 `NursePatientConfig.patients[]`(preferred name)与既往记忆喂给 `buildLiveInstructions`(当前是首次患者)。
4. **原生 Android 音频**:`MirrorAudio` 原生模块(AudioRecord 16k 采 / AudioTrack 24k 放 + AEC/NS),取代 Web-only 音频,走 EAS dev build。
5. **视频 + 认知评估**(蓝图 M3):录像 + 上传 + provider mesh(后续)。

> 完整目标架构与分阶段路线见 `scratchpad/android-mirror-blueprint.md`(蓝图 v2,已对抗验证)。
