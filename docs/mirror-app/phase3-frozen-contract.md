# Phase 3 最小统一后端契约（Frozen v1）

日期：2026-07-22
事实源：`docs/architecture/reflexion-api-v1.openapi.yaml`
运行时：`caregiver-server/src/v1`

Mirror 和 Caregiver 从本阶段起只依赖同一 API origin 的 `/api/v1`；不再新增 Expo API route，也不在客户端保存 Qwen、数据库、对象存储或搜索服务的账户密钥。

| 能力 | 冻结路由 | 身份 | 成功语义 |
| --- | --- | --- | --- |
| Device authentication | credential exchange / rotation | bootstrap → device JWT | 短期 access + 可轮换 refresh |
| Pairing v2 | `POST /device-pairings`、claim、status、exchange | bootstrap / human JWT | 原子 assignment；一次性交换票据 |
| Session create | `POST /sessions` | device JWT | `201` + `sessionId/stateVersion` |
| Realtime ticket | `POST /sessions/{id}/realtime-tickets` | device JWT | 短期、session-scoped Qwen ticket |
| Event batches | `POST /sessions/{id}/event-batches` | device JWT | 有序、幂等、可报告 duplicate/gap |
| Artifact upload | upload plan → direct PUT → commit | device JWT / signed URL | SHA-256、size、server verification |
| Session complete | `POST /sessions/{id}/complete` | device JWT + `If-Match` | `202` + `operationId` |
| Processing status | `GET /sessions/{id}/processing-status` | device/human JWT | `accepted/queued/processing/completed/failed` |
| Heartbeat | `POST /devices/{id}/heartbeats` | device JWT | 60 秒、AppState 恢复立即上报 |

## 不变量

- 所有成功响应使用 `{ data, meta: { requestId } }`，错误使用稳定 `code/message/retryable/details`。
- command/create 请求使用 `Idempotency-Key`；session 完成使用 `If-Match`。
- Device 只能访问 active assignment 的 patient；human 还要通过 active `care_relationship` scope。
- artifact 二进制不经过 API 进程；Mirror 只接受服务端签发的 HTTPS 上传地址。
- 本地失败会进入 Mirror durable outbox；后端 worker 使用 outbox lease、retry 和 dead-letter。
- processing status 不返回原始向量、内部异常分数或未批准诊断标签。
- review-case 通知按 `tenant + recipient + source case` 去重，并只物化给有 `monitoring:read` 关系的 caregiver。

## 异步状态机

`created → active → ingesting → processing → completed | excluded | review_pending`

处理错误进入 `processing_failed`，对外映射为 `failed + retryable`；worker 重试时回到 `processing`。Mirror 的 Session complete 页面只承诺“已同步”或“已安全排队”，不假装分析已经完成。

## 自动验收

`caregiver-server/src/v1/integration/phase3.api.integration.test.ts` 使用临时 MongoDB replica set 运行完整 HTTP/事务/worker 链路。发布前必须通过：

```bash
npm run typecheck
npm test
npm run test:coverage
npm run test:coverage:phase3
npm run test:coverage:api
npm run build
```

Phase 3 核心路由、平台边界以及完整 `/api/v1` route layer 的聚合行覆盖率和函数覆盖率均不得低于 90%。对象存储、Qwen、天气与搜索在测试中只模拟外部 provider 边界，签名、密钥封装、数据库状态机、事务、授权、工具审计和 worker 使用生产代码执行。
