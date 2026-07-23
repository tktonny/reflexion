# Mirror App 产品与技术基线

> 审查日期：2026-07-22
> 范围：`mirror-app`、`caregiver-app`、`reflexion-server`、`platform_April` 及相关产品、临床、合规文档
> 性质：代码现状审计 + 目标产品定义，不代表临床验证已经完成

## 文档导航

- [产品需求文档](./product-requirements.md)：产品定位、用户旅程、功能需求、验收标准和非功能目标。
- [目标系统架构](./system-architecture.md)：当前拓扑、目标服务边界、数据流、核心契约、安全与部署设计。
- [现状与缺口分析](./current-state-and-gap-analysis.md)：逐项对照代码，列明已完成、部分完成、缺失及风险级别。
- [开发路线图](./development-roadmap.md)：P0/P1/P2 优先级、12 周实施计划、后续临床与规模化阶段。
- [Realtime Turn-taking Contract](./turn-taking-contract.md)：Phase 0 音频状态机、不变量、参数基线和设备验收门槛。
- [Turn-taking 真机验收手册](./turn-taking-device-runbook.md)：APK、日志、30 回合 smoke、手动结束矩阵和 soak 步骤。
- [Phase 1 MuMu 验收报告](./phase1-mumu-acceptance.md)：30/100 回合、30 分钟 soak、结束矩阵、语言切换和已知硬件边界的实测记录。
- [Platform v2 API 与领域架构](../architecture/platform-v2-api-and-domain-architecture.md)：统一 API 域名、身份、配对、Session 状态机、异步事件和迁移边界。
- [Platform v2 数据库设计](../architecture/platform-v2-database-design.md)：MongoDB collection、索引、对象存储、向量版本和旧数据映射。
- [纵向监控与向量异常检测](../architecture/longitudinal-vector-anomaly-design.md)：个人基线、质量/身份门控、多分量异常、持续性和 reviewer 闭环。
- [OpenAPI v1 草案](../architecture/reflexion-api-v1.openapi.yaml)：Mirror、Caregiver、Provider 共用的关键接口契约。
- [Google 需求源对齐说明](./requirements-source-alignment.md)：将 `Updated metrics`、Signal-to-Status 和 Mirror APK 需求映射到 v2 API、数据库和两级 baseline。

## 一页结论

当前仓库已经不是从零开始的原型。它具备设备配对、Qwen 实时语音、多种对话传输、端侧唤醒词、摄像头抽帧、会话后 LLM 评估、MongoDB 保存、离线待上传队列，以及 caregiver 端的会话与完成率视图。

但它还不是可安全试点的完整产品，主要原因不是 UI，而是闭环尚未形成：

1. 日常助手目前没有真实天气、用药、日程或提醒工具，prompt 明确承认没有实时天气数据。
2. 单次对话的 LLM 输出被直接标为 `healthy / needs_observation / dementia`，但没有同意状态、质量门控、结构校验、模型版本、纵向基线或人工复核。
3. 评估数据虽然写入 MongoDB，caregiver 趋势却只看“当天是否完成对话”，没有消费认知变化数据。
4. caregiver 登录没有真正的会话令牌；多数 API 直接信任请求中的 `nurseId` 或 `patientId`。设备鉴权默认也是关闭的。
5. `mirror-app`、`reflexion-server` 和 `platform_April` 是三套未统一的数据与服务路径。Python 平台已有更完整的身份、质量和纵向能力，但没有接入镜面端的 MongoDB 会话链路。
6. Caregiver Express API 已部署到 Vercel，但 mirror 的生产 API host、自动同步、推送通知、目标唤醒词、全链路测试、审计和数据治理仍未闭环。

因此建议将产品措辞从“每日诊断”调整为“每日认知状态采集与纵向变化筛查”。单日结果只能作为质量合格的观察记录；达到基线和持续变化规则后，才生成供专业人员复核的风险提醒。

## 推荐的产品边界

- `Companion`：日常问答、天气、提醒、经 caregiver 配置的用药计划、日程和陪伴。
- `Daily Check-in`：2–5 分钟、协议化但自然的认知状态采集，不在镜面端显示诊断结论。
- `Longitudinal Monitoring`：以个人基线、质量覆盖和持续变化为核心，不用单次 LLM 分类替代临床诊断。
- `Caregiver`：展示依从性、简化摘要、待处理事项和经规则批准的提醒，不默认显示原始研究风险分数。
- `Provider / Reviewer`：查看完整评估、质量、模型版本、证据和复核状态；这是当前生态仍缺少的生产控制面。

## 代码核验结果

- `mirror-app` TypeScript 静态检查：通过。
- `mirror-app` 硬件决策表 smoke test：7/7 通过。
- `reflexion-server` 部署已确认：`https://reflexion-caregiver-app-server.vercel.app/health` 于 2026-07-22 返回 HTTP 200 与 `{"ok":true}`；caregiver App 已通过 `EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL` 读取该地址。
- 该部署目前只覆盖 `reflexion-server` 路由；mirror 所需的配对申请、设备状态、Qwen token、会话写入和评估路由仍在 `mirror-app/app/api/*`，不能直接把 mirror 的 `EXPO_PUBLIC_API_BASE` 指向 caregiver URL。
- `reflexion-server` 静态检查：未执行成功，当前工作区未安装其 TypeScript 依赖，`tsc` 不存在。
- `mirror-app` 已加入 Phase 1 turn-taking 确定性测试与 provider smoke；仓库仍缺少覆盖 Mirror/Caregiver/Server 的统一 CI 和端到端集成测试。
- 审查时已有 3 个用户修改文件，本文档没有覆盖它们：
  - `mirror-app/app/realtime-test.tsx`
  - `mirror-app/src/hooks/useConversation.ts`
  - `mirror-app/src/hooks/useDirectRealtimeConversation.ts`
