---
category: 技术文档
title: 接口与 API 接入说明
---

# 接口与 API 接入说明

## 概述

云策 OA 开放平台提供 RESTful API，用于组织架构同步、审批单读写、考勤数据查询与员工入离职事件订阅。企业版客户可申请开放权限，标准版仅开放只读的组织与人员接口。

接口根地址为 `https://<tenant>.yunce-oa.com/api/v1`，私有化环境为 `https://<host>:8443/api/v1`。所有接口使用 JSON 传输，字符集 UTF-8。

## 认证与鉴权

采用 OAuth 2.0 客户端凭证模式获取 Token：

1. 客户在管理后台创建应用，获得 client_id 与 client_secret；
2. 调用 `/oauth/token` 换取 access_token；
3. 后续请求在 Header 中携带 `Authorization: Bearer <token>`。

access_token 有效期 24 小时，refresh_token 有效期 7 天。Token 过期返回 401，需重新获取。client_secret 只显示一次，泄露后须立即重置。

## 限流与配额

| 版本 | 限流 | 单日配额 |
| --- | --- | --- |
| 标准版 | 60 次/分钟 | 5 万次 |
| 企业版 | 600 次/分钟 | 100 万次 |

超出限流返回 429，响应头包含 `Retry-After` 建议等待秒数。建议客户端做指数退避重试，初始间隔 1 秒，最多重试 3 次。

## 核心接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | /employees | 分页查询员工 |
| POST | /employees | 创建员工（触发入职流程） |
| PATCH | /employees/{id} | 更新员工信息 |
| GET | /departments | 查询部门树 |
| POST | /approvals | 发起审批单 |
| GET | /approvals/{id} | 查询审批详情与流转记录 |
| GET | /attendance/records | 查询考勤记录 |
| GET | /knowledge/search | 知识库检索 |

分页参数为 `page` 与 `page_size`，page_size 最大 100，默认 20。列表接口返回 `total`、`page`、`page_size` 与 `items`。

## 请求与响应约定

- 请求体统一为 JSON，时间字段使用 ISO 8601 格式，例如 `2026-09-15T14:00:00+08:00`；
- 金额字段以「分」为单位的整数传输，避免浮点误差；
- 写接口支持幂等键：请求头传 `Idempotency-Key`，24 小时内相同键的重复请求返回首次结果；
- 错误响应结构为 `{"code": "...", "message": "...", "request_id": "..."}`。

常见错误码：

| HTTP | code | 含义 |
| --- | --- | --- |
| 400 | INVALID_PARAM | 参数校验失败 |
| 401 | TOKEN_EXPIRED | Token 过期或无效 |
| 403 | NO_PERMISSION | 无接口权限或数据范围受限 |
| 404 | NOT_FOUND | 资源不存在 |
| 429 | RATE_LIMITED | 触发限流 |
| 500 | INTERNAL_ERROR | 服务内部错误，附 request_id |

排查问题时请提供 request_id，技术支持可通过它定位完整调用链。

## Webhook 事件订阅

支持订阅三类事件：员工入职（employee.created）、员工离职（employee.terminated）、审批状态变更（approval.status_changed）。

推送使用 HTTP POST，超时 5 秒；失败按 1 秒、10 秒、60 秒、10 分钟、1 小时重试 5 次，仍失败则进入死信队列。接收方须在 5 秒内返回 2xx，并保证幂等处理。

签名校验：请求头 `X-YC-Signature` 为 `HMAC-SHA256(secret, timestamp + body)`，接收方须校验时间戳在 5 分钟内。

## 接入与上线流程

1. 在管理后台创建应用并申请权限范围；
2. 在测试租户完成联调，联调通过后提交上线申请；
3. 安全团队评估数据范围与传输方式，2 个工作日内反馈；
4. 生产环境配置 IP 白名单，仅允许固定出口 IP 调用；
5. 上线后 7 天为观察期，每周同步调用量与错误率。

## 版本演进与兼容策略

开放平台接口按版本路径隔离，`/api/v1` 内的变更保持向下兼容：只新增字段，不删除字段，不改变既有字段语义。字段废弃时先标记为 deprecated 并在响应中保留至少 6 个月，期间通过发布说明通知客户，之后才从文档中移除。

新增不兼容的接口时另起 `/api/v2`，v1 与 v2 并行维护不少于 12 个月，便于客户平滑迁移。客户可以通过 `GET /api/v1/meta` 查询当前版本号、支持周期与已废弃字段列表。

## 最佳实践

- 增量同步使用 `updated_at` 游标，避免全量拉取；
- 批量写入单次不超过 100 条，失败项单独重试；
- 不要在客户端硬编码 Token，使用配置中心或密钥管理服务；
- 定期轮换 client_secret，建议 90 天一次。
