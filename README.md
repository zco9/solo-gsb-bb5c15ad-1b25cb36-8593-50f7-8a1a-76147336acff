# 饮用水混配优化 API（Drinking Water Blending API）

基于 **Python + FastAPI + SciPy + SQLite** 的饮用水混配优化服务。用户提交目标体积、
钙 / 镁 / 碱度区间以及 2～6 种水源的浓度、可用量与单位成本，服务求解满足全部区间
与供给上限的**最低成本配比**；成本并列时取**按水源 ID 排序后用量向量字典序最小**
的方案；问题不可行时返回**各指标的可达范围**。支持以**幂等键**保存方案并按 ID 查询。
对于已保存且状态为 `optimal` 的方案，还可通过**供给缩减预案接口**在不改动原方案的
前提下，按"先最小化总调整量、再最小化成本、并列取用量向量字典序最小"重新配水。

## 快速开始（单条命令：安装 + 建库 + 启动）

```bash
docker compose up --build
```

该命令完成依赖安装、SQLite 建库（服务启动时自动执行 `CREATE TABLE IF NOT EXISTS`）
并启动服务。

- 服务地址：**http://localhost:8000**
- 交互式文档（Swagger UI）：**http://localhost:8000/docs**
- 健康检查：**http://localhost:8000/health**

所有接口均挂载在 `http://localhost:8000/api/v1/...` 下（容器内监听 `0.0.0.0:8000`，
Compose 将宿主机 **8000** 端口映射到容器 **8000** 端口）。

不使用 Docker 时，本地等价的一条命令（Python ≥ 3.11）：

```bash
pip install -r requirements.txt && python -c "from app.db import init_db; init_db()" && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 配置

均通过环境变量配置，默认值如下：

| 变量      | 默认值              | 说明                                   |
| --------- | ------------------- | -------------------------------------- |
| `DB_PATH` | `./data/blends.db`（容器内 `/data/blends.db`） | SQLite 数据库文件路径 |
| `HOST`    | `0.0.0.0`           | 监听地址（`python -m app.main` 时生效） |
| `PORT`    | `8000`              | 监听端口                               |

Compose 默认将宿主机 `8000` 映射到容器 `8000`，并通过命名卷 `blends-data` 持久化数据库。

## 单位约定

- 体积：m³（`target_volume`、`max_volume`、响应中的 `volume`）
- 浓度：mg/L（钙、镁；碱度按 mg/L CaCO₃ 计）
- 成本：任意货币单位 / m³

## API

### `POST /api/v1/blends/solve` — 只求解，不落库

```bash
curl -X POST http://localhost:8000/api/v1/blends/solve \
  -H 'Content-Type: application/json' \
  -d '{
    "target_volume": 1000,
    "calcium":    {"min": 40, "max": 60},
    "magnesium":  {"min": 5,  "max": 20},
    "alkalinity": {"min": 100, "max": 200},
    "sources": [
      {"id": "W1", "max_volume": 600, "cost": 1.0, "calcium": 100, "magnesium": 30, "alkalinity": 250},
      {"id": "W2", "max_volume": 800, "cost": 2.0, "calcium": 20,  "magnesium": 5,  "alkalinity": 80}
    ]
  }'
```

最优响应（`200`）：

```json
{
  "status": "optimal",
  "plan_id": null,
  "total_volume": 1000.0,
  "total_cost": 1500.0,
  "blend": [
    {"source_id": "W1", "volume": 500.0, "cost": 500.0},
    {"source_id": "W2", "volume": 500.0, "cost": 1000.0}
  ],
  "indicators": {"calcium": 60.0, "magnesium": 17.5, "alkalinity": 165.0}
}
```

### `POST /api/v1/blends` — 求解并以幂等键保存

幂等键可放在请求体 `idempotency_key` 字段或 `Idempotency-Key` 请求头中（二者至少其一，
body 优先）：

```bash
curl -X POST http://localhost:8000/api/v1/blends \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: order-2026-0001' \
  -d '<同上请求体>'
```

- 首次创建返回 `201`，响应体含 `plan_id` 与 `created_at`；
- 以**相同键 + 相同请求**重发：返回 `200` 与已保存的方案，并带响应头
  `X-Idempotent-Replay: true`（不会重复求解或新建记录）；
- 以**相同键 + 不同请求**重发：返回 `409 Conflict`；
- 缺少幂等键：返回 `422`。

### `GET /api/v1/blends/{plan_id}` — 按 ID 查询已保存方案

```bash
curl http://localhost:8000/api/v1/blends/1
```

不存在时返回 `404`。

### `POST /api/v1/blends/{plan_id}/contingency` — 供给缩减预案

针对**已保存且状态为 `optimal`** 的方案，给定部分水源的**新供给上限**，重新求解
一份缩减预案。请求体是"已有水源 ID → 新供给上限"的映射（至少一项，
`0 ≤ 新上限 ≤ 原上限`）；**未列出的水源沿用原上限**。服务读取保存时的原始请求
与配比，保持目标体积、指标区间、水质（各水源浓度）和单位成本不变，按以下优先级
优化：

1. 先最小化各水源用量变化绝对值之和 `Σ|x_i − x_i⁰|`（**总调整量**）；
2. 在总调整量最小的配比中，再最小化**总成本**；
3. 仍并列时，按水源 ID 升序的用量向量取**字典序最小**。

预案**不会被保存，也不会修改原方案**；接口可对同一方案反复调用。

```bash
# 假设方案 1 的配比为 W1=500、W2=500（W1 原上限 600、W2 原上限 800），
# 将 W1 的供给上限缩减到 400：
curl -X POST http://localhost:8000/api/v1/blends/1/contingency \
  -H 'Content-Type: application/json' \
  -d '{"W1": 400}'
```

可行时返回 `200`，含新配比、各水源用量差、总调整量及新旧成本差：

```json
{
  "status": "optimal",
  "plan_id": 1,
  "total_volume": 1000.0,
  "total_cost": 1600.0,
  "cost_delta": 100.0,
  "total_adjustment": 200.0,
  "blend": [
    {"source_id": "W1", "original_volume": 500.0, "new_volume": 400.0,
     "volume_delta": -100.0, "new_cap": 400.0, "cost": 400.0},
    {"source_id": "W2", "original_volume": 500.0, "new_volume": 600.0,
     "volume_delta": 100.0, "new_cap": 800.0, "cost": 1200.0}
  ],
  "indicators": {"calcium": 52.0, "magnesium": 15.0, "alkalinity": 148.0}
}
```

- 新上限不构成约束时（例如等于或高于当前用量），原配比不变，
  `total_adjustment` 与 `cost_delta` 均为 `0.0`；
- 新上限为 `0` 合法，表示完全停用该水源。

新上限下无可行配比时返回 `200` 与不可行原因（不保存任何内容，也不改原方案）：

```json
{
  "status": "infeasible",
  "plan_id": 1,
  "message": "total available volume 900 is below the target 1000; ...",
  "achievable_ranges": {
    "volume": {"requested": 1000.0, "max": 900.0},
    "calcium": {"min": 20.0, "max": 20.0, "...": "..."}
  }
}
```

错误状态码：

| 情况                                       | 状态码 |
| ------------------------------------------ | ------ |
| 方案 ID 不存在                             | `404`  |
| 原方案状态不是 `optimal`（如 `infeasible`）| `422`  |
| 请求体为空映射、含未知水源 ID              | `422`  |
| 新上限为负 / 非有限值，或**超过原上限**    | `422`  |
| 请求体不是 JSON 对象                       | `422`  |

### 不可行响应

问题不可行时（求解与保存接口均）返回 `200` 及结构化说明，例如钙下限 70 超过可达上限：

```json
{
  "status": "infeasible",
  "message": "calcium: requested minimum 70 exceeds the achievable maximum 68",
  "achievable_ranges": {
    "calcium":    {"min": 36.0, "max": 68.0, "requested_min": 70.0, "requested_max": 90.0},
    "magnesium":  {"min": 10.0, "max": 20.0, "requested_min": 5.0,  "requested_max": 20.0},
    "alkalinity": {"min": 114.0, "max": 182.0, "requested_min": 100.0, "requested_max": 200.0}
  }
}
```

可达范围指：在供给上限与目标体积约束下（若总供给不足目标体积，则改用全部可用水量），
各指标在混配水中可实现的浓度区间（对每个指标分别求解 min/max 两个 LP 得到的投影范围）。
总可用水量不足目标体积时，额外返回 `achievable_ranges.volume.max`。不可行方案同样会被保存，
可按 ID 查询。

## 规则与语义

- **优化模型**：线性规划（SciPy `linprog`，HiGHS 求解器）。变量为各水源取水量
  `x_i`；目标 `min Σ cost_i·x_i`；约束 `Σx_i = 目标体积`、
  `lo_j·V ≤ Σ conc_ij·x_i ≤ hi_j·V`（钙/镁/碱度）、`0 ≤ x_i ≤ max_volume_i`。
- **成本并列**：在所有最优解中，按水源 ID 升序排列的用量向量取**字典序最小**者
  （依次最小化 x₁, x₂, …；通过一系列 LP 精确实现）。水源 ID 为字符串，按字典序排序。
- **区间缺省**：`calcium` / `magnesium` / `alkalinity` 任一区间可省略，表示该指标不设限。
- **校验**：水源数量 2～6、ID 唯一、`min ≤ max`、数值非负（目标体积为正）；不满足返回 `422`。
- **幂等**：以幂等键唯一索引存储；请求体（不含键本身）做 SHA-256 指纹比对，防止键被
  不同请求复用。
- **供给缩减预案**：对已保存的 `optimal` 方案，用 L1 线性化（引入偏差变量
  `d_i ≥ |x_i − x_i⁰|`，最小化 `Σd_i`）先求最小总调整量；再将其作为等式约束
  固定，依次求解成本最小化与字典序并列（共三级 LP，数值不稳时以紧不等式回退）。
  预案仅根据保存的原始请求与配比即时计算，**不落库、不改写原方案**；新上限下不可行
  时同样返回各指标可达范围与原因。未知方案 `404`，非 `optimal` 原方案或非法上限 `422`。

## 测试

```bash
pip install -r requirements-dev.txt && pytest
```

共 35 个用例，覆盖求解器（最优性、字典序并列、不可行范围、预案的最小调整 /
成本次级目标 / 字典序并列 / 不可行）与 API（幂等回放 / 冲突、按 ID 查询、
预案可行 / 不可行 / 校验错误、参数校验等）。

## 项目结构

```
app/
  main.py     # FastAPI 路由与生命周期（启动时自动建库）
  solver.py   # SciPy 线性规划：最低成本 + 字典序并列 + 供给缩减预案 + 不可行可达范围
  models.py   # Pydantic 请求/响应模型与校验
  db.py       # SQLite 持久化（幂等键唯一约束、原子写入）
  config.py   # 环境变量配置（DB_PATH / HOST / PORT）
tests/        # pytest 单元与端到端测试
Dockerfile    # python:3.12-slim 镜像，非 root 运行，含健康检查
docker-compose.yml
```
