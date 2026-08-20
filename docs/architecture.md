# 架构说明

版本范围：Waypoint Policy v1，代码基线
`724ead21be2c27d9b40c200375ee4ab49ccedc84`。批准语义以
[Waypoint Policy v1 合同](conveyorvla_waypoint_policy_contract_v1.md) 为准；本页说明
合同在仓库中的落点。旧 `state28 + 20×10 direct action` 结构只作为历史兼容面保留。

## 1. 系统边界

```text
模型输入
  完整任务 + head[t-0.20s,t] + wrist[t-0.20s,t]
      │
      ├─ Pass 1：受约束 Qwen generation
      │           ACTION + 单 route token + subtask，或 DONE
      │
      └─ Pass 2：同一 user 输入 + 模型自己的完整 Pass 1 prefix
                 第二次完整 Qwen forward，输出最后 16 层 hidden states
                       │
              route 只选择一个动作域
               ┌───────┴────────┐
               │                │
       Navigation FM        Manipulation FM
       [20,3]               [20,7]
       body waypoint        query-base absolute TCP target
               │                │
模型外         PCT → DWA         workspace → cuRobo/IK
执行侧         locomotion        joint controller
               └───────┬────────┘
                 首个目标完成/失败
                       │
                 新视觉 query
```

模型侧不接收 robot state、phase、operation、target truth、task FSM 或 semantic history。
执行侧必须读取 odometry、关节、TCP、局部地图和碰撞状态完成安全规划；这不构成模型
state 输入，也不得用来重写 Qwen route。

## 2. 仓库分层

```text
Contracts
├── conveyorvla/waypoint.py              token、shape、坐标、时间和安全常量
├── conveyorvla/waypoint_protocol.py     无 state 的 runtime/v1 request/response
└── configs/waypoint_v1.json             模型、loss、优化和置信度配置

Data / training
├── conveyorvla/waypoint_data.py         raw→派生 schema、audit、normalizer、loader
├── conveyorvla/waypoint_model.py        Qwen 接口与双 Layerwise FM head
├── scripts/build_waypoint_dataset.py
├── scripts/audit_waypoint_dataset.py
├── scripts/train_waypoint.py
└── scripts/check/evaluate/export_waypoint_*.py

Serving / execution
├── conveyorvla/waypoint_runtime.py      严格推理 session 与 RECOVER
├── conveyorvla/waypoint_execution.py    receding-horizon executor
├── conveyorvla/waypoint_planner_adapters.py
├── conveyorvla/waypoint_rollout.py      图像缓冲、HTTP client、frame/state adapter
├── scripts/serve_waypoint.py
├── scripts/serve_waypoint_curobo.py
└── scripts/run_waypoint_rollout.py

Collection / simulator
├── isaac/runtime.py / runtime_core.py    采集与现有 ConveyorBench 物理主循环
├── isaac/scene.py / workcell.py          Liangzhu、Go2-X5、相机与工位
└── schema/ / sidecar/                    canonical raw 与资产 provenance
```

稳定合同模块不依赖外部 planner checkout。PCT/DWA 与 cuRobo 的项目特定接线集中在
`waypoint_planner_adapters.py` 和启动脚本，核心协议与模型可以独立做静态测试。

## 3. Qwen 路由与双动作头

Pass 1 只能生成以下 active route token 之一，或精确的 `<|pred_done|>`：

| route | 动作域 | head |
|---|---|---|
| `NAV_TO_SOURCE` | `NAVIGATION` | Navigation FM |
| `PICK` | `MANIPULATION` | Manipulation FM |
| `NAV_TO_TARGET` | `NAVIGATION` | Navigation FM |
| `PLACE` | `MANIPULATION` | Manipulation FM |
| `DONE` | `NONE` | 不生成动作 |

route 是单 token；subtask 文本不参与专家 parser。Pass 2 不能复用 generation cache
代替完整 forward，也不能用 GT prefix。两个 16 层 Flow-Matching head 参数不共享；
Qwen 最后 16 层与对应 head block 逐层 cross-attention。两个 head 均没有 state encoder，
Qwen 主干、embedding/LM head 和双 head 一起训练。

训练主目标使用 GT route 选择 oracle-prefix 动作 loss；`lambda_self` 在总训练进度 5%
前为 0，之后线性升到 0.5，用模型自产 prefix 形成辅助目标。训练中的 self-conditioning
不会引入 previous-subtask history。

## 4. 动作、坐标与时间

所有 horizon 都固定为 20，并锚定 query 时刻同一个 `query-base-B_t`：

| 动作域 | shape | stride | 含义 |
|---|---:|---:|---|
| Navigation | `[20,3]` | 0.60 s | `[dx_body, dy_body, dyaw]`，不是速度 |
| Manipulation | `[20,7]` | 0.20 s | `[x,y,z,roll,pitch,yaw,gripper]` absolute TCP target |

Navigation 不是逐点 delta 积分；每行都是相对同一 `B_t` 的未来 base pose。ARM target
也是相对同一 query base 的绝对目标，而不是相对当前 TCP 的 delta。进入 cuRobo 前必须
显式执行 `query-base → curobo-planner-base` 变换。

数据允许 `action_valid_mask` 在 phase boundary 或 episode 末尾截断 horizon。在线
输出必须经过有限值、shape、frame、normalizer、workspace、segment 和 sequence gate。

## 5. 运行时与 fail-closed

`conveyorvla-waypoint-runtime/v1` request 只包含 request/episode/sequence 身份、完整
指令、两张有序 head 图、两张有序 wrist 图和 calibration ID。协议递归检查并拒绝
`state28`、phase、operation、locked route、pose truth 和 history 等字段。

以下情况返回 `RECOVER` 且不复用上次动作：请求过期/重放、标定不一致、非法 prefix、
route 置信度低于 0.55、反归一化失败、shape/数值不合法或 active route 没有动作。
`RECOVER` 是停机语义，不是把外部 FSM 的 route 填回模型。

单卡服务只加载由 `export_waypoint_inference.py` 从绑定 ZeRO checkpoint 生成的完整
inference export。export 同时绑定 source commit、resolved run、processor、special token
ID、normalizer hash 和 checkpoint step。

## 6. Receding-horizon 执行

Navigation executor：

1. 校验完整 `[20,3]` 与 prefix mask；
2. 选择第一个平移至少 0.03 m 或偏航至少 3° 的有效 waypoint；
3. body→world 后交给启用 PCT 且显式禁止 fallback 的 planner；
4. DWA 每个控制 tick 基于测量速度和局部地图输出有界 `[vx,vy,wz]`；
5. 首 waypoint 到达、超时、stall 或失败后停止并要求新 query。

纯旋转 waypoint 走限幅 terminal-yaw controller。导航时执行器可分别维持
`stow_open` 或 `carry_closed`，但 route 仍来自模型。

Manipulation executor：

1. 校验 `[20,7]` workspace 和相邻目标变化；
2. 仅采用第一个有效 absolute TCP target；
3. 转换到 cuRobo planner frame，以实时关节与碰撞场景规划；
4. 要求 reachable、collision-free 且末端误差在阈值内；
5. 底盘保持零，执行 joint path 后重新 query。

PCT/DWA 适配器绑定批准的 `arm-vla-grasp-sim@388b6818f4c605a707d13c519fbb58b1d07acd92`。
当前 cuRobo 参考 checkout 为
`87260212b891aaae8c157a1d9a3277439f602a65`；真实运行仍须记录干净状态与环境。实际
planner/Isaac 门禁状态见 [status.md](status.md)，不能从“代码已接线”推断为“闭环已通过”。

## 7. 旧采集与模型边界

ConveyorBench canonical raw、动态传送带教师和 Isaac 采集 runtime 仍是有效的数据证据
基础；它们不等于现行 Waypoint 模型输入。旧 `temporal_v3` / dense-transition view、
`state28`、`[vx,wz]`、TCP delta、`scripts/train_hierarchical.py`、`serve.py` 和
`evaluate.py` 只用于历史复现，不能与 Waypoint checkpoint、normalizer 或 runtime/v1
混用。

源码继续采用单一 live tree，不创建 `runtime_v4.py` 或并行版本目录。数据 schema、
动作语义、坐标、时钟或 checkpoint contract 变化时必须升级显式 ID，并提供拒绝或迁移
路径。
