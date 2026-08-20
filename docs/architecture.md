# 架构说明

> 版本范围：本文主要描述 2026-08-13 的采集/runtime 架构。第 4～5 节的
> `state28 + 20×10 直接动作` 模型是当前旧实现，不是下一代目标。2026-08-20 已批准的
> 目标架构以 [Waypoint Policy v1 合同](conveyorvla_waypoint_policy_contract_v1.md) 为唯一依据；
> 合同尚未实现，二者不得混称。

## 1. 仓库分层

```text
CLI
├── collect / run_benchmark / validate / convert / train / evaluate
│
Runtime
├── isaac/runtime.py        当前 NuRec 场景入口
├── isaac/runtime_core.py   唯一物理与采集主循环
├── isaac/scene.py          NuRec 与动态前景组合
├── isaac/workcell.py       Go2-X5、传送带、相机和投放盒
└── isaac/physics.py        共享 PhysX 小工具
│
Contracts
├── schema/                 canonical episode、记录、校验和导出
├── sidecar/                外部资产校验与组合层生成
└── task_coordinator.py     顺序目标状态
│
Policy
├── conveyorvla/temporal.py 双帧时序目标
├── conveyorvla/streaming.py 在线动作合并
└── conveyorvla/lerobot_v3.py LeRobot v3 适配
```

`runtime.py` 只负责当前场景特有的资产和 provenance；所有物理采样、专家控制、
记录和失败处理都在 `runtime_core.py`。这样场景适配不会复制一套采集主循环。

## 2. 版本策略

源码文件不带 V1/V2/V3 后缀。迭代通过 Git commit、branch 和 tag 保存，当前代码
直接覆盖旧实现。仅以下版本标识继续保留：

- `conveyor-bench-v1`：已经落盘的 canonical episode 协议；
- `conveyor-vla-al0-temporal-v3`：导航—抓取—配送联合训练记录格式；
- LeRobot `v3.0`：第三方数据格式；
- 历史 teacher/profile/scene ID：用于拒绝不兼容旧数据。

升级数据字段时新增 schema migration 或显式拒绝，不创建第二套 runtime。

## 3. 渲染与物理

NuRec 背景和 Isaac 前景由 RTX 单遍注册渲染。Gaussian 只负责静态外观；机器人、
传送带和物体都是 Isaac prim，并参与 PhysX。场景逻辑不会把动态物体后期贴到背景上。

sidecar 校验顺序：

```text
root containment
  → no symlink/special file
  → manifest membership
  → SHA-256
  → NuRec USDZ members
  → runtime USD layer
  → stage prim contract
  → object visual/collision fixture
```

任一门禁失败都在仿真开始前终止。

## 4. ConveyorVLA AL0

当前网络输入和输出：

```text
完整语言指令
head[t-5, t] + wrist[t-5, t]
当前 28 维机器人状态
          │
Qwen3-VL 第一次生成当前 canonical subtask
          │ 模型预测文本 + 原始观测
Qwen3-VL 第二次完整 forward（全量微调）
          │ hidden size 2560
按预测文本映射 Navigation DiT 或 Manipulation DiT
          │
未来 20 × 2 或 20 × 7 动作块（25 Hz，0.8 s）
```

Qwen3-VL 主干、embedding/lm head 和两个 DiT 都参与训练。Dispatcher 只做语言到专家
的映射，不保存外部任务顺序。主 prompt 在训练和在线都不接收 semantic history；阶段
判断只依赖当前双帧视觉与完整任务。动作专家 route 的 teacher forcing 单独衰减到 0。

28 维状态包含底盘速度、角速度、重力投影、机械臂/夹爪关节等 proprioception。
10 维动作包含底盘、TCP 和夹爪命令。overview、物体真值和教师 phase 都不能进入
策略输入。

## 5. 时序能力

动态抓取能力不是来自“把视频存成 MP4”，而来自三个合同：

1. head/wrist 各提供 `[-5, 0]` model tick、跨度 0.20 秒的有序短 clip；
2. 模型预测未来 20 个独立动作目标；
3. 在线执行按 episode、generation、observation tick 和 target tick 合并动作。

过期动作、旧 episode、旧 generation、倒序 observation 和不足两步的有效后缀全部
fail-closed。控制器不允许仅按数组重叠位置拼块。

Navigation DiT 只输出 `[vx,wz]`。动作组合器必须显式补全 joint-space reference：
`NAV_TO_SOURCE=stow+open`，`NAV_TO_TARGET=carry+closed`，并从实测关节限速过渡；空动作或
零 TCP delta 不能再表示导航阶段机械臂姿态。Manipulation DiT 激活时底盘速度硬置零。

## 6. 专家状态机

专家主路径按顺序执行：

```text
mobile_settle
→ mobile_approach
→ mobile_stabilize
→ arm_preposition
→ settle / select
→ pregrasp / track / descend
→ close / lift
→ carry_retract（底盘锁定，收回标准携带位）
→ carry_backoff / carry_backoff_settle
→ carry_turn（蓝框为左转）
→ carry_navigate / carry_settle
→ carry / preplace / place_descend
→ open / retreat / verify_place
```

目标物体从 episode 初始化起连续随传送带运动；导航和预摆臂只门控 oracle 激活，
不再门控物体生成。联合训练只接受 `whole_body_policy`。exporter 要求接近传送带至少 `0.20 m`、收臂后
负向直退至少 `0.30 m`、负载导航至少 `0.10 m`，并要求 `carry/preplace/place_descend/open`
阶段底盘动作严格为零。到达目标框后由低层 `root_pose_hold` 站立控制抵消机械臂反力，
且只有物体已经进入蓝框后才允许执行 `open`；上层 VLA 不再输出导航动作。固定底盘只保留为
机械臂消融，不再能生成联合训练记录。`m0_*assist` 参数只用于历史诊断，任何启用
assist 的 episode 都由 exporter 拒绝进入标准训练集。

## 7. 扩展规则

新增物品：扩展 `sidecar/objects.py` 和资产 manifest，并增加 fixture 测试；不要复制
scene 或 runtime。

新增速度：通过配置和采集参数扩展允许集合，并重新运行教师节拍测试；不要复制
collector。

新增任务：优先扩展 canonical task/phase 和 `task_coordinator.py`；只有数据字段变化
才升级 schema。

新增场景：若仍使用同一机器人、传送带和 episode 合同，把场景差异放入
`isaac/scene.py` 的配置；只有物理主循环确实不同才拆模块。
