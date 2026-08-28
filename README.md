# ConveyorVLA AL0

ConveyorVLA AL0 是面向 Go2-X5 移动操作机器人的视觉语言 waypoint 策略与 Isaac
评测框架。当前现行模型合同是
[`qwen3vl-layerwise-dual-fm-waypoint-v1`](docs/conveyorvla_waypoint_policy_contract_v1.md)：
模型只读取完整指令和 head/wrist 双帧图像，自主决定四阶段 route，并预测空间目标；
机器人状态只允许留在模型外部的规划与控制侧。

截至 2026-08-21，Waypoint Policy v1 的 runtime/eval 基线为分支
`feature/conveyorvla-waypoint-v1` 的提交
`ace7d6e9f2026b55be2f9cc55cf4a355b4dde339`。当前 durable checkpoint 为
`step_002000@a8d57a22c515`；四卡训练已按用户指令在 step 2090 后停止，checkpoint 仍按
每 500 effective optimizer step 保存。

step 2000 的 checkpoint load、inference export、服务和真实 cuRobo known-pose 已通过。
移除额外导航门控并改用原始 arm-vla 规则后，模型完成 22 次 NAV query 并自主切到 PICK；
但首点都小于 reference 的 0.18 m 到达容差，机器人没有接近可乐，首个 ARM target 又超过
原始 35° rate gate。三路视频与完整 trace 已下载。准确边界见
[当前状态](docs/status.md)和
[step 002000 reference 复测](docs/checkpoint_step2000_arm_vla_reference_evaluation_20260821.md)。

## 现行合同

```text
完整任务 + head/wrist[t-0.20s, t]
              │
Pass 1：受约束 Qwen 生成 ACTION/DONE、单 token route 与 subtask
              │ 模型自己的完整 assistant prefix
Pass 2：同一观测上的第二次完整 Qwen forward
              │ 最后 16 层逐层条件
       ┌──────┴──────┐
       │             │
NAV Layerwise FM   ARM Layerwise FM
[20,3] body        [20,7] query-base
waypoint           absolute TCP target
       │             │
PCT → DWA         safety → cuRobo/IK
       └──────┬──────┘
       只执行首个目标后重新观测
```

模型 request、batch 和 checkpoint 不得包含 `state28`、phase、operation、外部 FSM
状态或 previous-subtask history。执行器可以读取 odometry、关节、TCP、碰撞和局部地图，
但不得用这些信息覆盖模型 route。

## 仓库

```text
assets/                  可公开的小型机器人、工位和策略资产
configs/                 benchmark、数据、模型和分布式训练合同
docs/                    权威合同、架构、数据、操作、状态和历史证据
scripts/                 数据、训练、评测、服务、planner probe 与 rollout 入口
src/conveyor_bench/       采集/runtime 及现行 waypoint 实现
tests/                    静态合同、数据、模型、服务、planner 和 rollout 测试
```

数据集、checkpoint、日志、视频、3DGS sidecar、Conda 环境和 `handoff_private/` 不进入
Git。它们只能由 manifest、SHA-256 和外部不可变路径引用。

## Manipulation_Navi_v1 successor

本分支另行实现了代码候选：NAV 改为 `[10,3]@0.20s` reference，Mani 改为读取 13D 可测
关节状态并直接输出 `[10,7]@0.04s` joint/gripper trajectory；新 runtime 不使用 Mani IK、
cuRobo、DONE 或 learned K*/prefix selector。

2026-08-28 已完成首批 4 条 Gate-A review episode 的 raw 数据审计：50 Hz control 时钟、
applied joint/gripper command、三路图像、四 route 顺序和 Mani 抓取/放置时序基本通过；但
当前 materializer 在 1,585 个 query 中暴露 11 个合法 zero-prefix tail，NAV 教师速度包络与
冻结合同不一致，PLACE physical progress 又只覆盖 late bucket。为此已冻结严格的 `K=0`
审计语义：它只允许由 raw 时序证明的 `boundary` 或 evaluator-confirmed `success_tail`，输出
完整 10 点 hold；它不恢复模型 K* 或 runtime prefix 选择。该规则尚待代码和测试落地，正式
immutable 数据、overfit、训练和真实闭环仍未开始。准确边界见
[训练改进方案](docs/conveyorvla_joint_trajectory_training_improvement_plan.md)与
[全新数据采集规范](docs/conveyorvla_joint_trajectory_fresh_data_collection_spec.md)；已完成代码
边界见[实施报告](docs/manipulation_navi_v1_code_implementation_20260827.md)。

## 快速检查

从仓库根目录执行：

```bash
python -m pip install -e .

PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider
```

训练 source 曾在下列文件中验证 49 项静态门禁；现行 runtime/eval commit 还增加了后续
export 与 cuRobo lifecycle 回归测试，因此不要把 49 当作当前固定收集数：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider \
  tests/test_waypoint_contract.py \
  tests/test_waypoint_data.py \
  tests/test_waypoint_model.py \
  tests/test_train_waypoint.py \
  tests/test_waypoint_runtime.py \
  tests/test_waypoint_open_loop.py \
  tests/test_waypoint_planner_adapters.py \
  tests/test_waypoint_service.py \
  tests/test_waypoint_rollout.py
```

数据构建、四卡训练、checkpoint 校验、单卡服务和真实 planner 接线命令统一维护在
[操作手册](docs/operations.md)，不要从历史诊断文档复制命令。

## 文档

- [文档索引与权威性](docs/README.md)
- [Manipulation_Navi_v1 代码实施报告](docs/manipulation_navi_v1_code_implementation_20260827.md)
- [已批准且冻结的 Waypoint Policy v1 合同](docs/conveyorvla_waypoint_policy_contract_v1.md)
- [Waypoint v2 阶段切换执行与长训计划](docs/waypoint_v2_stage_transition_execution_plan.md)：
  已批准的 successor 计划；冻结 v1，使用全新 v2 schema，并按证据选择 terminal-hold、动态
  prefix、局部目标 CRL、训练 FM sample `1→4` 与 on-policy correction
- [当前状态、证据和未通过门禁](docs/status.md)
- [step 002000 原始 arm-vla 规则闭环复测](docs/checkpoint_step2000_arm_vla_reference_evaluation_20260821.md)
- [step 001000 开环与真实 Isaac 闭环评测](docs/checkpoint_step1000_evaluation_20260821.md)
- [模型、协议与执行架构](docs/architecture.md)
- [Waypoint 数据 schema 与质量门禁](docs/data.md)
- [数据、训练、评测和部署操作](docs/operations.md)
- [Benchmark 任务规范](docs/benchmark.md)
- [版本迁移与兼容策略](docs/history.md)

`assets/policies/go2_x5_pct_dog_only/policy.pt` 的再分发许可证尚未确认；公开发布前
必须取得授权或替换为许可证明确的权重。
