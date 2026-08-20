# ConveyorVLA AL0

ConveyorVLA AL0 是面向 Go2-X5 移动操作机器人的视觉语言 waypoint 策略与 Isaac
评测框架。当前现行模型合同是
[`qwen3vl-layerwise-dual-fm-waypoint-v1`](docs/conveyorvla_waypoint_policy_contract_v1.md)：
模型只读取完整指令和 head/wrist 双帧图像，自主决定四阶段 route，并预测空间目标；
机器人状态只允许留在模型外部的规划与控制侧。

截至 2026-08-21，Waypoint Policy v1 的 runtime/eval 基线为分支
`feature/conveyorvla-waypoint-v1` 的提交
`121512903667e16578525ec22dcfb2d0deca92e5`；正式训练 checkpoint 仍绑定干净提交
`724ead21be2c27d9b40c200375ee4ab49ccedc84`。4×H20 长训已按用户指令暂停在最后有效
step 1181，最后完整 checkpoint 为 step 1000，后续新训练默认每 500 step 保存。

step 1000 的四卡 load、route/格式开环、inference service 和真实 cuRobo known-pose
已经通过；NAV/ARM 动作质量仍差，完整自主 Isaac 测试在首个 NAV chunk 的 yaw 安全门
失败。三路视频已生成并下载，但不能据此声明闭环成功。准确边界见
[当前状态](docs/status.md)和 [step 001000 评测](docs/checkpoint_step1000_evaluation_20260821.md)。

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
- [已批准且冻结的 Waypoint Policy v1 合同](docs/conveyorvla_waypoint_policy_contract_v1.md)
- [当前状态、证据和未通过门禁](docs/status.md)
- [step 001000 开环与真实 Isaac 闭环评测](docs/checkpoint_step1000_evaluation_20260821.md)
- [模型、协议与执行架构](docs/architecture.md)
- [Waypoint 数据 schema 与质量门禁](docs/data.md)
- [数据、训练、评测和部署操作](docs/operations.md)
- [Benchmark 任务规范](docs/benchmark.md)
- [版本迁移与兼容策略](docs/history.md)

`assets/policies/go2_x5_pct_dog_only/policy.pt` 的再分发许可证尚未确认；公开发布前
必须取得授权或替换为许可证明确的权重。
