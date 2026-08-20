# ConveyorVLA AL0

ConveyorVLA AL0 是面向 Go2-X5 移动操作机器人的动态抓取基准与 VLA 训练框架。
当前任务在 Liangzhu NuRec/3DGS 静态背景中，用 Isaac Sim 驱动机器人、深绿色
传送带、刚体零件和投放盒，完成：

```text
导航到传送带 → 跟踪运动目标 → 抓取并抬升 → 收臂 → 后退左转
→ 导航到蓝框 → 驻车放置 → 松爪后再移动
```

仓库只维护一套现行实现。V1/V2/V3 仅表示历史协议或 LeRobot 数据版本，不是并列
源码；旧实现由 Git 历史保存。

## 目录

```text
assets/                 可公开的小型机器人、工位和策略资产
configs/                benchmark、数据、模型和时序合同
docs/                   架构、数据、操作、状态与历史说明
scripts/                采集、校验、转换、训练和测评入口
src/conveyor_bench/      当前唯一实现
tests/                   不启动 Isaac 的逻辑与静态接线测试
```

大体积 3DGS/物品资产、数据集、模型和运行环境通过 SSH sidecar 管理，不进入 Git，
也不允许运行时联网下载。

## 快速检查

```bash
conda activate env_isaaclab
python -m pip install -e .
python scripts/check_environment.py

PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider
```

校验远端资产：

```bash
export CONVEYOR_BENCH_ASSET_ROOT=/diff/wallx_workspace/dzb/assets/conveyorvla-v3
python scripts/validate_assets.py \
  --asset-root "$CONVEYOR_BENCH_ASSET_ROOT" \
  --allowed-root /diff/wallx_workspace/dzb
```

采集一条 whole-body smoke episode：

```bash
python scripts/collect.py \
  --asset-root "$CONVEYOR_BENCH_ASSET_ROOT" \
  --output-root outputs/smoke \
  --physical-gpu 2 \
  --robot-mode whole_body_policy \
  --episodes 1 \
  --seed 1101 \
  --belt-speed 0.01 \
  --require-all-success
```

使用 `--dry-run` 可以检查路径、命令和 GPU 约束而不启动 Isaac。raw episode 只有在
任务成功，并通过 validator、quality audit、camera gate 和无损 export 后，才允许
转换为 LeRobot v3 训练数据。

## 当前边界

> 2026-08-20 架构转换说明：用户已批准
> [`qwen3vl-layerwise-dual-fm-waypoint-v1`](docs/conveyorvla_waypoint_policy_contract_v1.md)
> 作为下一代模型、训练与推理合同。当前代码和正在运行的 dense-view7 训练仍属于旧的
> `state28 + velocity/TCP-delta` 合同，不能视为 waypoint v1 已实现。

- PCT 对齐的 Go2-X5、head/wrist 标定和第三视角已经接入；
- 目标从环境初始化开始连续运动；完整移动教师已通过单条成功 smoke；
- 抓取后必须先垂直抬升并锁底盘收回标准携带位，之后才允许底盘运动；
- 当前只完成 `cola` 的训练刚体合同，四零件 × 两速度 × 48 条的 384 条矩阵尚未
  启动；
- PCT Liangzhu n200/n250 使用独立 raw→LeRobot v3 适配器，可用于静态
  box1→box2 最小训练闭环，不计入上述动态传送带配额；
- 教师轨迹成功不能表述为 VLA 闭环成功。

## 文档

- [文档索引与权威性说明](docs/README.md)
- [已批准的 Waypoint Policy v1 合同](docs/conveyorvla_waypoint_policy_contract_v1.md)
- [Benchmark 规范](docs/benchmark.md)
- [模型与代码架构](docs/architecture.md)
- [数据格式与质量门禁](docs/data.md)
- [采集、训练与测评操作](docs/operations.md)
- [当前状态与下一步](docs/status.md)
- [版本迁移与兼容策略](docs/history.md)
- [Liangzhu seen dense-transition 合同](docs/liangzhu_seen_dense_transition_contract.md)
- [Seen 子任务数据问题分析与修正](docs/seen_subtask_data_analysis_and_remediation.md)

`assets/policies/go2_x5_pct_dog_only/policy.pt` 的再分发许可证尚未确认；公开发布前
必须取得授权或替换为许可证明确的权重。
