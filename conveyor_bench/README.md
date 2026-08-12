# ConveyorBench

ConveyorBench 是 ConveyorVLA AL0 的仿真、示教采集、数据门禁、LeRobot 转换、
训练和闭环测评框架。当前场景把 Liangzhu NuRec/3DGS 作为静态背景，把 Go2-X5、
深绿色传送带、刚体零件和投放盒作为 Isaac Sim 动态前景，在同一 RTX 渲染和物理
循环中采集数据。

## 设计原则

仓库只有一条现行代码路径：

```text
scripts/collect.py
  → scripts/run_benchmark.py
  → isaac/runtime.py
  → isaac/runtime_core.py
  → isaac/scene.py + isaac/workcell.py + isaac/physics.py
  → schema recorder / validation / export
  → scripts/convert_dataset.py
  → LeRobot v3 (H.264 / PyAV)
  → scripts/train.py
```

数据中的 `conveyor-bench-v1`、`temporal_v3` 和 LeRobot `v3.0` 是兼容协议标识，
不是三套并行运行时。不要为新实验复制 `runtime_v4.py`；直接修改当前实现，由 Git
提交保存历史，并在数据契约变化时升级 `schema_version`。

## 目录

```text
conveyor_bench/
├── assets/                 # 可进入 Git 的机器人、工位和小型策略资产
├── configs/
│   ├── benchmark.json      # 当前场景、机器人、相机和采集合同
│   ├── dataset.json        # raw → LeRobot v3 合同
│   ├── model.json          # 模型资产与训练超参数
│   └── temporal.json       # 双帧时序观测与 20×10 动作合同
├── docs/                   # 规范、架构、数据、操作、状态和迁移说明
├── scripts/                # 单一用途的可执行入口
├── src/conveyor_bench/
│   ├── schema/             # 已有 episode 的稳定数据协议
│   ├── sidecar/            # SSH 资产校验和运行时 USD 组合
│   ├── isaac/              # 当前场景、物理与采集主循环
│   ├── conveyorvla/        # 时序数据、流式控制与 LeRobot 适配
│   └── task_coordinator.py # 多阶段目标协调器
└── tests/                  # 不启动 Isaac 的逻辑和静态接线测试
```

脚本职责：

| 脚本 | 职责 |
| --- | --- |
| `check_environment.py` | 检查 Python 依赖和项目内机器人资产 |
| `validate_assets.py` | 完整校验 SSH sidecar 哈希与 NuRec 结构 |
| `probe_scene.py` | 场景、相机和注册渲染探针 |
| `probe_gripper.py` | FinRay 碰撞与夹爪资产探针 |
| `probe_mobile_locomotion.py` | whole-body 移动门禁 |
| `run_benchmark.py` | 单个 Isaac 采集进程 |
| `collect.py` | 小批次采集、逐条门禁和成功索引 |
| `validate.py` | canonical episode/run 严格校验 |
| `audit_episode.py` | 图像、动作和任务质量审计 |
| `check_camera_gate.py` | 相机完整性、时变性和策略可见性门禁 |
| `export.py` | canonical raw 到训练 JSONL 视图 |
| `convert_dataset.py` | 成功 episode 到 LeRobot v3 |
| `train.py` | ConveyorVLA AL0 训练 |
| `serve.py` | 本地推理服务 |
| `evaluate.py` | Isaac 闭环测评 |

## 环境

宿主环境需要 Python 3.10/3.11、Isaac Sim、Isaac Lab、PyTorch、NumPy 和 OpenCV。
LeRobot 转换使用独立 Python 3.10 环境和 `lerobot==0.4.4`。

```bash
conda activate env_isaaclab
python -m pip install -e .
python scripts/check_environment.py
```

大资产根目录必须包含 `TRANSFER_MANIFEST.sha256`、`liangzhu/` 和 `objects/`：

```bash
export CONVEYOR_BENCH_ASSET_ROOT=/diff/wallx_workspace/dzb/conveyorvla-v3-assets-20260811
python scripts/validate_assets.py \
  --asset-root "$CONVEYOR_BENCH_ASSET_ROOT" \
  --allowed-root /diff/wallx_workspace/dzb
```

兼容读取旧环境变量 `CONVEYOR_BENCH_V3_ASSET_ROOT`，新任务不要继续写入该变量。

## 测试

纯逻辑和静态接线测试不启动 Isaac：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider
```

一次最小采集命令生成检查后的 raw episode：

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

使用 `--dry-run` 可验证命令、路径和 GPU 限制而不启动 Isaac。当前采集器只接受
物理 GPU 2/3，每个进程最多 8 条；不会占用或管理 GPU 0/1 上的外部任务。

## 数据与训练

raw episode 保留三路 PNG、50 Hz 状态/动作和全部事件。只有 `success=true` 且通过
validator、quality audit、camera gate 和无损 export 的 episode 才可进入训练集。

```bash
python scripts/convert_dataset.py \
  --episode-list outputs/smoke/successful_episode_roots.txt \
  --output-root outputs/lerobot

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 scripts/train.py \
  --lerobot-root outputs/lerobot \
  --output-dir outputs/checkpoints/al0 \
  --model-root /path/to/model-assets \
  --allow-fixed-base \
  --belt-speed 0.01
```

转换后为 LeRobot v3、H.264/PyAV 视频。四个视频特征分别是 head/wrist 的
`t-2` 和 `t` 帧，不是四个物理相机。详细字段和验证规则见
[数据说明](docs/data.md)。

## 状态边界

截至 2026-08-12：

- PCT 对齐的 Go2-X5、head/wrist 标定和第三视角已接入；
- Liangzhu NuRec 背景与 Isaac 动态前景已完成注册组合；
- 固定底盘、动态可乐罐跟随抓取和投放已有完整成功证据；
- 联合专家的“接近—抓取—负载导航—投放”真实 Isaac 正例已通过训练数据硬门禁：
  49.46 秒、两段净位移 `0.307 m / 0.302 m`，释放后入蓝框即成功；
- 当前 sidecar 仅把 `cola` 注册为可训练刚体，其他下载物品尚未完成抓取 fixture；
- 384 条矩阵不能启动，直到四个训练零件和第二档速度都通过同样门禁。

不应把专家轨迹成功表述成 VLA 闭环成功，也不应把失败 episode 放进专家训练集。后续
步骤和历史失败证据见 [状态说明](docs/status.md)。

## 文档

- [Benchmark 规范](docs/benchmark.md)
- [代码与模型架构](docs/architecture.md)
- [数据格式与质量门禁](docs/data.md)
- [采集、训练与测评操作](docs/operations.md)
- [当前状态与下一步](docs/status.md)
- [版本迁移与兼容策略](docs/history.md)

whole-body `policy.pt` 的权重再分发许可证尚未确认。它可用于当前研究环境，但公开
发布前必须取得授权或换用许可证明确的权重；见
`assets/policies/go2_x5_pct_dog_only/PROVENANCE.md`。
