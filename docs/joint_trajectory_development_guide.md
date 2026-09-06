# Joint-Trajectory 5 Hz 开发指南

适用：`Manipulation_Navi_v1`，`conveyorvla-joint-trajectory-policy-5hz-v1`。
本文依据当前配置、训练入口、正式训练报告和全量开环报告编写。实验事实见
[2026-09-05 实验卡](experiments/formal_5hz_20260905.md)，操作细节见
[正式评估说明](formal_joint_trajectory_evaluation.md)。旧 Waypoint 和 SPCGVLA 草案不覆盖本合同。

## 1. 从哪个入口开始

| 工作 | 入口 | 核心实现 |
|---|---|---|
| 固定源数据构建 | `scripts/materialize_joint_trajectory.py` | `joint_trajectory_data.py` |
| ABot 初始化、分阶段训练、恢复 | `scripts/train_joint_trajectory.py` | `joint_trajectory_training.py`、`dit.py` |
| 正式开环与验证集冻结 | `scripts/evaluate_joint_trajectory_formal.py` | `formal_checkpoint.py`、`formal_metrics.py` |
| 最终模型本机服务 | `scripts/serve_joint_trajectory.py` | `joint_trajectory_model.py`、`joint_trajectory_runtime.py` |
| 迁移闭环任务准备与汇总 | `scripts/run_formal_closed_loop.py` | `formal_physics.py` |
| 单 episode 仿真执行 | `scripts/run_joint_trajectory_rollout.py` | `joint_trajectory_system.py`、`isaac/` |
| 无模型执行一致性 | `scripts/prepare_execution_validation.py`、`scripts/replay_sampled_joint_targets.py`、`scripts/audit_source_action_contract.py` | `execution_consistency.py` |
| validation 源/模型 PCT 对照 | `scripts/probe_validation_navigation.py` | `waypoint_planner_adapters.py` |
| 独立 LiDAR/分割诊断 | `scripts/probe_liangzhu_lidar.py`、`scripts/probe_sam2_coke.py`、`scripts/segment_sam2_lidar_coke.py`、`scripts/view_lidar_pointcloud.py` | `perception/`、`isaac/liangzhu_lidar_probe.py` |

表中未写目录的核心模块位于 `src/conveyor_bench/conveyorvla/`。
`evaluate_joint_trajectory_open_loop.py` 是另一诊断入口，不能替代 formal 的最终模型绑定和 test 解锁检查。
LiDAR/SAM2 工具目前独立于正式模型输入；[SPCGVLA](SPCGVLA/README.md) 描述未来方案，不表示语义点云已经接入此次模型。

当前调用方向为 CLI → 模型/数据/运行合同 → 仿真与外部规划适配器；模型只接受合同内观测。
评估器单独读取物理真值。一个已知实现耦合是 `formal_checkpoint.load_formal_policy` 复用训练脚本的
`_build_model`；未来可提取公共构建器，但须用 strict-load 与身份回归验证，不能在冻结评估中途改动。

## 2. 必须保持的输入、动作与时钟

| 项目 | 生效语义 |
|---|---|
| 视觉 | head/wrist 各两个时间点，`t-0.20s` 和 `t`，共四张 RGB |
| Qwen Pass 1 | 全局指令与 RGB → `NAV_TO_SOURCE / PICK / NAV_TO_TARGET / PLACE` 及 subtask |
| 阶段提交 | 两个新观测确认；等待时底座零速度并保持最后关节目标 |
| Qwen Pass 2 | 同一观测与模型自己的回答前缀 → 最后一层 hidden state |
| Mani 状态 | 六关节位置、六关节速度、连续夹爪开度，共 13D；只输入 Mani 专家 |
| NAV | `[10,3]`，查询时刻底座坐标系下 `(x,y,yaw)`，米/弧度，点间隔 0.20s |
| Mani | `[10,7]`，六关节相对查询时刻的目标增量与连续夹爪开度，弧度/无量纲 |
| Mani 执行 | 反归一化并恢复绝对关节目标，位置/速率/夹爪裁剪；每点保持十个 50 Hz tick |
| 动作专家 | 两套不共享参数的 M0 DiT，4 次 flow-matching 采样迭代 |
| 完整操作块 | 10 点 × 0.20s = 2s；块后重查询，操作阶段约 0.5 Hz 模型重规划 |
| 禁用项 | DONE、prefix、IK/cuRobo、CRL、on-policy correction、自条件辅助和图像增强 |

配置中的 `num_target_vision_tokens=32` 实际创建可学习 future tokens，不表示将 Qwen 图像压缩成 32 个 token。
旧 `ExpertConfig` 的默认采样步数不能替代训练入口真正构建的 `M0DiTActionHead` 配置。
进度头虽存在，但源数据无可用 physical-progress 标签；当前有效 mask 为 false、`lambda_progress=0`，
不可将它描述成已经训练好的任务状态估计器。

NAV 十点均被记录和坐标变换，PCT API 实际只接收第十点作为目标 A；PCT 返回路径的最后一点是 B。
B 是规划结果端点，不是机器人已经走到的位置。现行检查要求 XY 距离 `|B-A|≤0.10m`；
超过即拒绝该次规划。机器人随后由 DWA 跟踪整条 PCT 路径，最终实测站位还需另行记录。
新诊断分支记录名义操作位姿 G、请求 A、规划 B、实测 C，并显式区分超时、到达和规划/控制失败。
采样时序、20 cm 栅格量化、退化路径及无模型回放的证据边界见
[执行一致性验证](execution_consistency_validation_20260905.md)。

## 3. 数据、初始化与 checkpoint

正式数据来自 `OscarXu/liangzhuNeW_500` 固定 revision，500 个成功示范按 episode 分成
400 train / 50 validation / 50 test；行数分别为 77,213 / 9,587 / 10,056。
标签是与相机对齐保存的 5 Hz 控制向量；不能把它描述成原始 50 Hz 已应用命令，
也不能用未来实测关节或未执行 cuRobo 轨迹替代控制标签。normalizer 只由 train 拟合。
边界和成功尾段的 hold 必须由数据时序规则导出，不使用未来阶段真值给部署策略选路由。

模型采用 ABot-M0 预训练权重进行严格领域迁移：加载 Qwen 与动作骨干，重置本合同的 token 行、
动作维度边界和辅助头；正式恢复则使用完整训练状态。`Qwen3-VL-4B-Instruct` 提供本地结构/processor，
不应将此次训练称为仅从原始 Qwen 初始化。

所有数据和训练输出必须位于 Git 工作树外。推荐同级目录：

```text
workspace/
├── ConveyorVLA/                 # Git 代码
│   └── artifacts/              # 忽略：本机依赖、场景和预训练模型
├── data_releases/<release-id>/  # 不可变数据、manifest、normalizer
└── training_runs/<run-id>/      # resolved 配置、checkpoint、日志、评估证据
```

保留 `resolved_run.json`、`resolved_policy_config.json`、`source.patch`、`CHECKSUMS.sha256`、
checkpoint manifest、权重、优化器/调度器/RNG 状态及初始化加载报告。仅一个 `model.safetensors`
不足以解锁正式评估。训练 commit 加 dirty patch 表示训练源码，后续发布 commit 不会追溯成为训练源码。

当前 `resolved_run.json` 绑定数据绝对路径，且 formal 会校验它的哈希。跨机器复现需要恢复相同挂载布局，
或单独设计并审计路径迁移；不能手改 manifest 和校验和后继续声称是原运行。新训练用自己的全新路径。

## 4. 环境与训练命令

轻量合同检查依赖 Python ≥3.10、NumPy、pytest。完整 CPU 回归还需 Torch、Transformers、Pillow、
Accelerate 等可选依赖；训练/推理和 Isaac 使用独立环境。
本次训练实际环境为 Python 3.10.20、Torch 2.6.0+cu124、Transformers 4.57 系列、
Accelerate 1.14；仿真为独立 Python 3.11 / Isaac Sim 5.1 环境。
`pyproject.toml` 的版本范围是安装约束，不是精确可复现 lockfile；升级依赖须另做兼容验证。

```bash
python -m pip install -e '.[conveyorvla]' pytest
```

以下为本次单卡拓扑的可移植命令模板；先设置实际路径，并选择空闲 GPU。
`MODEL_ROOT` 下需有配置指定的 Qwen 目录和 `ABot-M0-Pretrain/checkpoints/ABot_M0_Pretrain.pt`，
ABot 文件必须通过固定 SHA 检查。此仓库不分发这些权重或完整数据。

```bash
: "${SOURCE_ROOT:?设置固定 ModelScope 数据快照目录}"
: "${DATASET_ROOT:?设置工作树之外的全新数据发布目录}"
: "${TRAIN_RUN:?设置工作树之外的全新训练目录}"
: "${MODEL_ROOT:?设置本地基础模型根目录}"
: "${TRAIN_GPU:?设置已确认空闲的 GPU 编号}"

python scripts/materialize_joint_trajectory.py \
  --modelscope-dataset-root "$SOURCE_ROOT" --output-root "$DATASET_ROOT"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" accelerate launch \
  --num_machines 1 --num_processes 1 --mixed_precision bf16 --dynamo_backend no \
  scripts/train_joint_trajectory.py \
  --dataset-root "$DATASET_ROOT" --output-dir "$TRAIN_RUN" --model-root "$MODEL_ROOT" \
  --config configs/manipulation_navi_v1.json \
  --micro-batch-per-rank 2 --gradient-accumulation-steps 32 \
  --save-interval-steps 250 --num-workers 4 --attention-implementation sdpa --seed 20260905
```

有效 global batch 是 `world_size × micro_batch × accumulation = 64`。改卡数时必须保持采样合同，
不能沿用不匹配的累积步数。正式计划为 302 个 Stage A action-warmup update 与 2,112 个 Stage B update，
总计 2,414；`--max-steps` 截短只用于明确标记的 overfit 模式。恢复入口为 `--resume-from`，
应先核对原配置、数据和训练状态身份，不在已有输出上开始另一实验。

## 5. 测试与实验晋级

从仓库根目录执行以下分层检查。CPU 测试不代表 GPU strict reload、相机渲染、PhysX 或真机验证。

```bash
# 轻量：不需要神经网络权重或 Isaac
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p no:cacheprovider \
  tests/test_formal_evaluation.py tests/test_formal_physics.py \
  tests/test_formal_closed_loop.py tests/test_locomotion_contract.py

# 完整回归：准备完整可选依赖；允许测试在 127.0.0.1 临时端口通信
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
  python -m pytest -p no:cacheprovider
```

`assets/policies/.../policy.pt` 已从发布文件树移出，来源与 SHA 保留在
[PROVENANCE](../assets/policies/go2_x5_pct_dog_only/PROVENANCE.md) 和 contract 中。
未安装时仅实际权重完整性测试跳过；合同和伪造权重拒绝测试照常执行。安装了错误权重则必须失败。
`assets/asset_lock.json` 锁定 Git 中的资源和来源文件，外部 locomotion 权重由其独立合同验证。

正式评估按 preflight → 独立 validation smoke → 全量 validation → 冻结协议 → test 执行。
闭环另需 stage/render/control smoke，然后固定任务、seed、辅助条件与预算。
50 个 test 任务来自现有 50 个 held-out episode 的源任务实例；并非另造 50 种技能。
300 次是 `50 × 3 seeds × 2 条件`，独立任务数仍为 50；95% 区间按任务聚类，不能把 300 次全当独立任务。
采用最终模型且保留 `saturation≤0.5%` 门槛，不通过并不阻止报告能力诊断，但必须明确 gate 失败。

## 6. 闭环资源和复现边界

正式启动器目前有四个本机资源约定，在 `run_formal_closed_loop.py` 顶部常量中可查：

| 资源 | 仓库相对安装位置 |
|---|---|
| 源任务运行时 | `artifacts/sources/checkouts/arm-vla-388b681` |
| Liangzhu 场景 | `artifacts/assets/conveyorvla-v3/liangzhu` |
| Isaac Python | `artifacts/dynamic-isaaclab-5.1-20260804/envs/conveyor_py311/bin/python` |
| 此次 PCT locomotion 权重 | `artifacts/runs/conveyorvla-al0-liangzhu-closed-loop-20260813-r1/pct_runtime/checkpoints/go2_x5/pct_multifloor/model_26000.pt` |

这些依赖、PCT 地图及所用源 tar 在 prepare/run 中核验。新机器必须准备对应依赖；仅安装 Python 包不足以启动闭环。
路径配置化属于后续改进；若修改源码或资源，需准备新的 manifest 和 validation，不能继续旧冻结 test。
此处 PCT 权重与仓库 `assets/policies/` 的另一 locomotion 合同不是可随意互换的文件。

两种条件均禁止中途重置物体，均保留操作底座/支撑锁定。`source_assisted` 在满足闭合、抬升、接近和
稳定条件后添加固定抓取关节，首次连续打开目标前释放；`no_grasp_assist` 从不添加抓取关节。
两者都属于 Sim6→Sim5.1 迁移条件，推理期间暂停仿真；不代表自由底盘纯物理或实时部署。
当前未提供经本合同验证的真机上线流程。

## 7. 失败定位与下一轮开发

| 观察到的现象 | 先检查的证据 | 不能直接得出的结论 |
|---|---|---|
| PCT endpoint snap 超限 | 第十点目标、返回端点、地图/坐标与容差 | 模型一定不会导航 |
| DWA 空数组异常 | PCT 路径、局部候选与 DWA 输入 trace | 模型生成了空动作 |
| PICK 请求耗尽 | 站位、TCP 距离、关节跟踪、视觉查询时间 | DiT 参数太少 |
| 未抓起即切换后续阶段 | 模型 route、物理事件与历史观测 | 高开环准确率等于阶段正确 |
| saturation 超门槛 | 原始预测、实际裁剪、标签自身事件率 | 只放宽门槛即可解决能力问题 |

优先做独立 validation 对照：源动作回放验证迁移一致性；名义抓取站位启动验证交接；
比较更短执行块验证反馈周期；再比较状态/历史、有界夹爪和几何监督。每次仅改一个可检验因素，
记录分阶段事件和完整任务结果。原因分析及证据限度见
[架构诊断](joint_trajectory_architecture_analysis_20260905.md)。

冻结评估以源码文件内容哈希为准。当前完整身份覆盖 `src/**/*.py` 和 `scripts/*.py`；
open-loop 使用自身范围。文档、测试和 Git 提交不会改变该源码摘要，但发布前仍应复核。
改变合同、执行点数、辅助物理或采样参数须创建新协议和输出目录；不得覆盖旧报告，也不得删除失败 attempt。

## 8. 维护边界

本仓库主要是研究平台，同时包含机器人仿真栈。代码/配置在 Git，实验证据和大工件独立存储，
第三方来源和许可证边界随工件保留。模型、动作/时钟、数据 split/normalizer、仿真辅助是关键审查面；
各改动提交者应提供可复核合同和测试，具体维护者由仓库维护方指定，本文不虚构 CODEOWNERS 或远端分支保护。

当前发布到 `Manipulation_Navi_v1`；旧 Waypoint 分支/合同保留复现，SPCGVLA 仍是实验设计。
采用独立功能分支和明确 commit，兼容修复可单独 backport；不跨合同盲目合并。
`pyproject.toml` 的 `0.1.0` 是现有包版本，本次代码快照用 commit 标识，不创建声称模型达标的版本 tag。
未来正式 release 应在验证完环境、工件获取与完整评估后补充不可变 tag 和迁移说明。
提交前检查和工件规则见 [CONTRIBUTING](../CONTRIBUTING.md)。

## 2026-09-06 执行接口补充

本维护线的下一轮开发与验收以[执行接口 v2](execution_interfaces_v2_20260906.md)为准：
评价和辅助控制分离，源目标先经部署转换做物理对照，PCT 连续末段先通过几何/控制验收，
固定模型的反馈周期对照每个条件使用独立仿真进程并检查首帧。旧阈值和冻结结果保留。
