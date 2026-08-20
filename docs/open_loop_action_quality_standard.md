# ConveyorVLA AL0 开环动作质量标准检测（basic-v1）

> 历史标准：本页阈值和命令针对旧 `[vx,wz] + TCP-delta` 动作空间。Waypoint v1 使用
> `scripts/evaluate_waypoint_open_loop.py` 的 route、ADE/FDE、absolute TCP 和 safety
> 门禁，见 [operations.md](operations.md)；不得把 basic-v1 结果当作 Waypoint 通过。

## 1. 目的与边界

本检测回答一个单独的问题：在数据集记录的专家观测上，模型实际采样出的动作，和同一时刻的专家动作有多接近、方向是否正确、是否稳定。

它不是以下任何一种测试：

- 不是 flow-matching 训练 loss 的重算；
- 不是把专家动作画在视频上的状态回放；
- 不是仿真闭环执行，也不能单独证明机器人能完成抓取和放置；
- 不是 VLM 路由准确率测试。

主检测使用标注的正确子任务文本完成第二次 Qwen forward，并据此选择 Navigation DiT 或 Manipulation DiT。这样可以隔离动作专家质量；VLM 自主路由和闭环成功率必须分别检测。

## 2. 固定合同

- 默认 split：完整 `test`，不得抽样替代主结果。
- 推理：真实 DiT diffusion sampling，不使用 teacher-forced action，不用专家动作代替预测。
- 路由：主检测固定正确标注路由（oracle route）。
- 动作监督：只统计 `action_valid_mask=True` 的同专家连续前缀；跨阶段后缀不得进入当前专家指标。
- 执行前缀：单独报告前 5 个有效动作。这对应当前在线控制一次 replan 前最先消费的动作段。
- 完整 chunk：同时报告整个有效动作前缀，不能只看第一个动作。
- 物理量：同时保存原始归一化输出和经过线上同款 clip/scale 后的物理动作。
- 边界：分别报告 phase boundary 1 秒窗口与 interior，禁止把边界样本静默删掉。
- 随机性：每阶段固定抽取 16 条（边界 8 条、时序分布的 interior 8 条），以 4 个 seed 重复采样。
- 可追溯：必须保存逐样本预测、目标、mask、代码 commit、数据 manifest SHA-256、checkpoint SHA-256、运行命令与日志。

## 3. 标准指标

每个阶段、每个动作维度都必须报告：

- finite rate、预测/目标的 mean、std、min、max；
- MAE、RMSE、bias、Pearson correlation、R²；
- 相对零动作基线的 skill：`1 - model_SSE / zero_action_SSE`；
- 目标绝对值超过 0.05 归一化 deadband 时的符号一致率；
- 超出归一化动作合同的比例和到达 clip 边界的比例；
- 第一个动作、前 5 个动作、完整有效 chunk、相邻动作差分四组指标。

Navigation DiT 另报：

- 以 25 Hz 积分得到的前 5 步线位移和 yaw 变化误差；
- 位移/转角方向一致率；
- 预测为反向的前缀比例；
- `vx < -0.02 m/s` 的动作步比例。

Manipulation DiT 另报：

- 以 `gripper_open_fraction >= 0.5` 为 open 的二值准确率；
- 预测和目标的 open rate；
- 前 5 步及完整有效 chunk 的夹爪准确率。

稳定性探针另报每阶段 mean/P95/max sampling std、相对首个 seed 的 RMSE；导航报告前缀方向是否随 seed 翻转，操作报告夹爪状态是否随 seed 改变。

## 4. basic-v1 门禁

这些阈值是训练回归/明显失效门禁，不是仿真部署成功标准：

| 门禁 | 阈值 |
|---|---:|
| 完整 split 覆盖 | evaluated rows = dataset rows |
| 所有采样动作有限 | 100% |
| 每阶段 normalized RMSE | `<= 0.75` |
| 每阶段相对零动作 skill | `> 0` |
| 每阶段 normalized out-of-contract rate | `<= 5%` |
| 每个导航阶段前 5 步方向准确率 | `>= 80%` |
| NAV_TO_SOURCE 前缀反向率 | `<= 5%` |
| PICK/PLACE 夹爪二值准确率 | `>= 90%` |
| 每阶段 mean sampling std | `<= 0.20` |

任一门禁失败时，结论必须写成“开环动作质量未通过 basic-v1”，但仍要保留全部报告。不得用有限 loss、动作形状正确或少数挑选样本替代通过结论。

## 5. 标准命令

在干净代码快照和新的输出目录中运行：

```bash
eval_tmpdir=$(mktemp -d /tmp/conveyorvla-open-loop.XXXXXX)
CUDA_VISIBLE_DEVICES=0 \
TMPDIR="$eval_tmpdir" \
python scripts/evaluate_open_loop_action_quality.py \
  --repo /path/to/clean/repo \
  --hierarchy-root /path/to/derived/hierarchy \
  --checkpoint /path/to/consolidated/checkpoint \
  --checkpoint-sha256 <64-hex-sha256> \
  --model-root /path/to/base/models \
  --initial-action-checkpoint /path/to/action_model_final.safetensors \
  --split test \
  --batch-size 16 \
  --num-workers 4 \
  --seed 20260818 \
  --stability-samples-per-phase 16 \
  --stability-seeds 20260819,20260820,20260821,20260822 \
  --attention-implementation sdpa \
  --output-dir /new/run/results \
  --fail-on-gate
```

`TMPDIR` 必须使用短路径；Python multiprocessing 的 Unix socket 路径有长度上限，不能把远端长 run 路径直接用作多 worker 临时目录。

脚本拒绝覆盖已有 `output-dir`。集成 smoke 可临时添加 `--max-rows-per-phase 1 --stability-samples-per-phase 1`，但该结果必然不满足完整 split 覆盖，严禁作为正式报告。

## 6. 标准产物

- `report.json`：完整聚合、逐阶段、逐维度、边界和稳定性指标；
- `report.md`：人可读报告和逐项门禁结论；
- `predictions.jsonl`：test split 每一行的真实采样动作、专家目标、有效 mask、物理动作；
- `stability_predictions.jsonl`：稳定性子集的全部多 seed 动作。

外层运行目录还必须保存：

- 原样评测脚本及其 SHA-256；
- 完整命令、Conda 环境、GPU index/UUID；
- stdout/stderr 日志、退出码、开始/结束时间；
- 代码状态，以及 `handoff_private/` 未进入 Git 的证明。

## 7. 推荐判读顺序

1. 先看 coverage、finite、shape 等合同门禁；
2. 再看 NAV_TO_SOURCE 前 5 步是否反向，这是闭环初段最直接的危险指标；
3. 比较前 5 步与完整 chunk，判断问题发生在在线会消费的前缀还是远端后缀；
4. 比较 boundary 与 interior，判断切换窗口是否显著退化；
5. 看逐维 bias、符号一致率与 clip 率，区分系统偏置、方向错误和尺度饱和；
6. 看多 seed 稳定性，区分确定性错误和 diffusion 方差；
7. 开环通过后，再进行自主 VLM 路由测试与真实仿真闭环 episode。三者不可互相替代。
