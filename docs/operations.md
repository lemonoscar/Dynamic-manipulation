# 数据、训练与测评操作

版本范围：Waypoint Policy v1，runtime/eval 代码基线
`121512903667e16578525ec22dcfb2d0deca92e5`。正式 step 1000 checkpoint 的训练 source
仍为 `724ead21be2c27d9b40c200375ee4ab49ccedc84`。所有命令从干净仓库根目录执行，输出必须
使用全新目录。数据、checkpoint、日志、视频、cache 和 `handoff_private/` 均不得进入
Git。

## 1. 环境与代码预检

```bash
git status --short --branch
git rev-parse HEAD
python -m pip install -e .

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

2026-08-20 的训练基线在上述九个文件中收集 49 项测试并全部通过；后续 runtime commit
另增加 export 和外部 cuRobo lifecycle 回归测试。测试通过证明静态合同和接线，不证明
模型收敛或 Isaac episode 成功。

4×H20 的工作根固定为 `/diff/wallx_workspace/dzb`。远端操作前必须先做非交互 SSH
探测，确认目标 worktree 干净、commit 精确一致、磁盘和四卡状态可用；不得 reset/clean
或占用其他任务的 GPU。

## 2. 构建无 state waypoint 数据

```bash
WAYPOINT_SOURCE_N200=/path/to/liangzhu_0815_n200
WAYPOINT_SOURCE_N400=/path/to/liangzhu_0815_n400
WAYPOINT_DATASET=/new/path/conveyorvla-waypoint-v1

python scripts/build_waypoint_dataset.py \
  --source-root "$WAYPOINT_SOURCE_N200" \
  --source-root "$WAYPOINT_SOURCE_N400" \
  --output-root "$WAYPOINT_DATASET" \
  --audit-only

python scripts/build_waypoint_dataset.py \
  --source-root "$WAYPOINT_SOURCE_N200" \
  --source-root "$WAYPOINT_SOURCE_N400" \
  --output-root "$WAYPOINT_DATASET"

python scripts/audit_waypoint_dataset.py \
  --dataset-root "$WAYPOINT_DATASET" \
  --output /new/run/waypoint_dataset_audit.json

python scripts/extract_waypoint_videos.py \
  --dataset-root "$WAYPOINT_DATASET" \
  --output-root /new/run/waypoint_review_clips
```

`--audit-only` 只列出 eligible source，不保留输出目录。materialize 拒绝覆盖已有目录。
正式运行前必须核对 [data.md](data.md) 记录的 schema、522 episodes、119,700 rows、
manifest hash 和 normalizer hash。review clip 只有 head+wrist，不得称为第三视角证据。

## 3. 四卡正式训练

```bash
WAYPOINT_DATASET=/path/to/immutable/waypoint-dataset
WAYPOINT_RUN=/new/path/to/formal-run

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file configs/accelerate_zero3_4gpu_waypoint.yaml \
  scripts/train_waypoint.py \
  --dataset-root "$WAYPOINT_DATASET" \
  --output-dir "$WAYPOINT_RUN/output" \
  --model-root /diff/wallx_workspace/dzb/models/base \
  --config configs/waypoint_v1.json \
  --max-steps 10000 \
  --batch-size 3 \
  --gradient-accumulation-steps 2 \
  --warmup-steps 200 \
  --save-first-checkpoint-step 20 \
  --save-interval-steps 500 \
  --log-interval-steps 1 \
  --num-workers 0 \
  --attention-implementation sdpa \
  --seed 20260820
```

四卡 effective global batch 为 `3 × 4 × 2 = 24`。训练从本地干净
Qwen3-VL-4B-Instruct 和两个全新随机 Layerwise FM head 开始：

- 不载入旧 action checkpoint；
- 不 resume 旧 optimizer/scheduler；
- 不使用 `scripts/train_hierarchical.py`；
- `--limit-train-rows` 只能用于明确标记的诊断/overfit run，正式 run 必须为 0。

训练入口会在 `resolved_run.json` 中绑定 commit/dirty state、完整 argv、配置 hash、
数据/normalizer hash、模型文件 hash、special token ID、batch 和环境。ZeRO checkpoint
另有 `waypoint_checkpoint_manifest.json`，不满足绑定时加载器必须拒绝。

## 4. 健康启动判定

至少观察 20 个连续 `train_step` event，并同时满足：

- step 连续且 `valid_optimizer_step=true`；
- total、answer、decision、active-route、NAV、ARM loss 和全部 learning rate 有限；
- Qwen、Navigation head、Manipulation head gradient norm 均有限且大于 0；
- 四个 rank 存活，四张 H20 均有真实计算利用率；
- checkpoint 已完整 commit，trainer state、四个 ZeRO model/optimizer shard 和 manifest
  都存在；
- 日志没有 traceback、OOM、NCCL、NaN/Inf 或提前退出。

`scripts/audit_training_events.py` 目前仍读取 legacy
`subtask_loss/action_loss/teacher_forcing_probability/routing` 字段，不能作为 Waypoint
v1 event 的自动门禁。2026-08-20 正式启动使用独立的严格 JSONL 检查完成 1–20 step
验证；后续若恢复自动化，应先扩展该脚本并加测试。

健康启动只说明训练过程正常。前 5% 进度 `lambda_self=0`，因此前 20 step 还没有
验证 self-conditioned route。

## 5. Checkpoint 与开环

ZeRO checkpoint 必须用相同四卡 accumulation 合同加载：

```bash
WAYPOINT_CHECKPOINT=/path/to/checkpoints/step_000020

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file configs/accelerate_zero3_4gpu_waypoint.yaml \
  scripts/check_waypoint_checkpoint.py \
  --checkpoint "$WAYPOINT_CHECKPOINT" \
  --report /new/run/checkpoint_report.json
```

开环评测同时检查自主 route 与 oracle-prefix waypoint/TCP 质量：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file configs/accelerate_zero3_4gpu_waypoint.yaml \
  scripts/evaluate_waypoint_open_loop.py \
  --checkpoint "$WAYPOINT_CHECKPOINT" \
  --report /new/run/open_loop_report.json \
  --split val \
  --profile diagnostic \
  --rows 40 \
  --batch-size 2 \
  --diffusion-seeds 17,29,43,71
```

`rows` 必须能被 `world_size × batch_size` 整除。`--profile overfit` 使用更严格的
route、RECOVER、ADE/FDE、方向、ARM pose、夹爪和 safety 阈值；失败报告必须保留，不能
改用 diagnostic 结果宣称通过。

正式全量 checkpoint 没有 `training_subset_indices`，因此不能伪装成 32-row overfit
checkpoint 运行 `--profile overfit`。step 1000 的实际协议是 val/diagnostic/40 rows/
四 diffusion seeds；它通过结构门禁，但动作误差和 violation rate 不具备闭环可用性。
准确数字见 [step 001000 评测](checkpoint_step1000_evaluation_20260821.md)。

## 6. 单卡 inference export 与服务

先把已绑定 ZeRO checkpoint 合并成不可变 inference export：

```bash
python scripts/export_waypoint_inference.py \
  --checkpoint "$WAYPOINT_CHECKPOINT" \
  --output-dir /new/path/waypoint-inference-export \
  --max-shard-size 5GB

CUDA_VISIBLE_DEVICES=0 python scripts/serve_waypoint.py \
  --export-dir /new/path/waypoint-inference-export \
  --device cuda:0 \
  --port 18081 \
  --seed 20260820
```

服务只监听 loopback，校验 export、processor、token ID、checkpoint、normalizer 和模型
合同。request 只能含完整指令、两路双帧图像和序列/标定身份；任何 state、phase、pose
truth 或 history 都应被拒绝。step 1000 已完成真实 consolidation 和单卡四图 request。
Qwen tied weights 必须由 Transformers 的 tied-weight 声明去重；不要手工复制或删除
共享 tensor key。

## 7. PCT/DWA、cuRobo 与 rollout

Navigation 参考 probe：

```bash
ARM_VLA_ROOT=/path/to/clean/arm-vla-grasp-sim-388b681

python scripts/probe_waypoint_navigation.py \
  --reference-root "$ARM_VLA_ROOT" \
  --report /new/run/navigation_probe.json \
  --server-python /path/to/reference/python
```

它在合成无障碍地图上运行批准的真实 PCT/DWA stack，要求 PCT fallback 明确关闭、
路径和 snap 合法、DWA 指令有限有界，并在首 waypoint 后 requery。

cuRobo 服务：

```bash
python scripts/serve_waypoint_curobo.py \
  --reference-root "$ARM_VLA_ROOT" \
  --workspace-root /path/to/arm-vla-runtime-assets \
  --curobo-source-root /path/to/clean/curobo-8726021 \
  --host 127.0.0.1 \
  --port 8766 \
  --ready-json /new/run/curobo_ready.json
```

`--reference-root` 是干净 arm-vla 代码 checkout；`--workspace-root` 是 robot/scene/config
运行资产根；`--curobo-source-root` 是干净 cuRobo checkout。三者不得再指向同一路径来
碰巧通过 import。服务 ready 后应先跑 known-pose gate，核对 reference commit、direct
absolute TCP capability、两个 frame、world collision、orientation fallback=false、
reachable/collision-free 和终端 pose error。

模型服务与 cuRobo 服务均 ready 后，`run_waypoint_rollout.py` 接管批准的 arm-vla
full-physics pipeline：

```bash
python scripts/run_waypoint_rollout.py \
  --reference-root "$ARM_VLA_ROOT" \
  --model-endpoint http://127.0.0.1:18081 \
  --curobo-port 8766 \
  --max-queries 400 \
  --max-control-steps 24000 \
  -- \
  <run_full_physics_pipeline.py 的批准场景、seed、资产和输出参数>
```

字面量 `--` 之后只能放批准 reference pipeline 的参数；wrapper 禁止外部 route gate、
`--remote-vla-eval`、dry-run/navigation-smoke 重写和额外 FSM。分阶段诊断可用
`--stop-after-route` / `--required-first-route`，但不能当作完整自主闭环。

rollout 不再启动 reference pipeline 的 legacy cuRobo 服务。它只复用指定 port 上已经
ready 的 Waypoint 服务，并逐项校验 capability；身份或 frame 不匹配立即 fail-closed。

step 1000 的真实 cuRobo known-pose 已通过；完整自主 Isaac 测试也已执行并生成三路视频，
但首个预测 NAV chunk 因 yaw segment 超限而安全失败。oracle-route 与四阶段 staged
rollout 仍未完成。测试链可运行不等于完整 episode 成功。

## 8. 2026-08-20 正式 run 记录

| 项目 | 值 |
|---|---|
| host / work root | `4xH20 (VM-0-3-ubuntu)` / `/diff/wallx_workspace/dzb` |
| source | `feature/conveyorvla-waypoint-v1@724ead21be2c27d9b40c200375ee4ab49ccedc84`，clean |
| worktree | `worktrees/ConveyorVLA-waypoint-v1-fe2b4ea` |
| dataset | `datasets/derived/conveyorvla-waypoint-v1-full-8fcccd9` |
| run | `runs/conveyorvla-waypoint-v1-formal-724ead2-s10000-20260820T1813` |
| tmux | 已按用户指令停止；原 session `cvla-wp-formal-724ead2-s10000` 不再存在 |
| GPUs | 4 × NVIDIA H20 |
| environment | `.conda-envs/conveyorvla-al0-lerobot044` |

step 1–1181 均产生有效训练事件；最后完整 checkpoint 为 step 1000。2026-08-20
23:07:20 CST 后由用户授权 Ctrl-C 暂停，日志尾部 `KeyboardInterrupt`/signal 2 是主动
停止证据。当前远端没有训练、模型服务、cuRobo 服务或 rollout 进程占用 GPU。

step 1001–1181 没有 checkpoint。现行 `train_waypoint.py` 没有 optimizer-resume CLI；
不要把 load/eval 通过描述为续训已验证。若以后新增 resume，必须保持相同数据、batch、
ZeRO 和 scheduler binding，并用新目录记录 parent checkpoint。后续新训练的 checkpoint
默认/显式间隔均为 500 effective optimizer steps。

## 9. ConveyorBench 采集链

动态传送带/三视角 canonical 采集仍由 `collect.py → validate.py → audit_episode.py →
check_camera_gate.py → export.py → convert_dataset.py` 管理。它用于生成可审计 raw 和
旧 LeRobot 数据，不是本轮 0815 Waypoint 正式训练来源。采集的任务、物品、速度、相机
和成功定义见 [benchmark.md](benchmark.md)；旧 state28 训练/服务命令只从 Git 历史或
对应历史文档复现，不能写入本节现行命令。

## 10. 中断与恢复

- 不覆盖已有 run、checkpoint、dataset 或 export；
- 失败/中断后保留日志、events、resolved run 和已 commit checkpoint；
- 当前 Waypoint 训练入口不支持 optimizer resume；新增该能力前不得用手工 state 拼接冒充；
- 若将来实现 resume，先运行 checkpoint binding/load gate，并记录 parent run；
- 不删除 source raw、已发布 episode 或远端 evidence；
- 不用 `git reset`、`git clean`、stash 或 force-push 处理远端差异；
- 只停止可精确识别为本任务启动的进程/tmux，其他进程不得控制。
