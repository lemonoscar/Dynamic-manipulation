# 采集、训练与测评操作

本文命令默认从仓库根目录执行。远端工作根为
`/diff/wallx_workspace/dzb`，本项目实验只允许使用物理 GPU 2/3。

远端只保留以下运行布局：

```text
ConveyorVLA/                         代码仓库
assets/conveyorvla-v3/              SSH 交付的 3DGS 与物品资产
datasets/conveyorvla-al0-grasp-v1/  392 条 LeRobot v3 基线数据
models/base/                         Qwen3-VL、ABot-M0 与配置登记的 VGGT
models/conveyorvla-al0/              已训练动作头、配置和统计量
results/joint-smoke-r23/             最新完整移动教师成功证据
dynamic-isaaclab-5.1-20260804/       不可搬移的 Isaac 运行环境
.conda-envs/conveyorvla-al0-lerobot044/  LeRobot 0.4.4 环境
workspace-manifest/                  清理与保留清单
```

## 1. 环境预检

```bash
conda activate /diff/wallx_workspace/dzb/dynamic-isaaclab-5.1-20260804/envs/conveyor_py311
python -m pip install -e .
python scripts/check_environment.py
```

纯逻辑测试：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider
```

## 2. Sidecar 资产

资产通过 SSH 放到服务器，不提交 Git。当前经过哈希校验的目录：

```bash
export CONVEYOR_BENCH_ASSET_ROOT=/diff/wallx_workspace/dzb/assets/conveyorvla-v3
python scripts/validate_assets.py \
  --asset-root "$CONVEYOR_BENCH_ASSET_ROOT" \
  --allowed-root /diff/wallx_workspace/dzb
```

正式采集不能使用 `--metadata-only`。全量 SHA-256 校验必须通过，且不能出现 symlink、
缺失文件或额外未登记成员。

## 3. 场景与移动探针

在采集前分别验证场景和 locomotion：

```bash
python scripts/probe_scene.py \
  --asset-root "$CONVEYOR_BENCH_ASSET_ROOT" \
  --output-dir outputs/probe_scene \
  --enable_cameras --headless

python scripts/probe_mobile_locomotion.py --help
```

检查 head、wrist、overview 是否清晰、目标和传送带是否完整可见、动态物体与背景遮挡
是否正确。Navigation gate 至少检查净位移、航向、驻车速度和 reset 稳定性。

## 4. 采集 dry-run

dry-run 会解析资产、路径、seed、速度、GPU 和最终 Isaac 命令，但不启动仿真：

```bash
python scripts/collect.py \
  --asset-root "$CONVEYOR_BENCH_ASSET_ROOT" \
  --output-root outputs/pilot \
  --physical-gpu 2 \
  --robot-mode whole_body_policy \
  --episodes 1 \
  --seed 1101 \
  --belt-speed 0.01 \
  --dry-run
```

当前限制：

- `--physical-gpu` 只能是 2 或 3；
- 单进程 `episodes` 在 1–8；
- 当前允许速度为 `0` 或 `0.01 m/s`；
- dynamic 默认目标为 `cola`；
- stationary 只接受预注册 seed。

## 5. Pilot

先在 GPU 2/3 各运行一个小批次，seed 不得重叠：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/collect.py \
  --asset-root "$CONVEYOR_BENCH_ASSET_ROOT" \
  --output-root outputs/pilot/gpu2 \
  --physical-gpu 2 --robot-mode whole_body_policy \
  --episodes 2 --seed 1101 \
  --belt-speed 0.01 --require-all-success

CUDA_VISIBLE_DEVICES=3 python scripts/collect.py \
  --asset-root "$CONVEYOR_BENCH_ASSET_ROOT" \
  --output-root outputs/pilot/gpu3 \
  --physical-gpu 3 --robot-mode whole_body_policy \
  --episodes 2 --seed 1201 \
  --belt-speed 0.01 --require-all-success
```

`CUDA_VISIBLE_DEVICES` 与 Kit 的 `activeGpu` 语义取决于服务器启动环境；正式运行前要用
`nvidia-smi` 和 Kit 日志证明进程确实落在授权物理卡。不要杀死、占用或修改 GPU 0/1
上的外部任务。

Pilot 通过标准：

- 所有请求 seed 都发布且成功；
- `collection_report.json` 的 eligible 数与请求数一致；
- 没有 `.inprogress`；
- validator、audit、camera gate 和 export 全部通过；
- Kit/stdout 无 traceback、OOM、CUDA error 或 fatal；
- 人工检查至少一条完整三相机视频。

## 6. 单条数据复核

```bash
python scripts/validate.py outputs/pilot/gpu2/raw
python scripts/audit_episode.py EPISODE_ROOT
python scripts/check_camera_gate.py EPISODE_ROOT \
  --output EPISODE_ROOT/camera_gate_report.json
python scripts/export.py EPISODE_ROOT --profile all --force
```

命令成功不代替视觉检查。需要观察机器狗先走到传送带并驻车、机械臂跟随缓降抓取、
垂直提起后直线后退、原地转向蓝框、负载导航、再次驻车、俯视放入并松爪。开始放置后
底盘不得再次移动。`carry_backoff` 或 `carry_navigate` 为零帧、三段位移低于门禁，
放置阶段出现非零底盘动作，或实际漂移超过 `0.05 m` 时，即使旧评分器给出成功也不得
进入训练。

## 7. 转换到 LeRobot v3

使用独立 Python 3.10 + `lerobot==0.4.4` 环境：

```bash
conda activate /diff/wallx_workspace/dzb/.conda-envs/conveyorvla-al0-lerobot044
python scripts/convert_dataset.py \
  --episode-list outputs/pilot/gpu2/successful_episode_roots.txt \
  --episode-list outputs/pilot/gpu3/successful_episode_roots.txt \
  --output-root outputs/lerobot_pilot
```

先使用 `--max-episodes 1` 做 smoke，再运行完整转换。转换后检查四个视频 feature 的
首帧，并抽查首/中/末 episode。

PCT Liangzhu 的 `liangzhu_0815_n200` 与 `liangzhu_0815_n400` 使用单独入口；raw
目录保持只读：

```bash
python scripts/audit_pct_source_overlap.py \
  --source-root DATASETS/liangzhu_0815_n200 \
  --source-root DATASETS/liangzhu_0815_n400 \
  --output RUNS/liangzhu_0815_source_overlap.json

python scripts/convert_pct_dataset.py \
  --source-root DATASETS/liangzhu_0815_n200 \
  --source-root DATASETS/liangzhu_0815_n400 \
  --require-hierarchy-eligible \
  --audit-only

python scripts/convert_pct_dataset.py \
  --source-root DATASETS/liangzhu_0815_n200 \
  --source-root DATASETS/liangzhu_0815_n400 \
  --require-hierarchy-eligible \
  --output-root outputs/lerobot_pct_pilot \
  --max-episodes-per-source 4
```

适配器只接收成功、执行来源可验证、双相机同步且采用 50 Hz 控制/5 Hz 图像时钟的
episode。PCT 状态记录是稀疏控制结点，因此 25 Hz 监督由结点间位置线性插值、四元数
最短弧归一化插值和控制命令零阶保持重建；转换 manifest 会明确记录这一事实。双帧
历史为真实的 `[-5, 0]` model tick（0.20 秒），不能误写成原生采集的 0.08 秒。

## 8. 训练

Liangzhu seen 四阶段训练必须使用 `scripts/train_hierarchical.py`。同一个 Qwen3-VL
依次执行“生成当前 subtask 文本”和“带预测文本重新编码原始观测”两次 forward，
Qwen 主干、LM head、Navigation DiT 与 Manipulation DiT 全部参与反向传播。主 prompt
在训练与推理时都不接收真实或预测的 semantic history；teacher forcing 只影响动作
专家路由，并按配置衰减到 0。旧的 `scripts/train.py` 是冻结 Qwen 的单动作头路径，
不得用于本合同的正式训练。

正式训练从干净的本地 Qwen3-VL 与发布的 ABot action 权重初始化，不恢复旧
optimizer/scheduler，也不使用旧 hierarchy view：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file RUNS/accelerate_zero3_4gpu_accum1.yaml \
  scripts/train_hierarchical.py \
  --hierarchy-root DATASETS/NEW_DENSE_TRANSITION_VIEW \
  --output-dir RUNS/NEW_FORMAL_RUN/output \
  --model-root /diff/wallx_workspace/dzb/models/base \
  --initial-action-checkpoint \
    /diff/wallx_workspace/dzb/models/conveyorvla-al0/action_model_final.safetensors \
  --max-steps 10000 \
  --warmup-steps 200 \
  --save-first-checkpoint-step 25 \
  --save-interval-steps 1000 \
  --log-interval-steps 1 \
  --batch-size 64 \
  --gradient-accumulation-steps 1 \
  --teacher-forcing-full-steps 100 \
  --teacher-forcing-end-step 4000 \
  --attention-implementation sdpa
```

该命令在四卡上的 global batch 为 `64 × 4 × 1 = 256`。动作物理尺度从新 view 的
train split P99.9 统计派生并冻结在 `configs/temporal.json`，线上 action composer
读取同一配置。Navigation DiT 仍只输出 `[vx,wz]`；composer 显式补齐
`stow + open` 或 `carry + closed` 的关节/夹爪命令。

健康启动至少要求：

- 四个 rank 都初始化完成；
- 数据帧数、state/action 维度和统计量正常；
- checkpoint transfer 报告无缺失/意外 key；
- 连续至少 20 个 step 的 loss、gradient norm 与 learning rate 有限且 step 递增；
- Qwen、Navigation DiT 和 Manipulation DiT 都有非零梯度；
- GPU 0/1/2/3 有真实计算利用率；
- tmux、事件日志、run metadata 与可加载 checkpoint 都存在。

## 9. 服务与闭环

推理服务：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/serve.py \
  --action-checkpoint CHECKPOINT.safetensors \
  --state-statistics state_statistics.json \
  --model-root /diff/wallx_workspace/dzb/models/base \
  --device cuda:0 \
  --port 18080
```

闭环测评：

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py \
  --endpoint http://127.0.0.1:18080 \
  --state-statistics state_statistics.json \
  --asset-root "$CONVEYOR_BENCH_ASSET_ROOT" \
  --episodes 10 \
  --belt-speed 0.01 \
  --output-dir outputs/eval/al0 \
  --enable_cameras --headless
```

时序训练产物旁的 `conveyorvla_al0_config.json` 会由 runtime 自动发现，用于 20-step
动作头和 PCT 归一化；服务端按连续 `sequence_id` 缓存上一组 5 Hz head/wrist 图像，
首个请求复制当前帧，随后严格使用 0.20 秒历史。在线协议执行前 16 行兼容动作前缀，
训练目标仍保留完整 20×10 动作块。

assist 参数只用于诊断，不得用于正式成功率或训练数据。

## 10. 何时开启 384 条正式采集

不要直接把 pilot 命令循环 384 次。先完成：

1. 四个物品的 sidecar fixture；
2. 两档动态速度的允许配置和教师门禁；
3. 8 个 cell 的独立 seed 池和可恢复总账；
4. GPU 2/3 双 worker pilot；
5. LeRobot round-trip；
6. 每 cell 至少连续多条成功。

正式协调器应以“48 条 training-eligible 成功”为 cell 终点，而不是 48 次尝试。
任务失败从预留 seed 补采，结构失败立即停止对应 worker。

## 11. 中断与恢复

- 不删除已发布 episode；
- `.inprogress` 保留作故障证据，复核后移到隔离区；
- 根据 manifest 中 seed 续跑，不靠目录数量推算；
- 一个输出根只由一个 coordinator 拥有；
- 重启前记录 commit、配置哈希、资产 manifest 哈希、环境和 GPU；
- 数据格式或 teacher 改动后必须开新 collection root。
