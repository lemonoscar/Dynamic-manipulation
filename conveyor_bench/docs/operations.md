# 采集、训练与测评操作

本文命令默认从 `Dynamic/conveyor_bench` 执行。远端工作根为
`/diff/wallx_workspace/dzb`，本项目实验只允许使用物理 GPU 2/3。

## 1. 环境预检

```bash
conda activate env_isaaclab
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
export CONVEYOR_BENCH_ASSET_ROOT=/diff/wallx_workspace/dzb/conveyorvla-v3-assets-20260811
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
收臂后负载导航到分类箱、再次驻车、俯视投放以及释放后进入指定框。`carry_navigate` 为零帧或
两段位移低于数据门禁时，即使旧评分器给出成功也不得进入训练。

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

## 8. 训练

模型资产根需要包含 `configs/model.json` 登记的 Qwen3-VL 与基线检查点。训练默认冻结
Qwen3-VL，适配 DiT 动作模型。

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 scripts/train.py \
  --lerobot-root outputs/lerobot_pilot \
  --output-dir outputs/checkpoints/al0_pilot \
  --model-root /diff/wallx_workspace/dzb/models/conveyorvla-al0 \
  --belt-speed 0.01
```

健康启动至少要求：

- 两个 rank 都初始化完成；
- 数据帧数、state/action 维度和统计量正常；
- checkpoint transfer 报告无缺失/意外 key；
- loss 有限并持续更新；
- GPU 仅为 2/3；
- `training_report.json` 和 safetensors checkpoint 可原子写入。

## 9. 服务与闭环

推理服务：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/serve.py \
  --action-checkpoint CHECKPOINT.safetensors \
  --state-statistics state_statistics.json \
  --model-root /diff/wallx_workspace/dzb/models/conveyorvla-al0 \
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
