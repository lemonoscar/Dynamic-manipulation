# Formal Joint-Trajectory 5 Hz 评估

本入口固定正式训练的最终 checkpoint。依赖及本机资源安装约定见[开发指南](joint_trajectory_development_guide.md#6-闭环资源和复现边界)，最新公开快照见[实验卡](experiments/formal_5hz_20260905.md)。协议将训练拟合诊断、模型自产路由动作和实际闭环分开；测试结果不会自动成为业务达标结论。

## 开环

```bash
repo="$(pwd)"
: "${RUN_ROOT:?设置已有正式训练运行目录，包含最终 checkpoint 和校验清单}"
: "${MODEL_ROOT:?设置本地基础模型根目录}"
: "${EVAL_GPU:?设置已确认空闲的开环 GPU 编号}"
run="$RUN_ROOT"
python_eval="${PYTHON_EVAL:-python}"
# 新复现使用独立目录，不覆盖现有正式评估
eval_root="${EVAL_ROOT:-$run/evaluation/reproduction-v1}"

CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$python_eval" "$repo/scripts/evaluate_joint_trajectory_formal.py" \
  --checkpoint "$run/checkpoints/step_002414" --model-root "$MODEL_ROOT" \
  --output-dir "$eval_root/validation_full_v1" --batch-size 16

CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$python_eval" "$repo/scripts/evaluate_joint_trajectory_formal.py" \
  --checkpoint "$run/checkpoints/step_002414" --model-root "$MODEL_ROOT" --split test \
  --freeze-test-from "$eval_root/validation_full_v1/report.json" \
  --output-dir "$eval_root/test_full_v1" --batch-size 16
```

- 用 `--max-rows 16` 在独立目录做 validation 分层 smoke；该结果不允许用于解锁 test。
- 用 `--preflight-only` 校验绑定及样本清单；它不代表模型推理成功。
- 原命令在同一输出目录重跑可以续接 JSONL 的完整前缀。批大小、seed、样本清单或相关源码变化会拒绝续跑。
- `--seeds 17 29 43` 做重复采样时，各 seed 必须全部计入，不能挑选最好结果。
- test 要求全量 validation 已完成，相关源码、checkpoint、配置、batch size、seed、指标协议全部相同。检查在打开 test 样本文件之前执行。
- `strict_load.json` 是本次真正构建和严格加载模型的记录；`preflight.json` 只有静态绑定证据。
- `report.json` 的 `status=complete` 表示所选推理与汇总完成；性能门槛单独见 `metrics.saturation_gate`，不能把运行完成解释为性能通过。

2026-09-05 运行有一项明确的统计修正：在不改模型推理和逐样本轨迹指标的前提下，将Wilson边界包络限定为二值观测。`evaluation/freeze_validation.py` 会验证所有其他代码哈希相同、AST仅`cluster_mean`函数变化，保留原始`report.json`/推理源码快照/rows SHA-256，并生成`report_frozen.json`作为本次test入口。全新运行当前版本则可直接使用其`report.json`冻结。不要把原始推理源码身份改写成新身份，也不要在原始运行目录用改变后的代码直接续跑。

`rows.jsonl` 保存路由概率、subtask、错误原因、原始物理单位轨迹、逐样本指标、episode 和 diffusion seed。`predicted` 只在动作域可比较时存在；跨域错误仍在路由和覆盖率总分母中。`oracle` 使用真实 route + canonical subtask，是条件诊断。`baseline` 是零位移/保持当前夹爪状态。真实未来点与 terminal-hold 点分别统计。

95% 区间以 episode 为聚类单位，在每个 episode 内合并重复 seed，再进行 bootstrap；不把相邻帧作为独立试验。只有一个 episode 时不提供区间。所有 episode 的二值结果均为0或1时，用 episode 数量计算 Wilson 边界包络，避免输出误导性的 [1,1]。总体帧加权均值与 episode 等权均值都保存。

饱和合约沿用 `(position events + rate events + gripper events)/(samples×10×7)`；同一位置可能被重复计数，该事件率可能超过1。分别报告各事件率、唯一受影响比例及真实标签裁剪率。0.005 门槛不随结果调整。clip事件按现有执行器的数值精度判定，小量越界也会计入。

## 服务

```bash
: "${SERVICE_GPU:?设置已确认空闲的服务 GPU 编号}"
CUDA_VISIBLE_DEVICES="$SERVICE_GPU" "$python_eval" "$repo/scripts/serve_joint_trajectory.py" \
  --checkpoint "$run/checkpoints/step_002414" \
  --model-root "$MODEL_ROOT" --port 18082
```

formal 不传 `--weights`，只严格加载绑定的 `model.safetensors`。`/health` 提供实际 step、权重/数据/normalizer/config 哈希、源码身份、10点及0.2s合约。`/infer` 仅接受相机、语言、操作状态和请求元数据；`diffusion_seed` 用于重复实验。服务只监听本机回环地址。一个实例串行服务一个活动 episode，不能让多个 rollout 交错调用同一实例。

## 迁移闭环

```bash
"$python_eval" "$repo/scripts/run_formal_closed_loop.py" prepare \
  --checkpoint "$run/checkpoints/step_002414" --split val \
  --output-dir "$eval_root/closed_validation_protocol" --seeds 17 29 43

"$python_eval" "$repo/scripts/run_formal_closed_loop.py" run \
  --manifest "$eval_root/closed_validation_protocol/manifest.json" \
  --output-dir "$eval_root/closed_validation" --isaac-device cuda:0
```

先使用 `prepare --limit 1 --seeds 17` 和 `run --limit 1 --max-queries 3` 做 smoke。query limit 结束不表示这个任务在完整预算内必然失败。完整协议默认50个固定源任务×3 seeds×2物理条件；按任务聚类汇总。实际启动前检查GPU占用，不能停止其他任务腾出显存。

准备器验证数据发布、source manifest 和所用 tar 的SHA-256，只提取任务及summary JSON；保持已经采样的起点、物体、两桌、目标区域和全局指令。它移除会覆盖这些几何的旧 annotation 引用，将机器人/物体资产绑定到本地副本，保留旧运行时识别的 visual/collision fallback token，由第二阶段绑定器解析真实资源。录视频必须有 overview，模型视觉显式为 full，传感器渲染间隔为10个50 Hz控制tick。

两组均禁止 episode 中途物体重置，保留并披露操作时的底座/支撑锁定：

- `source_assisted`：测得闭合、抬升≥4cm、距TCP≤8cm且速度≤0.30m/s后，创建固定抓取关节；连续夹爪目标首次打开前移除它。这是源辅助机制的迁移实现，闭合证据改用了连续目标及测量，不能宣称与源 binary-counter 核验完全一致。
- `no_grasp_assist`：同样记录物理事件，但从不创建固定抓取关节。

两个条件都不是完全自由底盘的纯物理实验。接触证据采用几何代理，不是接触传感器实测。原合约 released + target区域1s的成功率与包含pick、carry、release、无drop的事件链成功率分别保存。阈值及源语义差异在 `physics_evidence` 中显式记录。

推理等待期间仿真时钟停止。报告分别列出观测/动作点5 Hz、低层控制50 Hz、重规划率、墙钟推理延迟。此实验不能证明实时5 Hz部署能力。

每个attempt保留命令、完整日志、summary、视频与trace；异常不从分配总数中消失。`query_count`计数收到并通过请求身份校验的模型响应；零计数可能是启动失败或首请求未完成，不能据此断言服务端从未计算，也不能作为模型0%成功率的依据。失败类别及请求输入trace需共同解释。任何未完成attempt都需要先检查原进程，禁止盲目重复执行。

模型每次真正接收的四张JPEG、13维状态及请求元数据独立保存并记录哈希。正式视频以实际5Hz采样率编码；早期pilot仍保留原25fps视频，不能按其回放时长推断仿真时长。墙钟超时若没有summary，会从已flush的trace恢复已完成请求、裁剪计数及抓起/运输/释放事件，标为partial evidence，不自动记成零请求启动失败，也不补造整任务成功。汇总同时输出分阶段物理事件率、95%区间及saturation gate。

闭环启动器和长期服务可以具有不同的源码快照：manifest 的 `runner_source_sha256` 绑定启动器代码，`identity.source_sha256` 绑定服务启动时的代码。两者分别验证，不能通过删除identity gate强行复用错误权重或配置。

完整闭环使用test任务时，prepare必须传 `--freeze-validation <完整冻结validation报告>`；它会检查完成状态、权重和开环代码身份。场景、依赖资产、PCT地图及运动控制器权重也在准备时记录哈希，运行前复核。

以下仅说明原运行的本机管理材料，它们位于外部训练目录，未作为公共脚本分发；新复现不依赖这些辅助文件。原运行使用持久 tmux 会话，见 run 的 `evaluation/persistent_sessions.json`。逐步状态见 `open_loop_sequence_status.json` 和 `closed_loop_sequence_status.json`。根据完整pilot耗时估计，300次闭环可能需要约37–53小时；未完成前不能报告全测试集成功率。

只读查看本次进度：`python3 "$run/evaluation/status.py"`。开环结束自动生成`OPEN_LOOP_RESULTS.md`及PNG；闭环结束自动生成`CLOSED_LOOP_RESULTS.md`。运行中的逐attempt汇总在`closed_test_full_v1/report.json`，其`status=running`时统计只代表已完成子集。
