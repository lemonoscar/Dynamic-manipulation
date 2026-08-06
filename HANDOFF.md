# Dynamic Manipulation handoff — 2026-08-06

> 2026-08-06 命名更新：当前策略基线的正式名称为 **ConveyorVLA AL0**。
> 本文早期记录中的 M0/M1/M2、`m0_*` 路径和 schema 是历史实验或冻结兼容标识。
> 新操作入口使用 `train_conveyorvla_al0.py`、`serve_conveyorvla_al0.py` 和
> `run_conveyorvla_al0_closed_loop.py`。

## 仓库状态

- 本地仓库：`/home/lemon/research/Issac/Dynamic`
- 分支：`main`
- 远端：`https://github.com/lemonoscar/Dynamic-manipulation.git`
- 当前 production 运行提交：`d7b6f0963bef0864b8101571981b7d02e40c3122`
- 本文档对应的正式提交和远端状态以 `git log -1`、`git status -sb` 为准；全程
  直接推送 `main`，没有 PR。

## 最新状态：ConveyorVLA AL0 temporal v1 已正式采集

本节是当前操作真值；后文 2026-08-03 至 2026-08-05 的服务、旧矩阵和进程信息
只保留为历史证据，不再作为运行指令。

DynamicVLA 的代码级调研已经转化为 AL0 的兼容增量：head/wrist 各使用
`[t-2,t]` 双帧，当前 `state28` 对齐未来 `20×10@25 Hz` 独立动作目标；TCP 采用
观测时刻 root/TCP 下可跳行的 SE(3) 目标，在线 buffer 用 episode/generation/tick
拒绝旧块并只保留最新结果。旧 `m0_*` schema、checkpoint key 和三种 legacy
profile 没有改写；结构只有在闭环相对旧 AL0 确认提升后才允许命名为 AL1。

代码与文档已通过以下门禁：

- 本地完整测试：431 passed；
- 远端完整测试（首个正式提交）：430 passed；生产运行修正后定向测试：12 passed；
- G1 单回合在物理 GPU 3 成功：976 control steps、1464 PNG、300 temporal records；
- G2 8-cell pilot：8/8 success、8/8 fully gated、2438 temporal records；
- G3 首批 production：16/16 success、16/16 fully gated；四 profile 共 64 个哈希
  全部独立复核；另加载 9 个真实首/中/末时序 sample，均为 `state28`、
  `20×10` action、head/wrist 共四张 `224×224 RGB` 图。

### 远端冻结身份

```text
SSH alias:       4xH20
allowed root:    /diff/wallx_workspace/dzb
source root:     /diff/wallx_workspace/dzb/conveyorvla-al0-34318e3-20260806-r1
source commit:   d7b6f0963bef0864b8101571981b7d02e40c3122
source tree SHA: 4b353a1bd247c913daa096a762c2e54ad5a0a3af168f65151c44948cc0245a2e
asset lock SHA:  3351a6cf3ef7bb65fcd44245541c8cd044d5fb3e65434b18ebfb9ee488b2e075
environment:     /diff/wallx_workspace/dzb/dynamic-isaaclab-5.1-20260804
run root:        /diff/wallx_workspace/dzb/conveyorvla-al0-runs-d7b6f09-20260806
dataset root:    /diff/wallx_workspace/dzb/conveyorvla-al0-runs-d7b6f09-20260806/datasets/grasp-temporal-v1-d7b6f09-r1
tmux session:    al0-production-d7b6f09-r1
coordinator log: /diff/wallx_workspace/dzb/conveyorvla-al0-runs-d7b6f09-20260806/logs/production-coordinator.log
exit marker:     /diff/wallx_workspace/dzb/conveyorvla-al0-runs-d7b6f09-20260806/logs/production-coordinator.exit
```

生产命令固定为：

```bash
cd /diff/wallx_workspace/dzb/conveyorvla-al0-34318e3-20260806-r1/conveyor_bench
PY=/diff/wallx_workspace/dzb/dynamic-isaaclab-5.1-20260804/envs/conveyor_py311/bin/python

PYTHONDONTWRITEBYTECODE=1 "$PY" -B scripts/collect_conveyorvla_al0_grasp.py \
  --phase production \
  --output-root /diff/wallx_workspace/dzb/conveyorvla-al0-runs-d7b6f09-20260806/datasets/grasp-temporal-v1-d7b6f09-r1 \
  --python "$PY" \
  --isaaclab-source /diff/wallx_workspace/dzb/dynamic-isaaclab-5.1-20260804/source/IsaacLab/source/isaaclab \
  --kit-cache-root /diff/wallx_workspace/dzb/dynamic-isaaclab-5.1-20260804/kit-portable/cache \
  --runtime-library-dir /diff/wallx_workspace/dzb/dynamic-isaaclab-5.1-20260804/system-libs/root/usr/lib/x86_64-linux-gnu \
  --physical-gpu 2 \
  --physical-gpu 3 \
  --workers 2
```

`2026-08-06T12:01:22Z` 的稳定快照为 26 条 production、26 success、0 failure、
16 fully gated；第二批 seed `200009…` 与 `201009…` 已分别在物理 GPU 2/3 自动
接续，两个进程各约 3.7 GB 显存。GPU 0/1 仍是用户已有的 ABot-M0.5 进程
PID 14877/14885，本任务没有修改或复用。采集锁为 coordinator PID 159863，tmux
pane 存活，没有 exit marker；全部日志扫描为 0 条 traceback/OOM/runtime fatal。

首个双 worker 批次的 16 条仿真加门禁约 11–12 分钟。384 条无失败投影约
4.8 小时，保守预计从 2026-08-06 19:44 CST 启动后 5–6 小时完成；预留 seed
被消耗时可能延长到约 7.5 小时。按实测均值，最终 pilot 加 production 约
55–60 GB。这里只声明正式采集已稳定启动，不声明 384 条已经完成。

### 监控与恢复

```bash
DATA_ROOT=/diff/wallx_workspace/dzb/conveyorvla-al0-runs-d7b6f09-20260806/datasets/grasp-temporal-v1-d7b6f09-r1
tmux has-session -t al0-production-d7b6f09-r1
find "$DATA_ROOT/production" -type f -name summary.json | wc -l
find "$DATA_ROOT/production" -type f -name quality_report.json | wc -l
find "$DATA_ROOT/production" -type f -name camera_gate_report.json | wc -l
find "$DATA_ROOT/production" -type f \
  -path '*/exports/conveyorvla_al0_temporal.jsonl' | wc -l
find "$DATA_ROOT/production" -type d -name '*.inprogress' -print
nvidia-smi
```

总账按完整双 cell wave 原子更新，所以一个 cell 尚未达到 48 条时可能落后于即时
目录计数。锁 PID 存活或 tmux 仍在时，绝不能启动第二个 coordinator 或删除锁。
若意外中断，先确认精确 PID 已退出，再审计 orphan、`.inprogress`、源码/资产
指纹和已发布 seed；只有全部一致时才使用上述同一命令和同一 dataset root 恢复。
机器可读启动证据在
`conveyor_bench/docs/conveyorvla_al0_collection_launch_20260806.json`。

## 已完成的主体工作

新增了正式的 V1 `stationary_sort` 诊断：传送带命令与逐帧实测速度均为
`0.0 m/s`，完整执行抓取、携带、投放和 `0.5 s` 稳定判据，但不计入动态
benchmark 分数。固定合同为 `single_target`、一个 `part_red_block`、目标
`sort_bin_blue`，只允许以下场景：

| split | seeds |
| --- | --- |
| train | 1101, 1102, 1103 |
| val | 2101 |
| test | 3101 |

运行时已同步物体 spawn X/Y、静态 intercept、staging、oracle 和任务描述；配置
位于 `conveyor_bench/configs/v1.json`。协议、strict validator、AL0 exporter、AL0
dataset 和训练 CLI 均已加入静态任务支持。训练接口目标用法：

```bash
python conveyor_bench/scripts/train_conveyorvla_al0.py \
  --episode-root TRAIN_EPISODE_ROOT \
  --state-statistics TRAIN_STATE_STATISTICS \
  --initial-action-checkpoint ACCEPTED_M1_ACTION_MODEL \
  --model-root LOCAL_MODEL_ROOT \
  --output-dir NEW_EXPERIMENT_ROOT \
  --belt-speed 0 \
  --task-type stationary_sort
```

另新增默认关闭的 `--mobile-approach-assist`，只用冻结 service command 完成
`mobile_approach`，物体生成前抑制 AL0 请求，用于隔离判断静态抓取 primitive。
它是诊断功能，不能把 assisted 回合计作 policy-only 成功或训练数据。

## 已生成并验收的静态 oracle 数据

这些输出位于被 `.gitignore` 排除的 `conveyor_bench/outputs/`，canonical 数据不要
手工修改：

- train：`outputs/gate/v1_stationary_train_oracle_final_1101_1103`
  - 3/3 success
  - 2906 control steps，4356 张三相机 PNG
  - 1428 条 AL0 legacy-profile train records
- val：`outputs/gate/v1_stationary_val_oracle_2101`
  - 1/1 success，957 steps，1434 PNG，470 records
- test：`outputs/gate/v1_stationary_test_oracle_3101`
  - 1/1 success，966 steps，1449 PNG，475 records

五条 episode 均已通过 strict validator、quality audit 和 temporal camera gate；
实测最大带速为零，无跌倒和禁区碰撞。详细 episode ID、导出 SHA 和在线证据在
`conveyor_bench/docs/m0_stationary_followup_20260803.json`。

## ConveyorVLA AL0 结论

接受的基线仍是 M1：

- action SHA：`2f6c10a55f857cab198daa1886be0d8b2df5fbb7f93d3e8c648df3f3bd795024`
- state statistics SHA：`29bc9a04a9c0eb03947e21e3fb752959c3d6841097df52cbb001acb5558d0e66`
- training report SHA：`ac80d39d764412a4782ac94988e4f73c332869d83e6389682f507ac36eef8a02`
- 1183 records，1200 steps

静态 policy-only seed 1101 在 `mobile_approach` 超时，没有进入机械臂 full action。
只辅助 approach 的干净隔离回合证明 M1 确实能完成闭夹、112 个连续双侧持有
step、约 `2.24 s` 持有和约 `0.18264 m` 抬升；但它在 `7.38 s` 主动开爪，未完成
carry/place。canonical failure reason 是 `runtime_error`；对应 `summary.json` 的
`metrics.abort_metadata` 已持久化 pregrasp `IKConvergenceError` 及误差，因此它是
可复核的诊断原因，但不替代顶层标准 failure reason。

M2 `m2-carry-retract-step800-20260803-2100` 已拒绝：离线 grasp transition 从
M1 的 `1/3` 退化为 `0/3`，在线仍在 approach 超时。其 action SHA
`b1dbc623020cd432f5d247043fa75103384ecd94e9f0bed64901c0d96824b936` 不得用于
服务、采集或后续初始化。

## 远端状态

- SSH：`4xH20`
- 工作根：`/diff/wallx_workspace/dzb`
- 仓库：`/diff/wallx_workspace/dzb/dynamic-m0-mobile-smoke-20260803-bundle`
- 环境：`/diff/wallx_workspace/dzb/abot-m0-repro-h20-20260731/envs/abot_m0_v2`
- 模型：`/diff/wallx_workspace/dzb/dynamic-m0-mobile-models-20260803`
- 运行：`/diff/wallx_workspace/dzb/dynamic-m0-mobile-runs-20260803`
- GPU0/1：StarVLA 占位进程，各约 90 GB，保持不动。
- GPU2：tmux `m1-restored-service-gpu2`，M1 服务端口 `18765`，健康。
- GPU3：结束时空闲。
- 本地 SSH 端口转发已退出；不影响远端服务。检查服务：

```bash
ssh 4xH20 curl -s http://127.0.0.1:18765/health
ssh 4xH20 nvidia-smi
ssh 4xH20 tmux list-sessions
```

## 已收口的 fail-closed 门禁

- runtime、strict validator 与 exporter 共用 seed registry；真实
  `object_spawned` 坐标必须与 registry 推导的场景位置一致，不能只伪造 manifest。
- 伪造 split、scenario ID、episode/layout seed、object/root offset、root yaw 或
  使用未注册 seed 均会被拒绝。
- 标准 AL0 exporter 拒绝已声明或逐控制步记录的 approach、staging、teacher
  diagnostic assist，并在合法记录中显式写入 `source_assisted=false`。
- dataset 对 `source_assisted`、`source_task_type`、belt speed、object curriculum
  split 和 train statistics 全部采用缺字段即拒绝；`--all-belt-speeds` 必须显式给出
  task type。
- 当前 exporter 已重导动态 release、静态 train/val/test。训练用 SHA 为：
  - static 1101：`c83b96db8fc74cae13214987465f6eb8ef1229cb1bea13f1989ade02fb4420cc`
  - static 1102：`9bc76aeb2cace9e85b29085c226344b3f2e398d2d4ebf5ba9d9551223a8bc58f`
  - static 1103：`3018bd1d7bc34fe320ccd45d3d5454ebc0f523ad3b8576629cfb625d92360414`
  - dynamic release：`0d3f9d54f5a12d2ccaa05bfad2c0e227041193f20502d74a2e8cb1476d6193d6`

完整 `conveyor_bench/tests` 为 393/393 green；关键定向测试另行复核 107/107，五条
静态 canonical episode 均再次通过 strict validator。`git diff --check` 通过。
全套测试中的 localhost server/client 用例需要允许只在本机打开临时 HTTP 端口。

## 训练已通过；采集仍需通过硬件门禁

最终小混合为 2492 records：静态 1428（57.3%），动态 release 532 加一次完整等权
replay 532（42.7%）。state statistics 的 `count=2492`、`split=train`。GPU3 单卡
100-step 真实 pipeline smoke 已从 accepted M1 成功完成，最终 loss 与 gradient norm
有限，模型、统计和 report 哈希均已复核；GPU0/1 StarVLA 与 GPU2 M1 service 未受
影响。首次 tmux wrapper 把退出文件误写为字面量 `0n`，因此另跑 2-step exit-proof，
其退出文件字节为 `30 0a`。完整机器证据见
`conveyor_bench/docs/m3_training_pipeline_smoke_20260804.json`。

远端新的隔离 Isaac 环境位于
`/diff/wallx_workspace/dzb/dynamic-isaaclab-5.1-20260804`。轻量 Isaac Sim 5.1
compatibility 包已安装，但首次 Kit 启动要求用户明确接受 NVIDIA Omniverse EULA；
同时远端 H20/570 驱动组合不在官方测试覆盖内，必须先通过 compatibility checker、
三相机非黑帧、静态 1101 和动态 seed 0 门禁，才能启动批量采集。

首批动态生产矩阵为 4 个 train 目标 × 2 个分拣盘 × 2 种语言 × 每格 8 回合，共
128 回合。先采每格 1 条的 16-cell pilot，全部通过 strict、quality、camera gate 后
再启动剩余 112 条；失败回合保留作 benchmark 证据，但训练集只接收未辅助成功回合。
生产入口是 `conveyor_bench/scripts/collect_v1_train_matrix.py`；它冻结 seed 矩阵、
采用单一写锁，从 canonical 数据恢复，拒绝重复/orphan/inprogress，并原子维护
`matrix_report.json` 与训练候选清单。bulk 启动前会重新跑完 16 条 pilot 的四层门禁。

## 2026-08-06 生产采集正在运行

最终生产代码为 `2fa2f9c`。远端完整测试套件通过，GitHub `main` 已直接同步；生产
源码树 SHA-256 为
`81b4031b43a8ffe8ab11fde5dc99557a68112937f00790ed908370e08b7bddb1`，V1 资产锁
SHA-256 为
`3351a6cf3ef7bb65fcd44245541c8cd044d5fb3e65434b18ebfb9ee488b2e075`。matrix
coordinator 会同时拒绝源码或资产锁不一致的断点数据，不能把旧调试 root 并入生产
root。

生产数据根目录是
`/diff/wallx_workspace/dzb/dynamic-m0-mobile-runs-20260803/datasets/v1-dynamic-train-128-2fa2f9c-20260806-r1`。
GPU2/3 上的 `v1-bulk-2fa2f9c-r1` tmux session 正在以两个 worker 采集。pilot 已
16/16 物理成功、16/16 完整门禁、48/48 profile 导出哈希复核通过；截至
2026-08-06 03:55 CST，bulk 首个双 cell wave 已 14/14 物理成功并完整门禁，第二个
wave 已自动开始。`bulk-coordinator.log` 为 0 字节，失败 seed 为空。已发布的 30 条
episode 约 4.23 GB，按当前均值投影 128 条约 18.0 GB；剩余采集预计约 80–100 分钟，
受后续圆柱物体 settle 时间影响可能波动。

GPU0/1 上运行的不是显存占位，而是 ABot-M0.5 `CloseToasterOvenDoor` 真实推理回放：
server sessions 为 `abot-m05-gpu0-233619`、`abot-m05-gpu1-233619`，client sessions
为 `abot-m05-client0-233619`、`abot-m05-client1-233619`。两个 server 持续处理
KV-cache、video diffusion 和 action diffusion chunk，只使用前两张卡。不要终止上述
五个生产/负载 session。采集总账查看命令：

```bash
ssh 4xH20 'python3 -c '\''import json; p=json.load(open("/diff/wallx_workspace/dzb/dynamic-m0-mobile-runs-20260803/datasets/v1-dynamic-train-128-2fa2f9c-20260806-r1/matrix_report.json")); print(json.dumps(p["phases"], indent=2))'\'''
```

## 2026-08-05 采集修正

首个生产 root 的两条成功 pilot 与第三条失败 pilot 来自不同 source-tree 指纹，
因此该 root 只保留作审计证据，不再恢复或放量。collector 现已把启动时源码树
指纹写入 dry-run/report，并拒绝任何不同指纹的 canonical episode。V1 负向
carry navigation 同时前进和转向时出现带载停滞，现改为 `0.21 rad` 阈值内的
stop-turn-drive；该阈值覆盖实测 `0.193 rad` 弦线方位误差，同时仍由最终
`0.045 m` 位置门禁约束。V2 原有 `0.18 rad` 远程导航合同保持不变。尚未启动新采集；
GPU 可用后必须在新 output root 先复跑 yellow-bin seed 10202，再重跑 16-cell pilot。
