# M0-Mobile 在线闭环与验收

本文说明如何把本仓库的 M0-Mobile action head 接入 Go2-X5 横向传送带任务，
以及当前已经验证到哪里。模型权重不提交到 Git；训练、服务、离线门禁和 Isaac
闭环入口都在本仓库内。

## 控制边界

在线策略只接收 `head_rgb`、`wrist_rgb`、双语任务指令和 `state28`，输出未来
`16×10` 的 body/base-frame 动作块。`overview_rgb`、物体真值、接触真值和
oracle 目标不会发送给模型。

当前运行时属于可审计的 hybrid baseline：服务层负责起步前的底盘稳定、机械臂
预置、终止保持和 Go2 前向速度量化；M0 负责移动阶段的前向意图，以及抓取阶段
的 TCP、姿态和夹爪动作。oracle 仅用于阶段与成败评价。抓取测试不得通过实时
oracle 目标修正 M0 的 TCP，也不得强制下降或闭爪；否则只能标记为
`oracle-assisted`，不能记作 policy-only 结果。

默认每次重规划执行动作块前 2 步；进入 `track/descend/close/lift` 后可执行前
12 步，避免阶段切换只存在于动作块尾部。run summary 会分别记录配置的前缀、
实际 M0/service 控制步、请求数和时延。

## 运行顺序

以下命令均从 `conveyor_bench/` 执行。先为成功相机 episode 生成因果训练视图和
状态统计：

```bash
python scripts/export_v1.py EPISODE_ROOT --profile m0_mobile
python scripts/compute_m0_mobile_state_stats.py \
  EPISODE_ROOT/exports/m0_mobile.jsonl \
  --output STATE_STATISTICS_JSON
```

补强指定观察时间窗口时，创建独立 hard-link 数据子集；源 episode 不会被修改：

```bash
python scripts/build_m0_window_booster.py \
  --episode-root EPISODE_ROOT \
  --output-root BOOSTER_ROOT \
  --window 7.80:8.60
```

两张可见 GPU 的继续训练示例：

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc-per-node=2 \
  scripts/train_m0_mobile.py \
  --episode-root EPISODE_ROOT \
  --episode-root BOOSTER_ROOT \
  --state-statistics STATE_STATISTICS_JSON \
  --initial-action-checkpoint PREVIOUS_ACTION_MODEL \
  --model-root LOCAL_MODEL_ROOT \
  --output-dir NEW_EXPERIMENT_ROOT \
  --max-steps 1200 \
  --batch-size-per-device 2 \
  --gradient-accumulation-steps 1
```

服务端严格校验 action checkpoint、状态统计和训练报告的 SHA，并只监听回环地址：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/serve_m0_mobile.py \
  --action-checkpoint EXPERIMENT_ROOT/action_model_final.safetensors \
  --state-statistics EXPERIMENT_ROOT/state_statistics.json \
  --model-root LOCAL_MODEL_ROOT \
  --device cuda:0 \
  --port 18765
```

启动 Isaac 前先运行阶段门禁。示教要求在本次执行前缀内发生的下降/闭爪若没有
被模型及时预测，脚本返回 `2`：

```bash
python scripts/check_m0_grasp_transition.py \
  --episode-root EPISODE_ROOT \
  --state-statistics EXPERIMENT_ROOT/state_statistics.json \
  --endpoint http://127.0.0.1:18765 \
  --executed-prefix 12 \
  --report outputs/gate/m0_online/grasp_transition_gate.json
```

门禁通过后再运行一个 seed 的在线闭环：

```bash
python scripts/run_m0_closed_loop.py \
  --endpoint http://127.0.0.1:18765 \
  --state-statistics EXPERIMENT_ROOT/state_statistics.json \
  --actions-per-replan 2 \
  --transition-actions-per-replan 12 \
  --episodes 1 \
  --seed 0 \
  --belt-speed 0.06 \
  --max-duration 30 \
  --output-dir outputs/gate/m0_online_seed0 \
  --headless \
  --device cpu
```

若零速任务在 `mobile_approach` 就超时，可只辅助这一前置阶段，隔离判断静态
抓取 primitive。该诊断用冻结的 `0.20 m/s` service command 完成靠近，在物体
生成前抑制所有 M0 请求，并从 object-visible `sequence 0` 开始推理：

```bash
python scripts/run_m0_closed_loop.py \
  --endpoint http://127.0.0.1:18765 \
  --state-statistics EXPERIMENT_ROOT/state_statistics.json \
  --actions-per-replan 2 \
  --transition-actions-per-replan 12 \
  --mobile-approach-assist \
  --episodes 1 \
  --seed 1101 \
  --belt-speed 0 \
  --max-duration 30 \
  --output-dir outputs/gate/m0_stationary_approach_assist \
  --headless \
  --device cpu
```

manifest 只记录开关是否 `enabled`；每步 metadata 和 run summary 另行记录是否
实际介入、辅助步数、控制来源、交接时的 root 位置/速度、机械臂误差、生成前
请求数和首次推理 phase。该开关默认关闭；实际介入过的回合都是
`assisted diagnostic`，不能计作 policy-only 成功或写入训练集。

若 guard-off 回合一直停在 `pregrasp`，可在完全相同的模型、seed、速度和时长下
增加一次诊断性 A/B：

```bash
python scripts/run_m0_closed_loop.py \
  --endpoint http://127.0.0.1:18765 \
  --state-statistics EXPERIMENT_ROOT/state_statistics.json \
  --actions-per-replan 2 \
  --transition-actions-per-replan 12 \
  --pregrasp-workspace-guard \
  --episodes 1 \
  --seed 0 \
  --belt-speed 0.06 \
  --max-duration 30 \
  --output-dir outputs/gate/m0_online_pregrasp_guard_seed0 \
  --headless \
  --device cpu
```

该开关默认关闭，仅在 `pregrasp` 使用 robot-base frame 的固定单边边界
`x<=0.622 m`、`y>=-0.060 m`、`z>=0.250 m`。它不读取物体状态或 oracle TCP，
也不修改底盘、旋转、夹爪、下降和闭爪。manifest、每步 metadata 和 run summary
会记录 proposed/guarded/applied/realized TCP、裁剪轴、修正量及跟踪误差。
guard-on 只用于区分预抓取漂移与下游抓取能力，即使成功也必须标为
`assisted diagnostic`，不能混入正式训练集或计作 policy-only 成功。

若位置 guard 仍不能进入 `track`，第二级诊断可固定世界坐标下的公开工位
位姿，进一步隔离 M0 的下探、闭爪和抬升能力：

```bash
python scripts/run_m0_closed_loop.py \
  --endpoint http://127.0.0.1:18765 \
  --state-statistics EXPERIMENT_ROOT/state_statistics.json \
  --actions-per-replan 2 \
  --transition-actions-per-replan 12 \
  --pregrasp-staging-assist \
  --episodes 1 \
  --seed 0 \
  --belt-speed 0.06 \
  --max-duration 30 \
  --output-dir outputs/gate/m0_online_pregrasp_staging_seed0 \
  --headless \
  --device cpu
```

该模式在 `pregrasp` 将底盘命令置零、保持夹爪打开，并把 TCP 引导到由场景
注册表计算出的固定 world-frame 工位。固定目标本身不读取实时物体状态，也不
复用 shadow oracle 的 TCP 目标；但辅助的启停与交接明确使用 shadow oracle
的阶段判定，因此仍属于 privileged diagnostic。交接到 `track` 时会丢弃未实际
执行的旧 pregrasp action chunk，再从当前观测重新推理。原始 M0 action、实际
控制来源、固定目标、world-frame 位置/姿态误差和交接丢弃数都会被记录。该
模式比 workspace guard 更强，结果只能用于定位故障，绝不能计作 policy-only
成功或写入正式训练集。两个诊断开关互斥，且都默认关闭。

当 assisted staging 已经完成抓取、但 `carry_retract` 未通过 compact joint gate
时，可再做一次 executor 可实现性诊断：

```bash
python scripts/run_m0_closed_loop.py \
  --endpoint http://127.0.0.1:18765 \
  --state-statistics EXPERIMENT_ROOT/state_statistics.json \
  --actions-per-replan 2 \
  --transition-actions-per-replan 12 \
  --pregrasp-staging-assist \
  --carry-retract-teacher-executor \
  --episodes 1 \
  --seed 0 \
  --belt-speed 0.06 \
  --max-duration 30 \
  --output-dir outputs/gate/m0_online_teacher_executor_seed0 \
  --headless \
  --device cpu
```

该开关只在 `carry_retract` 将 shadow oracle 动作投影成 M0 physical10（夹爪为
`0/1`）后执行，但丢弃专家的直接 joint target，并通过与 M0 相同的 Cartesian
IK 执行器落地。现有
`0.060 rad` joint error、`0.35 rad/s`、`0.30 s` dwell 和 `6 s` timeout 均不
改变。若该诊断也失败，才有证据说明 Cartesian action 与 joint gate 合同可能
不可实现；若通过，则应先补 carry 数据。它显式使用 privileged teacher action，
因此整个回合仍不能计作 policy-only 成功或进入训练集。

## 2026-08-03 验收结果

本次使用 1,183 条有效训练记录、1,200 步补充训练的 action model：

```text
action model SHA-256:       2f6c10a55f857cab198daa1886be0d8b2df5fbb7f93d3e8c648df3f3bd795024
state statistics SHA-256:   29bc9a04a9c0eb03947e21e3fb752959c3d6841097df52cbb001acb5558d0e66
training report SHA-256:    ac80d39d764412a4782ac94988e4f73c332869d83e6389682f507ac36eef8a02
```

离线结果：移动意图 3/3 通过；抓取阶段边界仅 1/3 通过，整体门禁失败。失败的
两个窗口把下降预测晚了 8 个 control step，其中最后一个窗口还把闭爪放在
12-step 执行前缀之外。

在线 seed 0 完整运行至 `19.72 s`，结果为 `target_missed`：492 次推理请求，
无 HTTP/OOM 错误，但策略始终停留在 `pregrasp`，目标从出口离开；没有进入
`track/descend/close`，没有双指接触，也没有夹持。服务端推理均值
`81.26 ms`、P95 `83.10 ms`；端到端 RTT 均值 `117.77 ms`、P95
`128.76 ms`。该失败 episode 保留在：

```text
outputs/gate/m0_online_m1_multiphase_hybrid_seed0/episodes/
run-20260803T112917387345Z-34f6a577-ep0000-seed0-whole_body_policy
```

因此可以确认“相机/状态 → 远端 M0 推理 → 动作投影 → Isaac 物理推进 →
canonical 记录与判定”的在线链路已跑通，但当前 checkpoint 尚未达到可开启
M0 policy 数据采集的门槛。现阶段可以继续采集 oracle 成功示教；不能把该模型
生成的失败轨迹混入成功训练集。

随后运行的静态 world-frame staging assist 诊断把失败边界推进到了抓取之后。
TCP 在物体窗口前连续 `0.5 s` 的最大位置误差为 `9.94 mm`、最大姿态误差为
`0.0238 rad`；`8.34 s` 进入 `track`，交接后 `0.04 s` 开始执行新的 M0 动作，
`0.44 s` 闭爪、`0.50 s` 双指接触并 held、`0.56 s` 进入 lift。辅助在
`track/descend/close/lift` 中没有继续生效，因此可确认当前 checkpoint 已具备
从合格 staging 状态完成动态下探、闭爪和抬升的能力。

该回合仍不是任务成功：物体在整个 `carry_retract` 阶段保持夹持且机器人稳定，
但 6 秒后触发 `mobile_carry_retract_timeout`。M0 把 compact TCP 误差从
`174.4 mm` 降到最小 `30.2 mm`，而 compact joint gate 的最佳最大关节误差仍为
`0.298 rad`，没有达到 `0.060 rad`。完整机器可读结论位于
`docs/m0_assisted_staging_seed0_20260803.json`。

后续 teacher-executor A/B 已经排除了 executor 不可实现这一假设。该回合在
`carry_retract` 用 shadow teacher physical10 通过与 M0 完全相同的 Cartesian IK
执行路径，83 个控制步后满足 `0.060 rad / 0.35 rad/s / 0.30 s` compact joint
gate，并在 `11.40 s` 进入 `carry_turn`。因此旧回合的 carry timeout 是学习输出
和闭环分布问题，不应通过放宽 joint gate 掩盖。teacher 回合后来在
`place_descend` 高位提前开爪，仍不是成功轨迹。

一次针对 carry 的 M2 补训也已按回归门禁拒绝。其 action SHA 为
`b1dbc623020cd432f5d247043fa75103384ecd94e9f0bed64901c0d96824b936`：离线抓取
transition 从 M1 的 `1/3` 退化到 `0/3`，在线 seed 0 在 `mobile_approach` 超时，
从未执行 full action。该 checkpoint 不用于服务、采集或后续初始化；远端服务已
恢复为上文三项 SHA 对应的 M1。

### 静止传送带补足结果

正式 `stationary_sort` 仍执行完整抓取—携带—投放任务，但带速严格为零且不计入
动态分数。预注册的 3 个 train、1 个 val、1 个 test oracle 回合全部成功，并均
通过 strict validator 与 temporal camera gate：

- train：2906 control step、4356 张三相机 PNG、1428 条 M0-Mobile 记录；
- val：957 step、1434 PNG、470 条记录；
- test：966 step、1449 PNG、475 条记录。

exporter 使用 scenario split，val/test 明确标为 `val`/`test`，不会因为
`part_red_block` 属于 train 资产而泄漏。静态训练可用
`--belt-speed 0 --task-type stationary_sort` 精确筛选；完整机器可读证据位于
`docs/m0_stationary_followup_20260803.json`。

M1 的无辅助静态 seed 1101 在 `mobile_approach` 的 `4.0 s` 超时，100 次请求均
未进入机械臂 full action，说明静态语言条件下的底盘前置能力尚未学会。仅辅助
mobile approach 的干净隔离回合则满足以下事实：

- 物体生成前 M0 请求为 0，首次推理为 `sequence 0 / settle / 4.44 s`；
- 交接时 root `x=0.07355 m`、平面速度 `0.02694 m/s`、最大机械臂关节误差
  `0.06374 rad`，均处于冻结前置门槛内；
- 交接后 455 步全部为 M1 full action；双侧接触与 `in_gripper` 的 112 个
  50 Hz step 完全一致，连续持有约 `2.24 s`；
- 零件最高达到 `z=0.71493 m`，相对抓取前稳定高度抬升约 `0.18264 m`；
- M1 在 `7.38 s` 主动开爪，shadow phase 从未离开 `pregrasp`；canonical
  failure reason 为 `runtime_error`。对应 `summary.json` 的
  `metrics.abort_metadata` 已持久化 `IKConvergenceError`、`pregrasp` phase，以及
  `position_error=0.0892 m / orientation_error=0.2678`，因此 IK 越界是可复核的
  canonical 诊断原因；它仍不能替代顶层标准 failure reason。

因此 M1 已经具有静止物体的闭夹、双侧持有和明显抬升 primitive，但没有学会
相位对齐、高位保持、carry 和 placement；两条在线回合都不能计作任务成功。

### 下一次训练门禁

下一轮从已接受 M1 初始化，不从被拒绝的 M2 初始化。先只构造可审计的小混合：

- 静态三条 train 完整轨迹占约 `45–60%`；
- 现有动态完整成功 replay 不低于 `40%`；
- carry window booster 不高于 `5%`，避免再次覆盖早期能力；
- 先跑 `1000–1200` step，两张 H20，状态统计只由最终 train 混合计算。

验收顺序固定为：静态离线动作边界 → 静态无辅助 seeds 1101/1102/1103 必须
`3/3` 完整投放成功 → 复跑已接受的动态离线/在线门禁且不得退化 → 再检查 val
2101 和 test 3101。任何阶段失败都停止，不通过重复同一 seed、放宽 gate 或把
assisted 回合加入训练来“修正”结果。通过这些门禁前，可以继续小规模 oracle
示教采集，但不能开启 M0 成功轨迹采集或大规模放量。
