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

因此下一轮补足分成两个互不混淆的目标：先增加带初始 TCP/root 扰动的成功
pregrasp 恢复示教，使 guard-off 能稳定到达 staging；再提高成功示教中
`lift→carry_retract→carry_turn` 窗口的采样权重。当前 1,183 条训练记录只包含
一次完整成功轨迹，已有 booster 主要重复 approach 与 grasp transition。补训后
先复跑 seed 0 guard-off，并单独检查 Cartesian action 与 joint-space compact gate
是否仍不一致；在有数据证据前不放宽 gate。通过后再扩到至少 5 个 seed 和一个
不同高度物体。只有抓取、持有、搬运和投放均由策略完成后，才扩大采集规模。
