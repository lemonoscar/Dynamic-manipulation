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

下一轮最小验收矩阵是：增加多个预抓取扰动和接触时序的成功示教，重新通过
3/3 离线边界门禁，再对至少 5 个 seed 和一个不同高度物体运行 guard-off
闭环。只有抓取、持有、投放均由策略真实完成后，才扩大采集规模。
