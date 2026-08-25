# 当前状态、证据与剩余门禁

最后复核：2026-08-25 CST。冻结的 Waypoint v1 runtime/eval 基线为
`feature/conveyorvla-waypoint-v1@cfed498eff780d390426962f309a3002173e9ed3`，durable
checkpoint 为 `step_002000@a8d57a22c515`。Waypoint v2 最新根因、修正和训练决策见
[阶段无法切换：证据链、根因与修正](waypoint_v2_transition_root_cause_20260823.md)和
[操作时序与夹爪监督修正](waypoint_v2_manipulation_sequence_and_gripper_correction_20260823.md)。
当前正式训练参数由 [Waypoint Policy v2 合同](conveyorvla_waypoint_policy_contract_v2.md)
冻结。

## 0. Waypoint v2 最新状态

v2 `step_002000@02ee859` 已确认为 8-episode、1,877-row overfit checkpoint，而非完整
522-episode 长训；step 2000 等价约 68.19 个 sampling epoch。严格开环综合门禁失败，
prefix 在闭环中 140/157 次预测 `K=1`，CRL action-shuffle drop 仅 `0.00679`。删除废弃
local fatal stall 后的 seed 139 闭环仍连续 157 次输出 `NAV_TO_SOURCE`，PICK 概率最高
`0.201772`，证明 stall 不是不能切换的根因。

实现审计发现 B2 transition 软标签只存在于有效权重约 0.1 的辅助 crossover，主 answer CE
与 route CE 仍对同一 route token 做两份硬监督。当前修正把 transition route/decision token
从硬 answer CE 中屏蔽，并由独立 route loss 使用连续 old→new 目标；interior 与 B1 保持
原硬标签行为。原始 `K*=0` row 也不再被伪标为可表示的 `K=1` prefix。

证据支持的最小候选为 terminal-hold + B2 boundary/progress + S1；prefix、CRL 与
on-policy correction 均保持关闭。正式 full-data run 曾从官方 Qwen 初始化，仅双卡、
global batch 64 启动；连续 step 4–23 的健康审计通过。用户随后在有效 step 568 后停止，
最后 durable checkpoint 为 step 500；2026-08-23 最新实时核验没有本任务进程或 tmux。

step 500 的 seed 147 诊断后来暴露出两个新根因：旧 runtime 会从不可规划 target0 跳到
target1，且旧 v2 ARM 第 7 维是测得手指开度而不是专家夹爪命令。`8970dea` 已改为每次
MANI query 只按时间顺序执行 target0、无规划/timeout 安全重询，并从 raw 显式
`gripper_command` 构建全新 immutable schema。新数据 522 episode、119,700 row 的完整
audit 和逐行差分均通过，state field/tensor 为 0。旧 step 500 与新 schema/manifest 不兼容，
不得 strict resume。2026-08-24 首轮 command-gripper S4 full-data run 使用物理 GPU 2/3、
global batch 128、3,000 effective optimizer step 和每 250 step 保存；prefix/CRL/B5
on-policy correction 均关闭，GPU 0/1 的无关 StarVLA 未被触碰。

该正式 run 已从 clean `waypoint-v2@571f306154aa30d0544bb14cd2754b0c6e6e6637`
启动，run ID 为
`conveyorvla-waypoint-v2-b2-s4-command-gripper-full522-2gpu23-zero3-gbs128-571f306-s3000-20260824T002235CST`。
steps 1–10 连续健康审计通过，但继承自 v1 的 self-conditioned 调度在 step 152 激活，令
单步从约 60 s 增至约 305 s；step 152–160 没有有效自产 route match。用户已在有效 step
238 明确终止该 run；没有 checkpoint，不能 resume。完整证据见
[Waypoint v2 command-gripper S4 修正与正式训练启动](waypoint_v2_command_gripper_s4_launch_20260824.md)。

替代 run 使用全新不可变
`configs/waypoint_v2_b2_s4_command_gripper_self1500.json`：steps 1–1500 严格关闭
self-conditioned，step 1501–2550 线性升到 0.5；其余数据、S4、global batch、GPU 和
checkpoint 合同保持不变，并从官方 Qwen fresh 初始化。替代 run 已从 clean
`waypoint-v2@4fb50ffa8f0a05eeda5d9dcc34a898658ba8d9f3` 在物理 GPU 2/3 fresh 启动，run ID
为 `conveyorvla-waypoint-v2-b2-s4-command-gripper-self1500-full522-2gpu23-zero3-gbs128-4fb50ff-s3000-20260824T113345CST`。
steps 1–10 全部有效，self-conditioned 权重、loss 和样本计数均严格为 0；中位 step 时间
58.17 s、中位吞吐 2.20 samples/s，reserved 显存峰值 64,408 MiB。健康审计结束后训练保持
运行，首个计划 checkpoint 为 step 250。完整证据见
[Waypoint v2 self1500 修正与替代训练健康启动](waypoint_v2_self1500_retraining_launch_20260824.md)。

用户在 2026-08-25 指令停止并评测；最后 durable checkpoint 为 `step_001250`，之后未保存的
训练 event 不作为 checkpoint 使用。step 1250 完整 load 和 transition-centric 64-row
开环完成：route accuracy 93.75%、boundary AUROC 0.976834、flicker 0，但 NAV direction
accuracy 75%，MANI inter-target step violation 75.893%。seed 139 完整自主闭环在 92 次
NAV 后自主切到 PICK，切换后底盘全程为零，连续 7 个 MANI target0 经真实 cuRobo 成功执行；
第 8 个 pose 无规划后 fail-closed，物体未抓起。

闭环触发了新的数据硬阻塞：全量 522/522 个 PICK 起始边界 row 的 target0 均为 close，
471 个 episode 到 horizon index 5、51 个到 index 6 才首次 open。根因是 `plan_pick` 规划等待
帧沿用了上一显式 close，却被当成可执行 PICK 动作；runtime 的 NAV `stow_open` 则以 open
状态进入 PICK。继续在同一数据上训练只会强化该错误。完整证据和下一数据门禁见
[step 1250 严格开环与完整自主闭环评测](waypoint_v2_step1250_strict_evaluation_20260825.md)。

## 1. 冻结 Waypoint v1 总结

Waypoint Policy v1 的无 state 数据、两次完整 Qwen、双 Layerwise FM head、checkpoint、
inference export、PCT/DWA、真实 cuRobo/IK 和模型自主 Isaac rollout 均已有可执行实现与
证据。默认 checkpoint 间隔为 500 effective optimizer step。

四卡 resume run 已按用户指令在 step 2090 后停止，最后一个完整 checkpoint 是 step 2000；
step 2001–2090 只有训练 event、没有 durable checkpoint，不能恢复。2026-08-21 最后一次
远端核验时，4×H20 没有本任务进程或显存占用；这不是 2026-08-22 的实时状态。

针对 action horizon 20 的最新执行策略不再固定取首点：在可信前 10 点中寻找不小于
`2 × 到达容差` 且最接近 `0.50 m` 的候选，按顺序交给 PCT，执行首个可规划点后重新推理。
seed 139 完整自主复测产生 18 次真实导航，并由模型自主切换到 PICK；说明“首点容差内导致
不导航”的执行问题已经解决。但模型导航方向仍振荡，PICK 的第 3 个 TCP target 又违反原始
35° 相邻旋转步长规则，因此完整 pick-place 仍未通过。完整结果见
[step 002000 lookahead 选点策略完整自主闭环复测](checkpoint_step2000_lookahead_evaluation_20260821.md)。

## 2. 门禁总表

| 门禁 | 状态 | 证据/边界 |
|---|---|---|
| 无 state 数据与 schema | 通过 | 522 episodes、119,700 rows；模型输入 state field/tensor=0 |
| Pass 1/Pass 2 与双 Layerwise FM | 通过（实现/训练） | 两次完整 Qwen、模型自产 prefix、NAV/ARM 均有非零梯度 |
| 80-step 小样本 overfit | **未通过** | 旧 overfit route 置信度未过，不因长训自动改写 |
| 四卡长训健康启动 | 通过后停止 | resume 后有连续健康 step；用户在 step 2090 后主动停止 |
| checkpoint 间隔 | 通过 | 每 500 step；step 2000 load/export gate 通过 |
| step 2000 inference export | 通过 | `step_002000@a8d57a22c515`，state field=0，单卡服务健康 |
| PCT/DWA known-waypoint | 通过（接线层） | reference PCT/DWA 可调用；不能外推模型导航成功 |
| 真实 cuRobo known-pose | 通过 | reachable/collision-free，reference identity 与 frame capability 通过 |
| lookahead selector 单元/远端回归 | 通过 | 本地 targeted 33 passed；本地/4xH20 relevant suite 各 53 passed |
| seed 正面可见 preflight | 通过 | seed 139 settle 后 bearing `-0.609°`；首帧目检可乐居中可见 |
| lookahead 完整自主 Isaac | **未通过** | 18×NAV 后自主 PICK；ARM target 2 旋转 step 超过 35° |
| 完整 pick-place | **未通过** | 没有成功抓取、搬运和放置 |
| Waypoint v2 S4 early-self 双卡训练 | **用户终止** | step 238；无 checkpoint；self-conditioned 过早激活 |
| Waypoint v2 S4 self1500 替代训练 | **用户停止；step 1250 已评测** | durable step 1250；静态开环结构通过，但完整闭环失败 |
| Waypoint v2 command-gripper 时序 | **数据硬阻塞** | 522/522 个 PICK boundary target0=close；必须新 schema 修正后再训练 |

“实现/接线/诊断通过”只覆盖表中明确层级，不向模型收敛或完整物理成功外推。

## 3. 训练与 checkpoint

| 项目 | 值 |
|---|---|
| host | `4xH20`，实际 `VM-0-3-ubuntu` |
| run | `runs/conveyorvla-waypoint-v1-resume-step1000-a8d57a2-s10000-20260821T015929` |
| training source | clean `a8d57a22c515e46a9ad20be6f6892a067e02b3c3` |
| parent | 正式父 run 的 `output/checkpoints/step_001000` |
| data | 108,603 train rows，全量 |
| batch | micro 3/GPU × 4 × accumulation 2 = global 24 |
| precision / sharding | bf16 / DeepSpeed ZeRO-3，无 CPU offload |
| save | 每 500 effective optimizer step |
| stopped | 用户授权停止；最后有效训练 event 为 step 2090 |
| durable | step 2000；load/export 通过 |
| last observed GPU state | 2026-08-21 四卡 0 MiB，无本任务 tmux/端口占用；操作前须重查 |

resume 正确恢复了 Qwen、双 head、optimizer、scheduler 和随机状态；从 step 1000 后重新
计算，不声称保留旧父 run 中没有 checkpoint 的 1001–1181。若未来续训，必须从 step 2000
写入全新 run。

## 4. step 2000 闭环演进

| profile | query / control step | 结果 |
|---|---:|---|
| `contract`（旧首轮） | 1 / 58 | 第 19 段平移超过 0.8 m，完整 horizon fail-closed |
| `executable-prefix-diagnostic` | 26 / 473 | 本地旧 `navigation_stall` 终止 episode |
| `unbounded-translation-diagnostic` | 9 / 301 | 取消 NAV 平移上限后仍被同一本地 stall 终止 |
| `arm-vla-reference`（旧首点） | 23 / 140 | 22×NAV 都在 0.18 m 容差内，几乎不移动；PICK target 0 超过 35° |
| `lookahead-arm-vla-reference` | 19 / 3712 | 18×真实 NAV 后自主 PICK；target 2 超过 35° |

旧 `navigation_stall` 是本地 `a852b5b9` 引入的 3 s 内目标距离没有改善 1 cm 规则，不是
reference 原始 detector；其配置字段、进度状态和 fatal 执行分支现已从代码全局删除，任何
profile 均不能再启用。reference 对照 profile 继续保留 PCT/DWA、reference timeout/stall
重询语义和原始 ARM workspace/rate/collision 规则。

## 5. 最新 selector 与闭环结论

`trusted-prefix-target-lookahead-pct-v1` 的 reference 参数为：可信前缀 10 点、最小
lookahead `0.36 m`、目标 lookahead `0.50 m`。正式 run 的 18 个 selected index 为 2–9，
平移半径 min/mean/max 为 `0.3735/0.4948/0.5811 m`；PCT 成功 `18/18`，8 个 chunk 到达、
10 个在 250 step timeout 后正常重询，没有额外导航 gate 把 episode 标失败。

底盘到可乐的 query-anchor 距离从 `1.3164 m` 降至最低 `0.3804 m`，PICK 时为
`0.4931 m`。首末底盘净位移 `0.9352 m`，query-anchor 路径累计 `7.3074 m`。这说明机器人
确实导航，但物距非单调，问题已经从执行器“只取首点”转为模型 waypoint 方向振荡。

模型以 `0.9478` 的 route 概率自主切到 PICK。ARM target 0/1 合法；target 2 相对 target 1
平移 `0.04414 m`，但 roll/pitch/yaw step 为 `40.44/19.56/38.13 deg`，违反原始
`35 deg` 单轴限制。validator 在 cuRobo 前审计完整有效 ARM prefix，因此没有把非法 chunk
发送给 planner。

## 6. 初始化、输入与视频

seed 139 是按正式 r2 task 和 runtime randomization 重扫后选出的正面 seed。对象物理 settle
完成后才进行 preflight 与第一次模型 query；实际 bearing 为 `-0.60894°`、距离
`1.31635 m`。front frame 中红色可乐罐居中清晰可见。几何信息只用于评测启动 gate，
`source_truth_sent_to_model=false`；模型输入仍只有 instruction 和 head/wrist 双时刻四图，
state field 为 0。

最新三路视频为完整 74.2 s episode，而不是此前几乎不导航的 3 s 视频：

```text
artifacts/evaluation/waypoint_step002000_lookahead_20260821T033113Z/
```

front/overview/wrist 分别为 1856/1855/1856 帧，远端/本地 SHA-256 一致，并已完成首帧、
最小物距、PICK 和末帧目检。视频、trace、日志、checkpoint 和运行资产均在 Git ignore 中，
没有加入公开仓库。

## 7. 已批准的下一步边界

2026-08-22 用户已批准
[Waypoint v2 阶段切换执行与长训计划](waypoint_v2_stage_transition_execution_plan.md)：
冻结上述 v1 数据、checkpoint 和结果为基线，从相同只读 source 构建全新 v2
schema/manifest，并在保留内部门禁的前提下自主执行到 4×H20 正式长训健康启动。本次文档
更新没有实时连接远端，因此任何“空闲”判断都必须在操作前重新核验。

执行优先级为：

1. 修复跨 route suffix 的 train/inference 语义不一致，并保留 original prefix `K*`；
2. 依次验证 boundary/progress、可信 prefix、PRTS 方法启发的局部 CRL、FM 训练 sample
   `1→4` 和 on-policy correction，按证据选择最小有效组合；
3. 提高 NAV 方向一致性、route 切换校准和 ARM RPY/curobo 可执行性；
4. 完成多 seed 开环、planner、NAV/ARM staged 和完整自主 pick-place 门禁；
5. 冻结最终 v2 合同和 resolved config，在 4×H20 启动每 500 step 保存的正式长训；连续
   至少 20 个健康有效 optimizer step 后保持训练运行。

取消的是中途汇报和逐阶段审批，不是数据、overfit、分布式、开环或闭环门禁。
production `contract` 仍须单独通过，reference/diagnostic profile 不能替代正式门禁。
