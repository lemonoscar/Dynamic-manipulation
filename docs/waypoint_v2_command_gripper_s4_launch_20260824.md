# Waypoint v2 command-gripper S4 修正与正式训练启动

日期：2026-08-24 CST
训练源码：`waypoint-v2@571f306154aa30d0544bb14cd2754b0c6e6e6637`
状态：用户在有效 step 238 终止；无 checkpoint；已由 self1500 合同取代

## 1. 结论

Waypoint v2 已完成本轮训练前的代码、数据和合同对齐，并从官方标准
`Qwen3-VL-4B-Instruct` fresh 初始化正式 full-data run。训练只使用物理 GPU 2/3，配置为
S4、global batch 128、3,000 effective optimizer step、每 250 step 保存。steps 1–10 的
连续健康审计通过；额外观察的 step 11 也正常。后来确认 v1 self-conditioned 调度在 step
152 过早激活，用户于有效 step 238 终止该 run，未生成 checkpoint。

本次启动证明训练系统和损失接线健康，不等于动作质量、阶段切换或完整抓取已经通过。
第一个模型质量门禁仍是 step 250 的 checkpoint load、严格开环和真实闭环。

## 2. 从问题到修正的完整链路

| 发现的问题 | 关键证据 | 本轮修正 | 当前状态 |
|---|---|---|---|
| transition row 的 route token 同时承受硬 answer CE 和软 route 监督 | B2 软目标只在低权重辅助项生效，主 CE 仍把边界帧硬压到旧 route | transition route/decision token 从硬 answer CE 屏蔽，由独立 route loss 使用连续 old→new 目标；interior 保持硬标签 | 已实现、测试通过 |
| v1 boundary 后 action suffix 在训练中被 mask，推理却可能读取 | 训练/推理 horizon 语义不一致 | 新 v2 对真实 boundary suffix 做 full-horizon terminal-hold，同时保留 original `K*`；source-tail/episode-tail 单独标记 | 已进入新数据与 FM 监督 |
| 旧 MANI runtime 会跳过不可规划的 target0，直接尝试后续 target | seed 147 在到达首目标后过早闭爪，执行时序与 0.20 s chunk 语义不一致 | 每次 MANI query 只按时间顺序规划/执行 target0；不可规划或 timeout 时零动作并重新观察 | 已实现、回归通过 |
| 旧 v2 第 7 维使用测得手指开度，不是专家夹爪命令 | measured opening 会把“保持张开/何时闭合”的结果状态误当成控制命令 | 从 raw `gripper_command` 重建全新 immutable schema；raw `0=close,1=open`，normalizer 为 `-1/+1` | 全量数据审计通过 |
| MANI route 下底盘仍可能延续导航动作 | 阶段切换后底盘不站住会破坏抓取 | MANI route 强制底盘速度为零；不恢复已删除的 local fatal navigation stall | 已冻结进 v2 合同 |
| 单个 FM noise/time draw 方差较大 | S1 每个真实 chunk 只采一组 Monte Carlo 样本 | 一次 Qwen forward 后为每个 chunk 采四组独立 `(noise,t)`，四组 FM loss 取平均；推理 timesteps 仍为 4 | S4 正式训练中 |
| 旧 step 500 与新监督语义不兼容 | 旧 checkpoint 绑定 measured-opening schema 和旧 optimizer moments | 不做 strict resume；新 run、schema、manifest、config 和 checkpoint identity 全部隔离 | 已执行 |

learned prefix、local CRL 和 on-policy correction 本轮都关闭。NAV runtime 继续只在
`min(predicted_K,10)` 的可信范围内选点，但固定 10-step 上限不代表启用了 learned prefix
head。没有新增外部 phase/FSM、truth gate 或距离门控。

## 3. 数据与实现审计

| 项目 | 冻结结果 |
|---|---|
| schema | `conveyorvla-waypoint-dense-transition-v2-command-gripper-v1` |
| episode / row | 522 / 119,700 |
| train / val / test row | 108,603 / 5,771 / 5,326 |
| state field / tensor | 0 / 0 |
| manifest SHA-256 | `6f534e1b7ed456ab6595985d7148eea5e9ff214d4e6a308c5e34baa93fa2506f` |
| policy config SHA-256 | `a173ef0ec0ddb5e605f313f6759bf61ce0b26e2b214cdb105e5303e3771c043e` |
| distributed config SHA-256 | `075ea150b7272cd94b44a4a0468047dbfc0bdac6b71b651f46e0383939d55f57` |
| 本地回归 | 76 passed |

新旧 119,700 row 已逐行核对：除 schema/provenance 与 ARM gripper channel 的预期变化外，
episode split、视觉、route、边界和动作字段保持对齐。loader 的模型 batch 不含 state、GT
phase、operation、object truth 或 simulator target。

## 4. 正式 run 合同

| 项目 | 值 |
|---|---|
| run ID | `conveyorvla-waypoint-v2-b2-s4-command-gripper-full522-2gpu23-zero3-gbs128-571f306-s3000-20260824T002235CST` |
| GPU | 物理 GPU 2/3，2 × H20 |
| precision / sharding | bf16 / ZeRO-3，无 offload |
| micro / accumulation / global batch | 2/GPU / 32 / 128 |
| FM training draws / inference timesteps | 4 / 4 |
| warmup / max step | 200 / 3,000 |
| checkpoint interval | 250 effective optimizer step |
| seed | 20260824 |
| initialization | 官方 Qwen3-VL-4B fresh，无 resume parent |

启动前核验 GPU 2/3 无外部 compute process；GPU 0/1 的既有任务未被终止或共享。两张选定
GPU 各对应一个训练 rank，训练源码 worktree 在启动时为 clean，HEAD 与 upstream 均为
`571f306154aa30d0544bb14cd2754b0c6e6e6637`。

## 5. steps 1–10 健康审计

连续 10 个 event 均满足 `valid_optimizer_step=true`，run state 为 `running@step10`，无 failed
event、OOM、NaN/Inf、NCCL error 或 traceback。逐字段检查覆盖 answer/route、NAV/ARM 四个
S4 draw、boundary/progress、Qwen/NAV/MANI/辅助梯度、LR、吞吐和显存。

| 指标 | min | median | max |
|---|---:|---:|---:|
| total loss | 2,941.41 | 3,880.22 | 4,067.08 |
| answer loss | 7.538 | 8.375 | 9.318 |
| route loss | 2.073 | 2.079 | 2.079 |
| NAV FM loss | 739.87 | 1,523.58 | 1,868.37 |
| MANI FM loss | 2,033.27 | 2,274.35 | 2,550.73 |
| gradient norm | 2.97e6 | 1.09e7 | 4.12e7 |
| optimizer step time | 58.37 s | 59.74 s | 77.14 s |
| throughput | 1.659 | 2.143 | 2.193 samples/s |
| reserved memory | 63,530 | 64,318 | 64,318 MiB |

首步包含初始化开销；steps 2–10 的 step time 保持约 58–60 s。step 10 的 Qwen 梯度出现一次
尖峰，step 11 已从 `4.12e7` 回落到 `9.50e6`，loss、吞吐和显存没有同步失稳，因此判定为
非持续异常。10 步 total-loss CV 为 0.100，吞吐 CV 为 0.070，reserved-memory 峰值约占
H20 容量的 65.7%。

四组平均 draw loss 均非零：NAV 为 `1373.51/1437.58/1484.49/1411.29`，MANI 为
`2260.80/2223.97/2241.28/2386.77`。这证明 S4 接线真实生效；它尚不能证明 S4 的最终动作
质量优于 S1。prefix、CRL 和 self/on-policy loss 在整个窗口严格为 0，符合冻结合同。

## 6. 后续门禁

1. step 250 保存后先做 checkpoint load/export 和 identity 检查；
2. 严格开环覆盖完整 transition 的 lag、crossover、flicker、20-step NAV/MANI 图与动作连续性；
3. 用相同闭环协议验证底盘在 MANI route 下为零、target0 时序重询、机械臂对准—闭合—抬升；
4. 保存 head/wrist/overview 三视角未截断视频与逐 query trace；
5. 只有上述门禁通过，才能把“训练健康”升级为“模型动作或完整任务成功”。

corrected-data 8–16 episode overfit 仍是未补做的科学门禁；用户本次明确要求直接启动
full-data run，因此本文不会把它伪记为已通过。

## 7. 终止与替代决议

首轮 run 的 steps 1–151 使用 oracle-prefix 主训练；step 152 起原 v1 的
self-conditioned 分支开始执行。step 152–160 没有自产 route match，但 optimizer step time
由约 60 s 增至约 305 s。到 step 238 时 `lambda_self=0.04143`，仍未到首个 step 250
checkpoint。用户授权立即终止，两个 rank 正常退出，GPU 2/3 归零，旧 artifacts 保留。

替代 run 不 resume 旧权重或 optimizer，并使用全新 config identity：steps 1–1500
`lambda_self=0`，step 1501–2550 线性升到 0.5。该分支与 learned prefix head、B5
on-policy correction 数据混采严格区分。

替代 run 已通过 steps 1–10 健康启动并保持运行；后续状态、身份和审计数值见
[Waypoint v2 self1500 修正与替代训练健康启动](waypoint_v2_self1500_retraining_launch_20260824.md)。
