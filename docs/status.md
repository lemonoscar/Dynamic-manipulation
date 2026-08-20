# 当前状态、证据与剩余门禁

最后复核：2026-08-21 02:13 CST。现行 runtime/resume 代码基线：
`feature/conveyorvla-waypoint-v1@0deec5ec60f771826b4c5d2ff47fe731dfa7e477`；父
step 1000 checkpoint 的训练 source 为 `724ead21be2c27d9b40c200375ee4ab49ccedc84`，
当前 resume run 的训练 source 固定为 `a8d57a2`。

## 1. 总结

Waypoint Policy v1 的无 state 数据、两次完整 Qwen、双 Layerwise FM head、checkpoint、
开环、PCT/DWA、真实 cuRobo/IK 和模型自主管理的 Isaac rollout 均已有可执行实现与分层
证据。默认 checkpoint 间隔为 500 effective optimizer step。

step 1000 的 route/格式开环通过，但 20 点 NAV/ARM 数值质量仍不足。strict 自主闭环因
NAV 第 18 段 yaw 超限正确 fail-closed；显式 executable-prefix staged 诊断证明第一个合法
receding-horizon waypoint 可以在真实动力学中完成。完整 horizon、ARM route 和完整自主
episode 尚未通过，不能由 staged 成功外推。

四卡正式长训已从 durable step 1000 严格恢复。step 1001–1020 共 20 个连续 optimizer
step 全部通过有限值、正梯度、正 learning rate 和进程/GPU 健康门禁；训练仍在 4×H20
后台推进。详细 step 1000 诊断见
[开环与真实 Isaac 闭环评测](checkpoint_step1000_evaluation_20260821.md)。

## 2. 门禁总表

| 门禁 | 状态 | 证据/边界 |
|---|---|---|
| 无 state 数据构建与 audit | 通过 | 522 episodes、119,700 rows；state field/tensor=0；manifest/normalizer hash 冻结 |
| Pass 1/Pass 2 与双 Layerwise FM | 通过（静态/训练） | 两次完整 Qwen、模型自产 prefix、NAV/ARM 均有非零梯度 |
| 80-step 小样本 overfit | **未通过** | 旧 overfit route 置信度未过门禁；不能被长训替代为“通过” |
| 父正式训练 | 通过后主动暂停 | step 1–1181 有效；durable checkpoint 为 step 1000 |
| 严格 optimizer resume | 通过 | Qwen/双 head/optimizer/scheduler/random state 恢复；data/scheduler alignment 有记录 |
| 新长训健康窗口 | **通过并持续运行** | step 1001–1020 连续有效，四 rank 存活，四卡实际计算 |
| checkpoint 间隔 | 通过 | 默认和运行参数均为 500；下一新 checkpoint 为 step 1500 |
| step 1000 route/格式开环 | 通过 | val 40 rows；5 类各 8；accuracy=1.0，RECOVER/invalid=0 |
| step 1000 action 开环质量 | **未通过可用性判断** | NAV/ARM OOB、segment/step violation 和 pose error 偏高 |
| inference export + 服务 | 通过 | tied Qwen export 已修；约 21 GB；单卡四图 request 成功 |
| PCT/DWA known-waypoint probe | 通过 | fallback 关闭、snap/速度有界、首目标后 requery |
| 真实 cuRobo known-pose | 通过 | reachable/collision-free，41 点 path，pose error 约 `6e-8` |
| strict 完整自主 Isaac | 已执行、**未通过** | 无 GT phase/FSM/route gate；尾部 segment 18 yaw 超限时 fail-closed |
| executable-prefix staged NAV | 通过（诊断层） | 首点合法，terminal-yaw 后 step 75 `first_waypoint_reached`；完整 horizon 仍失败 |
| ARM staged / 完整自主 episode | 未完成/未通过 | 不能由 known-pose 或单 NAV route 外推 |

“通过（结构/接线/诊断层）”只覆盖表中明确层级，不向模型收敛或完整物理成功外推。

## 3. 四卡恢复长训

| 项目 | 值 |
|---|---|
| host | `4xH20`，实际 `VM-0-3-ubuntu` |
| run | `/diff/wallx_workspace/dzb/runs/conveyorvla-waypoint-v1-resume-step1000-a8d57a2-s10000-20260821T015929` |
| tmux | `codex_cvla_wp_resume_a8d57a2_20260821` |
| source | clean `a8d57a22c515e46a9ad20be6f6892a067e02b3c3` |
| parent | `conveyorvla-waypoint-v1-formal-724ead2-s10000-20260820T1813/output/checkpoints/step_001000` |
| train rows | 108,603，全量、无 subset |
| batch | micro 3/GPU × 4 GPU × accumulation 2 = global 24 |
| precision / sharding | bf16 / DeepSpeed ZeRO-3，无 CPU offload |
| max/save | step 10,000 / 每 500 step |
| health window | step 1001–1020，2026-08-21 02:05:41–02:12:26 CST |
| current state | 四 rank 与 tmux 存活，训练继续运行 |

resume binding 记录 `optimizer_resume=true`，parent manifest SHA-256 为
`8df7837797c5b27b72a1c3a77bfb5a995f2a59f815fd91d2a97240b8b3c84610`。
scheduler 从 step 1000 原样恢复，`repaired=false`；sampler 的每 pass 9,050 个 micro-batch
中跳过 2,000 个，对齐 step 1000 的数据位置。

20-step 健康窗口：

| 指标 | min / mean / max |
|---|---:|
| total loss | 0.4878 / 1.0809 / 1.9610 |
| NAV loss | 0.1233 / 0.3959 / 1.1312 |
| ARM loss | 0.1769 / 0.4396 / 0.7272 |
| VLM gradient norm | 88.02 / 240.00 / 685.89 |
| NAV gradient norm | 8.01 / 17.27 / 42.62 |
| ARM gradient norm | 6.54 / 11.16 / 15.00 |

全部 step 均为 `valid_optimizer_step=true`，所有 loss、四组 learning rate 和 gradient norm
有限；三个组件梯度严格大于 0。四卡同时计算采样曾达到 100%/100%/100%/100%，显存约
46–51 GB。日志在该窗口没有 traceback、OOM、NCCL error、NaN/Inf 或提前退出。

父 run 的 step 1001–1181 因暂停前没有新 checkpoint 而不可恢复；当前 run 正确从
step 1000 重算，不声称保留那 181 个仅存在于旧日志中的计算 step。下一个 durable 新
checkpoint 是 step 1500。

## 4. waypoint 非法的根因

训练集 63,350 个 NAV row 在原始 0.8 m/45° 合同下 sample/segment violation 都为 0，
最大相邻平移 0.2652 m、最大偏航 19.881°。因此批准阈值与 GT 兼容，不应靠放宽合同来
迁就 step 1000 输出。

64 个 NAV oracle row × 4 diffusion seed 的审计中，首点 violation rate=0，完整 20 点
violation rate=0.96875；最早坏点为 index 4。20 点全有效监督只有 20,125 row，而前部位置
有 60,506 row，尾部明显欠监督。即使诊断性把 yaw 放到 180°，完整 horizon 仍有 0.6875
失败，所以问题是尚未训练好的 horizon 数值质量，而不是单一 45° 阈值。

初始化不是根因：模型 query 在 control step 58、对象 settle 和机器人初始化之后；query
root z 约 0.1914 m，与训练约 0.1897 m 相当。StarVLA 的 Qwen3-VL/Layerwise FM、
Beta(1.5,1)、`noise_s=0.999`、4-step Euler 设置与当前实现一致；arm-vla 的
`pct_multifloor` 初始姿态、actuator 和 DWA/control dt 也与实际 rollout 一致。

## 5. executor 修复与真实仿真

默认 `contract` profile 继续审计全部 20 点并 fail-closed。显式
`executable-prefix-diagnostic` 先保留完整 violation，再只执行独立合法首点；没有修改
0.8 m/45°、PCT snap、workspace 或速度门禁。

诊断依次发现并修复两个 executor 语义冲突：位置已经进入 0.12 m 到达容差时，旧实现仍
等待 DWA 平移进展而 stall；约 0.03 m 的目标还会被送入 0.2 m PCT 栅格，使最近端点 snap
在 0.10 m 半格边界波动。`92ba25f` 处理 PCT/DWA 行进后的 terminal-yaw；`a8d57a2`
允许 staged 诊断直接检验位置已到的首目标。最终合同复核把后一旁路限定在 diagnostic，
production 仍保持 `<0.03 m` 的纯旋转边界和原 PCT snap 门禁。

修复后的 staged r3：

- 模型输入只有完整空间指令和 head/wrist 两时刻四图，`model_state_fields=0`；
- Qwen route=`NAV_TO_SOURCE`，无 GT phase、外部 FSM 或 route 覆盖；
- 首点 `[-0.03717,-0.00095,0.23649]`，自身合法；
- 完整 horizon 明确记录 `navigation segment 18 exceeds yaw limit`；
- `planner=terminal_yaw`，control step 75 达到 `first_waypoint_reached`；
- summary `success=true`、`query_count=1`、`state_trace=[NAV_TO_SOURCE]`；
- overview/front/wrist 为 37/38/38 帧，视频、summary 和 trace 已下载到本地 Git 忽略目录。

这只证明合法首点和 terminal-yaw 执行链通过。strict run 的完整 horizon 失败结论没有被
改写，ARM model target 与完整 pick-place episode 仍需新 checkpoint 再测。

## 6. 本轮实现节点

- `aa06479`：默认 checkpoint interval 1,000 → 500；
- `23afff4`：分离 cuRobo source 与 arm-vla runtime assets；
- `13f6e87`：正确导出 Qwen tied weights；
- `1215129`：capability-gate 外部 Waypoint cuRobo 服务；
- `22b186c`：严格 ZeRO resume 与显式 executable-prefix 诊断 profile；
- `92ba25f`：PCT/DWA 到达位置容差后的 terminal-yaw；
- `a8d57a2`：验证位置已到的首目标可绕过无意义的 PCT snap；后续合同复核将其限定为
  diagnostic profile。
- `0deec5e`：production 恢复 `<0.03 m` 纯旋转 PCT 旁路边界，0.03–0.12 m 仅供显式
  diagnostic profile 检验。

## 7. 下一门禁

1. 保持当前 4×H20 run 运行，并检查 step 1500 新 checkpoint 完整性；
2. 用更新 checkpoint 重跑 40-row/四 seed 开环，重点看 NAV 尾段和 ARM pose/step；
3. 依次补 ARM staged route、四阶段 staged 和 strict 完整自主 episode；
4. production 默认始终使用完整 horizon `contract` profile，不用诊断 profile 冒充成功。
