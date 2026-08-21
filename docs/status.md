# 当前状态、证据与剩余门禁

最后复核：2026-08-21 10:12 CST。现行 runtime/eval 基线为
`feature/conveyorvla-waypoint-v1@ace7d6e9f2026b55be2f9cc55cf4a355b4dde339`；当前 durable
checkpoint 为 `step_002000@a8d57a22c515`。

## 1. 总结

Waypoint Policy v1 的无 state 数据、两次完整 Qwen、双 Layerwise FM head、checkpoint、
inference export、PCT/DWA、真实 cuRobo/IK 和模型自主 Isaac rollout 均已有可执行实现与
证据。默认 checkpoint 间隔为 500 effective optimizer step。

四卡 resume run 已按用户指令在 step 2090 后停止，最后一个完整 checkpoint 是 step 2000；
当前 4×H20 没有本任务进程或显存占用。step 2001–2090 只有训练 event、没有 durable
checkpoint，不能恢复。

step 2000 的 strict 闭环仍因完整 NAV horizon 尾部超限失败。为回答“额外门控是否掩盖
模型能力”，现已增加 `arm-vla-reference` 对照 profile：只使用 reference 首点规则、
reference 到达容差/DWA/stall，并移除本地完整 horizon、PCT snap、重复 DWA 速度和本地
stall 拒绝。该复测运行到模型自主从 NAV 切换 PICK；但 22 个 NAV 首点都小于 reference
的 0.18 m 到达容差，机器人没有接近可乐，首个 ARM target 又超过原始 35° rate gate。
完整结果见
[step 002000 原始 arm-vla 规则闭环复测](checkpoint_step2000_arm_vla_reference_evaluation_20260821.md)。

## 2. 门禁总表

| 门禁 | 状态 | 证据/边界 |
|---|---|---|
| 无 state 数据与 schema | 通过 | 522 episodes、119,700 rows；模型输入 state field/tensor=0 |
| Pass 1/Pass 2 与双 Layerwise FM | 通过（实现/训练） | 两次完整 Qwen、模型自产 prefix、NAV/ARM 均有非零梯度 |
| 80-step 小样本 overfit | **未通过** | 旧 overfit route 置信度未过，不因长训自动改写 |
| 四卡长训健康启动 | 通过后停止 | resume 后有连续健康 step；用户在 step 2090 后主动停止 |
| checkpoint 间隔 | 通过 | 每 500 step；step 2000 load gate 通过 |
| step 2000 inference export | 通过 | `step_002000@a8d57a22c515`，state field=0，单卡服务健康 |
| PCT/DWA known-waypoint | 通过（接线层） | reference PCT/DWA 可调用；不能外推模型导航成功 |
| 真实 cuRobo known-pose | 通过 | reachable/collision-free，reference identity 与 frame capability 通过 |
| strict 完整自主 Isaac | **未通过** | 首 query 的 NAV segment 19 平移超过合同 0.8 m |
| 原始 arm-vla 规则对照 | 已完成、**未通过** | 22×NAV 后 PICK；未接近物体，首 ARM yaw step 47.74° > 35° |
| 完整 pick-place | **未通过** | 没有成功导航、抓取、搬运和放置 |

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
| current GPU state | 四卡 0 MiB，无本任务 tmux/端口占用 |

resume 正确恢复了 Qwen、双 head、optimizer、scheduler 和随机状态；从 step 1000 后重新
计算，不声称保留旧父 run 中没有 checkpoint 的 1001–1181。新 run 的 step 2001–2090
同样没有 checkpoint；若未来续训，必须从 step 2000 写入全新 run。

## 4. step 2000 四组闭环结果

| profile | query | control step | failure |
|---|---:|---:|---|
| `contract` | 1 | 58 | `navigation segment 19 exceeds translation limit` |
| `executable-prefix-diagnostic` | 26 | 473 | 本地旧 `navigation_stall` 终止 episode |
| `unbounded-translation-diagnostic` | 9 | 301 | 取消 NAV 平移上限后仍被同一本地 stall 终止 |
| `arm-vla-reference` | 23 | 140 | 自主 PICK 的 target 0 超过原始 rotation step limit |

前两种诊断暴露：此前的 `navigation_stall` 是本地 `a852b5b9` 引入的 3 s 内目标距离没有
改善 1 cm 规则，并会把整个 episode 标记失败；它不是 reference 原始 stall detector。
`ace7d6e9` 的 `arm-vla-reference` profile 改用 reference detector，stall 或 250-step
chunk timeout 只停车并重新 query。

该 profile 同时不再做以下本地叠加：完整 20 点 horizon 审计、0.10 m PCT snap 拒绝和
重复 DWA 速度拒绝。仍保留的规则来自 reference 或批准合同：首点 0.8 m/45°、协议有限值、
ARM workspace/0.15 m/35°、cuRobo reachable/collision、关节与夹爪限制。

## 5. 初始化、NAV 与 ARM 结论

初始化顺序通过：对象先自由 settle 59 step，位移 `0.00453 m`，然后模型才读取 control
step 58 的 query anchor。失败不能归因于“机器狗尚未落地就推理”。

原始 reference 复测中，22 个 NAV 首点的平移 min/mean/max 为
`0.0060/0.0819/0.1356 m`，全部在 `0.18 m` 到达容差内。每次都正确产生
`first_waypoint_reached`、`failed=false` 和新 query，没有 `navigation_failed`。底盘全程
XY 位移仅 `0.0440 m`，到可乐距离从 `1.5458 m` 增至 `1.5558 m`，所以 NAV 没有完成。

模型随后自主输出 PICK。首个 absolute TCP target 相对当前 TCP 平移仅 `0.04126 m`，但
roll/pitch/yaw step 为 `33.44/0.91/47.74 deg`；yaw 超过 reference/合同的 `35 deg`，在
cuRobo 前失败。这是原始 arm-vla rule，不是额外本地门控。若要完全忽略该门禁，必须另建
明确标记的 unsafe diagnostic，不能称作 reference 或 production。

## 6. 视频与证据

最新 `arm-vla-reference` 三路视频为 69/70/70 帧，时长 2.76/2.80/2.80 s；远端/本地
SHA-256 一致并通过 `ffprobe`、末帧目检。overview 显示机器人没有走到源箱附近。证据位于
Git 忽略目录：

```text
artifacts/evaluation/waypoint_step002000_arm_vla_reference_20260821T015504Z/
```

strict、prefix 和 unbounded 三组 step 2000 证据保存在：

```text
artifacts/evaluation/waypoint_step002000_20260821T084024/
```

这些视频、trace、日志、checkpoint 和运行资产严禁加入 Git。

## 7. 下一步边界

当前没有训练或评测任务在运行。下一 checkpoint/续训若由用户重新授权，应优先修正模型
本身而不是继续添加 runtime 门控：

1. 让 NAV 首点尺度与 reference 0.18 m 到达容差一致，并验证真实目标距离持续下降；
2. 约束 route 切换时机，避免未接近源物体就输出 PICK；
3. 改善 ARM RPY target 的相邻变化和 cuRobo 可达率；
4. 依次重跑多 seed 开环、原始 reference 对照、ARM staged 和完整自主 pick-place；
5. production 默认 `contract` 仍须单独通过，诊断 profile 不能替代正式门禁。
