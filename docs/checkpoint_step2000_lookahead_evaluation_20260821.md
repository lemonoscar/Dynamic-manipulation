# step 002000 lookahead 选点策略完整自主闭环复测

复核时间：2026-08-21 11:58 CST。被测模型为
`step_002000@a8d57a22c515`，runtime 为
`feature/conveyorvla-waypoint-v1@cfed498eff780d390426962f309a3002173e9ed3`，固定
arm-vla reference 为
`arm-vla-grasp-sim@388b6818f4c605a707d13c519fbb58b1d07acd92`。

## 1. 结论

新 lookahead 策略解决了“每个 20 点 chunk 只取首点，而首点落在 0.18 m 到达容差内，
导致几乎不导航”的执行问题。正式 seed 139 闭环中，18 个 NAV chunk 实际选取 index 2–9，
选点平移半径为 `0.3735 / 0.4948 / 0.5811 m`（min/mean/max）；机器人进行了 3712 个
control step、18 次真实 PCT/DWA 导航，并由模型自主从 `NAV_TO_SOURCE` 切换到 `PICK`。

这仍不是完整任务成功。底盘到可乐的 query-anchor 距离从 `1.3164 m` 一度下降到
`0.3804 m`，随后因模型 waypoint 方向振荡又增大，模型在 `0.4931 m` 时以 `0.9478`
概率自主选择 PICK。ARM chunk 的 target 0 和 target 1 合法，但 target 2 相对 target 1 的
roll/yaw 跳变为 `40.44 / 38.13 deg`，超过原始 arm-vla/批准合同的 `35 deg` 相邻旋转
步长上限，因此在 cuRobo 调用前 fail-closed。

本轮证明新选点策略已使 action horizon 20 被用作 lookahead 候选，而不是错误地退化为首点；
尚未证明 step 2000 模型具备稳定目标趋近、ARM chunk 连续性或完整 pick-place 能力。

## 2. 选点策略

`trusted-prefix-target-lookahead-pct-v1` 对每次模型返回的固定 `[20,3]` NAV chunk 执行：

1. 只读取 `action_valid_mask` 的连续有效前缀，并将候选限制在前 10 点；
2. 去掉平移小于 `0.03 m` 且转角小于 `3 deg` 的退化点；
3. 优先保留平移不小于 `2 × goal_tolerance` 的点；本次 reference profile 的
   `goal_tolerance=0.18 m`，所以最小有效 lookahead 为 `0.36 m`；
4. 按平移半径最接近 `0.50 m` 排序，同分时取时间更早的点；
5. 若没有点达到最小 lookahead，则退化为可信前缀中最远的非退化点；
6. 按排序依次调用 PCT，选择首个可规划候选，再由 DWA 执行；到达或 250 control-step
   chunk timeout 后停车并重新询问模型。

这不是固定取尾点，也不把完整 20 点一次性开环执行。后 10 点不会直接驱动机器人；每次只
执行一个经 PCT 验证的 lookahead 目标，然后以新视觉重新推理。默认 `contract` profile
采用同一选点逻辑，但保持自己的 `0.12 m` 容差，因此最小 lookahead 为 `0.24 m`。

新增 `lookahead-arm-vla-reference` profile 用于能力对照：保留原始 arm-vla 的 DWA、
0.18 m 到达容差、reference stall detector 和 250-step 重询语义；不恢复此前已经确认会
掩盖模型能力的本地完整 horizon、PCT snap、重复 DWA 速度或 3 s/1 cm stall 拒绝。PCT
仍为必需且不允许 fallback。

## 3. 实现与回归

实现提交为 `cfed498eff780d390426962f309a3002173e9ed3`，包括：

- `rank_navigation_waypoints` 的确定性候选排序和 PCT 候选回退；
- executor trace 中的 selection policy、可信长度、最小/目标 lookahead、候选顺序、PCT
  拒绝和最终 selected index；
- rollout 的初始源物体可见性 preflight；
- contract、architecture、operations 和对应单元测试同步更新。

本地 targeted suite 为 `33 passed`；本地及 4xH20 的 Waypoint relevant suite 均为
`53 passed`。冻结 Isaac Python 3.11 环境有 pytest、没有 accelerate；冻结训练 Python
3.10 环境有 accelerate、没有 pytest，因此没有为了强行同进程收集训练测试而修改远端环境。
模型服务仍直接通过 checkpoint export/load gate 验证。

## 4. seed 与输入边界

seed 不是沿用旧 task 的几何结果，而是对正式
`configs/tasks/waypoint_closed_loop_task_r2.json` 的 runtime annotation、box-pair
randomizer、collision PLY、task randomization 和 base-goal randomization 重新扫描 1000 个
seed 后选定。seed 139 的 settle 后 preflight 为：

| 项目 | 值 |
|---|---:|
| 可乐相对底盘 bearing | `-0.60894 deg` |
| 可乐相对底盘距离 | `1.31635 m` |
| body-frame XY | `[1.31628, -0.01399] m` |
| front RGB | `480×640×3`，存在 |
| source truth 发送给模型 | `false` |

几何 gate 只用于在评测启动前拒绝侧向/背向 seed；模型 request 仍只有 instruction、
head `[t-0.20,t]` 和 wrist `[t-0.20,t]` 四张图，state field 数为 0。首帧人工目检确认红色
可乐罐清晰位于头部画面中央，overview 确认机器人正对源桌。对象完成物理 settle 后才进行
preflight 和第一次模型 query，因此不存在未落地就推理。

## 5. 完整自主闭环结果

评测在 4xH20 上使用 GPU 0 运行 step 2000 模型服务、GPU 1 运行真实 cuRobo 服务，并由
Isaac/locomotion/PCT/DWA 完整闭环。没有 required-first-route、oracle route、GT phase、
外部 FSM 或 state 覆盖；最大预算为 400 query / 24000 control step，episode 自然退出。

| 指标 | 结果 |
|---|---:|
| route 序列 | `18 × NAV_TO_SOURCE → PICK` |
| query / control step | `19 / 3712` |
| 仿真 / 墙钟时长 | `74.24 / 744.58 s` |
| NAV selected index 范围 | `2–9` |
| selected 半径 min/mean/max | `0.3735 / 0.4948 / 0.5811 m` |
| NAV 到达 / timeout 重询 | `8 / 10` |
| PCT 成功 | `18 / 18` |
| PCT 候选拒绝 | `0` |
| 初始 / 最小 / PICK 时物距 | `1.3164 / 0.3804 / 0.4931 m` |
| query-anchor 路径长度 | `7.3074 m` |
| 首末底盘净位移 | `0.9352 m` |
| episode success | `false` |
| failure | `arm target 2 exceeds rotation step limit` |

18 个 query-anchor 的物距不是单调下降：

```text
1.316, 1.138, 0.809, 0.806, 0.460, 0.674, 1.105, 0.986, 0.530,
1.048, 1.133, 0.777, 1.193, 1.083, 0.699, 0.380, 0.836, 0.751,
0.493 (PICK)
```

因此选点执行层已经让机器人移动，但模型预测方向仍明显振荡。sequence 15 距离最小却因
底盘朝向导致可乐不在 front frame；sequence 18 的 PICK frame 中可乐位于画面左下并清晰
可见，说明 route 切换有视觉依据，但切换距离和 ARM target 质量尚不稳定。

## 6. ARM failure 诊断

PICK query 下当前 TCP 的 query-base pose 为：

```text
[0.375286, -0.000553, 0.206160, 0.000008, 0.001342, -0.000219]
```

前三个模型 target 的相邻变化为：

| target | translation | roll / pitch / yaw step |
|---:|---:|---:|
| 0 | `0.03390 m` | `12.03 / 7.83 / 5.93 deg` |
| 1 | `0.05346 m` | `7.15 / 7.88 / 4.24 deg` |
| 2 | `0.04414 m` | `40.44 / 19.56 / 38.13 deg` |

target 2 的 translation 合法，但 roll 和 yaw 超过 `35 deg`。validator 按批准合同审计
完整有效 ARM prefix，所以整个 chunk 在开始执行前被拒绝，cuRobo 没有收到非法 target。
该限制来自 arm-vla reference/批准合同，不是本轮新增的导航门控。

## 7. 视频与运行证据

三路完整视频已从远端下载，远端/本地 SHA-256 一致：

| stream | 分辨率 | 帧数 | 时长 | SHA-256 |
|---|---:|---:|---:|---|
| front | 640×480 | 1856 | 74.24 s | `8e80c076dfe59bfdfb898796e2226e469091a306b98b711e42d0c170150680d0` |
| overview | 1280×720 | 1855 | 74.20 s | `f92e8dc1e1348af30dc3d43c2633a75ffdf9d089665cd4f6f978e2cd0f0b75b4` |
| wrist | 640×480 | 1856 | 74.24 s | `be136d77981779d4a5b889be5c455b13061be3396f49949d9dcfb540b0617dae` |

视频、summary、完整 trace、启动脚本、抽帧和日志位于 Git 忽略目录：

```text
artifacts/evaluation/waypoint_step002000_lookahead_20260821T033113Z/
```

远端 run root 为：

```text
artifacts/runs/conveyorvla-waypoint-v1-step2000-lookahead-eval-20260821T033113Z
```

评测结束后仅停止本次创建的服务/tmux。GPU 0/1 服务均已退出，四张 H20 回到
`0 MiB / 0%`，没有残留本任务进程。

## 8. 当前判断

新 selector 可以作为后续闭环的通用 receding-horizon 策略：它使用可信前缀和物理尺度，
同时让 PCT 决定最终可执行候选。下一次模型训练/评测应关注模型本身的 waypoint 方向一致性、
接近源物体后的 route 切换距离，以及 ARM target 的相邻姿态连续性；不应再用“取第一个点”
或新增导航安全门控来掩盖这些模型问题。
