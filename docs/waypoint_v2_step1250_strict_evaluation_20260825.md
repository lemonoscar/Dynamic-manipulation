# Waypoint v2 step 1250 严格开环与完整自主闭环评测

- 评测日期：2026-08-25 CST
- checkpoint：`step_001250`
- 训练代码：clean `waypoint-v2@4fb50ffa8f0a05eeda5d9dcc34a898658ba8d9f3`
- 训练状态：已按用户指令停止；最后 durable checkpoint 为 step 1250
- 科学结论：结构与静态开环基本成立，但完整闭环失败；当前数据存在确定的 PICK 起始夹爪时序错误，禁止把本结果解释为“只需继续训练”

## 1. 身份与测试边界

step 1250 checkpoint 完整加载通过：trainer/scheduler 均为 1250，双 rank 共检查
`5,025,556,280` 个 parameter-partition value，非有限值为 0。冻结身份如下：

| 项目 | 值 |
|---|---|
| dataset manifest SHA-256 | `6f534e1b7ed456ab6595985d7148eea5e9ff214d4e6a308c5e34baa93fa2506f` |
| normalizer SHA-256 | `e781bfed2661befa77dc13cdc3d4a7b88a77ee2678562fc952089f6cc307dc4a` |
| resolved config SHA-256 | `f914462a34b210bc969386669594c3f23d07313c1cc31f8477f938b17bbf1401` |
| FM Monte Carlo draws | 4 |
| self-conditioned loss at step 1250 | 关闭；调度从 step 1500 后开始 |
| learned prefix / CRL / B5 on-policy correction | 全部关闭 |
| runtime NAV trusted-prefix cap | `min(model_K, 10)`；当前模型无 learned K，故实际 cap 为 10 |

开环读取 truth 只用于选样和计算指标；truth 没有写入模型请求或控制链。闭环使用预冻结
seed 139、settle 后可乐在 head view 可见的主场景；没有 required-first、stop-after、ARM
bypass、外部 phase/FSM 或已删除的 local fatal stall。NAV 使用真实 PCT/DWA，MANI 使用真实
cuRobo/IK 和 `chronological_target0`，route 始终只来自 Qwen Pass 1。

两次仅涉及 Isaac GPU ordinal 的启动失败没有进入模型、场景或控制证据。第三次启动完成
Vulkan physical device、CUDA visible ordinal 和 PhysX ordinal 的同卡映射后，才作为有效闭环。
该映射只存在于机器私有 launcher，没有修改公开控制代码。

## 2. 针对性严格开环

选取 val 64 row，覆盖 5 个 route、4 种 boundary 各 11 个窗口样本及 20 个 phase-interior
样本。动作使用 seeds `17/29/43/71`，固定 validation noise bank 使用 seed `20260822`、
4 draws；生成并检查 NAV `[20,3]` 与 MANI `[20,7]` 的完整 20-step target/predicted 图。

### 2.1 Route、boundary 与 progress

| 指标 | step 1250 |
|---|---:|
| route accuracy | 93.75%（60/64） |
| format invalid | 0 |
| RECOVER | 2/64 |
| phase-interior macro accuracy | 100%（20 row） |
| boundary AUROC / F1 | 0.976834 / 0.973684 |
| phase progress MAE | 0.058748 |
| time-to-boundary MAE | 0.793910 s |
| transition flicker | 0 |
| 四个 transition 的 switch lag | 均为 1 query，即 0.20 s |

静态序列中 `NAV_TO_SOURCE→PICK` 和 `NAV_TO_TARGET→PLACE` 没有 early/late switch；
`PICK→NAV_TO_TARGET` 与 `PLACE→DONE` 各有 `1/6=16.67%` late-switch window。该结果只证明
held-out 专家轨迹附近的排序，不证明闭环物理切换距离正确。

### 2.2 动作质量

| 指标 | NAV | MANI |
|---|---:|---:|
| mean position error | ADE 0.081099 m | TCP 0.015223 m |
| terminal error / orientation | FDE 0.147646 m | 0.439520 rad |
| first direction / gripper accuracy | 75.0% | 97.679% |
| segment / inter-target step violation | 3.125% | 75.893% |
| workspace violation | — | 0% |
| row-level normalization OOB | 47.917% | 100% |

NAV horizon MAE 从 step 1 的 `0.013594` 增至 step 20 的 `0.095588`；MANI 的平均相邻
RPY step 为 `0.249646 rad`。固定 bank 的 mean FM loss 为 NAV `0.051159`、MANI
`0.084587`。评测器的非 overfit profile 会把有限值、shape 和 coverage 通过记为
`status=pass`，但它没有对上表的高 MANI step-violation/OOB 设收敛阈值；因此这里的严格
科学判定不是“动作门禁通过”。

learned prefix 未启用，报告中的 predicted `K=20` 是 fallback，不是 prefix head 的预测；
`K MAE=9.04` 和 overrun `56%` 不能用来评价未训练的 prefix head。闭环只验证 runtime cap
没有越过 index 9。

## 3. seed 139 完整自主闭环

### 3.1 总结果

| 项目 | 结果 |
|---|---|
| 最终状态 | **FAILED** |
| failure reason | `no_active_manipulation_chunk` |
| model query / control step | 100 / 845 |
| route query | 92×`NAV_TO_SOURCE`，8×`PICK` |
| route flicker | 0；只发生一次 NAV→PICK |
| PCT/DWA / terminal-yaw chunk | 6 / 86 |
| selected waypoint index | 0–9；越过 cap 为 0 |
| deleted stall diagnostics | 全部为 null，未恢复废弃 stall 合同 |
| 抓取/搬运/放置 | 均未完成；物体未移动 |

初始 base 到可乐平面距离为 `1.259882 m`。导航期间最小达到 `0.664094 m`，随后回退；
Qwen 在 `0.720015 m` 处以 `PICK=0.562058`、`NAV=0.437731` 切换。累计 base XY 路径
`0.973914 m`，只减少约 `0.540 m` 直线距离，仍可见短 waypoint 反复重询与末端回退。
机器人没有跌倒。

一旦切到 PICK，底盘命令和实测速度持续为 `[0,0,0]`，没有再调用 NAV action head，满足
“MANI 时底盘必须站住”的执行合同。

### 3.2 MANI 执行链

sequence 92–98 连续完成 7 轮：

```text
重新观察 → Qwen 仍输出 PICK → MANI FM target0
          → 当前 target0 通过 safety/collision/IK
          → cuRobo joint path 执行 → first_tcp_target_reached → 重新观察
```

每轮只执行 index 0，没有扫描或跳选 suffix；joint path 长度依次为
`41, 41, 21, 21, 41, 41, 41`，orientation fallback 均未使用。第 8 轮 sequence 99 的
target0 为：

```text
[0.623657, -0.016330, 0.182156,
 -0.309914, 1.454237, -0.322041, 1.0]
```

cuRobo 对该 direct pose 返回无规划，原始 fail-closed 规则令 episode 终止。最终 TCP 到物体
3D 距离仍为 `0.354422 m`，其中 XY `0.229265 m`、高度差约 `0.270282 m`；视频确认机械臂
在可乐上方/后方抬起，没有形成接触或夹取。

## 4. 新发现的确定根因：PICK 起始监督反向

闭环切换前 NAV 的 `stow_open` 令两个夹爪关节稳定在约 `0.040 m`，即张开。进入 PICK
base-settle 时也继续保持 open。但前三个 MANI query 的预测 gripper target 分别约为
`0.076、0.089、0.196`，runtime 按冻结合同解释为 close，夹爪实际从 `0.040 m` 关到接近
`0 m`；之后预测升到 `0.568、0.997、0.964、0.971`，夹爪反而重新张开。

这不是单个 diffusion seed 的随机误差。对冻结 command-gripper 数据的全部 522 个 episode
重新审计得到：

| PICK 首边界监督 | episode 数 |
|---|---:|
| target0=`close` | **522/522** |
| target0=`open` | 0/522 |
| 首次 `open` 在 horizon index 5 | 471 |
| 首次 `open` 在 horizon index 6 | 51 |

原始专家轨迹在 route 切到 PICK 后先进入 `plan_pick`。规划等待期间显式 command 为空，派生器
按“沿用上一显式命令”保留了 NAV 末尾的 close；直到 `exec_pick` 才出现 `open`。以本次开环
代表样本为例，PICK boundary row 44 的 future source rows 45–49 全为 close，row 50 才 open，
于是其 20-step target 是“前 5 点 close → approach 时 open → 末端 close”。step 1250 的
预测曲线准确复现了这个 target。

因此旧审计只证明“第 7 维来自专家 command 而非 measured opening”，却漏掉了更高层的
可执行语义：`plan_pick` 是专家规划延迟，不应成为 VLA 的物理动作意图。闭环 runtime 又在
NAV 中明确使用 `stow_open`，两者构成确定的训练/推理 precondition mismatch。这直接违反
此前写下但未真正执行的门禁“PICK target0 在接近阶段保持 open，TCP 到 grasp pose 后才
close”。

## 5. 结论与下一门禁

step 1250 已证明：

1. state-free 两次 Qwen、B2 boundary/progress、S4 FM、terminal-hold、K≤10、真实
   PCT/DWA、MANI base lock、target0-only receding query 和真实 cuRobo/IK 均已接通；
2. 模型能自主完成 NAV→PICK，切换后底盘确实站住，且机械臂能连续执行 7 个自产 target；
3. 但 NAV 末段仍有短 waypoint 重询/回退，切换点偏远，MANI orientation/step continuity
   未过；
4. 最关键的是，当前 immutable 数据把所有 PICK 起始 target0 标为 close。继续在相同数据上
   从 step 1250 训练，最多会更牢固地学习错误时序，不能作为修复。

下一版数据必须使用全新 schema/manifest，显式排除或重标 `plan_pick` 非执行等待帧，并新增
全量硬审计：PICK boundary/approach 的 target0 必须 open；close crossover 只能发生在 TCP
已进入 grasp 区域之后；随后必须存在 close-hold 和 lift/retract。完成 8–16 episode overfit、
相同 targeted open-loop、cuRobo 可执行率与完整闭环前，不应恢复全量长训。

## 6. 证据完整性

- open-loop JSON SHA-256：`ff7ff0a28fc8c6932f3759b9b91a8ec6e43036ccb9bd2c1b4c81d0a6708bb01a`
- closed-loop summary SHA-256：`b3d8976e18b6623fcf2e22bed8ba23972157e94375bbef3f1036ea8d879a9a5f`
- full query/control trace SHA-256：`1737b6aa5cd0de459bcb229bcbe166802e63fa8d3efe61efa2f1aaba3acdac4b`
- 16.88 s、422-frame、1920×720 H.264 三视角拼接视频 SHA-256：
  `eea109dd96705adae3abba03406de7d97f7352b0e0a413c352804051f3eabbb1`
- 原始 overview/front/wrist 均完整解码；帧数分别为 422/423/423，overview 丢 1 帧，另外两路
  丢帧为 0。
- 评测结束后只停止并释放本次模型和 cuRobo 服务；物理 GPU 2/3 回到空闲，GPU 0/1 的无关
  进程未触碰。
