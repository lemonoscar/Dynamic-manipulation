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

cuRobo 对该 direct pose 返回无规划，当时 episode 随后以
`no_active_manipulation_chunk` 终止。最终 TCP 到物体
3D 距离仍为 `0.354422 m`，其中 XY `0.229265 m`、高度差约 `0.270282 m`；视频确认机械臂
在可乐上方/后方抬起，没有形成接触或夹取。

### 3.3 16.88 s 终止的后续根因与修正

后续逐层追踪证明，这不是“cuRobo 拒绝必然终止”的合同：

1. target0 已通过 TCP workspace/rate 验证；冻结 reference 的 `plan_pose`
   明确返回 `None`。
2. MANI executor 已正确返回 `failed=false`、`requires_requery=true`、零底盘且无新
   arm/gripper target，其语义是保持上一安全指令后重新询问。
3. rollout 执行了该保持周期，却没有处理 begin-level `requires_requery`，又调用了
   `_active=None` 的 `manipulation.step()`，才产生表中的终止原因。

当前修正严格限定为：只有合法 TCP 的结构化
`error_kind=plan_pose_unavailable` 会保持并完整重新推理；非法 TCP、cuRobo 服务异常和
其他规划异常仍 fail-closed。连续无规划只记录次数、sequence ID 与 TCP 差分，不引入新的
N 次重试/stall 终止门禁，仍由已有 episode/control-step 总 watchdog 给出最终上界。

2026-08-25 在 `waypoint-v2@78b545a1c1cbc00606d50feca9e638732f23f74d` 完成两层验证：

- 本地与 4xH20 的 48 项定向单测均通过。
- 新完整 seed 139 使用物理 GPU 2/3，运行 30 query / 1666 control step；第 29 个
  NAV 后模型在距物体 `1.012523 m` 时转 PICK，首 target 超过 `0.15 m` 平移限制，
  在 cuRobo 前以 `arm_target_rejected` 终止。因此它验证了“非法 TCP 仍 fail-closed”，
  但没有触发可恢复 None 分支，不得报告为任务或 None 恢复闭环通过。
- 为精确覆盖缺失分支，用历史 sequence 99 的真实当前 TCP/关节、预测 target 和
  两个 collision cuboid 调用实时 GPU cuRobo。planner 在 `30.690 s` 后再次返回
  `plan_pose=None`；新上层只记录 `manipulation_begin -> manipulation_requery`，下发零底盘、
  保持历史上一 arm target 和 gripper `open`，并返回 `completed=true`。没有
  `no_active_manipulation_chunk`。

私有证据 SHA-256：

- 精确 None 恢复 probe：
  `956e46e02d7b656fea79918c19ce61841d3547bd1e95383f9f2c0a3a27287f4a`；
- 新 seed 139 summary / trace：
  `97758c864e131042d023822cfe60aa5636dbdd181459284d3db7e18c1be9a144` /
  `816001bd571c8e6358fc15f35aa833008d521b8804b422a23ecb2dd2f817ed57`；
- overview / head / wrist 三路未截断视频：
  `987628c4ca319675f6878a5984d1f408f9996829332f20a05dec74ee7cd4df37` /
  `60c80859a17730d790637eed68a74098e3e7cf0e46e30a0aafbbd8e012c4a198` /
  `d749db9a43ecac748f1a3b7f2fde45bc53bc7aa8fd523a638a2cfff1c6a167e4`。

评测后只停止本次自有 model/cuRobo 服务，GPU 2/3 回到 `6/5 MiB`；GPU 0/1 的
无关 Ray 任务未被触碰。

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

## 7. 追加：seed 145/147 多种子闭环与硬门禁归因

### 7.1 预筛选与有效运行边界

2026-08-25 追加扫描 seed 140--151。所有 seed 先完成 object/base settle，只保存 head 首帧
和 truth-free 模型输入之外的评测几何；不查询模型、不执行 settle 后动作。只有 seed 145
（bearing `-14.7736 deg`、距离 `2.0306 m`）和 seed 147（bearing `2.3386 deg`、距离
`1.6198 m`）通过正前方 `±30 deg` 门禁，首帧目检也都直接看到红色可乐。其余 10 个 seed
的物体位于机身侧后方，因此没有拿来制造无意义的策略失败。

两类编排失败被排除在科学结果之外：首次预筛选同时关闭 dataset/video，导致相机根本没有
启用，三个 episode 均在 query 0 前以 `KeyError:'front'` 结束；seed 147 首次完整启动又因
常驻服务保留 seed 145 的相同 `episode_id=1, sequence=0`，在 query 0 返回
`stale_or_replayed_sequence`。前者通过只开启短视频渲染后重扫，后者通过重启同一冻结模型
服务、使用全新输出 identity 后重跑；两者均未进入下表。

### 7.2 三种子闭环对照

seed 145/147 与 seed 139 使用相同 step 1250、模型随机 seed、`K<=10`、真实 PCT/DWA、真实
cuRobo/IK、target0-only MANI 和原始 fail-closed 规则。没有 required-first、stop-after、
external truth/FSM 或 ARM bypass。

| 指标 | seed 139 | seed 145 | seed 147 |
|---|---:|---:|---:|
| 初始 base-object XY | 1.259882 m | 2.029790 m | 1.619777 m |
| 最小 base-object XY | 0.664094 m | 0.695448 m | 0.686932 m |
| 末 query base-object XY | 0.720015 m（PICK 切换） | 0.733920 m | 0.712646 m |
| base XY 累计路径 | 0.973914 m | 1.561529 m | 1.373009 m |
| route query | 92 NAV + 8 PICK | 175 NAV + 1 RECOVER | 151 NAV + 1 RECOVER |
| 是否进入 MANI | 是，7 个 target0 成功 | 否 | 否 |
| 直接终止事件 | 第 8 个 TCP pose 无规划 | route confidence `<0.55` | route confidence `<0.55` |

seed 145 最后一帧 route 分布为 `PICK=0.531089`、`NAV=0.468685`；seed 147 为
`PICK=NAV=0.499888`。两者都正处在模型的连续 crossover，却被部署配置中的
`route_confidence_min=0.55` 转换成 `RECOVER`。这不是外部 GT phase/FSM 覆盖 route，但它是
确定性的 runtime 硬阈值，直接阻止了下一次自主 PICK。该阈值在 v1 合同中只是初始提案，
正式值原本就要求在 validation reliability calibration 后冻结；当前多种子证据否定了直接
沿用 `0.55` 作为已校准正式值。

两个新 seed 的 NAV selector 均严格未越过 index 9，非空 stall diagnostics 为 0；机器人
没有跌倒。它们先接近到约 `0.69 m`，随后在可乐前反复短重询并回退，说明硬阈值是最后的
直接终止原因，但不是唯一根因：NAV 末段 waypoint/route 泛化本身仍不稳定。

### 7.3 夹爪初态不是直接失败门禁

seed 145/147 的 query 0 两指关节都接近 `0 m`，即 reset 初态接近闭合；但全部有效 NAV
control step 都明确命令 `open`，终止前两指均已达到约 `0.040 m`。因此当前 runtime 会在
进入 PICK 前把 reset 的 open/close 差异标准化为 open，两个新 seed 又都没有调用 ARM head，
夹爪初态不可能解释其失败。

seed 139 同样以 NAV `stow_open` 进入 PICK。其第 8 个被拒 target 的 gripper 分量还是
`1.0=open`，cuRobo 拒绝的是 6-DoF TCP pose 的 collision/IK/plan 可执行性，不是“夹爪闭合
门禁”。但是 522/522 个 episode 的 PICK boundary target0=close 仍是独立、确定的数据错误：
它会令已经打开的夹爪先关闭再重开，即使所有 TCP pose 都可规划，也会破坏接近--对准--闭合--
抬升的因果顺序。结论不是简单取消安全门禁，而是分别修正 route threshold calibration、
NAV 末段泛化、PICK command 时序和 MANI pose 可执行性。

### 7.4 追加证据完整性

- seed 145 summary / trace SHA-256：
  `418865b0501fecee38a5dd43558a30bb51672f3a4bed83ad7ee7e72cf20d088a` /
  `a39cbf4ab602601d05da9a6fae18b6a44d53658c3d77e9e6420ac466bf17bf24`；
- seed 147 summary / trace SHA-256：
  `3b01c06d445b38c388612a2a49aefbb56ec84e6e4983c37ba67ef04eb87c1e41` /
  `1f6929f549d955b9186609bf3122b83743cc451600d896a350f14f48aa3a0e8a`；
- 1920x720 三视角拼接视频：seed 145 为 247 frame、9.88 s、SHA-256
  `2f7cd622fd1dce4920eb256f6cce554c9253817851e3c13cd4145aab19739671`；seed 147 为
  325 frame、13.00 s、SHA-256
  `02bcee598df4e1a0eb0f0a5cc78075aad774c965ac64ebf5e052b9adb141760a`；
- 两个拼接视频和六路原始视频均完成全流解码；overview 各丢 1 帧，head/wrist 丢帧为 0；
- 最终只停止本次自有模型/cuRobo 服务；物理 GPU 2/3 回到 6/5 MiB，GPU 0/1 的无关进程
  与显存占用未改变。
