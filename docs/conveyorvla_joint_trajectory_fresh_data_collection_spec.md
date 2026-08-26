# ConveyorVLA 全新 Joint-Trajectory 数据采集规范

> 状态：grill 决策已确认；采集尚未开始，2026-08-27 复核
> 范围：Go2-X5、Liangzhu 场景、四 route 的完整导航—抓取—运输—放置任务
> 目的：为新的双 action-expert 方案采集全新、可追溯、可直接执行的专家数据
> 重要：本文定义的是新的 breaking-change 数据身份，不覆盖、不转换、也不混训冻结的
> Waypoint v1/v2 数据。

## 1. 结论先行

首个正式版本建议采集 **1,600 条完整成功 episode**：

| split | 成功 episode | 比例 |
|---|---:|---:|
| train | 1,280 | 80% |
| validation | 160 | 10% |
| test | 160 | 10% |
| 合计 | 1,600 | 100% |

建议采用分段发布，而不是一开始直接生成全部数据：

| 阶段 | 数量 | 是否进入正式集 | 用途 |
|---|---:|---|---|
| pipeline smoke | 32 条成功 | 否 | 检查 schema、时钟、视频、关节命令和磁盘占用 |
| overfit gate | 12 条成功 | 是，来自 train | 覆盖四 route、三 boundary，验证模型能够记住完整动作 |
| learning pilot | 200 条成功 | schema 完全一致时可进入 | 验证 direct-joint head、连续夹爪和速度包络 |
| formal release | 1,600 条成功 | 是 | 第一版正式训练、验证和测试 |
| optional extension | 每次增加 400 条，最多先到 2,400 | 新 immutable revision | 只有 held-out 曲线仍明显受数据量限制时才增加 |

1,600 不是理论上的“越多越好”，而是当前单场景、单目标物、预训练 Qwen 和
step1250 选择性 warm-start 条件下的第一版合理规模。它约为旧 522 episode 的三倍，且每条
成功 episode 都提供三次 route boundary 和一次独立 evaluator success。若 1,200→1,600 条的
held-out 动作误差、route flicker 和闭环成功率均已进入平台期，不应为了凑数继续采集；若仍
稳定改善，再按 400 条一批扩展到 2,400。

核心时钟和输出合同冻结为：

| 项目 | 建议值 |
|---|---|
| physics | 400 Hz |
| low-level control / raw state / applied command log | 50 Hz |
| camera | 25 Hz |
| 派生训练 query anchor | 5 Hz |
| NAV output | `[10,3] @ 0.20 s`，覆盖 2.0 s |
| Mani output | `[10,7] @ 0.04 s`，覆盖 0.4 s |
| Mani input state | 6 关节角 + 6 关节速度 + 1 夹爪开度，共 13D |
| route | `NAV_TO_SOURCE/PICK/NAV_TO_TARGET/PLACE` |
| runtime route commit | 初始和切换均需两次新观测一致确认 |

## 2. 为什么必须重新采集

旧 Liangzhu 数据保存了实测关节角和 Cartesian action，但没有保存机械臂控制器内部真正
下发的六轴 `_arm_target`。旧数据只能用未来实测关节轨迹近似 direct-joint 目标，无法严格
回答“模型输出与执行器收到的命令是否相同”。旧 command-gripper v2 还已被证明存在
522/522 个 PICK 起始边界 target0 错误闭合的问题。

全新采集必须同时记录：

1. 控制器请求的关节目标；
2. 经过 joint limit、rate limit 和插值后的实际下发关节目标；
3. 同时刻实测关节角与关节速度；
4. 请求和实际下发的连续夹爪目标；
5. base command、base pose/twist、相机和统一时间戳。

训练标签使用第 2 项——**实际下发的目标**。请求值只用于审计，实测值用于 Mani state 和
tracking-quality audit。若某一控制步缺少实际下发目标，必须拒绝该 row，禁止回退到未来实测
关节角进行静默替代。

新的推荐数据身份为：

```text
schema:  conveyorvla-joint-trajectory-v1
profile: conveyorvla-liangzhu-fresh-joint-trajectory-v1
```

最终 materialize 时必须生成唯一 dataset ID、manifest SHA-256、normalizer SHA-256 和 source
episode hash；上述名称只是 schema/profile，不是可变目录别名。

## 3. 模型可见数据合同

### 3.1 Pass 1

Pass 1 只能读取：

- 全局任务文本；
- head `[t-0.20 s, t]` 两帧；
- wrist `[t-0.20 s, t]` 两帧。

Pass 1 不得读取关节状态、base pose、GT route、teacher phase、物体位置、previous route 或
runtime pending route。它受约束生成 `ACTION`、四选一 route 和 subtask 文本；不生成 DONE。

### 3.2 Pass 2 与两个 action expert

Pass 2 的 Qwen 仍只读取相同视觉、任务和模型自己生成的完整 assistant prefix。两个动作头
共用 Qwen hidden，但参数和动作时钟独立：

- NAV expert：无 state 输入，输出 `[10,3]` body-frame base reference trajectory；
- Mani expert：额外读取 13D 连续 state token，输出 `[10,7]` direct-joint trajectory。

13D state 只能进入 Mani action expert，不能进入 Qwen、Pass 1、prompt 或 NAV expert。

### 3.3 Mani 动作定义

在 query 时刻 `t` 读取实测六轴关节位置 `q_measured(t)`。第 `k` 个训练目标为：

```text
delta_q[k] = q_command_applied(t + (k+1) * 0.04 s) - q_measured(t)
gripper[k] = gripper_command_applied(t + (k+1) * 0.04 s)
```

其中 `k=0..9`，夹爪约定 `0=完全闭合、1=完全张开`。runtime 还原：

```text
q_target[k] = q_measured(query) + delta_q[k]
```

这是 query-relative 关节位置轨迹，不是相邻动作增量，也不做逐步积分。夹爪是连续绝对目标，
runtime 不做二值阈值 gate，只进行有限值检查和 `[0,1]` 物理裁剪。

### 3.4 NAV 动作定义

第 `k` 个 NAV target 是 query body frame 下的未来 base reference：

```text
[dx_body, dy_body, dyaw]
time_offset = (k+1) * 0.20 s, k=0..9
```

所有点相对同一个 query pose，不是相邻 waypoint delta。runtime 信任完整 10 点，使用第 10 点
作为本次 PCT/DWA 的局部目标；前 9 点提供路径形状和连续性，不再做 K 搜索或 waypoint
selector。

### 3.5 边界后缀

两个 action expert 都完整监督 10 点，不再存在 `K*`、prefix head、`L_prefix` 或运行时
trusted prefix。若 10 点 horizon 穿过真实 route boundary：

1. 保留属于当前 route 的连续动作；
2. boundary 后重复最后一个当前 route 的合法 target；
3. 派生数据可记录 `terminal_hold_start_index` 供审计，但不得把它作为模型输入或 runtime
   selector。

episode tail 和 evaluator-success tail 同样使用 terminal-hold；不得通过 mask 让模型输出未监督 suffix。

## 4. 专家演示必须呈现的完整物理顺序

每条训练 episode 必须完成全部序列，失败 episode 可保留在 raw diagnostics，但不得进入专家
训练 split。

### 4.1 `NAV_TO_SOURCE`

1. 机械臂保持当前安全姿态，夹爪状态可以来自随机初始化；
2. PCT/DWA 驱动底盘接近 source；
3. 最后约 0.5 m 使用走—停式精确接近，不使用低于 locomotion 有效包络的连续微速；
4. 到达后 base command 为零，并满足稳定速度门禁至少 0.4 s；
5. 保留至少两张间隔 0.2 s 的新观测，支持 runtime 初始/route 双确认。

### 4.2 `PICK`

1. 全阶段 base command 强制 `[0,0,0]`；
2. 若夹爪初态不是 open，先明确打开；
3. 张开状态下完成 reach；
4. 继续对准并下降，不能把“到达第一个 target”解释为立即闭合；
5. 到达 grasp pose 后稳定 0.12–0.20 s；
6. 连续闭合夹爪，保持闭合；
7. 垂直抬升，再进入稳定 carry posture；
8. 只有 grasp、lift、carry-ready 都成立后才标注 `PICK→NAV_TO_TARGET`。

训练集硬门禁：任何 close command 早于 grasp alignment 都必须拒绝，不能依靠模型训练自行
修正错误教师时序。

### 4.3 `NAV_TO_TARGET`

1. 机械臂保持最后的 carry joint target；
2. 夹爪保持闭合；
3. loaded navigation 使用比空载更保守的前进和转向速度；
4. 到达目标箱后 base command 为零并稳定至少 0.4 s；
5. 保留连续两次新观测，再进入 PLACE。

### 4.4 `PLACE` 与 evaluator success

1. 全阶段 base command 强制 `[0,0,0]`；
2. 保持夹爪闭合，移动到目标箱上方；
3. 慢速下降到合法释放高度；
4. 稳定后连续打开夹爪并保持；
5. 必要时以关节轨迹安全回撤；
6. 物体脱离夹爪并在目标 box 有效区域内连续保持至少 1.0 s 后，由 evaluator 记录 success
   并结束；物体姿态不限。

GT 物体位置、接触、抓持和目标箱状态只用于教师、标签和 evaluator termination，不得写入
模型 request、route 或 action 控制链。

## 5. 时钟与整个运行速度

### 5.1 采集时钟

| 数据流 | 频率 | 说明 |
|---|---:|---|
| physics | 400 Hz | 不得通过增大 physics dt 加速采集 |
| low-level command/state | 50 Hz | 每 20 ms 保存 measured 和 applied target |
| camera | 25 Hz | head、wrist；overview 仅审计 |
| direct-joint target | 25 Hz | 每 40 ms 一个 Mani action，50 Hz 控制器做两 tick 插值 |
| 派生 query anchor | 5 Hz | 相邻 row 间隔 0.20 s，允许 action chunk 重叠 |
| NAV target | 5 Hz | 10 点覆盖 2.0 s |

所有流必须由同一个单调时钟和明确 tick ID 对齐，禁止按“最近帧”猜测配对。图像与 query
state 的时间差必须写入 manifest；正式数据要求绝对误差不超过一个 25 Hz tick，即 40 ms，
目标值应为不超过 20 ms。

### 5.2 runtime query 节奏

runtime 不会每执行一个低层动作就运行 Pass 1：

```text
NAV:  Pass1 → Pass2 → 执行 2.0 s reference/PCT-DWA → 新观测
Mani: Pass1 → Pass2 → 顺序执行 10 点/0.4 s → 保持末点 → 新观测
```

初始 route 和新 route 需要两次新观测确认。第一次出现新 route 时，base 零速、机械臂
保持最后 target；第二次同一新 route 的概率仍高于已确认 route 后，才用第二次的完整模型
prefix 运行 Pass 2。

建议性能目标：

| 指标 | 建议门槛 |
|---|---:|
| Pass1+Pass2 latency median | `<= 0.8 s` |
| Pass1+Pass2 latency p95 | `<= 1.2 s` |
| 单次 service timeout | `5 s`，超出按推理服务异常处理 |
| 完整 episode 目标时长 | median `60–90 s` |
| 正式 episode 上限 | `120 s` |

动作完成到新推理返回之间允许停顿。停顿是显式 inference hold，不是旧 PCT stall 合同，也不
触发外部 route 切换。为减少训练—运行差异，采集教师应在安全位置随机加入 `0.0/0.4/0.8/
1.2 s` 的 hold，其中 nominal 权重建议为 `10%/25%/45%/20%`；同一 hold 内保持最后的 base、
关节和夹爪目标，并在 hold 后根据新状态重新规划教师轨迹。

不要把 3 s 以上延迟当作正常训练随机化。它应首先作为部署性能问题修正，而不是用大量静止
数据掩盖。

### 5.3 采集吞吐估算

不能用 episode 数直接猜 wall time。正式开采前，先用 32 条 smoke 实测：

- 平均成功 episode 仿真时长 `T_episode`；
- real-time factor `RTF`；
- 成功率 `Y_success`；
- 每条 raw/derived 数据实际字节数。

预计 wall time：

```text
wall_hours = target_successes * T_episode
             / (3600 * RTF * parallel_workers * Y_success)
```

示例仅用于容量规划：若 `T_episode=75 s`、`RTF=0.7`、两个独立 worker、成功率 `0.85`，采集
1,600 条成功 episode 约需 34 小时。正式预算必须使用 smoke 实测值，并为磁盘预留至少
`1.3x` headroom；不得用这个示例替代实测。

并行 worker 必须使用不重叠 seed shard 和独立 staging/output 目录。合并只读取已完成且通过
canonical validator 的 episode，不能共享可写 episode 目录。

## 6. 行走速度建议

当前 locomotion 合同禁止横向 command，将 `|vx|<0.16 m/s` 的非零命令归零，并对
`vx/wz` 设置 `0.30 m/s / 0.35 rad/s` 硬上限。新数据首版不应同时修改这个已经验证过的
low-level policy 包络。

| 场景 | nominal | 采集随机范围 | 单条演示上限 |
|---|---:|---:|---:|
| 空载前进 `NAV_TO_SOURCE` | `0.20 m/s` | `0.16–0.24 m/s` | `0.30 m/s` |
| 空载转向 | `0.25 rad/s` | `0.15–0.30 rad/s` | `0.35 rad/s` |
| 负载前进 `NAV_TO_TARGET` | `0.18 m/s` | `0.16–0.20 m/s` | `0.24 m/s` |
| 负载转向 | `0.20 rad/s` | `0.12–0.25 rad/s` | `0.30 rad/s` |
| 横向 command `vy` | `0` | 不随机 | `0` |
| Mani 阶段 base command | `[0,0,0]` | 不随机 | `[0,0,0]` |

表中的“单条演示上限”用于约束首版专家数据；底层 locomotion 的全局安全硬包络仍为
`|vx|<=0.30 m/s、vy=0、|wz|<=0.35 rad/s`。负载演示采用更严格上限，不代表修改底层
policy 的能力边界。

精确靠近不建议持续输出 `0.05–0.10 m/s`，因为当前 locomotion guard 会将其归零。应由
PCT/DWA 使用合法的 `>=0.16 m/s` 短脉冲与零速 settle 完成最后距离，或者先单独重训并验证
low-level locomotion policy，再发布新的速度合同。

进入 PICK/PLACE 前的稳定门禁建议为：

```text
measured planar speed <= 0.02 m/s
measured yaw rate     <= 0.10 rad/s
continuous dwell      >= 0.40 s
```

这三个量仅用于教师边界标签和评测；runtime Pass 1 仍不能读取它们。

## 7. 机械臂与夹爪速度建议

### 7.1 关节目标速度

新的 Mani runtime 不使用 IK/cuRobo。采集教师可以使用离线 IK 或轨迹优化生成成功专家动作，
但 raw 必须记录最终实际下发的 joint target，模型训练和 runtime 均只使用 joint trajectory。

现有成功教师的 50 Hz joint target step 为：

```text
[0.008, 0.010, 0.010, 0.010, 0.008, 0.010] rad / 20 ms
```

等价速度上限：

```text
[0.4, 0.5, 0.5, 0.5, 0.4, 0.5] rad/s
```

建议把它保留为首版常规动作上限。25 Hz 输出相邻点的最大变化相应为：

```text
[0.016, 0.020, 0.020, 0.020, 0.016, 0.020] rad / 40 ms
```

靠近物体、下降、闭合期间使用上述上限的 `50–70%`；远离接触面的 reach、lift 和 retract
可以使用完整上限。任何成功演示中超过该包络的 target 必须标为采集缺陷，而不是交给
normalizer 裁剪。

### 7.2 Cartesian 审计速度

虽然 runtime 不执行 Cartesian target，仍应使用 FK 对 joint trajectory 做可解释的离线
质量审计：

| 运动段 | 建议 nominal | 最大值 |
|---|---:|---:|
| 空间 reach/retract | `0.08–0.12 m/s` | `0.15 m/s` |
| 最终下降/对准 | `0.04–0.06 m/s` | `0.075 m/s` |
| 抓取后垂直 lift | `0.06–0.08 m/s` | `0.10 m/s` |
| TCP 角速度 | `<=0.35 rad/s` | `0.50 rad/s` |

这些指标只审计轨迹连续性，不重新把 Cartesian pose 变成 runtime 动作，也不引入逐点 IK
可行性 gate。

### 7.3 连续夹爪

建议继续使用已验证的平滑时序：

| 参数 | 建议值 |
|---|---:|
| open→close 或 close→open 时间 | `0.70 s` |
| 到位保持 | `0.30 s` |
| 输出频率 | `25 Hz` |
| 输出范围 | `[0,1]` |

PICK 必须先保持 open，再对准、闭合、保持并抬升；PLACE 必须保持 closed 到合法释放位置，
再打开并保持。每条成功 episode 应恰好包含一次 PICK open→close 和一次 PLACE close→open
主转换；短暂数值回弹不得形成第二次语义转换。

## 8. 随机化原则

### 8.1 总原则

1. 只随机化真实部署中可能变化的量；不要用任意纹理和极端物理参数制造“看似多样”的数据。
2. 所有随机量按 episode 固定，禁止逐帧 lighting/camera flicker。
3. 使用确定性 seed 和分层抽样，不做所有参数全笛卡尔积。
4. 每条 episode 的 requested、resolved 和 realized randomization 全部写入 manifest。
5. 先保证 nominal 成功，再逐步加入 mild/edge；不能只保留随机化后最容易成功的样本。
6. 速度和 hold duration 必须随机化，使 progress 学习物理完成度，而不是记住固定秒数。

建议 train 中 `60% nominal/mild + 30% moderate + 10% edge`。validation 使用相同范围但完全
独立 seed；test 至少一半来自训练范围边缘或小幅外推区间。

### 8.2 推荐随机范围

下表是首版建议。任何范围在正式冻结前都必须用 32+200 episode pilot 验证成功率和画面合法
性。

| 因素 | train 建议 | test edge 建议 | 约束 |
|---|---|---|---|
| 初始 base—source 距离 | `1.2–2.0 m` | `1.1–1.2` 或 `2.0–2.2 m` | 目标须在可导航区域 |
| source 相对 head 方位 | `±20°` | `20–30°` | settle 后 head 必须可见目标 |
| cola 支撑面 XY | nominal 周围 `±30 mm / ±50 mm` | 扩至 `±40 mm / ±70 mm` | footprint 内且不得悬空/穿透 |
| cola yaw | `[-180°,180°]` | 同范围独立 seed | 圆柱标签朝向仍影响视觉 |
| cola roll/pitch | 由物理 settle 自然产生 | 不人工注入 | 必须保持可抓取直立状态 |
| destination | blue/yellow `50/50` | `50/50` | 每 split 独立平衡 |
| release point XY | 箱内合法区域 `±20 mm` | `±30 mm` | 保留完整物体 clearance |
| 初始 arm joint | nominal `±0.03 rad` | `±0.05 rad` | 先做自碰撞/关节限位检查 |
| 初始 gripper | 70% open、15% partial、15% closed | 各状态平衡提高 | PICK 教师必须能先打开再抓取 |
| belt speed | `0.008/0.010/0.012 m/s` | `0.007/0.013 m/s` | 只在真实部署包络内冻结 |
| base/arm 速度倍率 | `0.85/1.0/1.15` | `0.8/1.2` | 仍不得越过本文硬上限 |
| inference hold | `0/0.4/0.8/1.2 s` | 可增加少量 `1.5 s` | 每次 hold 后重新观察 |
| dome/key/fill intensity | nominal 的 `0.85–1.15x` | `0.75–1.25x` | episode 内固定 |
| light RGB/white balance | 每通道 `±5%` | `±8%` | 不改变目标/箱体基本颜色语义 |
| camera mount translation | `±3 mm` | `±5 mm` | 每 episode 记录 resolved extrinsic |
| camera mount rotation | `±1°` | `±2°` | head/wrist 独立但固定 |
| focal length | `±1%` | `±2%` | principal point 同时记录 |
| exposure/gamma | nominal 的 `±10%` | `±15%` | 禁止逐帧随机闪烁 |
| object mass | nominal `±10%` | `±15%` | 保持重心与碰撞几何不变 |
| contact friction | nominal `±10%` | `±15%` | support/object 成对审计 |

若真实传送带速度并非 `0.01 m/s` 附近，必须先实测部署分布，再替换表中 belt range。不能为了
沿用旧 config 而训练错误速度，也不建议首版直接加入旧 deferred `0.03/0.06 m/s`，除非教师
已经在这些速度上稳定完成全部任务。

### 8.3 首版明确不做的随机化

- 不随机修改 support height、碰撞几何或 TCP 定义；
- 不把物体随机下沉、穿透箱体或悬空；
- 不随机 route/boundary label；
- 不注入与真实部署无关的大幅 camera pose；
- 不做逐帧纹理、光照或时间戳抖动；
- 不添加任务中不存在的 object/distractor；
- 不把失败轨迹、人工修复点或 on-policy correction 混入首个专家版本；
- 不用 action noise 代替真实速度多样性。

这些项目只有在 nominal+recommended 数据通过完整闭环、且 held-out 证据指出具体缺口后才
增加。

### 8.4 语言随机化

建议为同一任务准备 6–10 条人工审核的等价表达，英文、中文和双语按部署需求平衡。所有表达
必须保持目标物、方向和目标箱语义不变。route token 和局部 subtask 标签不能随着措辞改变。
同一 paraphrase family 不能跨 split 复制完全相同的 episode seed。

## 9. split 与覆盖率

split 必须在采集前由 hash namespace 冻结，以完整 scenario/episode 为单位：

```text
train: hash(seed, "joint-trajectory-v1-split") in [0, 0.8)
val:   ... in [0.8, 0.9)
test:  ... in [0.9, 1.0]
```

同一基础 scenario 的 retry、速度变体或相机变体不得跨 split。失败后可以使用同一 split 的
attempt ID 重试，但正式集只保留一条通过全部门禁的 success，避免大量近重复样本提高表面
row 数。

1,600 条目标下，建议至少满足：

- 每个 split 的 blue/yellow 各约 50%；
- train 的 `destination × distance-bin × bearing-bin` 18 个主 cell，每个至少 40 条成功；
- 每条成功 episode 都含四个 route 和三个 boundary，因此 train 每类 boundary 至少 1,280
  个 event；
- nominal/mild/moderate/edge 的 attempted 与 successful 分布都要报告；
- 任一随机化 bin 成功率显著偏低时，先修教师或收窄不真实范围，不能只丢弃失败造成选择偏差。

## 10. raw canonical 必需字段

### 10.1 episode manifest

至少保存：

```text
dataset/schema/profile identity
episode/scenario/attempt IDs
split namespace and seed
task instruction and destination
all requested/resolved/realized randomization
physics/control/camera/action/query clocks
robot/camera/workcell calibration IDs
teacher version and Git identity
asset hashes
episode outcome and failure reason
```

### 10.2 每个 50 Hz control step

至少保存：

```text
sim_step, control_tick, model_tick, monotonic_time_s
base_pose_world, base_twist_body
base_command_requested, base_command_applied
arm_q_measured[6], arm_dq_measured[6]
arm_q_command_requested[6]
arm_q_command_applied[6]
gripper_measured_open_fraction
gripper_command_requested_open_fraction
gripper_command_applied_open_fraction
head/wrist/overview frame IDs and timestamps
teacher route/operation and boundary events
object/contact/goal truth for audit only
```

`arm_q_command_applied` 必须是写入底层 position controller 的最终值，而不是 IK 结果、未限幅
proposal 或 measured joint position。

### 10.3 派生训练 row

模型 loader 需要：

```text
sample_id, split, episode_id, query_timestamp
instruction
head_images[2], wrist_images[2]
assistant solution / route token / subtask text
route and action_domain
boundary transition / signed time (supervision only)
route-specific physical progress / valid mask (supervision only)
physical-progress provenance (audit only)
nav_action[10,3] | null
mani_action[10,7] | null
mani_state[13] | null
terminal-hold provenance (audit only)
```

loader 必须证明：

- Qwen/Pass1 tensor 中 state 数为 0；
- NAV expert tensor 中 state 数为 0；
- Mani expert 恰好读取 13 个允许的连续量；
- GT object、phase、operation、previous route 和 simulator target 均不进入模型输入。

## 11. route、soft CE、physical progress 与双确认所需数据

### 11.1 route 与 boundary 标签

阶段内部 route 使用硬 CE；每个真实 boundary 的前后窗口使用 old/new soft CE。按相对边界
时间和 transition-specific `tau` 构造：

```text
p_new_label = sigmoid(boundary_signed_time_s / tau_transition)
p_old_label = 1 - p_new_label
```

初始候选为：`NAV_TO_SOURCE→PICK` 与 `NAV_TO_TARGET→PLACE` 使用 `tau=0.20 s`，
`PICK→NAV_TO_TARGET` 使用 `tau=0.30 s`；最终值由 200-episode pilot 冻结。

采集必须保证每个 boundary 前后都有至少两条 5 Hz query row，并保存同一 transition ID。
runtime 不直接读取上述 label；它比较 Qwen 自己的 route probabilities：

1. 初始 route 连续两次为同一最高概率项才提交；
2. 新 route 连续两次为同一个候选，且每次 `P(new)>P(committed)` 才切换；
3. 待确认期间只 hold，不执行新 route 或继续旧动作。

训练数据不保存或监督 runtime 的 pending counter；它只是 model-output debounce，不是模型
history 输入。

### 11.2 route-specific physical progress

progress 只能从 route 对应的物理完成度派生：NAV 使用目标相对距离、朝向和最终停稳；PICK
使用 reach/alignment/descend/close/lift/carry-ready；PLACE 使用到释放位、下降、打开和物体
脱离。NAV_TO_TARGET 还必须在携带状态下计算接近与停稳。每个标签都要保存物理来源、单位、
归一化规则和 `valid` mask。

禁止使用 episode elapsed time、segment 内 timestamp 比例、row index 或固定帧比例替代物理
progress。hold 时标签保持，重新对准或物理倒退时允许下降；无法从可信 truth 构造时必须将
该 row 的 progress loss mask 掉。physical progress 和其来源只作监督/审计，均不得进入模型
输入或 runtime route/action 控制。

## 12. 质量硬门禁

### 12.1 episode 级

进入正式 expert 数据的每条 episode 必须：

- 完整成功，且未使用 diagnostic assist；
- route 物理顺序完整，三个 route boundary 均存在；
- 无 robot fall、forbidden collision、错误物体抓取或目标物丢失；
- head/wrist 全帧可解码、时间连续，overview 完整用于审计；
- source support 高度正确，物体无穿透/悬空；
- PICK/PLACE 全程 base applied command 精确为零；
- PICK 在对准前无 close，PLACE 在合法释放位置前无 open；
- 最终物体释放并稳定进入目标区域。

### 12.2 command/state 级

| 指标 | 门禁 |
|---|---:|
| `arm_q_command_applied` 缺失/非有限 | 0 |
| Mani 25 Hz 相邻 joint target 超速 | 0 |
| gripper command 越过 `[0,1]` | 0 |
| Mani base command 非零 | 0 |
| image/state/command timestamp 错配 `>40 ms` | 0 |
| joint command—measurement tracking error p95 | `<0.08 rad` |
| joint command—measurement tracking error p99 | `<0.12 rad` |
| normalizer train 单侧 clip rate | `<1%`，目标 `<0.5%` |
| duplicate sample/episode ID | 0 |

tracking error 超限不是简单删除单点的理由；它可能说明该整段教师动作不可可靠复现。必须检查
连续窗口，并在无法证明语义正确时拒绝整条 episode。

### 12.3 路径与时序审计

必须报告：

- 四 route 的 episode、row、duration 分布；
- 三个 boundary 的 before/after row 和 signed-time 分布；
- base path length、方向反转次数、最终到达误差；
- 六关节 command/measured 的速度、加速度、tracking error p50/p95/p99/max；
- PICK close 相对 alignment 的时间差；
- close→lift 与 PLACE open→release 的时间差；
- hold duration 和 inference-latency 分布；
- 每个随机化 bin 的 attempted/success/eligible 数量。

随机抽取 train/val/test × 四 route 的 head+wrist 视频，并叠加 route、commanded joint/gripper、
measured joint 和 boundary 时间。overview 只用于人工审计，严禁进入训练图像。

## 13. normalizer 与采样

所有 normalizer 只由 train split 拟合：

- NAV 三维使用共享 NAV-domain robust quantile；
- Mani 六维 `delta_q` 使用共享 Mani-domain robust quantile；
- Mani state 的 `q/dq` 使用 train-only z-score 或 robust scale；
- gripper action 保留已知物理范围 `[0,1]`，映射到 `[-1,1]`；
- gripper state 保留 `[0,1]`；
- PICK/PLACE 不建立不同单位或不同语义的 gripper normalizer。

训练 sampler 平衡四个 route、episode、progress bin 和 boundary event；每个完整 batch 保证
NAV、Mani 和三个 boundary 覆盖，并定期放入同一 transition 的 before/after 配对。首版不配置
on-policy correction mixture。

## 14. 采集发布流程

### Gate A：32 条 smoke

- 验证字段、时钟、command provenance、相机和存储；
- 手工重放至少一条完整成功 episode；
- 计算实际 RTF、成功率、字节/episode；
- 发现 schema 或教师问题时全部作废，不修补后混入正式集。

### Gate B：12 条 overfit snapshot

- 覆盖两个 destination、四 route、三个 boundary 和主要初态；
- 新模型必须能过拟合 route、连续夹爪和 joint trajectory；
- runtime replay 不得出现到第一个 Mani target 就提前闭合。

### Gate C：200 条 learning pilot

- 验证随机化成功率和每个 bin 的覆盖；
- 比较新 head selective warm-start 与 clean-head pilot；
- 开环检查 joint trajectory、gripper crossover、boundary lag/flicker；
- 至少多个预冻结 seed 完成 NAV→PICK→伸手→闭合→抬升的真实闭环能力验证。

### Gate D：1,600 条 formal release

- 只在 A–C 的 schema、teacher、速度和标签合同全部冻结后开始；
- 按预冻结 split/seed shard 采集；
- raw、derived、manifest、normalizer 均以 immutable ID 发布；
- formal audit 全通过后才允许长训。

### Gate E：是否扩展到 2,400

以 800/1,200/1,600 条 train-subset 学习曲线决定。只有以下至少一项仍随数据量稳定改善才扩展：

- held-out joint/gripper action error；
- boundary transition lag/flicker；
- 多 seed NAV 到达率；
- PICK close/lift 成功率；
- 完整任务闭环成功率。

不要用训练 loss 单独决定是否继续采集。

## 15. 与旧数据和旧执行器的边界

- 旧 522 episode、Waypoint v1、旧 v2 command-gripper 数据保持只读，只用于历史对照；
- 新 direct-joint 数据不得与旧 TCP target row 拼接或共享 normalizer；
- 新 Mani runtime 不调用 IK、cuRobo `plan_pose` 或逐点 feasibility selector；
- 教师内部可使用 IK/规划器生成专家，但其最终 applied joint command 必须完整记录；
- 不下载、加载或转换 PRTS 权重；
- K*、prefix head、`L_prefix`、CRL、on-policy correction 和 self-conditioned auxiliary 不进入
  首版数据/训练合同。

## 16. 最终推荐值汇总

```text
formal successes:        1600 (1280/160/160)
optional ceiling:        2400, +400 per evidence-backed revision
raw clocks:              physics400 / control50 / camera25 Hz
training query:          5 Hz
NAV:                     [10,3] @ 0.20 s, 2.0 s
Mani:                    [10,7] @ 0.04 s, 0.4 s
Mani state:              q6 + dq6 + gripper1
unloaded base nominal:   0.20 m/s
loaded base nominal:     0.18 m/s
base hard envelope:      |vx|<=0.30 m/s, vy=0, |wz|<=0.35 rad/s
arm joint max:           [0.4,0.5,0.5,0.5,0.4,0.5] rad/s
gripper move/hold:       0.70 / 0.30 s
base settle:             <=0.02 m/s, <=0.10 rad/s, >=0.40 s
normal runtime hold:     0.0–1.2 s, action endpoint held
route commit:            two fresh observations
training episodes:       complete unassisted successes only
```

## 17. 依据

- 旧数据规模、route 分布和已知 PICK 标签缺陷：[`data.md`](data.md)
- step1250 开环/闭环和 confidence crossover 证据：
  [`waypoint_v2_step1250_strict_evaluation_20260825.md`](waypoint_v2_step1250_strict_evaluation_20260825.md)
- 当前 physics/control/camera 时钟、教师速度、夹爪时序和 locomotion 包络：
  `src/conveyor_bench/isaac/runtime_core.py`、`src/conveyor_bench/isaac/locomotion.py`
- 官方 openpi action chunk 在 chunk 用尽后才重新推理：
  <https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py>

本文中的随机范围和数据规模是第一版工程建议；真正冻结值必须由 32 条 smoke、200 条 pilot、
真实部署测量范围和 held-out 学习曲线共同确认，并写入最终 immutable collection manifest。
