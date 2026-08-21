# ConveyorVLA Waypoint v2 阶段切换执行与长训计划书

- 文档版本：`waypoint-v2-stage-transition-execution-plan-v1`
- 状态：已批准执行；Waypoint v1 继续作为冻结基线，最终晋级组件由证据决定
- 编写与批准日期：2026-08-22 CST
- 适用分支：`feature/conveyorvla-waypoint-v1`
- 编写基线：`72355e9297a529639149a2046421711c97af6558`
- runtime/eval 代码基线：`cfed498eff780d390426962f309a3002173e9ed3`
- 对照 checkpoint：`step_002000@a8d57a22c515`
- 对照数据：`conveyorvla-waypoint-dense-transition-v1`，manifest SHA-256
  `0db6169d726b2165a90ec6e833403666179eb68135248af5681de92a400ec957`

## 0. 文档权威性与变更边界

本文是针对“模型不能在正确物理时刻稳定切换阶段”的已批准执行计划。它授权按本文门禁
实现、评测并启动 Waypoint v2 正式长训，但不原地修改
[Waypoint Policy v1](conveyorvla_waypoint_policy_contract_v1.md)。执行期间：

- 现有 `qwen3vl-layerwise-dual-fm-waypoint-v1` 模型输入、两次 Qwen forward、route 所有权、
  NAV/ARM 动作 shape、坐标系和 planner 边界保持不变；
- 现有 `conveyorvla-waypoint-dense-transition-v1` 数据、normalizer、checkpoint 和评测证据
  保持不可变；
- terminal-hold、新 prefix 预测、局部目标 CRL 和新 suffix 语义必须使用新
  schema/config/manifest，不得原地覆盖 v1；
- 已授权在完成实时身份、GPU、tmux/PID 和目录核验后同步至 4×H20、终止明确属于本项目且
  冲突的训练进程、运行门禁并启动正式长训；
- 任何阶段都不授权使用 GT 状态控制模型、外部 FSM 覆盖 route，或触碰无关远端任务。

执行不再需要中途向用户汇报或等待逐阶段审批；数据、overfit、单卡、分布式、开环和闭环
门禁仍然有效。Agent 应自主修复普通代码、数据、训练和评测问题，并按配对证据选择组件，
不得为了“一步到位”跳过可复现性检查。只有权限、数据或基础设施等无法自主消除的硬阻塞
才允许中断执行。正式长训前必须把最终晋级子集、schema 和 runtime 语义冻结为独立 v2
合同及 resolved config；未通过门禁的模块保持实验状态并可独立回滚。

## 1. 执行摘要

当前模型不是完全不会输出下一 route，而是没有可靠学习“何时完成当前物理子目标”。
step 2000 闭环已经自主执行 `NAV_TO_SOURCE → PICK`，但导航路径振荡，并在距可乐约
`0.493 m` 时切换；静态 route accuracy 因此不能代表正确的阶段切换能力。与此同时，v1
训练在未来跨 route 后 mask suffix，而推理服务无法预测该 mask，可能把未受监督的后段
waypoint 当作有效候选。

本计划把阶段切换重新定义为四个耦合问题：

1. 当前视觉是否包含足以判断物理边界的信息；
2. 20-step chunk 的每个位置是否具有一致的训练/推理语义；
3. 模型能否判断当前动作是否真的推进当前语言目标；
4. 模型能否联合预测本次可信 action prefix，而不是固定取首点或尾点。

待验证的 production candidate 池为：

```text
clean Qwen3-VL-4B
  + terminal-hold 全 horizon 监督
  + boundary/progress 时序损失
  + SparkVLA 式 prefix ranking
  + PRTS 式局部目标可达性 CRL
  + 每个 action chunk 4 组训练 FM Monte Carlo 样本
  + on-policy transition correction 数据
```

这不是强制全量开启清单。terminal-hold 建立 v2 suffix 语义基线，其余组件按顺序配对实验
晋级；正式长训使用通过内部门禁的最佳证据组合。PRTS 仅提供局部目标可达性训练思想，
不得加载 PRTS 发布权重，Qwen backbone 只使用官方标准 Qwen3-VL 初始化。

其中 FM “1→4”只指训练时每个动作样本的独立 `(noise, flow-time)` 数量；现有推理参数
`num_inference_timesteps=4` 保持不变。

## 2. 已知证据与失败定义

### 2.1 现有证据

- 无 state waypoint v1 数据已有 522 episode、119,700 row；模型 batch 中 state
  field/tensor 为 0。
- Qwen Pass 1/Pass 2、双 Layerwise FM、checkpoint export/load、PCT/DWA 和真实
  cuRobo known-pose 已有实现证据。
- step 2000 lookahead 闭环中，18 个 NAV chunk 均成功调用 PCT，模型自主切到 PICK；
  query-anchor 到可乐距离从 `1.316 m` 最低降到 `0.380 m`，但累计路径为 `7.307 m`，
  方向明显振荡。
- PICK ARM chunk 的 target 2 相邻 roll/yaw 跳变超过原始 35° 规则，完整任务未成功。
- 当前 80-step 小样本 overfit 门禁仍未通过，不能用长训步数替代该证据。

完整基线见 [当前状态](status.md)和
[step 2000 lookahead 闭环](checkpoint_step2000_lookahead_evaluation_20260821.md)。

### 2.2 本计划中的“阶段切换失败”

以下任一现象均计为阶段切换失败：

- 物理完成条件尚未满足时，Pass 1 提前输出下一 route；
- 物理完成条件已满足后，超过允许窗口仍输出旧 route；
- 连续 query 出现 `A → B → A` route flicker；
- chunk selector 读取超过真实同 route prefix 的动作；
- route 语义与 action chunk 动力学明显不一致；
- 开环 route accuracy 高，但闭环切换时间、路径或动作不可执行。

评估器可以读取仿真真值计算上述指标，但不得把它们写入模型 request、替换 route 或触发
外部 FSM。

## 3. 目标、非目标与硬约束

### 3.1 目标

1. 证明或否证现有四张视觉图是否足以辨别四个物理边界。
2. 消除训练 suffix mask 与推理全有效之间的语义不一致。
3. 让 Qwen 表征编码距当前物理子目标的进度和可达性。
4. 学习可信 prefix `K`，使 20-step horizon 兼顾远期信息与边界重规划。
5. 提高 NAV waypoint 方向一致性、ARM target 连续性和 route/action 对齐。
6. 在 episode-level split、固定 seed 和固定算力预算下给出可复现实验结论。

### 3.2 非目标

- 不通过增加外部 phase detector 或 FSM 修复模型；
- 不恢复 state28、joint/TCP/base pose、operation、grasp_done 或 previous-route history；
- 不以放宽全部 planner/机械臂动力学限制伪造策略成功；
- 不把 PRTS checkpoint 的规模优势误当作本地算法消融；
- 不同时修改数据、模型、FM 推理步数、planner 和 seed，导致结论不可归因。

### 3.3 保持不变的推理主链

```text
完整任务 + head/wrist[t-0.20s, t]
              │
Pass 1：受约束 Qwen 生成 ACTION/DONE、route、subtask
              │ 模型自己的完整 assistant prefix
Pass 2：相同视觉上的第二次完整 Qwen forward
              │
       ┌──────┴──────┐
       │             │
NAV FM [20,3]     ARM FM [20,7]
       │             │
模型内部 K̂        模型内部 K̂
       │             │
PCT/DWA          cuRobo/IK
       └──────┬──────┘
          重新观察
```

route 切换仍只能来自下一次 Qwen Pass 1。`K̂` 只限定本次 chunk 的可信执行范围，不能把
route 改成另一个阶段。

## 4. 可检验假设

| 假设 | 内容 | 判定实验 |
|---|---|---|
| H1 | 当前视觉 hidden 已包含边界信息，只是训练目标未利用 | frozen linear/MLP probe |
| H2 | masked suffix 是边界附近 waypoint 不稳定的主要来源之一 | v1 mask 对比 v2 terminal-hold |
| H3 | route CE 只能学静态分类，连续边界/进度损失能改善切换时机 | boundary-window sequence eval |
| H4 | 可信 prefix 排序能降低越界读取并避免固定首点/尾点 | `K̂` 校准与闭环 selected-index |
| H5 | PRTS 式局部 CRL 能把 route 语义与动作推进方向绑定 | goal margin、时序相关与闭环消融 |
| H6 | 每样本 4 组 FM draw 能降低随机梯度方差、改善动作连续性 | S=1/S=4 配对实验 |
| H7 | 专家成功数据不足以覆盖闭环振荡和误切换状态 | on-policy correction 消融 |

## 5. 新派生数据与标签计划

### 5.1 版本与不可变性

建议新 schema 名为：

```text
conveyorvla-waypoint-dense-transition-v2
```

名称在实现前由合同增补冻结。构建要求：

- 从相同只读 Liangzhu 0815 n200+n400 source 重新派生；
- 复用 v1 的完整 source episode split 和 split seed，保证公平对照；
- 输出到全新 immutable 目录，先 staging、审计后原子发布；
- v1 manifest、normalizer、JSONL、checkpoint 不得覆盖或就地迁移；
- v2 manifest 记录 source、split、transform、schema、config 和每个文件 SHA-256。

### 5.2 新增离线监督字段

| 字段 | 含义 | 模型推理输入 |
|---|---|---:|
| `route` | 当前 GT route | 否 |
| `next_route` | 下一相邻 route | 否 |
| `boundary_event` | 物理边界类型 | 否 |
| `time_to_boundary_s` | 距下一边界的秒数 | 否 |
| `phase_progress` | 当前阶段连续进度 | 否 |
| `original_valid_prefix_k` | 原始同 route target 数量 | 否 |
| `suffix_reason` | boundary、source-tail、episode-tail | 否 |
| `padded_action` | terminal-hold 后完整 target | 训练 target |
| `transition_window` | 是否在边界窗口 | 否 |
| `transition_id` | episode 内唯一切换事件 | 否 |

监督字段可以由 source pipeline state 和 pose 生成，但 loader 的模型 batch 仍禁止出现任何
robot state、GT phase、operation、object truth 或 simulator target。

### 5.3 四个功能边界

| route 转换 | 功能边界 |
|---|---|
| NAV_TO_SOURCE → PICK | `base_stopped_source_in_reach` |
| PICK → NAV_TO_TARGET | `grasp_lifted_carry_ready` |
| NAV_TO_TARGET → PLACE | `base_stopped_target_in_reach` |
| PLACE → DONE | `released_in_target` |

边界标签必须进行视频抽检。若 source pipeline 状态与肉眼可见事件存在系统性提前/滞后，
不能强迫视觉模型拟合不可见瞬间；应记录 interval label 和 uncertainty window。

### 5.4 Terminal-hold 规则

对因为真实 route boundary 导致的 suffix：

1. `K*` 定义为 20 个 future target 中属于当前 route 的连续前缀长度；
2. 前 `K*` 个 target 保留原始监督；
3. `K*...19` 重复最后一个当前 route 的合法 target；
4. NAV 重复最后一个 query-body waypoint；
5. ARM 重复最后一个 query-base absolute TCP target；
6. FM loss 对 padding 后的完整 20 点计算；
7. `K*` 单独用于 prefix selector 监督，不能因 padding 被改写成 20。

对 source 缺帧或意外 episode truncation，不得假装成任务完成。此类 row 必须用
`suffix_reason` 区分：可选择只保留 route supervision，或在审计后剔除 action loss；不能与
真实边界 terminal-hold 混为一类。

### 5.5 Transition-aware sampler

sampler 以 `transition_id` 和 episode 为基本单位，而不是把同一切换附近的相邻帧当作大量
独立事件。每个训练 batch 应覆盖：

- 四个 active route 与 DONE；
- 四种 transition window；
- phase interior 的不同 progress bin；
- 不同 source episode；
- NAV 与 ARM 两个动作域。

## 6. 模型与损失设计

### 6.1 Boundary/progress 辅助学习

在 Qwen Pass 2 hidden 上加入小型训练辅助头，预测：

- `time_to_boundary_s`；
- 当前 phase progress；
- 边界前/后概率；
- `K*` 或 prefix candidate scores。

这些是模型自产的内部信号，不是环境输入。训练策略：

- phase interior 使用标准 route CE；
- transition window 使用连续软标签；
- 对同一 transition 前后样本施加 pairwise temporal ranking；
- 旧 route logit 随时间下降，新 route logit 随时间上升；
- 标签加入小幅 boundary jitter，避免把单帧 pipeline 状态当作绝对真值。

辅助 progress 可以在正式推理中不导出；若 prefix head 使用其 hidden，也不得把外部 GT
progress 传入 runtime。

### 6.2 Prefix ranking

候选集合为 `K=1...20`。每个候选使用：

- Pass 2 Qwen hidden；
- predicted route embedding；
- candidate terminal action；
- prefix length embedding；
- FM action-head 表征。

排序目标：

- `K≤K*` 时，较长 prefix 通常更高效；
- `K>K*` 为越界候选，强惩罚；
- 边界清晰时最优为 `K*`；
- uncertainty window 允许 `K*±1`；
- phase interior 的 `K*=20` 应偏向长 prefix，而非反复一步重询。

推理使用：

- NAV 只在 `[1,K̂]` 内运行现有几何/PCT 候选排序；不得越过 `K̂`；
- ARM 只允许执行/规划预测可信 prefix，之后重新观察；
- `K̂` 不控制下一 route，也不读取外部完成状态。

### 6.3 PRTS 式局部目标可达性 CRL

PRTS 学习状态—动作表示 `φ(s,a)` 与语言目标表示 `ψ(g)`，使内积近似语言目标的
log-discounted reachability。原论文使用整任务目标和大规模多任务预训练；本项目必须改成
边界局部目标，避免只学 episode 时间。

建议的局部 goal：

| 当前 route | CRL goal 文本语义 |
|---|---|
| NAV_TO_SOURCE | source is reachable and the base is ready |
| PICK | object is firmly grasped, lifted and carry-ready |
| NAV_TO_TARGET | target is reachable while carrying the object |
| PLACE | object is released inside the target |

训练实现边界：

- `φ(s,a)` 读取允许视觉、Qwen hidden 和训练 target action；不读取 goal 文本或 state；
- `ψ(g)` 只编码局部 goal 文本；
- NAV/ARM action 可使用各自小型 encoder 投影到共同 CRL 空间；
- 使用 state-action→goal 与 goal→state-action 双向 InfoNCE；
- hard negative 包括错误 route goal、相邻阶段、跨 episode 打乱 chunk 和错误 goal/action
  配对；
- 同 goal 的不同进度样本为多正样本，并按距局部边界时间加权；
- CRL 梯度必须进入共享 Qwen 表征，但正式推理不使用 CRL 分数覆盖 route。

由于 NAV 为 0.60 s stride、ARM 为 0.20 s stride，时间权重按秒定义：

```text
w(t, route) = exp(-time_to_boundary_s / tau_route)
```

`tau_route` 由 train split phase-duration 分布冻结，并写入 resolved config。不得简单把所有
route 的离散 frame 数代入同一个 gamma。

PRTS 公开仓库目前完整开放的是下游 SFT/Flow-Matching；公开训练路径中 CRL heads 被冻结，
`forward` 不计算 CRL loss。因此本项目必须依据论文独立实现并测试，不得声称直接复现其
167B-token 预训练结果。

### 6.4 FM 训练 Monte Carlo 样本从 1 提升到 4

#### 精确定义

现有 config：

```json
"loss": {
  "repeated_diffusion_steps": 1
},
"action_model": {
  "num_inference_timesteps": 4
}
```

本计划的主候选改为：

```json
"loss": {
  "repeated_diffusion_steps": 4
},
"action_model": {
  "num_inference_timesteps": 4
}
```

即每个真实 action chunk 在一次 optimizer step 内采样四组独立噪声和 flow time：

```text
epsilon_m ~ Normal(0, I)
beta_m    ~ Beta(alpha=1.5, beta=1.0)
t_m       = (noise_s - beta_m) / noise_s
m         = 1, 2, 3, 4
```

训练 loss 为 Monte Carlo 均值：

```text
L_FM^(4) = (1/4) * sum_m MSE(v_theta(x_t_m, t_m), action - epsilon_m)
```

关键约束：

- 同一视觉样本只做一次 Qwen forward；
- 四组 draw 共享 Qwen hidden，但各自独立采样 `epsilon_m` 与 `t_m`；
- 四组 loss 取平均，不求和，因此不因为 `M=4` 隐式放大 FM loss 权重；
- global data batch、optimizer、learning rate 和 scheduler 首轮保持不变；
- 推理 denoising/integration 仍为 4 步，不在此实验中一起改变；
- NAV/ARM 分别报告四 draw 的 loss mean/std 和合并梯度。

#### 预期收益

- 降低单个随机 `(noise,t)` 对 FM 梯度的方差；
- 提高相同视觉/route 下 action target 的覆盖；
- 可能改善 NAV 方向一致性与 ARM RPY 连续性；
- 在小数据和边界样本上减少某一次极端 noise draw 的支配。

#### 计算与显存风险

当前实现通过重复 layerwise hidden/action batch 进入 FM head。`M=4` 不会把 Qwen forward
变成四次，但会增加 action-head cross-attention、DiT activation 和 Qwen hidden 梯度路径的
显存/算力。必须实测：

- GPU peak memory；
- optimizer-step wall time；
- samples/s 与 GPU-hours；
- Qwen/NAV/ARM gradient norm；
- 是否需要 action-head gradient checkpointing。

如果 micro-batch 需要下降，必须等量增加 gradient accumulation 保持 global data batch；
否则 S=1/S=4 实验会混入 batch-size 变化。若仍 OOM，可评估四 draw 分块/顺序累积，但不得
静默改变 loss 归一化或 Qwen gradient。

#### 必须保留 S=1 对照

S=4 是主候选，不是预设结论。至少进行：

- step-matched：S1 与 S4 相同 optimizer step；
- compute-matched：按实际 GPU-hour 对齐；
- 固定 validation noise/time bank，避免验证 loss 自身随机；
- 相同初始化、数据顺序、seed、global batch 和 checkpoint/eval cadence。

S4 只有在动作质量或梯度稳定性改善且计算代价可接受时才晋级后续完整实验。

### 6.5 总 loss

建议实验总目标：

```text
L = L_answer
  + L_route
  + lambda_fm       * L_FM^(M)
  + lambda_boundary * L_boundary_rank
  + lambda_progress * L_progress
  + lambda_prefix   * L_prefix_rank
  + lambda_crl      * L_local_CRL
```

初始候选权重：

| loss | 初始权重 |
|---|---:|
| answer/route CE | 1.0 / 1.0 |
| NAV/ARM FM | 1.0 / 1.0 |
| boundary ranking | 0.2 |
| progress | 0.1 |
| prefix ranking | 0.2 |
| local CRL | 0→0.1 warmup |

最终权重不能只靠数值大小决定。必须记录各 loss 对 Qwen、NAV head、ARM head 的梯度范数；
辅助目标合计不得长期压倒 route CE 与 FM。CRL 退化、prefix collapse 或 route accuracy
下降时应能单独关闭对应项。

## 7. 实验矩阵与变量隔离

采用顺序晋级，不做不可解释的全组合：

| ID | 数据/模型变化 | FM draw | 目的 |
|---|---|---:|---|
| B0 | 当前 v1 + step2000，只评估 | 1 | 冻结基线 |
| B1-S1 | v2 terminal-hold | 1 | suffix 语义基线 |
| B1-S4 | 与 B1-S1 完全相同 | 4 | FM 1→4 独立消融 |
| B2 | 晋级的 S + boundary/progress | winner | 切换时机 |
| B3 | B2 + prefix ranking | winner | 动态可信 horizon |
| B4 | B3 + local CRL | winner | route/action/goal 对齐 |
| B5 | B4 + on-policy corrections | winner | 闭环分布偏移 |

规则：

- B1-S1/B1-S4 必须在其他辅助 loss 均关闭时比较；
- B2、B3、B4 每次只新增一个机制；
- 每个晋级结论至少由相同 seed 的配对 run 支持；正式结论应补第二训练 seed；
- 失败实验必须保留 manifest、日志摘要和停止原因，不能只保存最佳 run。
- PRTS 权重不属于实验矩阵；只允许独立实现并验证其局部目标 CRL 思想。

## 8. 分阶段实施与门禁

### P0：可观测性 probe

冻结 step 2000 Qwen，用允许视觉 hidden 训练 linear probe 和两层 MLP probe，预测：

- current route；
- boundary before/after；
- `time_to_boundary_s`；
- `K*`。

episode-level split，四种 transition 分开报告。结果解释：

- linear 好、正式 route 差：主要是目标/解码问题；
- linear 差、MLP 好：边界可观测但非线性；
- 两者都差：优先审计标签与视觉窗口，继续长训没有依据。

probe 只用于诊断，不得成为外部 runtime phase detector。

### P1：v2 数据与小样本过拟合

1. 先在少量 episode 上生成 v2；
2. 审计 terminal-hold、`K*`、suffix reason 和 transition interval；
3. 生成人工可复核的阶段进度视频；
4. 在覆盖全部 route/boundary 的 8～16 episode 上 overfit；
5. 证明 answer/route、NAV/ARM FM、progress、prefix、CRL 各自可下降；
6. 证明 Qwen/NAV/ARM/辅助 heads 都有有限非零梯度；
7. state leakage、frame/pose round-trip 和 checkpoint save/load 通过。

过拟合门禁未通过时不得启动正式长训。

### P2：S=1/S=4 配对 pilot

- 使用 B1-S1/B1-S4；
- 先做短 smoke 验证显存与吞吐，再进行相同 step pilot；
- checkpoint 至少每 500 effective optimizer step 保存；
- 验证使用固定 noise/time bank；
- 同时报告 step-matched 与 GPU-hour-matched 结果。

S4 晋级条件至少满足一项，且没有明显回归：

- NAV/ARM open-loop aggregate error 有可重复改善；
- FM gradient/loss 方差显著下降并带来更好 horizon/channel 曲线；
- NAV 方向一致性或 ARM RPY 连续性改善。

若只有训练 loss 更平滑、动作指标不改善，则保留 S1。

### P3：boundary、prefix 与 CRL 顺序消融

按 B2→B3→B4 逐项训练和评估。每项必须同时给出：

- phase interior 指标；
- transition-window 指标；
- `K̂` 校准；
- action channel/horizon 图；
- closed-loop route/selected-index trace；
- 计算成本与失败案例。

### P4：on-policy correction

从当前最佳模型的闭环中收集：

- 提前/滞后 route；
- NAV 方向振荡；
- `A→B→A` flicker；
- prefix 越界；
- PICK 未抓稳便切换；
- ARM 姿态跳变；
- PLACE 提前结束。

由仿真 oracle 离线标注 route、boundary、K 和纠正 action；正式推理仍不读取 oracle。建议
初始 sampler 组成：60% 原始成功数据、25% transition window、15% on-policy correction，
之后按 validation 结果调整并记录 resolved ratio。

### P5：正式训练与闭环

只有 P0～P4 的晋级配置通过数据、overfit、单卡、分布式、开环和闭环门禁后，才允许开始
正式长训。正式 run：

- Qwen backbone 从官方标准 Qwen3-VL 初始化；不得使用 PRTS 权重，也不得加载不兼容的 v1
  schema checkpoint；
- 使用证据择优后的最小有效组件组合，不要求所有候选模块同时开启；
- 使用全新 run ID、config snapshot、manifest 和输出目录；
- 同步到 4×H20 前实时核验目标仓库、commit、Conda、GPU、tmux 和精确 PID；只可终止明确
  属于 ConveyorVLA 且与本 run 冲突的任务；
- checkpoint 每 500 effective optimizer step；
- 至少记录 route/answer、NAV/ARM FM、每 draw FM、boundary、progress、prefix、CRL、各模块
  gradient norm、LR、吞吐和显存；
- resume 只能从同 schema/model contract checkpoint；
- 长训总长度由冻结的正式 config 决定，不能把 20 step 当成训练总长度；
- 启动验收要求连续至少 20 个有效 optimizer step，loss/gradient/LR/吞吐有限且无持续异常、
  四卡均参与、checkpoint/output 路径正确；验收后保持训练进程运行；
- 训练健康不能替代开环和闭环门禁。满足上述启动验收后，本实施任务即可一次性交付并把
  goal 标为 complete，中途不再等待用户审批。

## 9. 开环评估协议

### 9.1 Transition-centric route 指标

- 四类 transition 的 early-switch rate；
- late-switch rate；
- switch lag median/P95，同时报告 query 和秒；
- old/new route logit crossover；
- boundary-window AUROC/F1；
- `A→B→A` flicker 次数；
- phase interior macro accuracy；
- route confidence/calibration。

随机抽取的 40 个平衡静态 frame 只能作为 smoke，不得作为阶段切换主结论。

### 9.2 Prefix 指标

- `MAE(K̂,K*)`；
- `P(K̂>K*)` 越界率；
- under-run 点数；
- 不同 route/progress bin 的 K 分布；
- K 与 time-to-boundary 的 calibration；
- runtime selected index 是否始终 `≤K̂`。

越界比保守更严重，报告必须分别列出 over-run 与 under-run，不能只给平均 MAE。

### 9.3 Action 指标与图片

每个 route/action domain 输出：

- 每个样本 20-step target/predicted 对比；
- NAV 3 个通道逐通道图；
- ARM 7 个通道逐通道图；
- per-channel MSE、MAE、相关系数；
- horizon-wise error；
- NAV endpoint、方向一致性和曲率；
- ARM translation、RPY、gripper 连续性；
- terminal-hold 区间误差；
- boundary/interior 分组结果。

### 9.4 FM S=1/S=4 专项指标

- 四 draw loss mean/std；
- optimizer-step FM loss rolling mean/CV；
- NAV/ARM/Qwen action-gradient norm rolling CV；
- fixed-bank validation loss；
- step time、peak memory、samples/s、GPU-hours；
- step-matched 与 compute-matched action quality。

### 9.5 CRL 指标

- correct-goal 与 wrong-goal similarity margin；
- value 与 phase progress 的 Spearman/Kendall 相关；
- transition 前后局部 value 曲线；
- 打乱 action 后 value 是否下降；
- 同 route 跨 episode 聚合与不同 goal 分离。

## 10. 闭环评估协议

### 10.1 Seed 与启动条件

主门禁使用固定、预先目检的 seed manifest：

- settle 后 head view 能直接看到可乐；
- 机器人不侧向或背对可乐；
- 可乐和目标仍保留合理位置变化；
- 几何 truth 只用于评测启动 gate，不进入模型 request。

主门禁建议 10 个正面可见 seed。不可见/大角度 seed 作为后续 stress set，不能混入当前
主结论。

### 10.2 每 episode 记录

- 每次 Qwen route/subtask/confidence；
- 四次评估用 GT boundary 时间；
- 每个 chunk 的 `K̂`、可信前缀和 selected index；
- NAV waypoint、PCT 结果、DWA control 和实际路径；
- ARM target 连续性、validator、cuRobo/IK 结果；
- early/late/flicker/failure taxonomy；
- 完整任务 success 与自然退出原因。

### 10.3 视频

保存未截断的 head、wrist、overview 三视角拼接视频，叠加：

- predicted route/subtask；
- 四阶段进度条；
- `K̂` 与 selected index；
- planner/validator 状态；
- evaluation-only GT 必须明确标记，且不进入控制。

视频、trace、日志、checkpoint、数据和运行资产全部留在 Git ignore 的 artifact backend，
公开仓库只保存不含私有路径的结果摘要与 manifest/checksum。

## 11. 晋级与最终通过标准

### 11.1 数据/实现门禁

- v2 与 v1 路径、schema、manifest、normalizer 完全隔离；
- episode split 无泄漏；
- boundary terminal-hold 与非边界 truncation 精确区分；
- state field/tensor 为 0；
- NAV/ARM round-trip 与 action shape/stride/frame 通过；
- Qwen、NAV、ARM、prefix/CRL heads finite loss、non-zero gradient；
- overfit、单卡、分布式、checkpoint save/load 通过。

### 11.2 开环门禁

建议初始阈值：

- 四类 transition median absolute lag 不超过 1 次 policy query；
- P95 不超过 2 次 query，并报告秒数；
- early-switch rate `≤5%`；
- `P(K̂>K*)≤5%`；
- `K` MAE `≤2` 点；
- phase interior route accuracy 相对 B0 下降不超过 1 个百分点；
- NAV/ARM action aggregate 指标不劣于 B0/B1；
- route flicker 和 CRL goal confusion 无系统性退化。

阈值须在首轮 validation 分布审计后冻结；不能看完 test 后反向调整。

### 11.3 闭环门禁

在 10 个正面可见主 seed 上建议要求：

- 至少 8/10 完成完整四阶段任务；
- 成功 episode 无 `A→B→A` route regression；
- runtime selected index 从不超过 `K̂`；
- 四次 route 切换都来自 Qwen Pass 1；
- 无 GT phase/state/FSM、required-first-route 或 expert action；
- 只使用批准的原始执行规则，不增加掩盖模型能力的行为阻断门控；
- 完整 trace、指标和三视角视频齐全。

## 12. PRTS 借鉴边界

PRTS 只作为阶段推理和局部目标可达性训练方法的参考，不进入权重或数据来源：

- 不下载、不加载、不转换 PRTS 发布 checkpoint；
- Qwen backbone 只允许从官方标准 Qwen3-VL 权重初始化；
- local CRL 必须在本项目无 state 视觉输入、四个局部物理目标和 NAV/ARM 双 head 上独立实现；
- 不复制 PRTS 的 proprioception/state prompt；
- CRL 只塑造训练表征，runtime 不得用其 value 覆盖 Pass 1 route；
- 结果只能归因于本地数据、实现和消融，不能声称复现 PRTS 的大规模预训练收益。

## 13. 复现、资源与 artifact 记录

每个 run 必须有机器可读 manifest，至少包含：

- experiment ID、父 run、创建时间和 owner；
- Git commit、branch、dirty flag 和必要时的 diff checksum；
- exact command、workdir、resolved config 及 SHA-256；
- dataset/schema/split/manifest/normalizer SHA-256；
- base/checkpoint URI、checksum 和模型合同；
- Python/PyTorch/Transformers/DeepSpeed/CUDA/driver/GPU；
- 所有随机 seed、determinism 和 distributed 设置；
- FM draw `M`、inference steps、global batch 和 gradient accumulation；
- evaluation sample/seed manifest；
- logs、checkpoints、tables、plots、videos 的外部 URI/checksum；
- failure/abort 也必须保留原因。

Git 禁止包含：`handoff_private/`、`artifacts/`、数据、checkpoint、日志、视频、环境目录、
机器私有路径和 secrets。公开结果摘要必须能够通过 manifest 解析到外部不可变证据。

## 14. 风险、回滚与决策树

| 风险/现象 | 判断 | 动作 |
|---|---|---|
| linear/MLP probe 都失败 | 视觉或标签不可观测 | 审计边界/相机；不盲目长训 |
| terminal-hold 改善动作但不改善 route | suffix 与切换是两个问题 | 保留数据修复，继续 B2 |
| S4 OOM | action-head activation 过大 | 保 global batch 调 micro/accum 或分块；保留 S1 |
| S4 只让 loss 更平滑 | 无下游收益 | 不晋级，恢复 S1 |
| prefix `K̂` collapse 到 1 | 效率/排序目标失衡 | 调整 ordinal target/采样，不用外部门控补偿 |
| CRL value 只跟随时间 | goal/negative 太弱 | 加 local boundary、hard negatives；仍失败则删除 CRL |
| 开环好、闭环差 | on-policy covariate shift | 收集 B5 corrections，不重复堆专家帧 |
| local CRL 无稳定收益或只跟随时间 | 方法不适配当前数据规模 | 关闭 CRL，晋级上一个有效组合 |
| 视觉确实不足 | 严格 v1 欠定 | 另提纯视觉长 clip/模型预测 memory 合同增补 |

每个模块都必须由独立 config 开关控制。回滚是切回上一个晋级实验及其 immutable
manifest/checkpoint，不允许把新 schema checkpoint 强行加载到旧 runtime。

## 15. 预期交付物

1. 可观测性 probe 报告和四类 transition 曲线；
2. v2 schema、manifest、数据审计和阶段进度样例视频；
3. B1-S1/B1-S4 step/compute-matched 报告；
4. B2～B5 顺序消融表和失败分析；
5. route/K/action/CRL 全套开环图；
6. 固定正面 seed 的完整自主闭环结果；
7. 三视角拼接视频和 trace checksum；
8. 最佳 run 的 resolved config、checkpoint identity 和实验 manifest；
9. 是否推广 terminal-hold、S4、prefix、CRL 和 on-policy data 的逐项决议；
10. 最终 Waypoint v2 合同、迁移说明和正式长训 resolved config；
11. 4×H20 正式长训连续至少 20 个健康有效 step 的启动证据，训练保持运行。

## 16. 参考工作

- PRTS：语言条件的 contrastive goal reachability，
  <https://arxiv.org/html/2604.27472>，代码 <https://github.com/TeleHuman/PRTS>
- SparkVLA：联合 Stop/action-prefix ranking，<https://arxiv.org/html/2608.16172>
- PALM：功能关键帧与连续 subtask progress，<https://arxiv.org/html/2601.07060>
- StarVLA：action chunk 边界 terminal-hold/padding 与 Layerwise FM，
  <https://github.com/vla-diff/starvla_base>

这些工作提供设计依据，不替代本项目在无 state、双 route-specific head、移动操作和 522
episode 数据规模下的独立消融。
