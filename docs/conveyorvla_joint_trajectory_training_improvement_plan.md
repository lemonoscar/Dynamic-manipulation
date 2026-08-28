# ConveyorVLA Joint-Trajectory 训练改进方案

- 文档版本：`conveyorvla-joint-trajectory-training-plan-v1`
- 状态：代码实现与合成门禁已完成，4 条 Gate-A review episode 已审计；正式数据 release、
  overfit、正式训练与真实闭环尚未开始
- 决策冻结日期：2026-08-27 CST
- 最新语义修订：2026-08-28 CST，冻结 raw-derived `K=0` full-hold 合同
- 适用开发分支：`Manipulation_Navi_v1`
- 目标模型合同：`conveyorvla-joint-trajectory-policy-v1`
- 目标数据 schema：`conveyorvla-joint-trajectory-v1`

## 0. 权威性与变更边界

本文定义 step1250 之后的 breaking-change successor，不原地修改或冒充现行
[Waypoint Policy v2 合同](conveyorvla_waypoint_policy_contract_v2.md)。旧 Waypoint v1/v2
数据、normalizer、config、checkpoint 和评测证据继续只读；新方案必须使用全新的模型合同、
数据 schema、resolved config、manifest、normalizer、run ID 和 checkpoint identity。

本文负责训练目标、模型结构、初始化、sampler、loss、optimizer、runtime 对齐和正式训练阶段。
全新数据的数量、随机化、采集时钟和教师速度见
[Joint-Trajectory 数据采集规范](conveyorvla_joint_trajectory_fresh_data_collection_spec.md)。
二者冲突时：

1. 本文决定模型、训练和 runtime 语义；
2. 数据采集规范决定 raw/derived 数据与采集质量；
3. 旧 v2 合同只决定旧 checkpoint 的复现，不约束新 joint-trajectory checkpoint。

本文是目标设计，不授权在文档提交时启动 GPU 任务。实现完成后，必须另行冻结实际代码
commit、数据 manifest、resolved config 和远端运行身份；机器私有目录、数据、checkpoint、
日志和视频不得进入 Git。

## 1. 结论摘要

首版采用以下最小组合：

| 项目 | 冻结决定 |
|---|---|
| Pass 1 | Qwen 继续生成 `ACTION + route + free-form subtask` |
| route | 四个原子 token：`NAV_TO_SOURCE/PICK/NAV_TO_TARGET/PLACE` |
| DONE | 删除；任务成功由 evaluator truth 判定 |
| Pass 2 | 使用模型自己的完整 assistant prefix 做第二次完整 Qwen forward |
| NAV | 独立 expert，`[10,3] @ 0.20 s`，2.0 s |
| Mani | 独立 expert，`[10,7] @ 0.04 s`，0.4 s |
| Mani state | `q6+dq6+gripper1=13D`，只进入 Mani expert |
| Mani action | query-relative `delta_q6` + continuous absolute gripper |
| action horizon | 完整 10 点监督和执行，无 `K*`、prefix selector 或 suffix mask |
| boundary suffix | `K>=1` 保留合法 prefix 后 hold；严格 raw-derived `K=0` 则完整 10 点 hold |
| FM 训练采样 | `M=1` |
| FM 推理积分 | 默认 10 步 |
| route 训练 | interior 硬 CE + route-specific transition 软 CE |
| route runtime | 初始和切换均需连续两次新观测确认 |
| progress | route-specific 物理完成度，不使用时间/row index |
| sampler | global batch 64，domain/route/progress/boundary/episode 分层 |
| 视觉增强 | 首版关闭 |
| 训练长度 | 约 2 个数据等效 epoch，不固定为 3,000 step |
| 解冻 | 前 0.25 epoch 只训动作模块，后 1.75 epoch 全量解冻 |
| 关闭项 | DONE、M=4、prefix、CRL、on-policy、self-conditioned auxiliary |

## 2. 模型输入、输出与控制所有权

### 2.1 Pass 1

Pass 1 只读取：

- 完整全局任务；
- head `[t-0.20 s,t]` 两帧；
- wrist `[t-0.20 s,t]` 两帧。

它受约束生成：

```text
ACTION + one atomic route token + free-form subtask
```

首版暂不删除 `ACTION` 或自由文本 subtask。route token 必须由 Qwen 的受约束 token logits
产生；不得使用外部分类器、GT phase、previous route、robot state 或 evaluator truth 覆盖。

Pass 1 不得读取：

```text
q/dq/gripper、base pose/twist、teacher operation、GT route、object truth、
previous subtask、pending route counter、simulator target
```

### 2.2 Pass 2

Pass 2 使用：

- 与 Pass 1 相同的任务和四张视觉图；
- 模型自己生成的完整 `ACTION + route + subtask` assistant prefix。

Pass 2 必须是第二次完整 Qwen forward，不复用 Pass 1 的 hidden。Qwen hidden 条件进入由
committed route 选中的一个 action expert。训练首版使用教师 committed-route prefix；正式
开环和 runtime 必须使用模型自产 prefix，首版不做 on-policy correction 混采。

### 2.3 route 与 action domain

| route | action domain | 动作 expert |
|---|---|---|
| `NAV_TO_SOURCE` | NAV | Navigation expert |
| `PICK` | Mani | Manipulation expert |
| `NAV_TO_TARGET` | NAV | Navigation expert |
| `PLACE` | Mani | Manipulation expert |

软 route 概率只训练 Pass 1；它不用于混合 NAV/Mani 输出。每个 row 的 FM loss 只进入教师
实际 committed route 对应的单一 expert。

### 2.4 不再输出 DONE

模型不预测 `DONE`，也没有 DONE row、DONE CE 或 `PLACE→DONE` route boundary。PLACE
持续执行，直到 evaluator 满足：

1. 可乐已经脱离夹爪；
2. 可乐主体位于任务指定 box 的有效内部区域；
3. 上述条件连续保持至少 1.0 s；
4. 可乐姿态不限；
5. 满足后立即终止 episode 并记录 success。

这些 truth 只属于 evaluator termination，不进入模型 request、route 或 action 控制链。瞬时
接触、仍被夹持、正在穿过或弹出目标区域均不得计为成功。

## 3. 两个动作 expert

### 3.1 共同结构

Qwen 由两个参数独立的 16-block action expert 共享。每个新 block 的顺序为：

```text
action/state self-attention
        ↓
cross-attention to matching Qwen hidden
        ↓
FFN
```

旧 Layerwise FM block 只有 cross-attention + FFN，且旧 `future_tokens` 无法影响最终 action
输出。新 expert 删除无效 future-token 分支；self-attention、动作位置表示和新 state/action I/O
必须重新初始化。

### 3.2 NAV expert

```text
shape:    [10,3]
stride:   0.20 s
horizon:  2.0 s
target:   query-body [dx_body,dy_body,dyaw]
state:    none
```

10 点都相对同一个 query body pose，不是相邻增量。NAV expert 不读取 q/dq/gripper、base
state 或外部 progress。runtime 把完整有效轨迹交给 waypoint→PCT→DWA 链；不得恢复 learned
`K*`、固定 K、prefix selector 或 GT 完成门控。

### 3.3 Mani expert

Mani state token 恰好包含：

```text
q_measured[6] + dq_measured[6] + gripper_measured[1] = 13D
```

该 token 只进入 Mani expert 的 self-attention；不得拼入 Qwen token、Pass 1 prompt、NAV
expert 或 assistant prefix。

Mani 输出为：

```text
shape:    [10,7]
stride:   0.04 s
horizon:  0.4 s
joint:    delta_q[k] = q_applied(t+(k+1)*0.04) - q_measured(t)
gripper:  continuous absolute open fraction in [0,1]
```

runtime 在整个 chunk 上使用同一个 query anchor：

```text
q_target[k] = q_measured(query) + predicted_delta_q[k]
```

`delta_q` 不是相邻点积分。夹爪不二值化，不增加提前闭合/释放硬门禁。

### 3.4 runtime 执行边界

- Mani route 下 base command 始终为 `[0,0,0]`；
- 顺序执行完整 10 点，chunk 用尽后保持末点并重新观察；
- 不调用 IK、cuRobo、`plan_pose` 或逐点 feasibility selector；
- policy 层不拒绝、不跳点；
- 底层保留不可关闭的 joint-position 和 joint-rate saturation；
- saturation 只保护执行器，不覆盖 route，不算模型成功；
- validation saturation rate 目标 `<0.5%`，超出表示模型/normalizer 缺陷。

## 4. 初始化与 checkpoint 兼容性

### 4.1 选择性 warm-start

目标初始化来自冻结 step1250 checkpoint，但它不是 resume：

| 参数区域 | 处理 |
|---|---|
| Qwen vision/language/multimodal | 加载 step1250 |
| route special-token embeddings/lm head | 加载 step1250；删除 DONE 监督但不原地改旧 tokenizer |
| boundary head | 仅在 shape 与目标语义都兼容时加载，否则重新初始化 |
| progress head | 重新初始化；旧 v2 的 elapsed-phase target 与新物理完成度不兼容 |
| 旧 expert cross-attention/FFN | 按明确 key map 加载到同域新 expert |
| 旧 flow-time encoder | shape/定义一致时加载 |
| action encoder/decoder | 重新初始化 |
| self-attention | 新增并重新初始化 |
| Mani state encoder | 新增并重新初始化 |
| horizon/position embedding | 重新初始化 |
| 旧 future tokens | 不加载，目标结构中删除 |
| optimizer/scheduler/RNG | 全部重新初始化 |
| normalizer | 使用新数据重新拟合，不加载旧值 |

loader 必须输出逐 key 的 `loaded/reinitialized/rejected` 报告，并拒绝静默 shape fallback。
新 checkpoint 与旧 waypoint/TCP checkpoint 不允许相互 resume。

旧 v2 `phase_progress` 由 segment timestamp 比例生成，属于本方案明令禁止的 elapsed-time
proxy。因此即使 tensor shape 相同，也不得加载旧 progress head；新 progress head 只能由
route-specific 物理完成度标签从头学习。

### 4.2 12-episode overfit 不进入正式初始化

12-episode overfit 是独立、可丢弃的架构门禁。通过后必须丢弃其模型和 optimizer state，
再从相同 step1250 选择性初始化启动正式 run，避免把 12 条轨迹和 Adam moments 带入正式
模型。

## 5. target、terminal-hold 与 normalizer

### 5.1 完整 10 点监督

NAV/Mani FM 始终监督全部 10 点。若第一个名义 target 仍属于 query 的 committed route，走
普通 `K>=1` 路径；horizon 跨过真实 route boundary 时：

1. 保留当前 committed route 的连续合法动作；
2. 后续位置重复最后一个合法 target；
3. gripper 同样保持最后连续目标；
4. action-valid mask 对 10 点全部为真；
5. 不拼接下一 route 动作；
6. 不保留 `K*`、`L_prefix` 或 suffix loss mask。

zero-prefix 只允许由连续的 50 Hz raw 时序证明：在第一个名义 target tick 到来前或正好到来
时，control log 已提交到合法直接后继 route，则记 `boundary`；若 control log 在该 tick 前
干净结束，且 evaluator、`summary.success=true`、`final_state=done` 同时确认，则记
`success_tail`。固定输出完整 10 点 hold、mask 全真、`terminal_hold_start_index=0`；NAV hold
为十个零 body reference，Mani hold 为 query 时刻 applied joint target 相对 measured q 的
delta 与当前 applied gripper 的十次重复。

raw `tail_reason` 不参与推导或兜底。tick/时间戳断裂、非法 route jump、pending/proposed-only
切换、失败/timeout/truncation、未证明 tail 或其他 future 缺失都必须拒绝。`K=0` 和
`terminal_hold_start_index` 只作审计，不恢复 learned K*/prefix head/runtime selector；runtime
仍信任全部 10 点。完整判定顺序以
[数据采集规范 3.5](conveyorvla_joint_trajectory_fresh_data_collection_spec.md#35-边界后缀)为准。

### 5.2 normalizer

所有统计只读取 train split：

- NAV 三个物理通道各自拟合，跨两个 NAV route、跨 10 个 horizon index 共享；
- Mani 六个 `delta_q` 各自拟合，跨 PICK/PLACE、跨 10 个 horizon index 共享；
- Mani state 的 q/dq 使用独立 train-only 统计；
- gripper action/state 使用已知 `[0,1]`，action 固定映射到 `[-1,1]`；
- 不建立 per-route 或 per-horizon normalizer；
- resolved normalizer 必须有 immutable ID 与 SHA-256。

## 6. Pass 1 与辅助目标

### 6.1 `L_answer`

`L_answer` 只训练：

- `ACTION` 与输出格式；
- free-form subtask 文本。

route token 从 `L_answer` 中 mask，避免与软 route CE 重复监督。route transition 软窗口内，
route-specific subtask token 同样 mask；阶段内部恢复完整 subtask CE。

### 6.2 `L_route`

`L_route` 是四个 Qwen 原子 route token 的唯一权威目标：

- phase interior 使用 one-hot 硬 CE；
- transition window 只在 old/new 两 route 上使用软 CE；
- 其他 route 的目标概率为 0；
- 同一个 row 不再叠加第二份硬 route CE。

软标签为：

```text
p_new = sigmoid(boundary_signed_time_s / tau_transition)
p_old = 1 - p_new
```

首版候选窗口：

| transition | 初始 `tau` | 约三倍 tau 的有效窗口 |
|---|---:|---:|
| `NAV_TO_SOURCE→PICK` | 0.20 s | ±0.60 s |
| `PICK→NAV_TO_TARGET` | 0.30 s | ±0.90 s |
| `NAV_TO_TARGET→PLACE` | 0.20 s | ±0.60 s |

最终 tau 必须由 200-episode pilot 的物理过程和视觉变化冻结；窗口外恢复硬 CE。

### 6.3 `L_boundary`

boundary ranking 只比较同一 episode、同一 `transition_id` 的 before/after rows，要求新 route
logit 随真实进度上升、旧 route logit 下降。不得跨 episode 拼 pair，也不得用全部 O(n²)
组合放大该 loss。

### 6.4 `L_progress`

progress 只表示 route-specific 物理完成度：

- NAV_TO_SOURCE：接近 source 与最终停稳；
- PICK：reach/alignment/descend/close/lift/carry-ready；
- NAV_TO_TARGET：携带状态下接近目标与最终停稳；
- PLACE：到释放位、下降、打开与物体脱离。

它禁止使用 elapsed time、phase row index 或固定帧比例。hold 时 progress 保持，重新对准或
物理倒退时允许小幅下降。无法构造可信物理标签的 route 必须 mask `L_progress`，不得伪造。
progress 只塑造训练 hidden，不进入 runtime route/action 控制。

## 7. Flow Matching 与总 loss

### 7.1 FM 采样与积分

首版严格采用官方 π₀ 风格的每 chunk 单 draw：

```text
loss.repeated_diffusion_steps = 1
action_model.num_inference_timesteps = 10
```

每个真实 action chunk 每次 optimizer exposure 采一组独立 noise 和一个 flow time。不同 batch
row 和不同 optimizer step 自然重采样。M=4 不进入首版，也不设置延迟激活 schedule。

### 7.2 Mani joint/gripper 分组

六个 joint 与一个 gripper 不按七维朴素平均：

```text
L_FM_Mani = 0.75 * L_joint_mean + 0.25 * L_gripper
```

Mani sampler 至少约 25% 覆盖 gripper 转换前、转换中或转换后窗口。仍预测连续 gripper，
不增加分类头或 runtime hard gate。

### 7.3 loss 归一化

每个目标先在自己的有效 token/row/domain/dimension 内求平均，再应用固定权重：

```text
L = 1.0 * L_answer
  + 1.0 * L_route
  + 1.0 * L_FM_NAV
  + 1.0 * L_FM_Mani
  + 0.2 * L_boundary
  + 0.1 * L_progress
```

- `L_answer`：有效文本 token 平均；
- `L_route`：有效 route rows 平均；
- `L_FM_NAV`：NAV batch×10×3 平均；
- `L_FM_Mani`：先按 joint/gripper 分组，再按上式组合；
- `L_boundary`：有效 transition pairs 平均；
- `L_progress`：有可信物理标签的 rows 平均。

首版不使用 GradNorm、PCGrad 或自动 loss balancing。记录各 loss 对 action expert、Qwen 和
vision 的梯度范数即可。

### 7.4 明确关闭

```text
L_prefix = off
local CRL = off
on-policy correction = off
self-conditioned auxiliary = off
M=4 = off
DONE/operation loss = removed
```

这些模块不得残留“权重为 0 但仍执行昂贵 forward”的代码路径。

## 8. 分层 sampler 与 global batch

正式 scientific contract 的 global batch 固定为 64，与使用 2 张或 4 张 GPU 无关；world
size、per-rank micro-batch 和 accumulation 在 resolved config 中换算，必须保持：

```text
world_size * micro_batch_per_rank * accumulation = 64
```

初始 global-batch 组成：

```text
约 28 NAV interior
约 28 Mani interior
约 8 boundary rows/pairs
```

boundary rows 仍只训练其 committed domain expert。sampler 的抽样顺序为：

```text
action domain → route → progress/boundary bucket → episode → row
```

硬要求：

- NAV_TO_SOURCE/NAV_TO_TARGET 在 NAV 内平衡；
- PICK/PLACE 在 Mani 内平衡；
- 每个 route 覆盖 early/middle/late progress；
- 三类 boundary 均有 before/after pair；
- 普通 row 在一个 global batch 内每个 episode 最多出现一次；
- 同 transition boundary pair 是唯一允许的 episode 重复；
- global batch 64 目标至少约 56 个 distinct episodes；
- gradient accumulation 窗口尽量不立即复用 episode；
- sampler 记录每 route/domain/progress/boundary/episode 的实际 exposure。

route CE 若因分层抽样改变自然先验，应使用已知 sampling probability 做权重修正；自然分布
validation 只用于校准检查，不改变训练 sampler。

## 9. 训练阶段

### 9.1 独立 overfit gate

先用 12 条成功 episode 建立临时 run，覆盖四 route、三 boundary、两个 destination、NAV
terminal-hold、PICK close/lift 和 PLACE open/release。它必须证明新 head 能记住 joint/gripper
时序和完整 10 点。通过后整个临时 run 丢弃；失败则修数据/实现，禁止靠全量长训掩盖。

### 9.2 正式 Stage A：动作模块 warm-up

正式 run 从相同 step1250 选择性初始化和全新 optimizer 开始。前约 0.25 数据等效 epoch：

- Qwen、vision、route、boundary、progress 参数冻结；
- 只训练 NAV/Mani expert、Mani state encoder 和 action I/O；
- Qwen 只做生成 action hidden 所需的 frozen forward；
- route/text/aux loss 可无梯度记录，但不得浪费额外解码路径。

若正式 train rows 约 384k–576k，Stage A 约 1,500–2,250 optimizer steps。

### 9.3 正式 Stage B：全量联合训练

剩余约 1.75 数据等效 epoch 全量解冻 Qwen，包括 vision encoder。所有模块都更新，但使用
差异化学习率：

| 参数组 | peak LR |
|---|---:|
| NAV/Mani expert、state/action I/O | `2e-5` |
| Qwen language/multimodal core | `2e-6` |
| vision encoder | `5e-7` |
| route token embeddings/lm head | `1e-5` |
| boundary/progress heads | `1e-5` |

optimizer/scheduler：

```text
optimizer:          AdamW
betas:              [0.9,0.95]
epsilon:            1e-8
weight_decay:       1e-8
global grad clip:   1.0
precision:          bf16
action warmup:      200 optimizer steps
Qwen/vision warmup: 解冻后 100 optimizer steps
decay:              cosine，末尾保留 peak LR 的 10%
```

action scheduler 从 Stage A 连续推进，不能在 Qwen 解冻时重启。Qwen/vision 在 Stage A 的
LR 严格为 0，解冻后独立平滑上升；它们的 optimizer moments 从空状态开始。

### 9.4 总长度

正式总长度由 materialized train rows 与 global batch 64 计算：

```text
equivalent_epoch_steps = train_eligible_rows / 64
target_steps = approximately 2 * equivalent_epoch_steps
```

预计 384k–576k rows 对应每 epoch 约 6,000–9,000 steps、总计约 12,000–18,000 steps。
3,000 step 只能作为中期观察点，不能称为完整训练。

### 9.5 保存与首轮非目标

- 每 250 effective optimizer step 保存；
- 保存主要用于恢复、加载检查和阶段观察；
- 首轮暂不设计复杂 checkpoint 排名或单一综合分数；
- 不以最后 step 或 total loss 自动宣称最佳；
- 不做训练时视觉增强；
- 不混入旧 waypoint/TCP action rows，旧数据只作历史对照。

## 10. runtime route commit

runtime 只使用 Qwen 自产 probabilities：

1. 初始 route 连续两次新观测为同一最高概率 token 后提交；
2. 新 route 连续两次为同一个候选，且两次均满足 `P(new)>P(committed)` 后切换；
3. 第一次新 route 胜出时进入 pending，base 置零、Mani 保持最后 target；
4. pending 不作为模型输入，不保存为 previous-route history；
5. 切换后只激活新 route 对应 expert；NAV→Mani 时 base 保持严格零速；
6. 没有固定 85% 阈值，也没有 GT 距离/phase/FSM 覆盖。

soft CE 只让概率随物理边界平滑 crossover；runtime 始终执行离散 committed route。

## 11. 训练与实现门禁

### 11.1 静态合同

至少覆盖：

- Pass 1/Qwen/NAV state tensor 数为 0；
- Mani state 恰好 13D，字段顺序和单位冻结；
- route 集合恰好四个且无 DONE；
- NAV/Mani shape、stride、frame、normalizer round-trip；
- action 10 点全部 valid，terminal-hold 无 suffix mask；
- 第一个 target 仍属于当前 route 时必须走普通 `K>=1` 派生；
- 合法直接后继在首 target 前/当时 committed 时，严格派生 `K=0 boundary` full hold；
- evaluator-confirmed clean end 严格派生 `K=0 success_tail` full hold；
- 其他 missing tick/tail、非法 jump、pending-only route 和 query `tail_reason` fallback 全部拒绝；
- `repeated_diffusion_steps=1`、inference steps=10；
- route token 不进入 `L_answer`；
- transition subtask mask 正确；
- soft CE 只分配 old/new route；
- physical progress 禁止 row-index fallback；
- Mani joint/gripper 0.75/0.25 分组；
- sampler domain/route/episode/pair 约束；
- Stage A/Stage B requires-grad 与 LR schedule；
- checkpoint key map 和新旧 checkpoint 拒绝规则；
- runtime 无 IK/cuRobo/prefix selector；
- evaluator success 的 release/inside/1.0 s 规则。

### 11.2 训练健康

正式启动后至少连续观察 10 个有效 optimizer steps，确认：

- 所有预期 GPU/rank 参与；
- effective global batch 恰好 64；
- total/component loss、gradient、LR、throughput、显存均有限；
- Stage A 中 Qwen/vision 梯度和 LR 严格为 0；
- NAV/Mani 都得到真实非零样本；
- 无 NaN/Inf、OOM、NCCL error、traceback 或错误 output path；
- resolved config、manifest、checkpoint identity 与本合同一致。

### 11.3 动作与闭环阶段证据

正式训练不能替代：

- 12-episode overfit；
- 固定 noise 的 NAV/Mani 完整 10 点开环图；
- gripper open→close、close→open 时序；
- 三个 boundary 的 early/late、crossover 和 flicker；
- 多 seed NAV→PICK→close→lift milestone；
- 完整可乐释放到目标 box 的 evaluator success。

checkpoint 排名策略当前明确 deferred；先验证训练主链和动作能力。

### 11.4 2026-08-28 代码与数据落地边界

当前 `Manipulation_Navi_v1` 分支已经以独立 module/config 实现 Wave 1–4 的可离线部分：

- 四 route、无 DONE、NAV/Mani 两种 10 点动作合同；
- fresh raw applied-command 校验、派生、terminal-hold、train-only normalizer 和 immutable
  materializer；
- 两个独立的 self-attention→Qwen cross-attention→FFN action expert；
- selective warm-start、answer/route mask、hard/soft route CE、boundary/progress、M=1 FM；
- global batch 64 分层 sampler、Stage A/B 解冻、参数组 scheduler 和新 checkpoint identity；
- Pass 1 双确认、pending hold、NAV reference、Mani direct-joint executor 与 evaluator success。

以上代码只由 synthetic fixtures、静态合同和旧 Waypoint 定向回归证明。2026-08-28 审计的
4 条 Gate-A review episode 已证明 raw 时钟、applied command、图像和 Mani 时序基本可用，
但也暴露了 11 个合法 zero-prefix rows、NAV 教师包络不合规与 PLACE progress 只覆盖 late
bucket。新冻结的 `K=0` 规则尚未在当前 materializer/validator/tests 中实现；因此 review 数据
不是正式 immutable release，仓库中仍没有正式 manifest/normalizer、新 checkpoint、overfit、
训练日志或真实 Isaac 闭环证据。纯 Python runtime 也尚未证明已接入现场 PCT/DWA 与机器人
底层 joint controller。完整实现清单和复现命令见
[Manipulation_Navi_v1 代码实施报告](manipulation_navi_v1_code_implementation_20260827.md)。

## 12. 实施波次与回滚

### Wave 1：合同与 schema

- 新建 versioned model/data/config identity；
- 删除目标 schema 的 DONE、K* 和旧 TCP action；
- 加入 Mani 13D state 与 applied joint labels；
- 验证 manifest 和 normalizer hash。

回滚：不发布新 manifest，旧 v2 完全不变。

### Wave 2：模型与 checkpoint loader

- 实现两个 10-step expert；
- 加入 action self-attention 与 Mani state token；
- 删除目标模型 future tokens；
- 实现显式 warm-start key map 和拒绝报告。

回滚：新 model contract 不加载，旧 checkpoint loader 不修改语义。

### Wave 3：loss、sampler 与 trainer

- 实现 route/text mask、route-specific soft CE、boundary/progress；
- 实现 Mani joint/gripper 分组 FM；
- 实现 global-batch 分层与 episode 去重；
- 实现 0.25/1.75 epoch 解冻和参数组 scheduler。

回滚：新 trainer/config 独立失效，不污染旧 config。

### Wave 4：runtime 与 evaluator

- Mani direct-joint 顺序执行，不调用 IK/cuRobo；
- NAV 完整 10 点接入 PCT/DWA；
- route 双确认与 pending hold；
- 删除 DONE，加入 evaluator success termination；
- 保留底层 joint/rate safety 与 saturation trace。

回滚：只切回旧 runtime/model contract，不转换新 checkpoint。

### Wave 5：overfit 与正式训练

- 独立 12-episode overfit 并丢弃；
- 从相同初始化启动 full-data 正式 run；
- 按数据 rows 解析 2 epochs 和每 250 step 保存；
- 远端启动前使用实时身份、目录、Git、Conda、GPU、tmux 和 PID guardrails。

回滚：停止新 run，保留 manifest/log/checkpoint 证据；不修改旧基线。

## 13. resolved run 必需身份

正式 run manifest 至少绑定：

```text
Git commit + dirty flag
model/config/data schema versions
resolved config SHA-256
dataset manifest/split/normalizer SHA-256
step1250 initialization URI/SHA-256 + key-map report
tokenizer/processor identity
optimizer/scheduler and stage boundary
global batch derivation
seed and distributed topology
CUDA/driver/GPU/environment identity
simulator/asset/calibration identity
checkpoint/output/log locations
evaluator protocol and success definition
```

不得用 mutable `latest`、复用旧 output 目录或手工拼接 optimizer state。2-GPU 与 4-GPU 只要
global batch、seed/sampler 语义和 resolved topology 明确，可以属于同一 scientific contract；
实际启动仍须遵守远端实时资源归属。

## 14. 参考证据

- 冻结旧 v2 合同与 step1250 缺陷：
  [Waypoint Policy v2](conveyorvla_waypoint_policy_contract_v2.md)
- step1250 route/action/闭环证据：
  [step1250 严格评测](waypoint_v2_step1250_strict_evaluation_20260825.md)
- 新数据与执行速度：
  [Joint-Trajectory 数据采集规范](conveyorvla_joint_trajectory_fresh_data_collection_spec.md)
- 官方 π₀ 每个 action chunk 使用单组 noise/time，默认 10 个推理积分步：
  <https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py>
- 官方 action chunk 顺序执行：
  <https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py>

## 15. 最终冻结清单

```text
KEEP:
  ACTION + atomic route + free-form subtask
  two full Qwen forwards
  Qwen-only route ownership
  NAV→PCT→DWA
  route hard/soft CE + boundary/progress
  complete 10-point terminal-hold supervision
  raw-derived K=0 boundary/success-tail full hold (audit-only)
  low-level joint/rate protection

CHANGE:
  NAV [20,3]@0.60 → [10,3]@0.20
  Mani TCP [20,7]@0.20 → joint trajectory [10,7]@0.04
  state-free Mani → 13D state token inside Mani expert only
  cross-attn-only block → self-attn + cross-attn + FFN
  fixed 3,000 steps → approximately 2 data-equivalent epochs
  partial execution/planner rejection → full chunk direct execution

REMOVE:
  DONE
  learned K*/prefix/suffix mask (K=0 audit sentinel is not a model K)
  Mani IK/cuRobo runtime
  M=4
  CRL
  on-policy correction
  self-conditioned auxiliary
  training image augmentation
  old action-data mixing
```

只有 fresh data、overfit、训练健康和真实 runtime/evaluator 测试全部通过后，才能把当前代码
候选晋升为独立正式模型合同。本文及 synthetic tests 不证明模型已经训练成功。
