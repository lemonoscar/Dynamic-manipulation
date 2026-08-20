# ConveyorVLA AL0 `step_002500` 闭环问题简报

> 历史失败证据：受测 checkpoint 使用旧 `state28 + velocity/TCP-delta` 合同，不是
> Waypoint v1 checkpoint。本文保留用于说明架构迁移原因，不提供现行训练或部署命令；
> 当前结论见 [status.md](status.md)。

- 日期：2026-08-18
- 范围：Liangzhu seen，`step_002500`，真实无辅助 Isaac 闭环
- 训练代码：`b2a3a25d68ee88fe5f761528acb1229595a0c5f7`
- 闭环代码：`ce56ce9dabf3101ca809950aecaac832404635c8` 加本地适配层

## 1. 结论

`step_002500` 不能完成抓取、搬运和放置。闭环中机器人没有接触可乐，而是导航反向、频繁切换阶段，最终因姿态恶化终止。

当前最明确的问题有两个：

1. **Prompt 的 semantic memory 训练—在线不一致**：训练注入的是标注的“上一不同阶段”，在线注入的却是模型“上一 query 预测”；同一个文本槽位代表了不同变量。
2. **Navigation DiT 有效训练不足**：批量生成 padding 被严格 parser 判成 invalid，大量样本没有进入任何 DiT；导航 loss 和动作裁剪率也明显偏高。

Prompt 错位已由代码确认，但它对反向动作的准确因果占比仍需固定输入消融。`step_002500` 只是中途诊断 checkpoint，不是通过闭环门禁的正式模型。

## 2. 当前 Prompt 的完整结构

### 2.1 一次 query 有哪些输入

在时刻 `t`，策略接收：

- head camera 两帧：`[t-0.20 s, t]`；
- wrist camera 两帧：`[t-0.20 s, t]`；
- 完整四阶段任务指令；
- 可选 semantic memory `M`；
- `state28_t`。

需要严格区分：

| 输入 | 内容 | 进入位置 |
|---|---|---|
| visual history | 两个相机各两帧，`[-5,0]/0.20 s` | Pass 1、Pass 2，必须保留 |
| semantic memory `M` | 一条上一子任务文本或空 | Pass 1、Pass 2，当前存在冲突 |
| `state28_t` | 关节、TCP、夹爪等状态 | 只进入 DiT，不进入 Pass 1 的阶段判断 |

### 2.2 Pass 1：生成当前子任务

`add_generation_prompt=True`。送入 Qwen chat template 的逻辑内容如下，方括号中的 memory 整行可以不存在：

```text
<user>
Head camera, oldest to newest:
[head frame at t-0.20 s, head frame at t]

Wrist camera, oldest to newest:
[wrist frame at t-0.20 s, wrist frame at t]

Task: Walk to the box holding the Coke can. Keep the base still and pick up
the can. Lift it and retract the arm. Turn around and walk to the other empty
box. Keep the base still and place the can on top of it.
The head and wrist videos are ordered from oldest to newest.
[Previous model prediction (may be wrong): M]
What should the robot do now? Output exactly one canonical subtask as
<|pred_action|><|subtask|><subtask><|end_subtask|>
</user>
<assistant generation starts here>
```

Qwen 必须严格生成以下四种答案之一：

| 阶段 | `<subtask>` 的精确文本 | Dispatcher |
|---|---|---|
| NAV_TO_SOURCE | `Walk to the box holding the Coke can.` | Navigation DiT |
| PICK | `Pick up the Coke can, lift it, and retract the arm.` | Manipulation DiT |
| NAV_TO_TARGET | `Turn around and walk to the empty box.` | Navigation DiT |
| PLACE | `Lower the Coke can onto the empty box and release it.` | Manipulation DiT |

完整输出格式例如：

```text
<|pred_action|><|subtask|>Walk to the box holding the Coke can.<|end_subtask|>
```

严格 parser 不接受未知子任务或任意尾随文本。

### 2.3 Pass 2：用当前子任务提取动作条件

Pass 2 重新使用**相同视频、相同 user prompt 和相同 memory `M`**，然后追加 Pass 1 的当前预测 `P_t`：

```text
<user>
[与 Pass 1 完全相同的两路视频和文字]
</user>
<assistant>
<|pred_action|><|subtask|>{P_t}<|end_subtask|>
</assistant>
```

此时 `add_generation_prompt=False`，Qwen 做完整 forward，输出 hidden states `H_t`。后续流程是：

```text
P_t ──> Dispatcher ──> Navigation DiT 或 Manipulation DiT
H_t + state28_t ──> 被选中的 DiT ──> action chunk
```

因此 `M` 不只是给 Dispatcher 看的提示。它会同时改变 Pass 1 的阶段预测和 Pass 2 的 `H_t`；即使 Dispatcher 最终选对专家，DiT 的动作条件仍可能被错误 memory 改变。

### 2.4 训练时的 memory

Dataset 最初生成 empty-memory prompt；训练循环随后可能注入 `previous_subtask_text`。这个字段来自标注中的“上一不同阶段”，不是模型上一 query 的预测：

| 当前阶段 | 训练可注入的 `M` |
|---|---|
| NAV_TO_SOURCE | 空 |
| PICK | NAV_TO_SOURCE 文本 |
| NAV_TO_TARGET | PICK 文本 |
| PLACE | NAV_TO_TARGET 文本 |

但 render 后它被写成：

```text
Previous model prediction (may be wrong): {ground-truth previous phase}
```

也就是说，token 声称它是“模型上一预测”，实际却是 ground truth。训练还让 history 和当前 route 共用同一个 teacher-forcing 调度：step 100 后线性下降，step 4,000 为 0；history 另有 50% dropout 和 25% corruption。

对存在 previous label 的样本，Prompt 中 `M` 的期望分布为：

| 时点 | Empty | 正确上一阶段 | 随机错误阶段 |
|---|---:|---:|---:|
| 训练早期 | 50.00% | 37.50% | 12.50% |
| step 2,500 | 80.76% | 14.43% | 4.81% |
| step 4,000 后 | 100% | 0% | 0% |

### 2.5 在线时的 memory 与冲突

在线适配层使用真正的递归 self-history：

```text
query 0: M=None       ──> 预测 P_0
query 1: M=P_0        ──> 预测 P_1
query 2: M=P_1        ──> 预测 P_2
...
```

在线没有 dropout、corruption 或衰减；第一次之后 `M` 始终存在，错误还会连续反馈。冲突可概括为：

| 维度 | 训练 | 在线 |
|---|---|---|
| `M` 的来源 | ground-truth 上一不同阶段 | 模型上一 query 预测 |
| 出现率 | empty/clean/corrupt 混合，最终全 empty | 首次 empty，之后始终存在 |
| 时间结构 | 独立 shuffle | 连续递归，错误相关 |

最直观的未见组合是连续 NAV_TO_SOURCE：

```text
训练：M 永远为空
在线 query 0：M 为空
在线 query 1 以后：M=NAV_TO_SOURCE
```

因此在线从第二次导航 query 起就进入该阶段训练未见过的 Prompt 条件。若 user memory 写着 PLACE，而 Pass 1 当前预测为 PICK，Dispatcher 虽会按 PICK 选择 Manipulation DiT，但 Pass 2 hidden states 会同时编码 PLACE 和 PICK。

## 3. Step、epoch 与训练阶段

本次 run 使用 73,557 个 train rows：

```text
global batch = 32 micro batch × 4 GPU × 2 accumulation = 256
equivalent epoch(step) = step × 256 / 73,557
1 equivalent epoch ≈ 287.33 step
```

| 时点 | 累计抽样 | 采样等价 epoch | 阶段 |
|---|---:|---:|---|
| step 2,500 | 640,000 | 8.70 | 联合训练中段，teacher forcing≈38.49% |
| step 2,738 | 700,928 | 9.53 | 源训练进程停止位置；不是受测 checkpoint |
| step 4,000 | 1,024,000 | 13.92 | teacher forcing 归零 |
| step 10,000 | 2,560,000 | 34.80 | 计划训练终点 |

`step_002500` 已完成 25% 的 optimizer steps，学习率约为峰值的 88.3%，但尚未进入占总预算 60% 的 zero-teacher-forcing 稳定阶段。它只是每 500 step 自动保存的 **mid-run diagnostic checkpoint**，不是按验证结果选择的合格模型。

“34.80 epoch”只能理解为总体抽样曝光量。`WeightedRandomSampler(replacement=True)` 会重复抽取高权重阶段、导航末端和边界样本，并遗漏另一些行；invalid route 还不会进入任何 DiT，所以不能说每条数据或每个 DiT 都被完整训练了 34.8 遍。

## 4. 评测结果与根因证据

真实闭环没有专家、FSM、CuRobo 或 teacher 辅助：

| 指标 | 结果 |
|---|---:|
| 初始/最小/最终目标距离 | `0.5905 / 0.5895 / 1.2076 m` |
| TCP—可乐最小距离 | `0.7765 m` |
| 可乐位移 | `0 m` |
| NAV_TO_SOURCE 负 `vx` | `58 / 140` |
| 最终倾角 | 约 `59.35°` |
| 结果 | `robot_environment_terminated` |

关键证据：

- 训练中 NAV_TO_SOURCE 的 208,226 个有效 action step 没有负 `vx`，闭环却采样到最低 `-0.253 m/s`。
- 第一次 empty-memory 导航前 5 步平均 `vx=+0.0467 m/s`；下一次 `M=NAV_TO_SOURCE` 时为 `-0.2309 m/s`。两次图像也变化了，所以这是支持证据，不是固定输入因果证明。
- oracle-route、empty-memory 开环中，NAV_TO_SOURCE 前 5 步方向一致率 98.61%，反向前缀为 0。
- 严格批量生成累计判出 16,821 个 invalid；这些样本没有进入 DiT。step 2001–2500 的单个 32 样本微批平均只有 7.07 个样本进入 Navigation DiT。
- test action loss：NAV_TO_SOURCE `0.09591`、NAV_TO_TARGET `0.10164`、PICK `0.01729`、PLACE `0.00686`。
- NAV_TO_SOURCE 监督中，32.2% 的 `vx` 和 55.6% 的 `wz` 超出现有 scale 并被裁剪。

优先级判断：

| 优先级 | 问题 | 状态 |
|---|---|---|
| P0 | semantic-memory 训练/在线合同错位 | 代码已确认 |
| P0 | batch padding 导致格式 invalid 和 DiT 样本饥饿 | 日志与离线测试已确认 |
| P1 | Navigation DiT 未收敛、动作 scale 饱和 | loss/统计已确认 |
| P2 | 导航缺少全局目标方位，存在部分可观测性 | 架构风险，需在 P0/P1 修复后复测 |

## 5. 最短修复路径

对现有 `step_002500`：

1. 在线关闭 **semantic self-history**，始终用 empty `M`；保留两帧 visual history。
2. 在 token 级将每个生成结果裁到第一个 `<|end_subtask|>`，保持严格 parser，要求 batch invalid=0。
3. 固定相同图像、`state28` 和 diffusion noise，只切换 `empty/current/previous/wrong M`，分别比较 Pass 1、Pass 2 hidden states 和 `vx/wz`。

重新训练：

1. 从干净 Qwen3-VL/ABot action 初始化，不从 `step_002500` 或旧 optimizer resume。
2. Prompt 不得使用 annotation 的 previous completed phase。
3. 若保留 memory，必须用连续序列中的 previous-query prediction 和 scheduled sampling；history 与 route teacher forcing 分开调度并最终归零。
4. 修复导航 `[vx,wz]` scale，记录裁剪率；持续记录两个 DiT 的有效样本、loss 和非零梯度。
5. 增加偏航、过冲和离轨恢复数据；低层保留倾倒、IK 和不可达目标保护，但不增加外部任务 FSM。

## 6. 重新训练门禁

- train/online 的 Pass 1、Pass 2 渲染后 token 合同一致；
- Prompt 中没有 annotation previous phase，batch invalid=0；
- teacher forcing=0 时 VLM、Navigation DiT、Manipulation DiT 均有有限 loss 和非零梯度；
- 固定输入 history 消融不再造成高频方向翻转；
- 导航动作裁剪率接近 0，诊断导航总体接近目标；
- Navigation 和 Manipulation 分阶段闭环分别通过；
- 多 seed 完整无辅助闭环完成导航、抓取、搬运和放置。

本地证据：

- `artifacts/evaluation/step_002500_closed_loop_20260818/r5_true_closed_loop/VALIDATION_REPORT.md`
- `artifacts/evaluation/step_002500_full_test_20260818/TEST_REPORT.md`
- `artifacts/evaluation/step_002500_open_loop_action_quality_20260818/OPEN_LOOP_ACTION_QUALITY_REPORT.md`
