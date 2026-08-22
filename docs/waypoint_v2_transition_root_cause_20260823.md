# Waypoint v2 阶段无法切换：证据链、根因与修正

- 复核日期：2026-08-23 CST
- 范围：Waypoint v2 `step_002000` 严格开环、seed 139 真实闭环、训练分布和实现计算图
- 结论状态：根因已定位；运行时不增加 gate；候选组件回退到 B2 最小组合

## 1. 结论

当前失败不能简单归因于“训练步数不够”。被评测的 `step_002000` 并不是全量数据长训，
而是从 8-episode overfit checkpoint 续训，在 1,877 个训练 row 上以 global batch 64
训练到 step 2000，等价重复约 68.19 个 sampling epoch。继续在同一分布上增加 step，只会
加强记忆，不能提供缺失的视觉状态覆盖。

更关键的是，B2 的批准语义没有完整进入主监督：transition window 的 old/new route
连续标签只存在于低权重 `route_crossover_loss`；同一个 Qwen route token 仍同时受到完整
assistant CE 和独立 route CE 两份硬标签监督。软目标实际在对抗更强的离散瞬切目标。

严格证据还否决了当前 prefix 与 CRL 的晋级：prefix 大面积塌缩到 `K=1`，CRL 的 action
shuffle drop 接近零且多个 route 的 progress 相关方向错误。因此下一训练候选应为：

```text
官方 Qwen3-VL-4B 初始化
  + v2 terminal-hold 全 20-step 监督
  + 修正后的 B2 boundary/progress 与连续 route 监督
  + S1 Flow Matching
  - learned prefix ranking
  - CRL
  - on-policy correction
```

这不是通过外部 FSM、truth、stall 或距离门控补偿模型。Pass 1 route 所有权和
NAV→PCT/DWA、ARM→原始 arm-vla→cuRobo/IK 主链不变。

## 2. 相互独立的关键证据

### 2.1 checkpoint 身份证明它是 overfit，不是全量长训

`step_002000@02ee859` 的 resolved run 记录：

| 项目 | 实际值 |
|---|---:|
| `training_subset` | `true` |
| episode 数 | 8 |
| train rows | 1,877 |
| global batch | 64 |
| max step | 2,000 |
| sampling epoch @ step 2000 | 68.1939 |
| FM train draws | S1 |

同一份 immutable v2 数据实际包含 522 episode、119,700 rows，其中 train 为 108,603
rows。当前 checkpoint 没有在这份完整训练分布上收敛过。

### 2.2 严格开环门禁已失败

严格开环报告 `open_loop_v2_overfit64.json` 虽显示 route accuracy `1.0`，但 64 个样本来自
训练 subset；四种 transition 各只有 1 个事件，且全部来自同一个训练 episode。零 lag、
零 flicker 因此只是记忆证据，不能外推。

动作和表征门禁的实际结果为：

| 指标 | 结果 | 判定 |
|---|---:|---|
| NAV direction accuracy | 0.75 | 低于 0.90 门槛 |
| NAV normalized OOB | 0.34375 | 失败 |
| ARM orientation mean error | 0.344 rad | 高于 0.20 rad 门槛 |
| ARM step violation | 0.6339 | 失败 |
| ARM normalized OOB | 1.0 | 失败 |
| prefix overall MAE | 0.538 | 未校准 |
| prefix overrun | 0.173 | 未通过 |
| CRL action-shuffle drop | 0.00679 | 近似未使用 action |

CRL progress Spearman 在 `NAV_TO_SOURCE` 为 `-0.018`，在 `PICK` 为 `-0.521`；局部目标
表征没有形成跨 route 稳定的物理进度关系。报告的综合 `quality_pass` 为 `false`。

### 2.3 删除废弃 local stall 后，闭环仍由模型自锁

seed 139 的 no-local-stall 闭环共进行 157 次模型 query、349 个 control step：

- 157/157 次 Pass 1 都输出 `NAV_TO_SOURCE`；
- 156 次正常以 `selected_waypoint_reached` 重询，没有 local `navigation_stall` fatal；
- 到可乐平面距离从 `1.316350 m` 最低降至 `0.713143 m`，最终为 `0.741201 m`；
- `PICK` 概率最大仅 `0.201772`，没有发生 logit crossover；
- 最终停止原因为 trusted prefix 没有非退化 waypoint，不是 stall 合同。

预测 `K` 的分布为：`K=1` 140/157、`K=2` 14/157，其余只出现 3 次。149/156 个导航
chunk 只执行一个 control step 就重询。后段 waypoint 越来越短，物距反而回升，形成
“短 waypoint→立刻重询→仍判旧 route”的闭环自锁。

作为独立对照，冻结的 v1 `step_002000@a8d57a2` 在相同 seed 139 上曾于第 19 次 query
由 Qwen 自主切到 `PICK`，切换时物距 `0.4931 m`、最低 `0.3804 m`。这证明场景、Pass 1
route 生成和执行接线本身具备切换能力；当前 v2 checkpoint 的训练分布/监督才是差异项。

### 2.4 计算图与数据语义的直接证据

实现审计得到两处可直接复现的不一致：

1. 完整 assistant CE 监督硬 route token；独立 `route_loss` 再监督一次同一硬 route；
   连续 crossover 仅以 `0.5 × lambda_boundary(0.2)` 的有效权重加入。
2. terminal-hold 正确保留 original `K*`，但 prefix candidates 只有 `1..20`；原始
   `K*=0` 的边界 row 被映射成训练 target `K=1`。这会要求模型信任一个数据已明确表示
   “无合法同 route target”的 waypoint。

训练 event 与该结论一致：step 2000 时 answer loss 已降至 `0.00179`，route loss
`0.0211`，但 crossover loss 仍为 `1.118`；prefix loss 仍为 `1.7149`。硬分类记忆已经
完成，连续切换与 prefix 排序却没有被学会。

## 3. 框架修正

修正只作用于 B2 及其后续训练计算图：

1. 对 active→active transition，完整 assistant CE 不再对当前硬 route token反向传播；
2. 独立 route loss 使用 `sigmoid(boundary_signed_time / 0.2 s)` 构造 old/new 连续分布；
3. 对 `PLACE→DONE`，同样屏蔽硬 ACTION/DONE token，并用连续 ACTION/DONE 分布监督；
4. phase interior 仍使用原有硬 route CE；B1 未启用 boundary/progress 时保持原行为；
5. original `K*=0` row 不参与 `K=1..20` prefix ranking，但 terminal-hold FM 监督仍保留；
6. runtime、request schema、route 解析、planner 和执行器均不改动。
7. 当某个 ZeRO-3 rank 没有 transition/time target 时，rank/time auxiliary head 仍以精确
   零损失留在 autograd graph，避免不同 rank 的 trainable parameter participation 不一致。

双卡 full-data 首轮进一步给出了该分布式缺陷的精确复现：step 3 的首个 batch pair 中，
rank 0 的 sampler batch 16 有 5 个 transition row，rank 1 的 batch 17 为 0 个；前两步
两边均有 transition。训练因此恰在 step 2 后进入持续 NCCL/NVLink collective，SM 100%
但不再产生 event。修正后的回归专门覆盖 transitionless rank 的所有 boundary 参数均获得
有限显式梯度（允许数值为零）。

新增回归覆盖 hard-token masking、连续 old/new 目标、B1 rollback 和零长度 prefix。针对
data/model/runtime/planner 的本地相关测试为 64 passed；完整远端环境仍需在提交后复核。

## 4. 训练决策

不从 8-episode `step_002000` 续训。若静态和远端门禁通过，则从官方标准
Qwen3-VL-4B 初始化，在完整 522-episode v2 immutable 数据上仅使用两张 H20 启动全新 B2
run：global batch 64、S1、每 500 effective optimizer step 保存、总长 2,000 step。
双卡使用 ZeRO-3 且不 offload optimizer，并以 micro 8/GPU、accumulation 4 保持 global
batch 64；两张 96 GiB H20 有足够显存，也减少 micro-batch 间重复参数通信。

这个训练把单一变量收敛到“正确数据覆盖 + 修正的 B2 监督”。prefix、CRL 和 on-policy
correction 均保持关闭，避免把已被证据否决或尚未批准启用的变量混入归因。

## 5. 正式双卡长训健康启动

最终 run 身份为：

```text
conveyorvla-waypoint-v2-b2-soft-transition-full522-s1-2gpu-nooffload-
gbs64-7ec8424-s2000-20260823T011748CST
```

| 项目 | 冻结值 |
|---|---|
| source | `7ec8424cc7d1e0e23b767eb1350986359d09c536`，clean |
| dataset manifest | `5361ed00f808d56537503cb2bfde25ee0ba8cbf9e7e85d7c6e1c35924c3ba56d` |
| policy config | `waypoint_v2_b2_s1.json@c43ecde7384372f382312d589ddcc2e3fad7c37fdb26c6677dffb75b377ff290` |
| world / batch | 2 GPUs；micro 8/GPU × accumulation 4 = global 64 |
| data coverage | 108,603 train rows；`training_subset=false` |
| initialization | 官方 `Qwen3-VL-4B-Instruct`，不 resume overfit checkpoint |
| components | terminal-hold + corrected B2 + S1；prefix/CRL/correction off |
| length / save | 2,000 effective optimizer step；每 500 step 保存 |

机器审计覆盖连续 step 4–23，结果 `ok=true`、problems 为空：

- 20/20 均为有效 optimizer step，run state 在 step 23 时仍为 `running`；
- step time min/median/max 为 `17.676/18.282/18.835 s`；
- throughput min/median/max 为 `3.398/3.501/3.621 samples/s`；
- peak reserved memory 为 `79,720 MiB`，占单卡容量 `81.45%`；
- total/Qwen/NAV/ARM/boundary/progress loss、五组 LR 和所有模块 gradient 均有限；
- 没有 OOM、NaN/Inf、traceback、NCCL error 或持续吞吐坍塌。

达到健康门槛后没有停止训练。该启动只证明训练过程和身份健康，不外推最终 checkpoint 的
开环、planner 或闭环能力；step 500 以后仍必须按正式门禁评测。
