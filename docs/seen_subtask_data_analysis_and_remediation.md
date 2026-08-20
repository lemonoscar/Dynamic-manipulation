# Seen 子任务数据问题分析与修正方案

> 历史诊断：本文分析旧 step 7000 / state28 direct-action 路线。其问题推动了
> Waypoint v1，但第 4～8 节不是现行模型合同；冲突时以
> [Waypoint Policy v1](conveyorvla_waypoint_policy_contract_v1.md) 和
> [当前状态](status.md) 为准。

更新时间：2026-08-17。

## 1. 结论

`step_007000` 的主要问题不是 train/val/test 把同一条轨迹按帧切开，而是训练监督与
闭环任务不一致：两个导航阶段与相邻操作阶段连接不够致密，真实历史子任务泄露当前
答案，phase-pure 过滤移除了切换监督，时间元数据不统一，导航执行器又把机械臂空动作
解释成世界坐标保持。

因此不得在原 hierarchy view 上直接续训。应保留旧 checkpoint 作为诊断基线，重新
构建数据视图、修正在线动作合同，再启动新训练。

## 2. 审计范围与事实

本次审计对应的实际训练入口是梁渚 seen 任务，不是早期 384 条传送带矩阵：

- 原始来源：`liangzhu_0729_n200`、`liangzhu_0729_n250`；
- base LeRobot v3：373 条成功 episode、80,285 个 5 Hz query frame；
- hierarchy view：74,168 个 phase-pure row；
- 四阶段：`NAV_TO_SOURCE → PICK → NAV_TO_TARGET → PLACE`；
- 双动作专家：Navigation DiT 为 2 维 `[vx, wz]`，Manipulation DiT 为 7 维末端与夹爪；
- VLM：Qwen3-VL-4B-Instruct 全量微调，第一轮生成子任务语言，第二轮提取 hidden states
  并路由到对应 DiT；
- 动作 horizon：20×10，25 Hz，覆盖 0.8 秒；视觉 query 为 5 Hz。

数据按完整 `source_episode_id` 切分：

| split | episode | row | n200 / n250 episode |
|---|---:|---:|---:|
| train | 331 | 65,930 | 118 / 213 |
| val | 15 | 2,947 | 6 / 9 |
| test | 27 | 5,291 | 6 / 21 |

审计没有发现 source episode 跨 split、selected index 重复、annotation/manifest split
不一致或完全相同 state+action episode 跨 split。实际比例约为 88.9%/4.0%/7.1%，存在
有限样本偏差和 test 的轻度来源偏移，但不足以解释闭环完全失败。

四阶段原始 row 分布为：

| 阶段 | train | val | test | 合计 |
|---|---:|---:|---:|---:|
| `NAV_TO_SOURCE` | 9,371 | 434 | 734 | 10,539 |
| `PICK` | 8,577 | 388 | 705 | 9,670 |
| `NAV_TO_TARGET` | 30,384 | 1,325 | 2,416 | 34,125 |
| `PLACE` | 17,598 | 800 | 1,436 | 19,834 |

训练使用 phase-balanced `WeightedRandomSampler`，所以 raw 数量不平衡不是唯一原因。

## 3. 已确认的问题

### 3.1 两个导航阶段不够致密

当前 hierarchy view 把完整轨迹拆成独立 phase-pure 样本。`NAV_TO_SOURCE` 包含较多与
抓取准备关系弱的远距离移动，`NAV_TO_TARGET` 又明显更长；模型能学习一般移动，却难以
学习“导航终点就是下一操作的可执行起点”。

期望的数据边界应是：

- `NAV_TO_SOURCE` 结束于底盘停稳、源物体进入机械臂工作区，下一帧立即进入 `PICK`；
- `NAV_TO_TARGET` 开始于抓稳、抬升并回到携带位，结束于目标箱进入放置工作区，下一帧
  立即进入 `PLACE`。

“致密”不是复制更多相同导航帧，而是保持 episode 连续、视觉目标明确，并提高临近
操作边界的有效采样密度。

### 3.2 真实历史子任务泄漏答案

每个样本当前携带真实 `subtask_history`。例如 `PLACE` 样本已经被告知前三阶段完成，
VLM 无需从视觉判断当前任务。使用真实历史时，64 个平衡 held-out 样本的生成准确率是
100%；清空历史后，64 个样本全部生成 `NAV_TO_SOURCE`，准确率降到 25%。

这说明当前近乎为零的 subtask loss 主要不能证明视觉子任务识别已经学会。

### 3.3 切换监督被 phase-pure 过滤

为避免一个动作块跨 phase，当前 builder 删除了边界附近的样本。20-step horizon 覆盖
0.8 秒，再叠加视觉历史，恰好把“何时结束当前阶段、何时开始下一阶段”的关键监督排除。
模型只看到阶段内部的容易样本，没有充分看到状态发生改变的瞬间。

### 3.4 时间合同没有统一

梁渚数据的真实视觉历史是原始 model tick `[-5, 0]`，跨度 0.20 秒。base 与 hierarchy
manifest 均记录了这一事实；仓库通用 `configs/temporal.json` 仍写 `[-2, 0] / 0.08s`。
训练入口虽可从 manifest 覆盖配置，但文档、测试、默认参数与部分 runtime 仍同时存在
两种含义，容易再次生成错误元数据或错误视频提示。

### 3.5 导航阶段的机械臂语义错误

Navigation DiT 只预测 `[vx, wz]` 是合理的，但在线 executor 不能把缺失的机械臂动作
补成零 TCP delta，并把零解释为世界坐标保持。已有闭环中每条 episode 出现 172～249
次 IK failure，物体从未抬起。

导航阶段必须由 action composer 显式补全机械臂与夹爪参考：

- `NAV_TO_SOURCE`：机械臂收纳位，夹爪打开；
- `NAV_TO_TARGET`：机械臂安全携带位，夹爪保持闭合。

参考值应是 joint-space 或明确的 body-frame pose，不得再依赖“空置/零增量”的隐含语义。

### 3.6 次要域差异

373/373 条来源 episode 均通过成功、视觉和训练质量门禁，但均为
`stable_physics_success=true`、`pure_physics_success=false`，操作阶段使用 base/support
lock。这符合“操作时底盘静止”的任务设计，但仍会造成 expert 与无辅助 executor 之间的
动力学差异。它不是本轮第一优先级，但必须保留为后续消融项。

## 4. 目标模型合同

双 DiT 结构保持不变，子任务由 VLM 自主生成：

```text
完整指令 + head/wrist[t-0.2s, t] + state28
                     ↓
       Qwen3-VL 第一次完整 forward/generation
                     ↓
        显式子任务语言（模型自己的预测）
                     ↓
   原始输入 + 预测子任务文本，第二次 Qwen3-VL forward
                     ↓
             最后若干层 hidden states
                     ↓
        NAVIGATION DiT 或 MANIPULATION DiT
                     ↓
               连续 action chunk
```

Dispatcher 只把 VLM 输出映射到动作专家，不维护任务顺序表，也不替 VLM 修正判断。

## 5. 数据修正方案

### 5.1 新建派生视图，不覆盖原数据

保留 base LeRobot v3 和旧 hierarchy view 只读，新建带新 schema version 的 dense
transition view。split 继续以完整 `source_episode_id` 为单位，并复用同一 split seed，
以便与 `step_007000` 做公平对照。

每个 row 至少增加或保留：

```text
episode_id
timestamp
subtask_label
next_subtask_label
seconds_to_boundary
is_boundary_window
transition_reason
action_chunk
action_valid_mask
history_offsets_model_ticks = [-5, 0]
history_span_s = 0.2
```

### 5.2 致密导航采样

- 保留完整导航覆盖，防止模型只会最后一米；
- 降低远离目标、视觉变化小的巡航帧权重；
- 提高每个导航终点前 2～4 秒和相邻操作开始后 1 秒的采样权重；
- `NAV_TO_TARGET` 只有在抓取确认、抬升、收臂到携带位后才开始；
- 每条 episode 必须通过连续性审计：前一 phase 的末帧和后一 phase 的首帧时间连续，
  不得跨 episode 拼接。

### 5.3 去除真实历史捷径

- 主 prompt 不再包含真实 completed-phase 列表；
- 第一次 VLM 推理只依赖完整指令、当前时序视觉和 state28；
- 若保留任务记忆，推理时只能使用上一时刻模型自己的文本输出，并 `stop_gradient`；
- 训练早期允许少量 teacher forcing，但其概率必须按训练进度衰减到 0，同时进行 history
  dropout 和错误历史扰动；
- 正式评测必须分别报告空历史、模型自回归历史和错误历史结果。

最小实现可先完全移除历史输入；只有无历史模型在视觉相似状态上出现真实歧义时，再加入
模型自产生的短记忆。

### 5.4 边界样本与动作 mask

不得为了保证 action chunk 纯净而删除整个边界 row。每个切换点保留前后至少 1 秒：

| 切换 | 新阶段开始条件 |
|---|---|
| `NAV_TO_SOURCE → PICK` | 底盘停稳，物体进入抓取工作区 |
| `PICK → NAV_TO_TARGET` | 夹爪闭合，物体抬起，机械臂到达携带位 |
| `NAV_TO_TARGET → PLACE` | 底盘停稳，目标箱进入放置工作区 |
| `PLACE → DONE` | 夹爪打开，物体已释放到目标区域 |

VLM 的 subtask language loss 对边界 row 正常计算；DiT loss 只对当前专家仍有效的 action
前缀计算。跨 phase 的后缀通过 `action_valid_mask=false` 屏蔽，而不是删掉观察样本。

### 5.5 时间统一

梁渚 seen 任务统一为：

```text
previous image timestamp = t - 0.20 s
current image timestamp  = t
raw/model tick offsets   = [-5, 0]
query rate               = 5 Hz
action rate              = 25 Hz
```

manifest 是权威来源。dataset builder、processor video metadata、训练 config、policy prompt、
服务端缓存和测试必须逐项断言相同的 0.20 秒合同。兼容字段名 `tminus2` 可以暂时保留，
但不得再据此推断 0.08 秒。

## 6. 在线动作修正

动作组合必须显式完成：

```text
NAV_TO_SOURCE action
  = Navigation DiT [vx, wz]
  + stow joint reference
  + gripper open reference

NAV_TO_TARGET action
  = Navigation DiT [vx, wz]
  + carry joint reference
  + gripper closed reference
```

Manipulation DiT 激活时底盘速度严格为零。导航和操作之间切换时，应以当前测量关节状态生成
短暂、限速的姿态过渡，避免一步跳到 canonical pose。

## 7. 验收门禁

### 数据门禁

1. episode-level split 无泄漏；
2. 四个 phase 每个 split 都有样本，并单独统计 source 分布；
3. 每条 episode 的四阶段时间连续，边界窗口没有被过滤；
4. `action_valid_mask` 与专家域一致，mask 后无跨专家 loss；
5. 所有视觉元数据均为 `[-5, 0] / 0.20s`；
6. 随机抽取 train/val/test × 四阶段视频，人工确认动作与标签；
7. 导航样本中的机械臂参考全部可解释为 stow 或 carry，不允许空语义。

### 模型门禁

1. 正常历史、空历史、错误历史分别测量 balanced subtask accuracy；
2. 空历史不能再退化为全部 `NAV_TO_SOURCE`；
3. 单独报告边界窗口准确率和切换时延，目标时延不超过两个 query，即 0.4 秒；
4. 分别报告 Navigation DiT 和 Manipulation DiT 的 held-out action loss；
5. 闭环 trace 必须记录每轮生成文本、路由专家、动作 mask、机械臂参考模式和切换原因；
6. 先通过每个阶段的闭环消融，再运行完整无辅助 episode。

## 8. 实施顺序

1. 冻结并记录 `step_007000` 为旧监督基线；
2. 修改 hierarchy builder，生成 dense transition view 和边界 mask；
3. 移除真实历史，统一 0.20 秒时间合同；
4. 修改 action composer，加入 stow/carry 明确参考；
5. 运行数据审计、四阶段视频抽查和 loader smoke；
6. 运行小规模训练，先验证空历史与边界指标；
7. 指标通过后才启动正式全量训练；
8. 最后运行无辅助闭环，并将 VLM 路由错误和 DiT/执行错误分开统计。

旧数据和旧 checkpoint 不删除、不改写；它们用于确认新方案的增益是否来自数据与接口修正，
而不是评测条件变化。
