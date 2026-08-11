# ConveyorVLA AL1：面向动态抓取的时序与流式控制方案

状态：研究依据与后续晋级候选，尚未替代 AL0。原方案冻结于 2026-08-06；
2026-08-11 已把 P0 数据合同和慢速夹爪执行融合为
`conveyorvla_al0_temporal_v2`，模型仍命名为 AL0。

当前正式执行方案已经收敛到
[CONVEYORVLA_AL0_EXECUTION_PLAN.md](CONVEYORVLA_AL0_EXECUTION_PLAN.md)。为了不把
未经闭环验证的结构提前命名成新一代模型，本文件中的机制先在 AL0 的 temporal
profile 中验证。`temporal_v1` 只保留为历史数据；当前 `temporal_v2` 只有取得
规定的动态成功率增益后才晋级为 AL1。

## 结论

当前模型已正式命名为 **ConveyorVLA AL0**。AL0 是一个可运行的兼容基线：
Qwen3-VL-4B 对当前时刻的头部/腕部图像和指令编码，DiT-B 根据单步机器人状态
预测 `16×10` 的 50 Hz 动作块。它没有显式物体运动观测，也没有把推理时延纳入
动作有效期，因此不能仅靠增加动态示教就宣称具有动态抓取能力。

建议把下一代模型命名为 **ConveyorVLA AL1**。AL1 不是重写整个模型，而是在
AL0 可继承的 Qwen/DiT 主干上同时补齐四个缺口：

1. 两帧视觉运动表征；
2. 可独立跳过的未来相对位姿动作；
3. 推理与执行重叠的时延感知 action streaming；
4. 由静止到低速、包含恢复状态的动态数据课程。

这四项必须联合验证。只有多帧而没有时延对齐，模型仍会执行过期动作；只有流式
推理而没有运动线索，模型仍无法判断目标速度；只有更多同分布成功示教，则无法
解决当前已经观察到的闭环分布偏移。

## 证据边界

本次分析基于 DynamicVLA 仓库的干净镜像，提交为
`a27a06d2ca74d0e987a5e552e01013073e93cfd8`。分析只读取源码，没有复制其实现，
AL1 也不在运行时依赖该仓库。2026-08-06 已重新连接 `lab-server` 并核对远端
`/data4/duanzhibo/xhq_workload/DynamicVLA`：分支 `master`、HEAD 为上述提交、
工作树干净；配置、模型、数据集和推理五个关键文件与本地镜像 SHA-256 完全一致。
远端保存的 1,800 episode 测评中，单次推理通常为 `0.33–0.42 s`，进一步验证
动作分块和过期前缀处理不是可选优化，而是闭环必要条件。

远端源码还暴露了四个不能原样复制的边界：stream identity 缺少 episode
generation、整块过期缺少安全分支、负切片合并存在边界脆弱性、size-one 输出
队列满时会丢弃新结果。AL0 temporal runtime 已把这些问题列为强制测试项。

DynamicVLA 的动态能力并非来自单个“时序层”，而是以下闭环组合：

其模型本体是约 0.4B 的紧凑 VLA：卷积 FastVLM/FastViT 视觉编码器、
SmolLM2-360M 语言主干和 flow-matching action expert 联合工作；官方配置冻结
vision/connector/text，训练 action 侧参数。紧凑模型和 KV cache 主要降低推理
时延，本身并不能替代时序观测。

| 机制 | 源码证据 | 实际作用 |
|---|---|---|
| 两时刻双相机输入 | `configs/dynamicvla.yaml`：`observation: [-2, 0]`、`N_OBS_STEPS: 2` | 在 25 Hz 数据上提供相隔 80 ms 的运动线索 |
| 时序视觉融合 | `modeling_dynamicvla.py`：把 `[B,T,C,H,W]` 变成 `[B,T*C,H,W]`，默认 `TEMPORAL_FUSION: attn` | 卷积视觉骨干联合编码两帧；不是 RNN，也不是显式光流 |
| 未来 delta-action chunk | `utils/datasets.py` 与 `configuration_dynamicvla.py`：相对当前 state 的 20 步动作 | 让每个未来目标带有观察时刻参照，支持预测性抓取 |
| 连续推理 | `modeling_dynamicvla.py::_inference_loop` | 独立进程持续推理，只保留最新观察，推理和执行重叠 |
| 过期前缀剔除 | `modeling_dynamicvla.py::_get_streaming_action` | 用 observation/action index 跳过已经错过的动作，再合并剩余 chunk |
| 动态训练分布 | Dynamic Objects Manipulation 数据与仿真配置 | 模型在训练时真正见过速度、方向、遮挡和接触时机变化 |

两个容易误读的点：

- `observation.environment_state` 中虽然记录了物体状态，但不在 DynamicVLA 的
  `REQUIRED_FEATURES` 中，因此物体真值速度不是策略输入；
- 仓库存在 3D RoPE 支持代码，但当前 DynamicVLA 路径使用的 position IDs 不能
  单独解释动态能力。主要时序来源仍是两帧视觉、未来动作和流式执行。

## AL0 的具体断点

| 维度 | ConveyorVLA AL0 | 对动态抓取的影响 |
|---|---|---|
| 视觉 | 每相机仅当前一帧 | 单帧无法区分“物体静止在此处”和“正经过此处” |
| 状态 | `state_sequence_length=1` | 只能使用当前本体速度，不能显式对齐观察历史 |
| 动作 | 16 个 50 Hz 增量命令，覆盖 320 ms | horizon 短；跳过增量动作会改变后续轨迹参照 |
| 因果偏移 | 训练标签从观察后 1 个 control step 开始，即 20 ms | 现有在线 P95 RTT 约 129 ms，首段标签到达时已过期 |
| 推理 | HTTP 同步请求后执行固定前缀 | 观察、推理和执行串行，动作没有有效 tick |
| 数据 | 主要是成功 oracle replay，接触窗口快且恢复样本少 | 容易学到时刻记忆，并在闭环轻微偏移后无恢复能力 |

当前辅助诊断已经证明 AL0 从合格 staging 位姿能够下探、闭夹和抬升；完全闭环
仍失败。这更符合“观察/时机/闭环分布不匹配”，而不是机械臂执行器完全不可用。

## AL1 冻结设计

### 1. 输入：两帧而不是堆更多无标签图片

每个 25 Hz model tick 使用：

- `head_rgb[t-2]`、`head_rgb[t]`；
- `wrist_rgb[t-2]`、`wrist_rgb[t]`；
- 当前 `state28[t]`；
- 明确的 `camera_id`、`model_tick`、`capture_time_s` 和 `Δt=80 ms`；
- 当前单目标抓取指令。

第一版直接使用 Qwen3-VL 的短视频输入能力，把每个相机的两帧作为一个有序 clip，
而不是把四张图当作无时序关系的独立图片。这样可以继续使用冻结的 Qwen 主干，
无需先引入第二套大型视觉模型。仓库已有 DynamicVLA 导出中的 `[-2,0]` history
和 25 Hz 相机时钟，AL1 exporter 应复用这条 canonical 数据路径。

必须增加一个反事实 motion probe：当前帧完全相同，只替换历史帧为“从左来”或
“从右来”，动作预测的横向拦截方向必须随之改变。若原生 video token 在冻结
Qwen 下不能通过该门禁，再增加零初始化的轻量 temporal residual adapter；在
门禁失败前不先引入额外视觉骨干。

### 2. 动作：25 Hz、20 步、可随机访问

AL1 输出 `20×10`，覆盖 0.8 s，与 model/camera 的 25 Hz 对齐；Isaac 的底层
控制仍保持 50 Hz。每个预测行的语义是：

- `base_vx/base_vy/base_wz`：该未来 tick 的速度命令；
- `tcp_xyz/tcp_rot`：相对观察时刻 TCP pose 的未来目标，而不是“再走一步”的
  增量；旋转用四元数复合后转 rotation vector，不能简单跨大角度相加；
- `gripper`：未来实测的连续开度 `0=close, 1=open`。

采集到的 50 Hz 示教在导出时组合成 25 Hz waypoint：速度取时间平均，TCP 由
真实 future pose 计算相对目标，gripper 保留完整的平滑开闭轨迹。运行时由
50 Hz Cartesian servo 插值到下一个 waypoint，不能把同一个位移增量重复执行
两次。

这一设计使 chunk 的第 `k` 行是独立的未来目标。推理迟到时可以安全丢弃前 `k`
行；AL0 的逐步增量 action 不具备这个性质。现有 DiT 的权重形状不依赖 horizon，
因此从 16 改到 20 可以继承 AL0 action trunk，不需要重置整个 action head。

### 3. 执行：观察 tick 决定动作有效期

同步的“请求一次、执行固定 N 步”改成单生产者/单消费者流：

```text
25 Hz camera/state ──> latest-only inference worker ──> tagged 20-step chunk
        │                                                │
        └──────────── 50 Hz simulator/controller <───────┘
                            skip stale prefix
```

每个请求携带 `observation_model_tick` 和 `observation_control_tick`；返回值携带相同
identity、`inference_started/finished` 和 `valid_from_control_tick`。控制器按真实
control tick 计算 stale prefix，绝不按固定 RTT 猜测：

```text
skip = current_control_tick - observation_control_tick
```

由于 AL1 action 是 25 Hz，而控制是 50 Hz，control tick 必须先映射到 action
index。已过期行直接丢弃，新旧 chunk 按目标 tick 合并；无新动作时继续执行旧
chunk 的尚未过期后缀。若剩余少于安全下限或 chunk identity 倒退，则保持当前
安全姿态并记录 fail-closed 原因。跨 chunk 只选择具有最新时钟身份的夹爪目标；
连续运动由底层夹爪轨迹生成器负责。

### 4. 训练：先保留语义，再学习时机

训练分三段，每段不通过门禁就停止：

1. **迁移烟测**：加载 AL0-M1；冻结 Qwen；仅训练新输入适配、state/action
   boundary 和 DiT。用“重复当前帧作为历史帧”检查 AL0 静态输出没有结构性退化。
2. **动态课程**：只做单物体抓取，不做分类和投放。速度分布先取
   `0 / 0.01 / 0.02 m/s`，方向固定为从左到右；再扩到 `0.03 / 0.06 m/s`。
3. **闭环修正**：把 policy rollout 中的 pregrasp、late-close、miss 和 drop
   状态交给 oracle 接管生成 recovery continuation；失败轨迹不能伪装成成功，
   但其观察可用于 DAgger/recovery 监督。

首轮 pilot 建议 300–500 条，不立刻大规模采集：约 25% 静止、25% 低速、25%
中低速、25% 扰动/恢复。采样按 phase 平衡，`descend/close/first_contact/hold`
窗口提高权重；gripper event 单独加权，避免长时间 open/hold 淹没闭爪边沿。

只有 motion probe 和低速闭环明显优于 AL0，才扩到多物体、速度随机化、移动底盘
和放置。否则先定位输入/延迟问题，而不是用更多相同轨迹掩盖结构缺陷。

## 实施顺序与改动边界

### P0：数据与时钟合同

- 使用独立 `conveyorvla_al0_temporal_v2` exporter，保留历史
  `m0_mobile_v1`/`temporal_v1` 数据不变；
- 复用 canonical `history_offsets_steps=(-2,0)`；
- 输出 capture/control/model tick、20 步 25 Hz future-pose action；
- 增加时钟、边界 padding、pose composition 和无观测泄漏测试。

### P1：离线 temporal policy

- 在 `Qwen3VLInterface` 增加两个短视频输入；
- 新建 AL1 policy/config，AL0 类和 checkpoint 继续可加载；
- 保持 Qwen 冻结，继承 DiT trunk；
- 完成 duplicate-frame regression 与 counterfactual motion probe。

### P2：流式 runtime

- 协议新增 observation/action tick，不改写 AL0 在线 schema；
- latest-only 推理 worker、stale-prefix skip、chunk merge 和安全 hold；
- 人工注入 1–8 个 control tick 延迟，验证任何过期 action 都不会执行。

### P3：小数据训练与闭环

- GPUs 2、3 作为本项目实验区，先跑 300–500 条 pilot；
- 固定 seed 集比较 AL0 与 AL1，不用 assisted 回合计 policy-only success；
- 通过后才开始更高速度和更大规模数据采集。

## 验收门槛

| 门槛 | 最低要求 |
|---|---|
| legacy regression | AL0 原测试全通过；旧 `m0_mobile_v1` 数据和旧 health payload 可读取 |
| temporal sensitivity | 相同当前帧、相反历史运动的预测拦截方向显著不同 |
| latency safety | 注入 1–8 control tick 延迟时，过期 action 执行数严格为 0 |
| 静止回归 | 50 个未见 seed，完整抓取成功率不低于 90% |
| `0.01 m/s` | 50 个未见 seed，抓取成功率不低于 85% |
| `0.02 m/s` | 50 个未见 seed，抓取成功率不低于 80% |
| 时序统计 | 记录 end-to-end P50/P95、平均 skip、有效后缀长度和 close timing error |
| 失败可解释性 | miss、late-close、IK、drop、timeout 分开计数，不合并为泛化的 failure |

达到上述前三档速度后，再定义移动底盘 + 动态抓取、carry 与 placement 的 AL1
完整任务；在此之前，单物体 pick 是唯一主任务。

## 明确不做

- 不把物体真值位置/速度作为策略输入；它们只用于监督、评价或辅助 loss；
- 不把 overview camera 输入策略；
- 不因更名重写旧 schema、profile、checkpoint key 或历史证据路径；
- 不先下载另一套大型 VLM；Qwen 原生两帧路径未被证伪前保持依赖最小；
- 不在低速闭环门禁前启动大规模轨迹采集。
