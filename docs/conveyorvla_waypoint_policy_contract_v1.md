# ConveyorVLA Waypoint Policy v1：模型、训练与推理合同

- 状态：`APPROVED / 已审核通过`，实施尚未开始
- 日期：2026-08-20
- 批准记录：用户于 2026-08-20 明确确认 waypoint 版本设计方案审核通过
- 模型合同：`qwen3vl-layerwise-dual-fm-waypoint-v1`
- 数据合同：`conveyorvla-waypoint-dense-transition-v1`
- 在线协议：`conveyorvla-waypoint-runtime/v1`
- 兼容性：与现有 `[vx,wz] + TCP delta + state28` checkpoint、数据和在线协议不兼容

## 1. 决策摘要

ConveyorVLA 从“直接预测当前控制量”改为“预测下一组空间目标”：

- Navigation Flow-Matching head 预测机器狗机体系未来 waypoint；
- Manipulation Flow-Matching head 预测基座系未来 TCP pose 与夹爪目标；
- Qwen3-VL 不再接收任何 robot state；
- 导航 waypoint 由外部 `PCT global planner -> DWA -> [vx,vy,wz] -> locomotion policy` 执行；
- TCP waypoint 由外部 `workspace/safety gate -> cuRobo/IK -> joint controller` 执行；
- Qwen3-VL 自主输出四阶段 route，外部模块不得用 operation、真实 phase 或 FSM 覆盖模型决策；
- 四个 active route 共用 `<|pred_action|>` 前缀，并使用单 token route，避免自由文本 parser 导致动作样本失效；
- 两个动作专家都使用 StarVLA 风格的 Layerwise Flow-Matching action head，Qwen 最后 16 层与 16 个 DiT block 逐层对齐；
- 旧速度动作、state28 投影器和旧 optimizer/scheduler 全部退出新合同。

一句话定义：**模型决定“机器狗接下来去哪里、夹爪接下来去哪里”，外部规划器决定“如何安全到达”。**

## 2. 依据与适用边界

本合同依据以下可复现代码状态，而不是只依据项目名称或 README：

| 参考 | 版本 | 本提案采用的部分 |
|---|---|---|
| ConveyorVLA 旧合同干净基线 | `1923b39ea50333cb0b0a3a7e48486662aa7cda4d` | 双相机时序、四阶段语义、episode split、boundary mask 和旧长训合同 |
| StarVLA WallX route-subtask | `a91b444548214f65d2020d888f7ff55622e35477` | `<|pred_action|>` route prefix、受约束首 token 路由、预测 subtask 回填、Layerwise Qwen→DiT、Flow-Matching head |
| `arm-vla-grasp-sim` | [commit `388b681`](https://github.com/BoZhiStudying233/arm-vla-grasp-sim/tree/388b6818f4c605a707d13c519fbb58b1d07acd92) | body-frame waypoint、PCT `NavPlan`、DWA、`[vx,vy,wz]`、base-frame TCP target、外部安全规划 |

关键代码证据为 StarVLA 的 `QwenPI.py`、`LayerwiseFM_ActionHeader.py`、
`wallx_cotrain_datasets.py` 和 `train_starvla_cotrain_router.py`，以及参考仓库的
[`vla_sim_real_action_contract.md`](https://github.com/BoZhiStudying233/arm-vla-grasp-sim/blob/388b6818f4c605a707d13c519fbb58b1d07acd92/docs/vla_sim_real_action_contract.md)、
[`evaluation/adapters.py`](https://github.com/BoZhiStudying233/arm-vla-grasp-sim/blob/388b6818f4c605a707d13c519fbb58b1d07acd92/source/evaluation/adapters.py)、
[`pct_adapter.py`](https://github.com/BoZhiStudying233/arm-vla-grasp-sim/blob/388b6818f4c605a707d13c519fbb58b1d07acd92/source/navigation/pct_adapter.py) 和
[`executor.py`](https://github.com/BoZhiStudying233/arm-vla-grasp-sim/blob/388b6818f4c605a707d13c519fbb58b1d07acd92/source/navigation/executor.py)。

当前代码、数据和 checkpoint 的旧合同仍由
[seen 数据整改文档](seen_subtask_data_analysis_and_remediation.md) 和
[step 2500 问题文档](step_002500_closed_loop_failure_analysis_and_remediation.md) 描述。
本文件已成为后续实现的唯一目标合同；在形成经过门禁验证的实现提交前，不得把现有
旧代码、旧数据、旧 checkpoint 或旧长训称为 waypoint v1。

适用范围为 Liangzhu seen 的完整抓取、搬运和放置。本文不定义真机 CAN、驱动器急停或
校准实现，但规定它们必须位于模型之外，并服从相同的 waypoint/pose 接口。

## 3. 总体链路

```text
完整任务 + head/wrist 时序图像
                 |
                 v
        Qwen3-VL Pass 1
  constrained route + subtask generation
                 |
                 v
  <|pred_action|><|route_*|><|subtask|>...<|end_subtask|>
                 |
                 v
        Qwen3-VL Pass 2
  相同观测 + 模型自己生成的完整 assistant prefix
                 |
        last 16 hidden layers
                 |
          route dispatcher
        /                    \
       v                      v
Navigation Layerwise FM   Manipulation Layerwise FM
 [Hn, 3] body waypoint     [Hm, 7] TCP target
       |                      |
       v                      v
 body->world local goal    workspace/rate gate
       |                      |
       v                      v
 PCT global planner         cuRobo / IK
       |                      |
       v                      v
 DWA -> [vx,vy,wz]          joint trajectory
       |                      |
       +---------- robot -----+
                    |
              新图像后重规划
```

Dispatcher 只做 route token 到动作专家的静态映射。PCT、DWA、cuRobo、IK 和安全控制器
使用机器人状态，但这些状态不得进入 Qwen 或任一 Flow-Matching head。

## 4. 坐标系、单位和时间定义

### 4.1 坐标系

| 名称 | 定义 |
|---|---|
| `W` | Isaac/真机定位系统世界系，右手系，单位 m/rad |
| `B_t` | query 时刻 `t` 的机器狗基座系，`x` 前、`y` 左、`z` 上 |
| `TCP` | FinRay 夹爪工具中心，标定版本必须写入 manifest |
| `C_head/C_wrist` | 两相机光学系，外参版本必须写入 manifest |

所有角度使用 rad，平移使用 m。偏航统一通过 `wrap_to_pi` 落在 `[-pi, pi)`。

### 4.2 视觉时间

- 模型 query 观测保留 head/wrist 两相机；
- 每个相机输入两帧，顺序为 oldest→newest；
- 时间为 `[t-0.20 s, t]`，即现有 `[-5,0] / 0.20s` 合同；
- 不输入 annotation history、previous subtask、previous route 或模型上一轮文本；
- online 与 train 使用完全相同的图像排序和 prompt 模板。

### 4.3 动作时间

两个专家都使用 `H=20`，但目标间隔不同：

| 专家 | shape | waypoint 间隔 | 覆盖时间 | 每次默认执行 |
|---|---:|---:|---:|---|
| Navigation | `[20,3]` | `0.60 s` | `12.0 s` | 第一个非退化 waypoint，抵达后重推理 |
| Manipulation | `[20,7]` | `0.20 s` | `4.0 s` | 第一个可达 TCP target，抵达/超时后重推理 |

Navigation 的 `0.60 s` 取代旧动作的 `25 Hz` 速度时序，使第一个 waypoint 具有可规划的
空间距离，并与 StarVLA 的稀疏长时域 waypoint 设计接近。该值必须先通过数据分布和
PCT 短程规划 probe；若变更，必须升级 resolved config 和 normalization hash，不能静默修改。

## 5. 模型输入合同

一次 `PolicyObservation` 只包含：

```text
protocol_version
request_id / episode_id / sequence_id / timestamp
global_instruction
head_images[t-0.20s, t]
wrist_images[t-0.20s, t]
camera_calibration_id
```

下列内容禁止进入 tokenizer、Qwen embedding、DiT condition 或模型服务 request：

```text
state28
base pose / velocity
joint position / velocity
TCP pose
gripper position
真实 phase / operation
真实或预测的 previous subtask history
task FSM state
目标物体的 simulator ground truth pose
```

“模型不输入 state”不等于系统不读取 state。执行侧必须继续读取里程计、关节、TCP、夹爪、
碰撞和 watchdog 状态，用于坐标变换、PCT、DWA、IK、限幅和验收；这些字段属于
`ExecutorObservation`，与 `PolicyObservation` 是两个独立接口。

## 6. Prompt 与 route 语法

### 6.1 固定 token

新增并严格验证以下 single-token special token：

```text
<|pred_action|>
<|pred_done|>
<|route_nav_to_source|>
<|route_pick|>
<|route_nav_to_target|>
<|route_place|>
<|subtask|>
<|end_subtask|>
```

active action 的形式语法为：

```text
ACTION := <|pred_action|> ROUTE <|subtask|> TEXT <|end_subtask|>
ROUTE  := <|route_nav_to_source|>
        | <|route_pick|>
        | <|route_nav_to_target|>
        | <|route_place|>
TERMINAL := <|pred_done|>
```

TEXT 是简短的当前动作语言，不承担专家解析。即使 TEXT 措辞不完全匹配训练 canonical
文本，只要 route token 合法，样本和推理都能确定动作专家。

### 6.2 User prompt 模板

图像由 processor 插入对应位置；文字逻辑模板固定为：

```text
You control a Go2-X5 mobile manipulator using only the ordered camera images.

Task: {global_instruction}

The head-camera images and wrist-camera images are each ordered from oldest
to newest. Decide what the robot should do now from current visual evidence.

For an active action, output exactly:
<|pred_action|><one route token><|subtask|>one short action command<|end_subtask|>

Valid route tokens:
<|route_nav_to_source|> approach the source object;
<|route_pick|> move the gripper to grasp and lift the object;
<|route_nav_to_target|> carry the object toward the destination;
<|route_place|> move the gripper to place and release the object.

If the whole task is visibly complete, output exactly <|pred_done|>.
Otherwise select the best active route from current visual evidence. Do not
describe the scene or output any other text.
```

Prompt 不重复 task 三次，不注入当前 phase，也不使用“Previous model prediction”槽位。
StarVLA 中值得保留的是明确输出语法和 route token，而不是特定无人机 prompt 中的外部
GRASP/PLACE phase 提示。

### 6.3 Canonical 训练答案

```text
<|pred_action|><|route_nav_to_source|><|subtask|>Walk toward the box holding the Coke can.<|end_subtask|>
<|pred_action|><|route_pick|><|subtask|>Move the gripper to grasp, lift, and retract the Coke can.<|end_subtask|>
<|pred_action|><|route_nav_to_target|><|subtask|>Carry the Coke can toward the empty destination box.<|end_subtask|>
<|pred_action|><|route_place|><|subtask|>Move the gripper to place and release the Coke can.<|end_subtask|>
```

可在训练集中加入经过审核的同义改写，但 route token、动作域和目标语义不得变化。

### 6.4 受约束解码

Pass 1 不再自由生成后再用字符串 parser 猜测：

1. 首 token 只比较 `<|pred_action|>/<|pred_done|>` 的 Qwen LM logits；
2. 若为 action，强制写入 `<|pred_action|>`；
3. 下一 token 只比较四个 `<|route_*|>` logits；
4. 强制 `<|subtask|>`，生成最多 24 个文本 token，遇 `<|end_subtask|>` 停止；
5. 未在上限内结束或 route confidence 低于阈值时，由 runtime 返回 `RECOVER` 和零动作，
   不得 fallback 到默认 NAV；`RECOVER` 是执行状态，不是训练中的模型 route token；
6. confidence 来自受限候选上的 softmax，不由模型生成数字字符串。active route 的
   `route_confidence=P(pred_action)*P(route_token|pred_action)`，DONE 为 `P(pred_done)`。

`route_confidence_min` 是 resolved deployment config 的必填项，初始提案值为 `0.55`；正式值
必须在 validation split 上完成 reliability calibration 后冻结并写入 run manifest。

目标门禁是 route 格式 invalid rate 精确为 0。

## 7. Qwen3-VL 到 Flow-Matching head

### 7.1 两次 Qwen 推理

Pass 1 显式生成 route 和当前 subtask。Pass 2 使用相同图像、相同 user prompt，并追加
模型自己生成的完整 assistant prefix：

```text
<assistant>
<|pred_action|><|route_*|><|subtask|>{predicted_text}<|end_subtask|>
</assistant>
```

Pass 2 设置 `output_hidden_states=True`，不使用 Pass 1 generation cache 代替完整 forward。
这样 action head 的条件与模型实际 route 文本一致。

### 7.2 Layerwise 条件

- Qwen3-VL-4B 主干全量微调，不冻结；
- 取 Qwen 最后 16 个 Transformer layer 的完整 hidden-state sequence；
- 两个 action head 都有 16 个 DiT block；
- Qwen 第 `L-15+i` 层 hidden states 通过 cross-attention 送入第 `i` 个 DiT block；
- `cross_attention_dim=2560`，DiT hidden size 默认 `1024`；
- action loss 允许梯度回传 Qwen；
- 不创建 state token、state projection，也不向模型传零 state 占位；
- 两个 head 使用相同的 Layerwise Flow-Matching 实现，但参数不共享。

这取代现有“只给两个 DiT 最终 Qwen hidden layer + state28 projection”的设置。

### 7.3 专家映射

| route token | action head | 外部执行模式 |
|---|---|---|
| `<|route_nav_to_source|>` | Navigation FM | PCT/DWA；机械臂 stow，夹爪 open |
| `<|route_pick|>` | Manipulation FM | cuRobo/IK；底盘保持零速度 |
| `<|route_nav_to_target|>` | Navigation FM | PCT/DWA；机械臂 carry，夹爪 closed |
| `<|route_place|>` | Manipulation FM | cuRobo/IK；底盘保持零速度 |
| `<|pred_done|>` | none | 零动作并正常结束 |

stow/carry 是由模型 route 选择的安全执行模式，不是外部 FSM 对模型 phase 的替换。

## 8. 动作合同

### 8.1 Navigation waypoint

Navigation head 输出：

```text
A_nav[t,k] = [dx_Bt, dy_Bt, dyaw_Bt],  k=1..20
```

整个 horizon 都锚定 query 时刻的同一个 `B_t`，不是逐 waypoint 连乘的 sequential delta。
从数据中的未来世界位姿生成标签：

```text
d_world = [x_(t+k) - x_t, y_(t+k) - y_t]
[dx_Bt, dy_Bt] = Rz(yaw_t)^T d_world
dyaw_Bt = wrap_to_pi(yaw_(t+k) - yaw_t)
```

在线 body→world 变换必须与此完全互逆：

```text
x_goal = x_t + cos(yaw_t)*dx_Bt - sin(yaw_t)*dy_Bt
y_goal = y_t + sin(yaw_t)*dx_Bt + cos(yaw_t)*dy_Bt
yaw_goal = wrap_to_pi(yaw_t + dyaw_Bt)
```

约束：

- 每个 waypoint 有 `action_valid_mask[k]`；跨 phase、episode 尾部和无标签后缀为 false；
- 相邻有效 waypoint 平移不得超过 `0.80 m`；偏航不得超过 `45 deg`；
- 非有限值、全部零/退化 chunk 或越界 segment 直接进入 runtime `RECOVER`，不得复用旧动作；
- executor 选择第一个平移至少 `0.03 m` 或偏航至少 `3 deg` 的有效 waypoint；
- 不直接取第 20 个 waypoint，也不把 `[dx,dy,dyaw]` 当 `[vx,vy,wz]`。

### 8.2 Manipulation TCP target

Manipulation head 输出：

```text
A_arm[t,k] = [x_Bt, y_Bt, z_Bt, roll_Bt, pitch_Bt, yaw_Bt, gripper]
```

- pose 是未来 TCP 在 query 时刻基座系 `B_t` 下的绝对目标；
- orientation 为 `R_Bt^T R_TCP_future` 的 roll-pitch-yaw，并固定使用
  `R = Rz(yaw) * Ry(pitch) * Rx(roll)`；
- `gripper=0` 为 closed，`gripper=1` 为 open；
- 整个 horizon 同样锚定 `B_t`；操作阶段要求底盘由 executor 保持静止；
- 每个 target 有同一套 `action_valid_mask`，不得让 PICK suffix 跨入 NAV 或 PLACE；
- online 默认只规划并执行第一个通过 workspace/rate gate 的 target，然后重新观测。

初始安全限值沿用参考实现作为提案默认值：相邻 TCP target 平移不超过 `0.15 m`，单轴
旋转不超过 `35 deg`；workspace、碰撞、关节、速度和夹爪限值由仿真/真机 profile 定义。

### 8.3 归一化

- NAV 的 3 个连续维度使用每个 horizon step 独立的 `q01/q99`，统计 shape `[20,3]`；
- ARM pose 的 6 个连续维度使用每 step `q01/q99`，统计 shape `[20,6]`；
- gripper 使用固定映射 `g_norm = 2*g - 1`，不从常量分布计算 quantile；
- 连续值线性映射到 `[-1,1]`；训练和推理必须记录 clip rate；
- normalization 文件必须带 schema、split、单位、frame、shape 和 SHA-256；
- 任一维度 `q99-q01` 过小必须显式拒绝或使用审核过的固定 scale，不得静默除零。

## 9. 新数据合同

### 9.1 派生原则

- 原始 LeRobot/PCT episode 只读；
- 新建独立目录和 manifest，禁止覆盖旧 velocity/TCP-delta 派生集；
- split 继续按完整 `source_episode_id`，优先复用当前公平对照的 train/val/test episode 列表；
- phase boundary 前后至少保留 1 秒观测；跨 phase suffix 只通过 mask 屏蔽；
- 数据导出可以读取 base/TCP/joint 状态来计算标签和审计，但 loader 不得把它们返回为模型输入；
- state 和动作标签来源必须可追溯到 source row/timestamp/calibration。

### 9.2 每行最小字段

```text
schema_version
source_dataset_id / source_episode_id / source_row_id
split
timestamp
global_instruction
head_images[t-0.20s, t]
wrist_images[t-0.20s, t]
route_token
subtask_text
action_domain                 # NAVIGATION、MANIPULATION 或 NONE(DONE)
nav_waypoints_body[20,3]      # 仅 NAV 行有效
arm_targets_base[20,7]        # 仅 ARM 行有效
action_valid_mask[20]
waypoint_time_offsets_s[20]
label_frame_id
calibration_id
label_provenance
```

成功 episode 必须保留 PLACE 完成后至少 `1.00 s` 的稳定观测作为 `<|pred_done|>` 监督；
DONE 行的 `action_domain=NONE`、两个动作字段为空且 mask 全 false。没有经人工或协议验证的
失败语义数据时，不训练模型生成 RECOVER；低置信度、协议错误和 planner 拒绝统一由 runtime
fail-closed。

模型 loader 的 batch schema 不得出现 `state28`、`observation.state`、joint、TCP current
pose 或 simulator object pose。审计 sidecar 可以保存这些证据，但必须位于不被 collator 读取的
provenance namespace。

### 9.3 数据门禁

1. train/val/test 无 `source_episode_id` 泄漏；
2. 四个 active route、DONE 及四个相邻切换边界都有连续样本；
3. 双相机均严格满足 `[t-0.20s,t]`；
4. NAV 标签 world→body→world round-trip 误差 `<1e-5 m/rad`；
5. ARM 标签 world→`B_t`→world round-trip 误差 `<1e-5 m/rad`；
6. mask 后无跨动作域监督，padding 不参与 loss；
7. 模型 batch 中 state 字段和 state tensor 数量均为 0；
8. normalization clip rate：train `<1%`，val/test 单独报告；
9. 每个 split×四个 active route 及 DONE 抽取视频，并对 active route 叠加 GT waypoint/TCP target；
10. manifest 记录 episode/row/route/boundary 数量和 SHA-256。

## 10. 训练合同

### 10.1 初始化和可训练参数

- production baseline 从干净 Qwen3-VL-4B 与两个新 Layerwise FM head 开始；
- 两个 head 的输入/输出层按 `3D` 和 `7D` 分别初始化；
- Qwen 主干、embedding、LM head、两个 Flow head 全部训练；
- 旧 Conveyor `[vx,wz]`、TCP-delta、state28 checkpoint 不得 optimizer-resume；
- 旧 Qwen 权重或 StarVLA/ABot DiT block 只能作为明确记录的 weight-only 初始化消融，输入/输出层重建，不能作为默认 production baseline；
- checkpoint 中不得残留可被误用的 state projection。

### 10.2 主训练 forward：StarVLA 风格 oracle-prefix 联合训练

每个样本把 canonical assistant answer 作为 teacher-forced solution，一次 Qwen forward 同时
计算语言 loss 和输出 hidden states：

```text
images + user prompt + GT assistant route/subtask
  -> Qwen CE
  -> last 16 hidden layers
  -> 按 GT route 选择正确 Flow head
  -> masked Flow-Matching loss
```

动作专家永远由 GT single-token route 选择，不由自由文本 parser 或当前模型预测选择。
因此所有合法 action 样本都能训练对应 head，错误 route 不会把动作送入错误专家，也不会
造成主动作 loss 饥饿。

### 10.3 Self-conditioned 辅助训练

只做 oracle-prefix 会产生训练/推理差异，因此加入独立的 self-conditioned 辅助项：

1. 使用与 online 完全相同的 constrained decoder 生成模型 route/subtask；
2. 用该预测 prefix 做第二次完整 Qwen forward；
3. 仅当预测 route token 与 GT route token **完全相同**，计算 self-conditioned action loss；
4. 动作 loss 仍送入 GT 专家；route 不一致时只训练 route/language loss，不把错误语义条件送入任何 action head；
5. oracle-prefix 主动作 loss 始终保留，所以 self-conditioned mismatch 不会饿死任何专家；
6. 训练中不引入 previous-subtask history，因此不存在 history teacher forcing。

默认调度按总 optimizer progress 定义：前 5% `lambda_self=0`，5%–40% 线性升到 `0.5`，
后 60% 保持 `0.5`。任何修改必须进入 resolved config，不能通过临时环境变量改变。

### 10.4 Loss

```text
L = lambda_answer * L_answer_ce
  + lambda_route  * L_route_token_ce
  + lambda_nav    * L_nav_flow_masked
  + lambda_arm    * L_arm_flow_masked
  + lambda_self   * L_self_conditioned_flow_masked
```

- `L_answer_ce` 覆盖完整 assistant answer；
- `L_route_token_ce` 单独覆盖 action/done 首 token 和四个 active route token，按类别频率加权；
- Flow loss 只统计 `action_valid_mask=true`；
- NAV/ARM loss、有效样本数、有效 action 数、clip rate 和 gradient norm 分开记录；
- action loss 必须允许梯度回传 Qwen；
- 多卡上某 rank 缺少某专家样本时，用零权重 dummy graph touch 参数，不能伪造动作样本。

### 10.5 采样与优化

- episode split 固定后，train sampler 对四个 active route、DONE 和四类相邻 boundary window 做显式平衡；
- sampler 报告“抽样等价 epoch”，但不得声称每行被完整训练相同次数；
- global batch、LR、总 step 在实现阶段通过小样本 overfit 和 4-GPU smoke 决定；
- resolved config 必须记录每个 optimizer parameter group：Qwen core、embedding/LM head、NAV head、ARM head；
- production 长训前，必须在 `lambda_self>0` 时证明 Qwen、NAV head、ARM head 均有有限 loss 和非零梯度。

## 11. 在线推理合同

### 11.1 Request

模型服务 request 不得包含 `phase`、`operation`、`locked_route`、真实 target pose 或 robot
state。最小 request：

```json
{
  "protocol_version": "conveyorvla-waypoint-runtime/v1",
  "request_id": "...",
  "episode_id": "...",
  "sequence_id": 12,
  "instruction": "Pick up the Coke can from box1 and place it on box2.",
  "images": {
    "head": ["t-0.20", "t"],
    "wrist": ["t-0.20", "t"]
  },
  "camera_calibration_id": "..."
}
```

### 11.2 Response

仅返回被选专家的 action 字段：

```json
{
  "protocol_version": "conveyorvla-waypoint-runtime/v1",
  "request_id": "...",
  "route": "NAV_TO_SOURCE",
  "route_token": "<|route_nav_to_source|>",
  "action_domain": "NAVIGATION",
  "subtask": "Walk toward the box holding the Coke can.",
  "route_confidence": 0.88,
  "decision_probs": {"ACTION": 0.97, "DONE": 0.03},
  "route_probs": {"NAV_TO_SOURCE": 0.91, "PICK": 0.03, "NAV_TO_TARGET": 0.04, "PLACE": 0.02},
  "nav_waypoints_body": [[0.22, 0.01, 0.03]],
  "arm_targets_base": null,
  "action_valid_mask": [true],
  "checkpoint_id": "...",
  "normalization_sha256": "...",
  "timing": {}
}
```

协议 validator 要求 route、action domain、shape、mask、finite、frame、单位和 checkpoint
metadata 一致。任何异常返回零动作并终止当前 chunk，不允许沿用上次 waypoint。
wire response 可以删除 mask=false 的固定长度 suffix，但 checkpoint raw trace 必须保存完整
`[20,D]` 输出、完整 mask、反归一化前后数值和实际被选中的 waypoint index。

## 12. Navigation 执行合同

完整链路固定为：

```text
NAV Flow head
 -> denormalize [dx_Bt,dy_Bt,dyaw_Bt]
 -> waypoint safety validator
 -> 当前 odometry 将 body local goal 转为 world goal
 -> PCTNavPlanner(current world pose, predicted world goal)
 -> NavPlan/path_world
 -> DWA(current pose, measured body velocity, local map)
 -> [vx_body,vy_body,wz]
 -> locomotion policy
 -> robot
```

执行规则：

1. 每次只执行第一个非退化有效 waypoint，不跳到 horizon 第 20 点；
2. PCT 的目标是模型预测的 local goal，不是数据或 FSM 提供的 pick/place GT goal；
3. PCT 失败默认 fail-closed，不静默切换 A* 或直线速度；
4. PCT 对终点的 snap 距离必须记录且不超过 `0.10 m`，超过即拒绝该 chunk；
5. 纯旋转 waypoint（平移小于 `0.03 m`）绕过 PCT 路径采样，交给带限幅的 terminal-yaw controller；
6. DWA 在本地地图上逐控制周期输出 `[vx,vy,wz]`，其中 `wz` 即 `vyaw`/yaw rate，模型永远不输出速度；
7. 达到 `0.12 m / 0.14 rad`、chunk 超时、stall 或 route 切换后发零速度并重新 query；
8. DWA/PCT 可绕障，但不得改变语义 route 或替换模型 local goal；
9. NAV_TO_SOURCE 使用 stow/open，NAV_TO_TARGET 使用 carry/closed；
10. planner、DWA、locomotion 每层都记录输入目标、输出、限幅、失败原因和耗时。

参考仓库当前 `RemoteVLANavPlanner` 是“body waypoint 直接形成稀疏 NavPlan，再交 DWA”；
本合同在其前后接口之间明确插入 PCT：body waypoint 先形成 world local goal，再由 PCT
生成可碰撞检查的 path。实现时应新增组合 adapter，不能误以为参考代码已经自动调用 PCT。

## 13. Manipulation 执行合同

```text
ARM Flow head
 -> denormalize TCP targets in B_t
 -> finite/workspace/step/rotation/gripper validator
 -> current measured joints/TCP + scene collision
 -> cuRobo/IK plan to first valid target
 -> rate/torque/joint/workspace limits
 -> arm and gripper controller
 -> robot
```

- 模型不输出关节角、关节速度或 CAN 命令；
- cuRobo/IK 使用实时 state，但不得把 state 回传模型；
- PICK/PLACE 时底盘保持零速度；
- planner unreachable、collision、过期 target 或 watchdog 触发时 fail-closed；
- route 切换只能来自下一次 Qwen 推理。执行器可以拒绝危险 route，但不能自动改成另一阶段；
- gripper 动作必须与 TCP target 同步记录，避免只移动夹爪位姿却未闭合/释放。

## 14. 训练前和长训前门禁

### 14.1 模型与开环

1. constrained route format invalid=`0`；
2. empty semantic history 是唯一模式，train/val/online prompt token-by-token 一致；
3. state leakage test 证明 Qwen、NAV head、ARM head 均未接收 robot state；
4. 两个 Flow head 均有 finite loss 和 non-zero gradient；
5. NAV 报 ADE/FDE、偏航误差、首 waypoint 方向、segment 越界率和 PCT 可规划率；
6. ARM 报 TCP position/orientation error、gripper accuracy、workspace/step 越界率和 cuRobo 可达率；
7. route confusion matrix 分别报告四阶段与 DONE，不用一个“总体有效率”掩盖类别失败；
8. 多 diffusion seed 报均值、方差和最坏样本。

### 14.2 Planner integration

1. 已知 body waypoint 的 body→world 变换单测通过；
2. 预测 world local goal 能生成至少两个点的 PCT path；
3. DWA 全程输出有限 `[vx,vy,wz]` 且在限幅内；
4. 首 waypoint 完成后发生新模型 query，不继续盲走剩余 chunk；
5. 零、过大、NaN、过期 waypoint 均零速 fail-closed；
6. 已知 TCP target 能通过 validator、cuRobo/IK 和执行 smoke；
7. 仿真与真机 adapter 共享 frame/unit/schema tests，但各自保留独立安全门。

### 14.3 端到端

依次通过：loader smoke、32样本 overfit、单 GPU train smoke、4 GPU distributed smoke、
checkpoint save/load、oracle-route planner rollout、自主 route 分阶段闭环、完整无辅助 episode。
完整 episode 必须保存 head、wrist、第三视角视频，以及逐 query route、subtask、waypoint、
PCT path、DWA command、TCP target、planner 状态和实际轨迹。

## 15. Checkpoint 与运行证据

每个 checkpoint 必须绑定：

```text
model_contract_id
dataset_schema_version + manifest_sha256
processor/tokenizer + special token ids
Qwen commit/base checkpoint sha256
NAV/ARM head shapes and horizon/stride
normalization file + sha256
camera/calibration contract
resolved train config + sha256
source Git commit + dirty-state artifact
```

loader 必须拒绝以下组合：旧 state28 config、新无 state checkpoint；旧 `[vx,wz]` normalizer、
新 waypoint head；旧 TCP-delta 数据、新 absolute TCP head；horizon/stride 或 token ID 不一致。

## 16. 实施波次

| 波次 | 内容 | 验收 | 回滚点 |
|---|---|---|---|
| 0 | 冻结旧 run 证据，记录本合同批准状态和 schema 名称 | 批准记录可审计 | 旧代码/数据不变 |
| 1 | 新 waypoint/TCP 标签 exporter 与 manifest | 数据/坐标/mask/video 门禁 | 删除新派生目录即可，raw 不变 |
| 2 | special tokens、prompt、受约束 router、双 Layerwise FM head | unit/overfit/gradient | 独立 config/model ID |
| 3 | body→world→PCT→DWA adapter 与 TCP→cuRobo adapter | integration/safety smoke | 保留旧 evaluator，不覆盖 |
| 4 | 开环、planner、分阶段和完整闭环评测 | 视频、trace、成功定义 | 不启动长训 |
| 5 | 新 run 名称从零启动正式长训 | 健康 step、checkpoint、GPU/日志证据 | 停止新 run，旧 run 不受影响 |

每个波次单独提交语义修改；不得把旧 schema 就地迁移，不得让新 evaluator 兼容性 fallback
静默接受旧 checkpoint。

## 17. 已知风险

1. 无 state 的 ARM head 只能从图像推断当前几何，部分可观测性高于现有模型；必须用 wrist
   视角、相机标定和 planner rejection 率验证，不能假设移除 state 一定更好。
2. `0.60 s` NAV waypoint stride 与 PCT 短程目标容差需要先做数据统计和 planner probe；
   stride 太短会被 goal tolerance 吞掉，太长会增加 route/碰撞风险。
3. Layerwise 双 head 与 self-conditioned 辅助 forward 显著增加显存和训练时间。
4. PCT/DWA/curobo 使用 state，因此系统不是“无状态”；只是模型接口无 proprioception。
5. RPY 在接近奇异姿态时不连续。若实施前数据门禁发现 wrap/gimbal 问题，必须暂停实现、
   修订本合同并重新审核后才能改为 6D rotation 或 quaternion；不得在训练中途悄悄改变
   action dimension。
6. 外部 planner 提高运动安全性，但无法修复错误 route 或错误目标 waypoint。
7. 现有 raw 若缺少 PLACE 后 `1.00 s` 稳定帧或 NAV 的 `0.60 s` 长时域标签，不得用复制
   padding 伪造；必须补采或把缺失 suffix/DONE 标为无效并在数据审计中明确报告。

## 18. 已批准决议

用户已批准以下八项决议，实施不得静默改变：

1. 接受两个独立 Layerwise Flow-Matching head，而不是一个带 mask 的统一 10D head；
2. 接受 NAV=`[20,3] @ 0.60s`、ARM=`[20,7] @ 0.20s`；
3. 接受 NAV 全 horizon 锚定当前 `B_t`，ARM 为当前 `B_t` 下的 absolute TCP targets；
4. 接受每次只执行第一个非退化 waypoint/target，然后重新观测；
5. 接受模型 request 完全移除 state、phase、operation、history，但 executor 继续使用 state；
6. 接受 oracle-prefix 主 loss + self-conditioned 辅助 loss，而不是让预测错误直接饿死动作专家；
7. 接受 PCT 默认 fail-closed，不自动 A* fallback；
8. 接受旧 checkpoint 仅作历史基线，不 resume、不作为 production 默认初始化。

任何需要改变上述决议、动作维度、坐标系、horizon/stride、route 语法或 state 边界的实现，
必须先修订本合同并重新取得审核；不得以临时配置或兼容 fallback 绕过。
