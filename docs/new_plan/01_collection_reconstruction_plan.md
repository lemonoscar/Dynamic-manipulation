# 文档一｜完整可乐搬运任务的数据采集重构与交付方案

**版本：** 1.0 · 2026-09-06  
**状态：** 设计与实施计划；不表示代码已修改、数据已重建或实验已完成。  
**适用仓库：** `lemonoscar/arm-vla-grasp-sim`，分支 `docs/liangzhu-50hz-collection-plan-20260906`，审读提交 `6e1b3856614862dd9eca88fbf8d2e3e6d7941d67`。  
**下游模型：** `Dynamic-manipulation@bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31` 的 ABot-M0 初始化、共享 Qwen、独立 NAV/Mani DiT。  
**配套文档：** [文档二：VLA 分阶段修正与验证](02_vla_staged_revision_validation.md)；[文档三：开发接口、理论与验收](03_developer_system_acceptance_theory.md)。

> **主决策：保留现有仿真、PCT、CuRobo 与执行教师；重构教师行为覆盖、控制记录来源和训练视图。原始 50 Hz 是事实记录层，不是模型必须以 50 Hz 推理的要求。完整 episode 是主数据单位，任务级和动作级样本是两个派生视图。**

## 1. 目标、范围与证据等级

### 1.1 要交付的能力

目标是 Go2-X5 在良渚场景中，按照原始指令完成一次连续搬运：导航到源物体附近、抓取并保持可乐、携物导航、放置到目标区域、确认释放和稳定放置。正常任务中不允许中途重置物体或机器人；受控扰动实验可以改变状态，但必须记录干预，不能将干预形成的跳变伪装成策略动作。[C01] [C06]

第一阶段不扩大为同时移动底盘并精细操作的 whole-body VLA，不引入多个物体类别，不强制增加放置后回撤。如果后续任务需要这些能力，建立新任务版本。现有采集不执行 place retreat，而旧模型文本包含 retract；新监督应改成“lower, release, and verify placement”，不能要求动作数据中不存在的行为。[C06] [C11]

### 1.2 四种证据标签

| 标签 | 含义 | 本文使用方式 |
|---|---|---|
| **CODE** | 在指定提交的代码或文档中可核实 | 记录器缓存逻辑、10D 导出、50 Hz 采样、任务配置 |
| **REPORT** | 用户最近汇报，完整新工件未独立读取 | train 的 27,871 / 2,022 / 11,819 chunk 统计、18 个分叉窗口、最新 NAV 结果 |
| **LITERATURE** | 原论文、官方文档或官方开源实现支持 | 层次化条件、动作前缀、任务化采集、几何表示 |
| **DESIGN** | 面向本任务的拟议修改和实验参数 | 新 schema、300 条数据对照、采集分层、门禁 |

REPORT 中：2,022/27,871≈7.25% 的 MANI chunk 含失效缓存；包含未知来源的隔离集合为 11,819/27,871≈42.41%。**42.41% 不是污染率；剩余集合也不是自动认证的干净数据。**本文件不声称已复核新报告文件或其全部轨迹。[U02]

### 1.3 保留项与新增项

保留可重复的场景随机化、现有完整教师、双 RGB、实际关节状态、50 Hz 原始控制记录和独立物理评价。新增：可靠生效命令记录、任务事件与事实、计划进度、恢复示范、可标定 RGB-D 候选、模型专用派生器、按任务家族划分的数据发布。[C01] [C02] [C03] [C04] [C05] [C06] [C07]

不在本轮同时引入 World Model、视觉大模型替换、语义点云长期地图和四个独立动作网络。它们不是实现本文数据目标的必要条件。

## 2. 现有采集与模型之间的关键缺口

| 缺口 | 当前事实 | 对学生的影响 | 重构决策 |
|---|---|---|---|
| 动作来源 | `_update_full_action()` 用显式目标、缓存或实测值补全 | 完整向量不等于真正生效的控制；reset 边界可能混入旧目标 | 在控制应用边界记录请求、解析后目标和实际响应 |
| 导出动作 | 默认 10D 使用下一帧实际底盘/TCP 位姿和实测夹爪 | 与 joint 命令学生不兼容，可能重新混淆闭合意图与物体宽度 | 新增专用 7D Mani / 3D NAV 派生视图 |
| 教师计划 | 教师维护 segment、tick index、闭合和收敛上下文 | 学生逐次重规划可能没有等价的推进信息 | 保存计划与事件，构造任务级、动作级条件样本 |
| 覆盖 | 成功配额、单场景几何随机化 | 倾向保留教师易完成区域，恢复状态不足 | 正常、边界扰动、恢复三个数据层 |
| 时钟 | 原始 50 Hz；旧模型 0.2 s 历史与目标间隔 | 相邻行读取会把 2 s 时域变成 0.2 s | 以真实秒和控制 tick 定义派生 |
| 评价 | 接触标签可来自运动学/动作启发式 | 不应直接作为严格抓持或完成真值 | 分开命令事件、几何代理、接触和完成条件 |
| 输入 | 教师用 Mesh/PhysX 真值定位，学生只看有限 RGB | 需要明确可观测性与真值边界 | 真值用于监督和分析；部署输入只用合法传感信息 |

上述是代码与模型合同的比较，不代表每个新 50 Hz episode 都存在所有问题。[C02] [C04] [C05] [C06] [C08] [C10] [C11]

## 3. 重构后的教师：以任务完成为中心，但保留可解释的执行结构

### 3.1 名义任务骨架

| 高层任务 | 学生可获得的条件 | 教师/评价可使用的证据 | 完成后仍需维护的事实 |
|---|---|---|---|
| `NAV_TO_SOURCE` | 目标身份、图像、合法定位/地图与本体状态 | 实际到达位姿、稳定性、下游可操作性诊断 | 源位置身份和当前站位 |
| `PICK` | 当前目标、视觉/可选深度、q/dq、夹爪状态、执行历史 | 接触、抬升、相对保持、是否仍受支撑 | `carrying(cola)`，可被后续掉落推翻 |
| `NAV_TO_TARGET` | 目的地身份、当前 carrying 估计、场景与本体状态 | 运输中的物体跟随和到达 | carrying 必须持续更新 |
| `PLACE` | 目标区域、当前持物状态、局部几何 | 释放、物体进入区域并稳定 1 s | `placed(cola,target)`，完成时仍需成立 |
| 高层 `VERIFY/FINISH` | 最新可用观察和执行反馈 | 独立任务评价，不输入策略 | 停止继续操作并保持安全状态 |

`VERIFY/FINISH` 是拟议高层协议，不表示旧动作模型已经支持 DONE，也不要求再建一个连续动作 expert。当前任务本身较简单，初始四步计划可固定；高层学习的重点是继续、推进、取消失效事实和局部修复。[C11] [U01] [R03] [R09]

### 3.2 不把教师内部状态直接当成部署事实

必须区分：

- `controller_phase`：源状态机内部执行位置，可用于溯源。
- `active_subtask`：当前希望完成的任务，可作为策略条件。
- `event_history`：此前发生过的事件，不因后来失败而删除。
- `current_facts`：现在是否持物、目标是否还在原位，允许变为 false 或 unknown。
- `last_model_output`：上一个查询的模型输出，不等于最后完成的阶段。

禁止使用含义混合的 `previous_prediction` 同时表示“上次模型回答”和“上一个已完成任务”。任务级训练样本与在线状态必须使用同一字段定义；可用于标注的未来信息不能泄漏进历史条件。[U01] [ENG]

### 3.3 教师可以知道更多，但不能悄悄改变学生的物理世界

使用真值目标规划专家动作是允许的。机械底座锁定、固定抓取约束、对象重新放置、碰撞代理和外部干预属于另一层，应逐项记录。

将 episode 标为明确的物理配置，例如 `base_locked_no_grasp_constraint` 或 `source_assisted_v1`。具体名字是拟议命名，不得根据类名自动判断实际干预。仅启用“无抓取固定约束”不等于底座也未锁定。读取运行时生效报告后再分类。[C06] [C14] [C15]

## 4. 完整 episode 的采集分层与数量规划

### 4.1 三种训练数据，不混淆失败与纠正

**N：正常完整搬运。**在既有安全几何随机化范围内完成完整流程，保持源教师和当前学生所需的主要控制模式。

**B：边界扰动后继续完成。**在接近、导航交接、释放等阶段施加受控且可恢复的变化，专家从实际状态继续完成。扰动参数、时刻和外部作用必须写入 `interventions`；受干预的物理转移不得当成纯动作导致的转移。

**R：执行偏离后恢复完成。**来自学生访问状态或预先定义的空抓、局部未对准等状态，由专家生成修正。学生的失败动作保留为诊断，不能无条件进入动作 BC 损失；专家有效纠正区间具有独立 provenance mask。[R06] [R07]

只改变初始状态的样例，与中途外力/目标移动的样例必须分层。不能把“完整成功 episode”要求用来删除所有失败附近的正确动作；也不能为了增加恢复数量，向同一标准状态反复复位后声称采到了连续恢复。

### 4.2 建议初始预算：同总量、不同覆盖的 300 条对照

| 数据集 | 正常 N | 边界 B | 恢复 R | 总数 |
|---|---:|---:|---:|---:|
| `D-normal-300` | 300 | 0 | 0 | 300 |
| `D-task-300` | 180 | 60 | 60 | 300 |

这是 **DESIGN**，不是理论最优比例或已采结果。两组可共享 180 条正常示范，因此最多需要 420 条不同训练 episode；额外正常样例与扰动/恢复样例按任务家族匹配。两组优化步数、有效 batch 和事件采样权重一致，另报专家查询量、轨迹总时长和采集成本。

在此之前先完成 12 条合同 pilot：6 条冷启动、6 条进程复用，覆盖正常/边界/恢复及规划等待、闭合切换、终止尾部。12 条是有效诊断目标，不承诺尝试 12 次全部成功；失败尝试必须保留。

开发集建议另取 50 个任务家族；架构初筛从中预先选择 20 个家族，每个含正常和一种扰动条件。最终测试在方法冻结后新建 100 个任务家族。所有同源恢复片段、分叉和渲染变化都必须跟随父家族划分，不能跨 train/validation/test。

### 4.3 采集分布应怎样报告

同时报告 `attempted`、`physically_completed`、`command_verified`、`quality_accepted`、`training_eligible`，不要只报成功配额。

用下式解释成功筛选后的分布，而不假设 accepted 仍均匀：

$$p(x\mid S)=\frac{p(x)P(S\mid x)}{P(S)}.$$

至少按机器人起点、两桌位置、源目标位置/朝向、交接误差、遮挡程度和失败阶段统计接受率。只增加采样 FPS 不增加这些独立因素的覆盖。[C01] [C06] [ENG]

## 5. 原始数据合同：观察、有效命令、执行结果三者分离

### 5.1 每个控制 tick 的因果记录

规范的概念顺序是：

```text
冻结 pre-action 观察 o_t
→ 生成请求 command_requested_t
→ 控制器解析、保持、限幅并报告 command_effective_t
→ 推进一个控制周期
→ 记录 post-action 响应 o_(t+1)
```

`command_effective` 指实际进入当前控制器/执行器的目标，不是电机力矩的同义词，也不要求隐式执行器提供不存在的显式力矩。可读取的目标、实际限幅和控制模式要与 PhysX/actuator 实际层次对应。[C02] [C03] [R16]

### 5.2 必须增加的最小记录字段

| 字段组 | 必需内容 |
|---|---|
| 身份 | `episode_uuid`、`task_family_id`、`reset_generation`、源码/配置/资产哈希 |
| 时钟 | `control_tick`、真实仿真时间、墙钟单调时间、相机采集时间、command apply 序号 |
| 观察 | 两相机 RGB、q/dq、实测夹爪；合法本体量与相机标定 |
| 请求 | 允许为空的显式 arm/gripper/base 请求；来源控制模块 |
| 生效目标 | 当前 arm/gripper/base 完整目标、目标ID、继承来源、控制模式、裁剪原因 |
| 实际响应 | post-step q/dq、夹爪、底盘状态；真值物体/接触进入评估区 |
| 计划 | teacher plan ID、segment、实际已执行索引、切换/等待原因 |
| 干预 | 外力、目标移动、辅助约束、base/support lock 的实际启停事件 |

必须能将每一个 action label 追溯到某个生效命令区间。ID 连续、shape 正确、最终成功，都不能替代这个检查。

### 5.3 reset、保持和重复仿真时刻

reset 发生时旧缓存失效。控制器若确实安装了新保持目标，应记录它来自本次 reset 的目标同步；不能让记录器把旧 pre-reset observation 当成新目标。

同一物理 tick 中可能有多个不推进物理的状态切换。它们进入事件流；只有最终被下一次物理推进消费的目标进入控制流。使用 `(reset_generation, control_tick, apply_sequence)` 区分，禁止凭 `frames.jsonl` 行号与 `samples.jsonl` 行号直接 zip。

合法保持必须引用当前 reset 周期内仍然生效的 `parent_command_id`。若无法恢复目标，标记 `unknown`，不使用 measured q、上一阶段标签或未来轨迹补齐。[C02] [C03] [ENG]

### 5.4 建议目录（全部为拟议新格式）

```text
episode_<uuid>/
  manifest.json
  task.json
  observations.jsonl
  control_effective_50hz.jsonl
  task_events.jsonl
  teacher_plan_trace.jsonl
  interventions.jsonl
  evaluator_truth.jsonl
  images/front/ ...
  images/wrist/ ...
  depth/ ...                    # 候选模态；可缺省但有 valid mask
  summary.json
```

不要求立即删除或重命名旧文件。可在原始目录旁生成这一规范视图，并保留原文件及哈希，实现审计和回滚。

## 6. RGB-D 与状态：先留存可用信息，再决定模型采用哪些

现有前/腕双 RGB 是主模态。建议新批次保留标定矩阵、相机外参、实际拍摄姿态、深度单位、depth definition（z-depth 或 ray range）及有效像素 mask。Depth 是候选输入，不能宣称当前仓库已经记录了合格 RGB-D。[C01] [C06] [R10] [R11]

若全量 50 Hz depth 成本过高，可在明确时间网格上采集较低频深度，但 RGB-D 实验只能使用有同步深度或经过验证的对齐样本；禁止默认复用上一帧而不记录年龄，禁止使用未来深度填补当前观察。

记录相机 FOV、裁剪平面和几何来源。良渚视觉层与碰撞几何可能不是同一表示，必须验证 RGB 中物体与 depth 几何的投影一致性。若深度来自真值碰撞几何，应明确其是传感器模拟还是诊断上界，不能把完美世界几何伪装成真实相机深度。

训练视图可以只读取 RGB，但底层保留 q/dq、实际命令历史与标定，有利于后续几何耦合和计划条件实验。单目预测深度属于 RGB 派生先验，与新增独立测量分组报告。

## 7. 两种训练视图的设计

### 7.1 任务级视图：学习局部更新，而非每次从头规划

输入是原始指令、当前合法观察、截至当前可获得的执行历史、当前计划与任务事实；标签是 `CONTINUE / ADVANCE / REPAIR_SUFFIX / FINISH / UNRESOLVED` 及结构化编辑。

首次主实验使用同一名义四步初始计划，隔离 rolling repair 的作用。初始计划自由生成可以作为后续独立实验，不与修复能力混在一起。[U01] [R03] [R09]

历史事件与当前事实分离。例如先前 `PICK_COMPLETED` 仍存在，但掉落后 `carrying=false`；修复后缀增加一个新 attempt 的 PICK，不重写历史。`now_subtask` 指当前意图，不代表任务已经完成。

教师使用真值生成标签时，输入仍只能包含部署可获得的观测和估计。对无法从给定输入确定的状态，标记不确定/待观察，不制造确定性目标。合成的 stale-plan 和错误历史必须标为训练增强，不能计作新物理示范。

### 7.2 动作级视图：以当前任务和执行上下文生成未来控制

当前 Mani 保留 7D 命令：

$$A_{t,k}^M=[q^{cmd}_{eff}(t+\tau_k)-q_t^{measured},\ g^{cmd}_{eff}(t+\tau_k)].$$

13D 本体状态由关节名称映射生成 `q6+dq6+measured_gripper1`，不把默认导出 state 的底盘/TCP 等所有量直接拼进旧模型。新增本体量应建立新 observation schema。[C08] [C10] [C11]

NAV 的旧视图可继续生成十个 query-body reference；目标 head 候选另行派生其末点及参考时域。它是局部期望状态，不是底盘速度命令。不得用世界坐标目标直接替代 body-frame label。

对 RTC 训练保存执行到新 query 时的旧剩余动作、有效长度、已承诺前缀 mask、起始锚点和模拟延迟。旧 delta 必须先转成绝对目标，再换到新 query 锚点。PICK→NAV 或目标语义改变后，旧 Mani prefix 不能跨任务继续使用。[R01] [R02]

### 7.3 不确定来源与边界的损失处理

第一版采用保守方案：真实未来区间包含失效或未知控制点时，隔离整个动作 chunk；该观察可否用于任务级监督另行判断。后续若引入逐点 mask，必须同步处理加噪、self-attention、loss 和归一化，不能只把 loss 乘零却仍让错误动作进入网络条件。

真实的终端保持与缺失数据不是同一类。成功尾部有真实保持证据时可派生 hold；未知控制、失败截断和 reset 边界不能伪造同一种 padding。

## 8. 统一时钟：50 Hz 原始流不改变旧模型的物理含义

| 项目 | 旧基线保留值 | 50 Hz 原始对应 | 候选改动 |
|---|---|---|---|
| 视觉历史 | 0.2 s | 间隔 10 tick | 长历史是独立消融 |
| Mani 目标间隔 | 0.2 s | 每 10 tick 一个目标 | 0.04 s、H=50 的候选 |
| 预测时域 | 2 s | 最远约 100 tick | 比较时保持物理时域 |
| 低层控制 | 0.02 s | 每 tick | 与动作目标频率独立 |
| 重规划周期 | 旧完整执行为 2 s | 不由 FPS 决定 | 0.4 s 候选，与 RTC 配对 |

必须显式存储 `first_target_offset_s`：旧模型未来一拍标签与新同刻命令视图不能同名。原始记录可派生 `legacy_future_5hz` 和 `causal_command_5hz` 两个有独立身份的视图；前者用于冻结模型兼容，后者需要独立训练和回放验证。此处不是认定旧一拍偏移导致失败。

`frame_index/fps` 仅为展示/容器时轴；原始事实以 `simulation_timestamp` 和控制 tick 为准。正式 50 Hz 完整性逐 tick 检查，不能仅沿用允许约 2% 缺帧的覆盖门。任务分段的 `min_segment_frames` 与 `hysteresis_frames` 应换成秒级定义，避免 FPS 提升十倍后防抖时间缩短十倍。[C02] [C07] [R15]

## 9. 数据切分、采样与归一化

先按 `task_family_id` 切分，再生成扰动、恢复和子片段。同一个祖先示范的所有分叉留在同一 split。`000006/000024/000030/000109` 作为已知机制回归集，不是未见能力测试。

训练采样至少平衡 NAV/Mani、完整任务阶段、关键夹爪事件、正常/恢复。原配置已有分域与夹爪切换采样，不能把它当作全新方法；新数据改变后重新验证其实际分布。[C08]

Normalizer 只由允许的 train 目标拟合。比较 D-normal/D-task 时，先冻结一个由共同可信训练池拟合的 normalizer，隔离覆盖因素；各自重拟合可作为补充系统比较。旧 checkpoint 永远使用其绑定的旧 normalizer，禁止静默替换。

训练预算报告优化步数、unique episodes、有效动作点、关键事件次数和墙钟成本。增加原始帧数不等于独立样本变多，不能自动按“旧两 epoch × 新十倍帧数”认为训练预算相同。

## 10. 代码修改清单与跨仓库交付

以下“新增”路径为拟议模块，不表示现有仓库已具备。

| 现有位置 | 修改内容 | 新增候选模块/测试 |
|---|---|---|
| `source/pipeline/full_physics_pipeline.py` | 分离 reset/event 与物理控制 tick，绑定 pre/apply/post | `source/recording/control_transaction.py`（新增） |
| `source/recording/lerobot_dataset.py` | 不再自行猜测有效目标；记录 provenance 与真实时间 | reset 缓存失效、同 tick 多 apply、队列阻塞测试 |
| 仿真实际 action 应用边界 | 输出 resolved effective target 与模式 | `source/recording/effective_command.py`（新增） |
| `source/manipulation/arm_executor.py` | 暴露 plan ID、cursor、等待/闭合切换原因 | 教师进度记录测试 |
| `source/recording/subtask_segmentation.py` | 秒级防抖；启发式 contact 不冒充物理真值 | 5/25/50 Hz 时长一致性测试 |
| `source/recording/training_action.py` | 保留旧10D导出，禁止被新学生误读 | 新 joint 视图独立 schema 测试 |
| `scripts/pipeline/run_full_physics_success_quota.py` | 恢复时比较完整运行合同；报告全部尝试 | 参数/资产/源码变更拒绝测试 |
| 下游 `joint_trajectory_data.py` | 消费统一原始规范，构造 task/action 视图 | 编解码黄金样本与时间偏移回归 |

不要在运行中的 worker 热改代码后继续累计同一 release。已有数据封存；新 schema、新物理条件和新源码建立新批次。本文不授权停止服务器上的其他进程，也不提供未经实现的可执行采集 CLI。

## 11. 采集侧实验与验收门

### D-01：控制记录可信度

执行 12 条 pilot，覆盖冷启动/复用及不同任务区间。所有进入“可信动作视图”的点必须有当前 reset 周期内有效来源；任一失效缓存进入有效监督即失败。未知可以保留在 raw，但不能作为已确认动作。使用 reset 前后不同目标的专门测试，不能只用静止场景。

### D-02：原始到部署的回放一致性

先 10 个可信源实例，比较 R0 原生50Hz生效命令、R1模型专用低频目标、R2归一化往返加部署限制。匹配初态、夹爪/底座辅助与时间合同。抓取通过后再扩完整 episode。报告差异首次发生位置，不只给最终成功布尔值。

原始50Hz控制不可得的历史episode只能做“采样目标回放”，不得声称复现完整源教师。与Sim5.1比较时另立迁移因素，不同物理版本的差异不全部归到采样。

### D-03：任务级标签因果性

对正常、掉落、空抓、目标改变、保持等待、完成六种情形，检查高层输入仅含截至 query 时可用信息；事件保留而事实可撤销；无未来真值泄漏；last_model_output 不替代 last_completed_task。离线构造可发现 schema 错误，不能替代物理恢复验证。

### D-04：覆盖与恢复

比较 D-normal-300 / D-task-300 时，以完整任务成功、恢复成功、错误推进为主结果。物理失败保留，不计为专家动作；预算、任务家族与源示范数量可追溯。正式训练实验见文档二 V-03。

### D-05：模态、接触覆盖与同步

检查 RGB-D 投影、z-depth/range含义、标定、相机历史、数据缺失和深度年龄。完美深度、退化深度、缺失深度分组。未知/缺失不填零当作有效距离；零值需配valid mask。

接触作为独立评价模态，先验证finger-object和object-all两侧覆盖，再校准稳定夹持、空抓、仍受桌面支撑、单侧托举、撞起和滑落等物理情形。缺少覆盖时保持unknown；几何代理仍可用于过程诊断。详细接触校准规模与判定见文档三第16.3节，不因RGB-D通过而默认接触覆盖通过。

### D-06：数据发布

发布 manifest、数据字典、来源分类、隔离清单、任务家族 split、训练池 normalizer、回放摘要、失败分布、源码/资产/参数哈希。对原始控制完整性要求确定性检查；对成功率不预设“100%可完成”的虚构结果。

数据来源门与策略能力门分开：历史 saturation≤0.5% 仍按冻结口径报告；研究诊断可在安全限制生效且明确 gate fail 的条件下进行，但不能据此批准部署。安全越界未被拦截、来源身份错配或伪造保持是硬阻断。

## 12. 推荐实施顺序

| 批次 | 工作 | 退出条件 |
|---|---|---|
| C0 | 冻结旧数据和运行身份；确认 task/action schema | 不再混用10D位姿与joint命令 |
| C1 | 生效命令记录、reset处理、真实时钟 | D-01通过；raw未知项可明确隔离 |
| C2 | 双训练视图、任务事件、模型专用派生 | D-02/D-03具有可解释证据 |
| C3 | 正常/边界/恢复pilot，RGB-D留存 | 采集因素和反馈边界一致 |
| C4 | 发布D-normal/D-task，开展架构实验 | D-04/D-05/D-06报告齐全 |

C1与高层/低层架构原型可并行。不要等每个历史失败都解释完才研究架构，也不要让新的架构吸收来源错误。

## 13. 理论依据与采用边界

本文的数据重构借鉴任务结构化生成、on-policy状态覆盖和层次化条件，但不复制其他论文的整套系统。MimicGen支持按任务/对象关系扩展示范；DAgger支持为学生访问状态补专家纠正；π0.5支持高层语义条件化；RTC相关工作支持记录和利用动作前缀。[R01] [R02] [R03] [R06] [R07]

50Hz日志、reset_generation、哈希、隔离策略和上述数量配比是本项目工程/实验设计，不是这些论文提出的通用定理。所有收益均需通过文档二实验验证。


## 来源与引用索引

来源核查日期：2026-09-06。用户仓库链接锁定上述提交；外部代码如链接到main/master，开工时仍需锁定SHA。本文不分发代码权重、场景、原始数据或外部图片。

### 项目代码与已发布实验

**C01｜采集方案与运行手册**  
[来源](https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/docs/collection_50hz/README.md)。审读代码基线；50Hz定义、成功配额、数据和物理环境说明。

**C02｜采集记录器：DwaEpisodeWriter**  
[来源](https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/recording/lerobot_dataset.py)。record、_update_full_action、report：缓存、同步、真实时间与允许缺样口径。

**C03｜采集主循环**  
[来源](https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/pipeline/full_physics_pipeline.py)。pre-observation、state machine、apply、step与StepRecord的顺序。

**C04｜旧10D训练动作导出**  
[来源](https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/recording/training_action.py)。build_vla_training_actions：下一采样实际位姿与实测夹爪，不是本模型joint命令。

**C05｜分段机械臂执行教师**  
[来源](https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/manipulation/arm_executor.py)。计划、执行索引、gripper context、等待和切换反馈。

**C06｜良渚可乐搬运任务**  
[来源](https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json)。任务流程、真值定位、辅助条件、相机门、无place retreat。

**C07｜子任务分段**  
[来源](https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/recording/subtask_segmentation.py)。基于帧的防抖与heuristic contact来源。

**C08｜当前VLA生效配置**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/configs/manipulation_navi_v1.json)。ABot初始化、Qwen、双DiT、5Hz目标、损失、关闭项与saturation门。

**C10｜当前数据派生与normalizer**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory_data.py)。旧sampled5Hz数据合同、future目标、route映射、训练输入边界。

**C11｜当前模型、动作与语言合同**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory.py)。H=10、0.2s间隔/历史、四route、canonical subtask、无DONE。

**C14｜已发布执行接口v2实验文档**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/docs/execution_interfaces_v2_20260906.md)。reset缓存反例、源/部署回放、条件PICK及接触未知；不含用户后续新实验的完整工件。

**C15｜独立物理事件评价器**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/physical_events.py)。relative geometry与三态contact，独立于辅助控制。

### 理论、论文与官方文档

**R01｜Black, Galliker, Levine. Real-Time Execution of Action Chunking Flow Policies (2025)**  
[来源](https://arxiv.org/abs/2506.07339)。采用动作重叠inpainting/异步接续思想；不提供任务级rolling plan或本任务安全保证。官方代码见K01。

**R02｜Black et al. Training-Time Action Conditioning for Efficient Real-Time Chunking (2025)**  
[来源](https://arxiv.org/abs/2512.05964)。采用训练期干净动作前缀与模拟延迟思路；当前M0适配仍需实现。官方代码见K01。

**R03｜Physical Intelligence et al. π0.5: a Vision-Language-Action Model with Open-World Generalization (2025)**  
[来源](https://arxiv.org/abs/2504.16054)。高层subtask与动作条件的相关先例。openpi见K02；其公开组件范围不等于论文完整高层系统。

**R06｜Mandlekar et al. MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations (2023)**  
[来源](https://proceedings.mlr.press/v229/mandlekar23a.html)。任务/对象结构化示范扩展的相关依据。 [官方代码](https://github.com/NVlabs/mimicgen)。

**R07｜Ross, Gordon, Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (2011)**  
[来源](https://proceedings.mlr.press/v15/ross11a.html)。DAgger：面向学习策略访问状态的监督；理论依赖专家/在线学习条件，不是任意恢复数据的收益保证。工程上由本项目教师查询实现，不声称移植原作者机器人代码。

**R09｜Huang et al. ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation (2024)**  
[来源](https://arxiv.org/abs/2409.01652)。阶段、反馈与回退的结构参考；本方案不采用其整套约束优化。 [官方代码](https://github.com/huangwl18/ReKep)，关注`main.py`。

**R10｜Ke, Gkanatsios, Fragkiadaki. 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations (2024)**  
[来源](https://arxiv.org/abs/2402.10885)。RGB-D空间特征与动作生成依据；本文不复用其性能数字。 [官方代码](https://github.com/nickgkan/3d_diffuser_actor)。

**R11｜Ze et al. 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations (2024)**  
[来源](https://arxiv.org/abs/2403.03954)。紧凑3D表示与diffusion policy的相关依据。 [官方代码](https://github.com/YanjieZe/3D-Diffusion-Policy)。

**R15｜Hugging Face. LeRobotDataset v3 官方文档**  
[来源](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)。数据、时间窗口与schema工程参考；不是当前自定义raw合同已经实现的证明。

**R16｜NVIDIA / Isaac Lab. Sensors API 官方文档**  
[来源](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html)。接触传感与过滤接口参考；版本/后端不同，必须现场验证覆盖，不从零值推断无接触。

### 论文相关的关键代码入口

**K01｜RTC官方可核对实现：real-time-chunking-kinetix**  
[来源](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py)。重点：`get_prefix_weights`、`realtime_action`、`loss`。main可变，实现前锁commit；这里未运行该项目。

**K02｜openpi动作模型实现**  
[来源](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)。重点：`embed_prefix`、`embed_suffix`、`compute_loss`、`sample_actions`；[仓库说明](https://github.com/Physical-Intelligence/openpi)。main可变，不表示完整高层planner已开源。

### 用户方案、最新汇报与本项目设计

<a id="source-u01"></a>
**U01｜用户上传《从RTC而来的High level planning.md》**  
本次对话原文，SHA256：`458146cff372a37a19e0caa872c19a410640bdfe6ead43d9e5f3e7f6d5e0fae3`。本文保留其原始目标、已完成任务、当前任务、剩余计划与基于观察/执行结果局部修改后缀的结构。本文对事件/事实、版本、动作条件和实验协议的细化属于拟议扩展；未复用原文的外部图片链接。

<a id="source-u02"></a>
**U02｜用户本轮对话中的最新实验汇报**  
训练集27,871个MANI chunk中2,022个含失效缓存，包含未知来源后建议隔离11,819个；源标签裁剪0.7303%；两套A/B/C共18个局部窗口；000006闭合推迟1.2s；NAV 15/12cm冲突及262个零命令tick。完整新报告/模型/数据/视频未独立读取，以上作为用户提供的报告证据，不声称新工件已经公开或独立复现。

<a id="source-eng"></a>
**ENG｜本项目工程推导与待验证设计**  
包括schema、版本隔离、回滚、样本数量/比例、研究晋级规则、控制合同、评估分母和安全监控组织。它们由任务需求、代码事实及明示的数学关系推出；不是论文原始贡献或已有的成功/安全定理。


[C01]: https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/docs/collection_50hz/README.md
[C02]: https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/recording/lerobot_dataset.py
[C03]: https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/pipeline/full_physics_pipeline.py
[C04]: https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/recording/training_action.py
[C05]: https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/manipulation/arm_executor.py
[C06]: https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json
[C07]: https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/recording/subtask_segmentation.py
[C08]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/configs/manipulation_navi_v1.json
[C10]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory_data.py
[C11]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory.py
[C14]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/docs/execution_interfaces_v2_20260906.md
[C15]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/physical_events.py
[R01]: https://arxiv.org/abs/2506.07339
[R02]: https://arxiv.org/abs/2512.05964
[R03]: https://arxiv.org/abs/2504.16054
[R06]: https://proceedings.mlr.press/v229/mandlekar23a.html
[R07]: https://proceedings.mlr.press/v15/ross11a.html
[R09]: https://arxiv.org/abs/2409.01652
[R10]: https://arxiv.org/abs/2402.10885
[R11]: https://arxiv.org/abs/2403.03954
[R15]: https://huggingface.co/docs/lerobot/lerobot-dataset-v3
[R16]: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html
[K01]: https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py
[K02]: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
[U01]: #source-u01
[U02]: #source-u02
[ENG]: #source-eng

