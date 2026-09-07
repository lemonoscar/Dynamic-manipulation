# 文档三｜开发者系统规范、验收计划与理论来源

**版本：** 1.0 · 2026-09-06  
**状态：** 目标系统设计规范与研究验收草案，不是已实现API或已完成验收报告。  
**基线：** 采集仓库`6e1b3856614862dd9eca88fbf8d2e3e6d7941d67`；VLA仓库`bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31`。  
**上位方案：** [文档一：数据采集](01_collection_reconstruction_plan.md)；[文档二：分阶段架构实验](02_vla_staged_revision_validation.md)。

> 本文把“预计修正后的体系”定义成可以实现、测试和回滚的接口。算法部分有论文依据；版本、数据流、安全隔离与验收阈值中大量内容是本项目工程设计，不能伪称已有论文证明。宏观rolling plan来自用户方案，是对层次化决策的项目化扩展，不是标准RTC在离散任务上的现成定理。

## 1. 读者、术语与设计级别

采集开发者重点看第4、5、13节；模型开发者重点看第6—9、12节；推理/控制开发者重点看第10、11节；实验负责人重点看第14—17节。

“必须”表示候选系统的硬合同；“建议”表示初版选择；“候选”表示仍须消融，不默认最终采用。所有新类名、JSON字段、目录和伪代码均为**拟议接口**。不应直接把本文代码块当成已有仓库命令执行。

### 1.1 基础证据

CODE事实来自指定提交。[C01] [C02] [C03] [C04] [C05] [C06] [C07] [C08] [C09] [C10] [C11] [C12] [C13] [C14] [C15] [C16] 用户最新train审计、18个分叉窗口和源NAV实测属于REPORT；相应新工件未独立读取。[U02] 用户上传的高层方案明确要求保留原目标、已完成任务、当前任务和剩余计划，必要时局部更新后缀。[U01]

不宣称本设计已改善成功率。旧严格接触评分未知，旧saturation门未通过，这两个限制不因写出新架构而消失。

### 1.2 目标系统的最小组成

| 模块 | 必须/候选 | 职责 |
|---|---|---|
| 原始记录与统一codec | 必须 | 可信观察、命令、时间和身份 |
| TaskMemory与rolling planner | 必须比较后采用 | 任务持续性、事实更新和局部修复 |
| NAV局部目标expert | 候选，旧十点保留基线 | 生成下游真正消费的局部导航目标 |
| Mani clean-action expert | 保留 | joint/gripper序列及动作上下文 |
| 动作RTC | 必须比较后采用 | 承诺前缀与重叠条件，避免无条件独立替换 |
| 实时视觉/本体条件 | 必须 | 当前几何与实际执行状态 |
| RGB-D几何分支 | 候选 | 提供米制局部空间信息 |
| 独立执行安全与评价 | 必须 | 监督安全和客观任务结果，不反向泄漏真值 |

## 2. 总体数据流与权限边界

```text
                       原始用户任务
                            │
                            ▼
  合法传感器 ──► ObservationCodec ──► CurrentObservation
                       │                    │
                       │        ┌───────────┴─────────────┐
                       │        ▼                         ▼
                       │  FeedbackEstimator        当前视觉/几何编码
                       │        │                         │
                       │        ▼                         │
                       │    TaskMemory ◄── Planner        │
                       │        │          Edit           │
                       │        ▼                         │
                       └────► TaskCondition ──────────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
                 NAV expert           Mani expert/RTC
                     │                     │
                     ▼                     ▼
                 NavGoal          AbsoluteActionQueue
                     └──────────┬──────────┘
                                ▼
                   ExecutionSafety + 控制器
                                │
                                ▼
                         机器人实际执行
                                │
                       新观察/命令反馈

  仿真真值 ──► 独立Evaluator ──► 实验记录
  仿真真值 ──► 专家标签生成器 ──► 离线监督
  辅助控制器（若启用）单独权限与配置，不由Evaluator布尔值驱动
```

普通策略只能读取部署可获得的传感器、自己实际发出的命令、合法地图/定位和用户任务。仿真object pose、真实成功、隐藏phase、完整外部接触真值只用于离线监督、独立评价或明确oracle组。

“合法地图”必须注明来源：预先提供的公开地图与在线LiDAR允许单列场景；直接读取全场景真值地图的组不能伪称纯视觉无地图。所有比较使用相同权限。

## 3. 核心设计不变量

1. **历史不是事实。** `PICK_COMPLETED`事件存在不意味着当前仍持物；current fact可以失效。
2. **预测不是反馈。** 模型回答“完成”不直接生成物理完成事件。
3. **向量完整不是命令可信。** 每个有效监督点必须有当前reset周期的命令来源。
4. **新计划不是自动接管。** 只有任务身份、时间、锚点和执行条件一致的候选可以进入队列。
5. **安全中止不是任务成功。** 无候选、超时和越界必须带原因，不能变成到达/完成。
6. **旧计划不是永远正确。** 已承诺动作受安全优先级覆盖，未承诺重叠可修正。
7. **50Hz不是VLM频率。** 采样、目标点率、执行tick与重规划分别定义。
8. **严格unknown不能转成true。** 几何代理、完整transfer和严格接触各有独立结果。
9. **工程修复与算法收益分开。** 架构实验以共同B0-runtime或B0-learned比较。
10. **原数据和旧成绩不覆盖。** 新协议、新权重、新物理条件都建立新身份。

上述是本项目工程约束[ENG]，不是RTC或π0.5给出的安全性定理。

## 4. 数据合同与跨仓库所有权

### 4.1 数据分层

| 层 | 所有者 | 记录内容 | 可否直接训练 |
|---|---|---|---|
| raw | 采集仓库 | 全部请求、有效命令、观察、事件、干预、真值 | 否，先审计 |
| verified view | 数据派生器 | 可追溯合法时间片与来源mask | 可以按任务筛选 |
| task view | planner训练 | 截至query的计划/事件/事实与编辑标签 | 可训练高层 |
| action view | action训练 | 当前条件、未来控制、前缀、有效mask | 可训练低层 |
| evaluator view | 评价器 | 物理成功与安全证据 | 禁止作为普通策略输入 |

### 4.2 建议RawControlRecord

下面是字段类型示意，不是某条真实轨迹：

```yaml
schema: raw-control-v2                 # 拟议新schema
identity:
  episode_uuid: string
  task_family_id: string
  reset_generation: integer
  source_commit: sha40
  resolved_config_sha256: sha256
clock:
  control_tick: integer
  sim_time_s: float
  wall_monotonic_ns: integer
  apply_sequence: integer
  physics_advanced: boolean
observation_ref: string
requested:
  base_twist: nullable_vector3
  arm_target: nullable_vector6
  gripper_target: nullable_vector2
resolved:
  command_id: string
  arm_target: nullable_vector6
  gripper_target: nullable_vector2
  base_twist: nullable_vector3
  source_class: issued | held_valid | stale | unknown
  parent_command_id: nullable_string
  actuator_mode: string
  limiting_report: object
post_observation_ref: nullable_string
teacher_plan_ref: nullable_string
intervention_refs: list_of_string
```

某通道可以unknown而另一通道valid，来源mask按通道保存。第一版动作训练可保守隔离整chunk，不等于原始记录需要丢弃。

### 4.3 来源状态转换

`issued`必须来自本tick有效下发；`held_valid`必须引用同reset周期内有效目标；reset后所有旧目标ID作废；无证据时为unknown，不能用measured q兜底。发生reset但不推进物理时，写事件并重新冻结观察，不把旧pre-reset观察当新初态。

同一tick多次apply由`apply_sequence`排序。只有最终被该物理步实际消费的目标进入生效控制视图；中间命令保留审计，不重复成训练时间步。[C02] [C03]

### 4.4 文件与数据发布身份

一份release至少绑定`schema_version / action_contract / time_profile / observation_contract / split_manifest / normalizer / source+patch / assets / assistance_profile / evaluator_profile`。相同目录不能因修复代码而继续写出不同合同数据。

配额恢复必须检查完整身份，不只比较seed和目标数量。已有采集进程不在本文中被启动、终止或重配；部署新版本时由负责人员明确切换批次。

## 5. ObservationCodec与ActionCodec

### 5.1 CurrentObservation

包含当前与历史RGB/可选depth、每相机实际时间、q/dq/gripper、合法base twist与定位、相机标定、有效mask和观察序号。`observation_id`只标识一次不可变快照，不能更新内容但保持ID不变。

当前模型13D本体状态仍按关节名取值。扩展高层/实时几何本体输入需要新schema和训练适配，不能把更多数值偷偷拼到旧向量。

### 5.2 模型动作的时间profile

| profile | 第一目标含义 | 用途 |
|---|---|---|
| `legacy_future_5hz` | 未来0.2s控制目标；旧执行解释保持不变 | 原checkpoint兼容诊断 |
| `causal_command_5hz` | 从当前控制区间开始使用的目标 | 新候选，需独立训练和回放 |
| `causal_command_25hz` | 每0.04s的目标，H=50 | 高分辨率候选 |

不论profile，系统必须显式存储预测目标时间、预计开始执行时间和观察时间。不能用一个`timestamp`掩盖所有含义。研究比较若改变profile，它就是实验因素。[C11] [C04]

### 5.3 Mani编解码

$$a^M_k=[\Delta q_k,g_k],\qquad q^{target}_k=q_{anchor}+\Delta q_k.$$

这里delta相对同一query锚点，不是逐点增量积分。Normalizer作用于规定通道；先反归一化，再恢复绝对关节目标，然后执行统一限制。有限且在范围内不等于碰撞安全。

旧动作换新锚点：

$$q^{old,target}_{j+m}=q_{old}+\Delta q^{old}_{j+m},$$
$$\widetilde{\Delta q}_j=q^{old,target}_{j+m}-q_{new}.$$

真正已下发的承诺前缀以effective target为准；未下发软重叠应使用经过当前执行限制的候选，并保留与原预测的差别。若限制改变了动作，不能再假设与原分布完全一致。

### 5.4 NAV编解码

对query-body目标：

$$p^W_{goal}=p^W_B(t_o)+R_{WB}(t_o)\,\Delta p^{B(t_o)}.$$

必须使用产生预测时刻的query pose，而不是推理返回时的最新pose。当前pose用于路径起点和在线重规划，两者职责不同。

若NAV继续输出reference，明确PCT到底消费末点还是整条路径。局部目标head不经过连续动作RTC的同一条件化接口，除非另做研究；本方案的动作RTC主要作用于Mani。

## 6. TaskMemory、任务事实与编辑协议

### 6.1 TaskPlan字段

```yaml
schema: task-plan-v2
mission_id: string
instruction_version: integer
original_instruction: string
plan_version: integer
active_task_epoch: integer
active_task_id: string
tasks:
  - task_id: string
    attempt_id: string
    primitive: NAV_TO_SOURCE | PICK | NAV_TO_TARGET | PLACE | VERIFY
    target_ref: string
    destination_ref: nullable_string
    preconditions: list_of_predicate
    completion_conditions: list_of_predicate
    status: pending | active | completed | superseded
completed_event_refs: list_of_string
current_facts:
  carrying_cola:
    value: true | false | unknown
    source: sensor_estimator | model_estimator
    evidence_refs: list_of_string
    observed_at_s: float
    validity_until_s: nullable_float
```

事实confidence可以记录，但未经校准的概率不作为完备真值。判据阈值与有效期在开发集预注册；来源缺失则unknown。过去完成事件不会被重写，当前事实过期后也不会因为历史成功而自动恢复true。

### 6.2 PlannerEdit字段

```yaml
parent_plan_version: integer
observed_active_task_epoch: integer
observation_id: string
operation: CONTINUE | ADVANCE | REPAIR_SUFFIX | FINISH | UNRESOLVED
current_task_candidate: nullable_task  # H0响应式候选；H1/H2可为空
new_suffix: nullable_list_of_task
evidence_refs: list_of_string
changes_active_task_semantics: boolean
reason_code: string
```

这里`reason_code`是简短可审计分类，不要求输出长链式推理。引用不存在、来自未来或其他episode的证据应拒绝。schema合法不代表语义正确，仍需实验评价。

H0仅预测`current_task_candidate`；适配器负责附加并发版本和执行引用，不能把服务器保存的任务历史反馈给H0模型。H1/H2输出上述计划编辑。对H0，`operation`可由适配器按“保持当前候选/提交不同候选”确定，但不允许适配器用评价真值补选正确route。共同封装不等于三组拥有相同的历史输入。

### 6.3 两个版本号的必要性

`plan_version`在任何任务编辑时递增，用于并发写入校验。`active_task_epoch`仅在当前任务/目标/关键约束发生语义变化时递增。

编辑未来PLACE任务而当前PICK不变，不应丢弃有效PICK动作。当前可乐丢失或目标物体改变时，必须递增epoch并取消旧动作。使用单一全局version会导致不必要停顿；完全没有version会允许过时动作接管。

### 6.4 提交与失效处理

采用compare-and-swap式提交：planner响应的parent version匹配当前内存才可提交，否则重新评价。不能在生成过程中直接修改共享列表。

`ADVANCE`需要部署可获得的证据，`FINISH`需要当前任务事实支持；独立evaluator不参与普通路径。持续事实失效时，可将当前任务标为superseded并加入retry，不“删除过去曾经成功”。

为了检验宏观方案，H1固定计划与H2可修复计划使用相同反馈估计器。若采用规则推进H1，应公开规则，并确保没有oracle真值特权。[U01] [R03] [R09] [ENG]

## 7. 条件耦合与模型边界

### 7.1 当前模型接口不被误描述

现有M0最后层条件是token序列，DiT已有cross/self交替。不存在“没有self-attention所以失败”的新结论；32个future tokens也不是图像压缩至32token。[C08] [C09]

### 7.2 候选耦合

$$C_t=\mathrm{concat}(Z^{task}_n,Z^{RGB}_t,Z^{geom}_t,E(s_t),E(A^{overlap}_t)).$$

为不同模态保留type、time、frame和valid mask。若将条件直接拼接，必须防止无效depth零值被当合法距离；若分bank cross-attention，也要固定访问预算，避免仅因额外token增加而误判结构收益。[R04] [R05] [R10] [R11]

高层计划token可以跨多个动作更新缓存。当前视觉必须更新；prefix KV cache只能对同一不可变观察复用，不是对未来时刻自动有效。[R03] [R13]

### 7.3 两服务研究原型与目标共享网络

研究阶段冻结低层原checkpoint，单独适配高层planner，六组共享同一高层模型。最终目标是共享Qwen/视觉模块，但迁移会改变条件分布，需要新的训练身份和回归。

不得在给出的系统框图中只画一个VLM，却在性能实验中运行两个大模型并隐藏成本。两服务资源成本、调度和延迟必须随报告交付。[ENG]

## 8. 理论：FM、clean-action与RTC适配

### 8.1 时间约定

本节定义τ=0为噪声、τ=1为数据。物理时间用t，不能与τ混用。外部openpi代码可能使用相反扩散时间约定；迁移必须显式转换，不照抄积分方向。[R01] [R12] [C09]

采用直线条件路径：

$$x_\tau=(1-\tau)\epsilon+\tau A^*,\qquad \epsilon\sim\mathcal N(0,I),$$
$$v^*=A^*-\epsilon.$$

如果网络预测clean action $f_\theta(x_\tau,c,\tau)$，在τ<1且没有额外clamp时：

$$v_\theta=\frac{f_\theta-x_\tau}{1-\tau},$$
$$\|v_\theta-v^*\|^2=\frac{\|f_\theta-A^*\|^2}{(1-\tau)^2}.$$

这说明clean-action参数化与速度误差之间的加权关系，不构成“所有输出都在可行流形上”的证明。实际M0包含time_epsilon与特定采样分布，必须按现有代码保持或显式版本化，而不能将上式当完整实现。[C09] [R04] [R12]

### 8.2 推理期RTC的指导形式

令Y为时间对齐后的旧剩余目标，W为承诺/软重叠权重。最终数据估计一般写为：

$$\hat A_1=x_\tau+(1-\tau)v_\theta(x_\tau,c,\tau).$$

RTC使用近似引导：

$$\tilde v=v_\theta+\lambda(\tau)\,J_{\hat A_1}(x_\tau)^T\,\mathrm{diag}(W)(Y-\hat A_1).$$

原论文给出随τ变化并截断的指导强度；官方`realtime_action()`实现VJP与重叠权重。M0在未触发denominator clamp时可以用clean预测构造对应估计；触发clamp的区间不能无审计地假设等价。[R01] [K01] [C09]

**必须保留的区分：**已承诺目标由实际执行队列保证不会被新结果追溯改写；指导采样对匹配是近似约束，并非数学上强制全prefix精确一致。不能因为guidance存在，就忽略队列对已经过去的动作时刻的处理。

### 8.3 实现骨架（伪代码，不是可直接替换现有函数的完整补丁）

```python
# 模型参数冻结；仅为x构建VJP图。W、Y必须已换锚点、归一化并对齐时间。
for tau in integration_grid:
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        clean = expert.predict_clean(x, live_condition, tau)
        velocity = clean_to_velocity(clean, x, tau, exact_time_contract)
        estimate = x + (1.0 - tau) * velocity
        residual = (weights * (previous_target - estimate)).detach()
        correction = torch.autograd.grad(
            outputs=estimate,
            inputs=x,
            grad_outputs=residual,
            create_graph=False,
        )[0]
    guided_velocity = velocity.detach() + gain(tau) * correction.detach()
    x = integrate_one_step(x.detach(), guided_velocity)
# 结果仍经过正常decoder、限制、安全与版本校验；不改写已执行前缀。
```

检查current inference路径是否被外层`inference_mode`包围；局部`enable_grad`不一定能撤销所有inference tensor限制。需要独立测试VJP非零、有限差分一致性、参数无梯度积累、显存释放和原关闭RTC路径逐位回归。

### 8.4 权重区域与时间对齐

推理开始时，将旧队列尚未执行的部分对齐到新计划的**实际命令应用时间网格**。源标签的取样时间与控制器开始保持该目标的时间必须分别保存：旧模型可能采用“未来一拍命令作标签、第一点立即开始执行”的已冻结合同，不能在RTC接线时无声明地改成到时才下发。使用实际预计延迟转换为承诺长度，保守调度可用ceil加明确余量；这与论文的简化整tick定义不同，属于工程适配。

W在已承诺区为高权重，在后续重叠区衰减，在无旧目标尾部为0。允许衰减并不等于允许跨任务沿用。若生成期间触发安全中止或active epoch改变，丢弃候选，重建条件。[R01] [ENG]

### 8.5 Training-time RTC

对每个训练样本采样合法前缀长度d，前缀输入为干净动作，后缀输入加噪。模型需要知道前缀/后缀不同噪声状态，可以采用token级time加mask。只对合法后缀计算损失：

$$\mathcal L_{suffix}=\frac{\sum_{k\ge d}m_k\|v_{\theta,k}-v_k^*\|^2}{D\sum_{k\ge d}m_k}.$$

D为动作维度；分母为0的样本不训练。旧动作前缀应与当前样本语义/任务一致。用同条专家轨迹制造前缀仅覆盖理想情况，后续还应比较模型自身实际承诺前缀；不把错误route与正确动作强配。[R02]

本项目M0现有标量time路径并非已经具备token级前缀噪声条件，因此这是实际模型修改，不是一个已存在的配置开关。推理期RTC与训练期RTC先分别比较，不默认叠加。

## 9. 理论：任务持续性、数据覆盖与空间表示

### 9.1 宏观rolling plan的来源与边界

π0.5提供语义subtask条件化的相关先例；ReKep展示分阶段约束与回退；用户方案提供剩余计划、当前指针与局部后缀编辑。[R03] [R09] [U01]

本项目将这些组合为：

$$M_{n+1}=\mathcal U(M_n,o_n,\text{execution feedback},\Delta P_n).$$

其中U必须维护事件/事实区别、版本和任务引用。这是DESIGN，不是对标准RTC的离散数学推导；也没有证明可以消除planning drift。通过H0/H1/H2比较验证额外价值。

### 9.2 数据选择与部署分布

成功筛选：

$$p_D(s)=p(s\mid\mathrm{accepted}).$$

策略部署：

$$s_{t+1}\sim P(\cdot\mid s_t,\pi(s_t)).$$

二者不天然相同。DAgger支持在学习策略访问状态上询问专家并聚合监督，但其理论依赖在线学习/专家条件，不能直接套成“恢复数据一定提高当前VLA成功率”。[R07]

MimicGen支持以任务结构与对象相对关系扩展示范；本方案借鉴组织方式，不声称已实现MimicGen或使用其资产。[R06]

### 9.3 面向操作的导航站位

固定Mani策略与执行器，定义：

$$V_M(b)=P(\text{操作成功}\mid b,\pi_M,\mathcal E_M).$$

导航交接理想上进入高$V_M$区域，而不只是接近物体。Mobi-π研究操作策略对底盘/视角分布的敏感性，支持这一研究方向。[R08]

本项目先用真实C上的rollout估计，不立即增加value network。IK可达、规划终点合法和对该视觉策略容易操作是不同条件。

### 9.4 深度与坐标

若depth是光轴z深度：

$$p_C=Z(u,v)K^{-1}[u,v,1]^T.$$

若记录的是ray range r，则：

$$p_C=r\frac{K^{-1}[u,v,1]^T}{\|K^{-1}[u,v,1]^T\|}.$$

用采集时刻外参变换到base/EEF参考系，不能用后续q替换腕相机姿态。深度数据必须保存定义、单位、mask和标定；RGB-D几何与语义融合由3D Diffuser Actor、DP3、ABot-M0等研究提供动机，但无当前任务必然改善的保证。[R10] [R11] [R04]

### 9.5 FK、误差度量与有界夹爪

$$\delta p\approx J(q)\delta q,\qquad \|\delta p\|^2\approx\delta q^T J^T J\delta q.$$

因此joint MAE不能替代任务空间误差。FK辅助监督使用同一机器人模型、同一坐标和示范对应时刻，不把所有动作拉向物体中心。[R14]

有界clean-gripper输出是项目候选：如$g=\sigma(z)$再映射到normalizer范围。它不约束高斯噪声或速度场；边界梯度、完全张开/闭合的标定和4步采样终值仍需测试。降低数值越界不等于提高物理抓取。[C09] [ENG]

## 10. 导航执行合同与在线验证

### 10.1 目标和停止条件的唯一来源

`ReachConfig`绑定goal ID、目标坐标、position/yaw tolerance、稳定性判据和计时器。DWA与外层不能分别维护15cm和12cm的互不一致停止逻辑。正常到达必须同时满足位置、朝向与冻结稳定性要求。

```text
位置未到达                 → 平移/重新连接
位置到达、朝向未到达       → 经验证的转向，不能记成功
位置/朝向均到达但未稳定    → 保持并验证
满足完整到达条件           → REACHED
无合法控制/地图失效        → SAFETY_STOP 或 FAILURE，不能记REACHED
```

转向时重新检查位置；漂出位置容差后回到调整位置。安全停止允许在目标外发生，但必须有原因。

### 10.2 A/B/C与连续末段

G为名义操作站位，仅评价；A为模型/源期望目标；B_raw为粗规划端点；B_exec为实际送入控制的修正路径端点；C为实测状态。保存所有项，不用A替代C。

对20cm栅格，平面最近中心最大量化误差为$0.2/\sqrt2\approx0.1414$m。这解释原0.10m门槛与端点量化的冲突，但不授权隐藏原snap失败。[C16] [ENG]

连续末段修复需经过完整机器人及携物包络、地面可通行性、转向/停车和控制可行性验证。原raw snap门保留为历史统计；新修复路径以独立协议记录最终endpoint误差。只有新协议审批才可改变旧“直接拒绝”的执行路径，绝不伪称旧门通过。

### 10.3 在线覆盖

离线路径通过只证明计划轨迹在已建模假设下受检查；实际C偏离后不能继续引用同一结论。在线监控检查当前包络、近未来控制窗口及可验证的停车轨迹是否在可信几何范围内。

地图未知、姿态不可信、动态障碍进入、携物包络改变或制动模型不足时，执行明确安全行为并请求重验证。此系统是监控/约束设计，不是形式化碰撞零风险证明。

## 11. 推理调度与队列协议

### 11.1 拟议ActionPlan

```yaml
schema: action-plan-v2
mission_id: string
instruction_version: integer
source_plan_version: integer
active_task_epoch: integer
active_task_id: string
observation_id: string
query_time_s: float
query_anchor_q: vector6
query_base_pose: pose7
action_time_profile: string
first_target_offset_s: float  # 标签采样偏移，不等于下发延迟
first_apply_time_s: float
sample_period_s: float
raw_targets: array_H_by_D
effective_absolute_targets: array_H_by_D
committed_until_s: float
valid_until_s: float
normalizer_id: string
model_id: string
safety_context_id: string
```

`first_target_offset_s`描述相对query时刻的监督目标采样偏移；`first_apply_time_s`描述部署队列实际开始下发第一点的时间。两者不允许混成一个字段。所有RTC重叠、过期丢弃和承诺长度按实际应用时间计算，同时保留标签语义用于训练/部署一致性检查。

`valid_until`根据观察年龄、任务和控制协议定义，不是凭模型confidence自动产生。输入prompt、当前任务target和模型/normalizer身份同时匹配才能消费。

### 11.2 主调度伪代码

```python
while running:
    obs = observer.freeze_current_snapshot()
    feedback = estimator.update(obs, command_log)  # 不读evaluator真值
    memory.update_current_facts(feedback)

    if planner_triggered(memory, feedback):
        planner.submit(snapshot(memory, obs))
    if planner.has_result():
        edit = planner.take_result()
        if edit.parent_version == memory.plan_version:
            task_change = memory.validate_and_apply(edit)
            if task_change.invalidates_active_actions:
                queue.cancel_unexecuted_task_epoch(task_change.old_epoch)

    if action_update_due(queue, obs):
        condition = build_condition(obs, memory.active_task, queue.overlap())
        action_worker.submit(condition)
    if action_worker.has_result():
        candidate = action_worker.take_result()
        queue.accept_only_if_identity_time_and_epoch_valid(candidate)

    command = queue.next_for_control_time(obs.control_time)
    command = safety.validate_or_controlled_stop(command, obs)
    applied = controller.apply(command)
    recorder.append_transaction(obs, applied)
```

安全线程不等待VLM。初次请求尚无队列时使用明确初始化保持，不默认沿用上一episode动作。候选过期、队列耗尽、服务异常、任务取消均有独立reason code。

### 11.3 时间与异步承诺

记录capture、request、VLM、action采样、response、apply全部时刻。开始新推理时保证剩余队列足以覆盖预计延迟和余量；返回后丢弃已经过去的目标点，不能从候选第0点重放。

实时实验报告实际p50/p95/p99及超时率。有限样本分位数不是硬实时上界；极端延迟必须触发明确定义的停止/保持，而不无限沿用旧计划。宏观planner可以低频，但其GPU排队不得阻断低层服务而未计入延迟。

当推理期间仿真暂停时，报告simulation-time能力，不宣传为达到真实实时频率。

## 12. 训练体系与迁移流程

### 12.1 三条损失路径分开

高层planner使用任务编辑和事实估计监督；低层使用FM/suffix损失；FK与有界夹爪是独立候选。无可用标签的progress项保持关闭，不用episode行号替代物理进度。

$$\mathcal L=\lambda_T\mathcal L_{task}+\lambda_A\mathcal L_{action}+\lambda_{aux}\mathcal L_{aux}.$$

这是组织式，不规定把所有项同时打开。各损失的合法样本、mask和分母单独报告，避免高层样本多而掩盖动作梯度。

### 12.2 分阶段训练

先在冻结低层上验证planner及推理期RTC。随后使用可信新数据适配action边界、必要的状态/任务编码器；再小学习率解冻共享Qwen。训练期RTC作为单独候选，不沿用未经适配的旧time接口。

从旧checkpoint继续训练与从相同ABot初始化重训是不同实验。数据比较使用相同初始化、相同步数和采样预算；架构比较如重置输出层应显式列出允许的权重差异。

### 12.3 模型选择与数据泄漏

所有候选先用开发集；选择规则在看到最终测试前冻结。父episode、恢复片段、分叉和视觉重渲染使用同一task family split。模型自产prefix只有在任务语义与标签一致时进入动作训练。

训练中用teacher未来计划直接作为当前条件是一种oracle训练环境，必须明确其与部署差异；正式训练应逐步覆盖可实际获得的计划/反馈。[R02] [R03] [R07]

## 13. 工程实施包与文件级职责

下表“新增”代表需要实现；已有路径仅指审读基线。

| 包 | 修改/新增位置 | 必须交付 | 对应实验 |
|---|---|---|---|
| I0 合同层 | 新`contracts/observation.py`、`action.py`、`task_plan.py`；接入现有joint合同 | 版本、codec、权限、golden fixtures | 所有实验 |
| I1 采集 | `source/recording/lerobot_dataset.py`、`full_physics_pipeline.py`、应用边界 | effective command、reset处理、真实时钟 | D-01/D-02 |
| I2 数据 | 新`task_view`/`action_view`派生；下游`joint_trajectory_data.py` | 来源mask、split、normalizer、两视图 | D-03/V-03 |
| I3 高层 | 新`task_memory.py`、`rolling_planner.py` | H0/H1/H2同权重协议、CAS编辑、事实失效 | V-01/V-02 |
| I4 动作 | `dit.py`与新`rtc_sampling.py`/`action_queue.py` | VJP引导、换锚点、时间队列 | V-01/V-04 |
| I5 几何 | 新`geometry_encoder.py`与condition builder | RGB-D投影、mask、当前帧融合 | V-05 |
| I6 导航 | `joint_trajectory_system.py`、`continuous_endpoint.py`与DWA适配 | 同ReachConfig、在线覆盖、完整G/A/B/C | V-06 |
| I7 评价 | 已有`physical_events.py`、独立assistance；新统一score | unknown覆盖、任务完成、分母清楚 | D-05/V-08 |
| I8 实验 | 新experiment runner/manifest；保留已有diagnostic入口 | 配对试验、任务聚类、失败完整记录 | V-01—V-08 |

不得将新模块名用于声称仓库已实现。CLI应在各包合入后再提供`--help`和最小可运行示例，禁止写一组不存在的命令让开发者直接启动量产。

## 14. 验收分层：研究许可、算法采用、部署发布不是同一门

### 14.1 四个层级

| 层级 | 判定对象 | 必须满足 | 不能据此宣称 |
|---|---|---|---|
| G0 合同/代码 | schema、时钟、权限、数值边界 | 所有确定性硬合同测试通过；未知保持未知 | 仿真抓取能力 |
| G1 研究诊断许可 | 固定身份与受控物理实验 | 无未拦截硬安全违规、数据/评价可追溯；失败门明确标注 | 所有发布门通过 |
| G2 方法采用 | 架构/数据比较 | 任务效用、机制与成本有可解释证据；反例不被隐藏 | 跨场景或真机普遍有效 |
| G3 部署/正式能力发布 | 完整系统 | 源/部署合同、完整任务、所声明物理条件、saturation和在线安全均通过 | 未测试的环境与机器人能力 |

0.5%门仍失败时可进行G1诊断，但G3不能通过。严格接触unknown时可以发布明确命名的几何transfer结果，不能发布严格抓取成功率。框架已跑完不等于任务成功。

### 14.2 继承阈值与新参数

继承：saturation≤0.005按原事件口径；历史raw snap≤0.10m；外层导航位置≤0.12m；最终区域内释放并稳定1s。朝向和速度要求从冻结配置读取，不在不同模块各写一份。[C08] [C12] [C13]

新参数：planner最大上下文、事实有效期、RTC权重/重叠长度、反馈周期、depth退化范围和停止安全余量，必须在开发集预注册。本文不把任何未标定的阈值当成机器人通用安全值。

## 15. 确定性测试与物理验收矩阵

| ID | 对象与测试 | 通过条件 | 理论/来源 | 实验关联 |
|---|---|---|---|---|
| ACC-01 | reset前后不同缓存目标，首个无显式命令 | 旧目标不进入当前有效监督；unknown不补值 | C02/C03, ENG | D-01 |
| ACC-02 | 同物理tick多个状态切换和apply | 仅实际消费目标进入控制流；事件不丢失 | C03, ENG | D-01 |
| ACC-03 | 50Hz缺帧、队列阻塞、进程复用 | 真时间/tick检查发现缺样；写盘无静默丢弃 | C02, R15 | D-01/D-05 |
| ACC-04 | 10D实际位姿误送joint模型 | schema明确拒绝，不按维度猜测转换 | C04/C11 | D-02 |
| ACC-05 | codec round-trip/关节顺序/单位 | 在预注册数值容差内一致，错序/错单位拒绝 | R14, ENG | D-02 |
| ACC-06 | 0.2s历史与目标时间在50Hz中派生 | 相机间隔/first_target_offset正确，缺样不偷用未来 | C11/R15 | D-02 |
| ACC-07 | 父示范、扰动、恢复、重渲染分组 | 同family不跨split；normalizer无test信息 | R07, ENG | V-03 |
| ACC-08 | 完成后掉落、错误上一输出、stale facts | 历史事件保留；当前carrying失效；不误推进 | U01/R03/R09 | V-01/V-02 |
| ACC-09 | 并发两次PlannerEdit | parent mismatch拒绝/重评价，不覆盖新状态 | ENG | V-01 |
| ACC-10 | 改未来计划 vs 改当前目标 | 前者保留当前动作；后者旧epoch候选拒绝 | ENG | V-01/V-07 |
| ACC-11 | evaluator真值注入检测 | 改动eval-only字段不改变普通策略输入；oracle单列 | C15, ENG | 所有闭环 |
| ACC-12 | RTC关闭回归 | 原采样路径和输出合同不改变 | C09 | V-01 |
| ACC-13 | RTC VJP数值梯度检查 | FP32 toy符合有限差分；真实模型梯度有限、无参数积累 | R01/K01 | V-04c |
| ACC-14 | 承诺前缀时间对齐/换锚点 | 绝对目标不变；已执行点不重放；异任务不沿用 | R01/ENG | V-02/V-04c |
| ACC-15 | 训练期前缀与suffix mask | prefix为有效干净条件；零有效后缀不训练 | R02 | V-04c |
| ACC-16 | NAV 12/15cm间隙、仅yaw未到 | 未达不得正常到达停止；安全停车带原因 | U02/C13, ENG | V-06 |
| ACC-17 | 原量化B、修复B_exec、实际C | 三者独立记录；新路径证据不覆盖旧raw门结果 | C16, ENG | V-06 |
| ACC-18 | 路径预检内/外与动态包络 | 实际越出可信范围前触发控制保护或重规划 | ENG | V-06/V-08 |
| ACC-19 | depth定义、K/T、孔洞和错外参 | 单位/投影校验；无效mask；错误标定被发现 | R10/R11/R14 | V-05 |
| ACC-20 | 真接触正负例与覆盖漏读 | 缺测unknown；支撑不误判悬空；旧score保留 | C15/R16 | D-05/V-08 |
| ACC-21 | 几何完成、strict unknown、模型FINISH | 三种结果分开；提前FINISH记错；延迟结束记录损害 | ENG | V-08 |
| ACC-22 | sample mean与episode mean跨门槛反例 | 原saturation gate仍按指定分母，不混用CI | C20, ENG | 所有报告 |
| ACC-23 | 延迟、out-of-order、重复响应、队列耗尽 | 不消费过时chunk；安全行为明确且计入失败 | R01, ENG | V-07/V-08 |
| ACC-24 | 完整episode真实运行 | 无中途reset或隐藏强制切换；失败不从系统分母删除 | 文档一/二 | V-08 |

代码通过仅满足其对应层级。ACC-13通过不证明RTC有效；ACC-19通过不证明depth提升；ACC-08通过不证明高层恢复能力；它们只是支持实验结论可解释。

## 16. 物理与方法验收的具体规模

### 16.1 最小合同pilot

数据12条有效诊断episode（冷启动/复用各6）与10个源回放实例，来自文档一D-01/D-02。失败尝试保留，不计作成功配额。两种频率或解码回放必须从相同物理条件出发，不能用完美源动作运行证明错误缓存也有效。

### 16.2 主架构初筛

H0/H1/H2×L0/L1，20个开发family×2种条件，240次初筛。已知000006/24/30单列机制回归。实际使用高层适配模型时明确其训练数据和成本，不能声称全模型冻结。完整episode为主，局部分叉1.6s及延长保持为独立诊断。

### 16.3 接触校准

至少覆盖稳定夹持、空抓、桌面支撑、单侧托举、撞起、滑落/释放等物理情形。每类建议先5种初始化；这不是统计上足够证明零误报。比较finger-object与object-all接触覆盖；单一桌面正样本不能认证articulation接触覆盖。

### 16.4 最终确认

从新冻结100个family、3个有效环境seed起步。主要基线与最终方法2组共600次/训练seed；3训练seed为1800次。资源不足时减少并报告CI与未覆盖条件，不将重复帧充作任务样本。

方法采用建议预注册最小有意义增益，例如完整任务+5个百分点，正常任务回退容忍不超过5个百分点，同时安全与合同不劣化。这些是研究决策尺度，可在试验前由负责人调整；绝非论文保证或机器人安全限值。CI包含零/成本明显增加时允许结论“不确定”或“不采用”。

## 17. 指标定义与报告模板

### 17.1 层次化结果

`execution_completed`只表示run按规定退出。`geometry_hold_proxy`表示局部物体-末端相对几何保持。`geometry_transfer_success`要求从源到目标的连续任务证据和释放/稳定。`strict_full_success`要求所声明严格物理证据充分；`task_finish_correct`另衡量planner是否在正确时机结束。

必须报告开始的全部任务、产生模型请求的任务、可评价任务、oracle/条件任务和最终系统任务。任何“条件成功率”旁都应给分母和起始条件。

### 17.2 核心指标

| 类别 | 指标 |
|---|---|
| 整任务 | transfer成功、strict成功/unknown、耗时、恢复完成率、物体掉落/错误放置 |
| 宏观 | 错误ADVANCE、重复/遗漏任务、无效repair、stale fact保留、FINISH过早/过晚 |
| 微观 | 实际闭合时间、闭合未发生比例、计划反复推迟、FK误差、跨块变化、持有观察窗口 |
| 导航 | A/B_raw/B_exec/C、位置/朝向/稳定、无原因零命令tick、偏离覆盖和重规划 |
| 系统 | capture到apply延迟、过期响应、队列耗尽、推理成本、服务/启动失败 |
| 数据 | 唯一失效命令点、受影响chunk、来源未知、阶段覆盖、有效episode和专家查询成本 |
| 约束 | 原saturation事件率、unique被改元素、每通道幅度、实际执行点与完整预测分别统计 |

saturation旧定义：`(position_events+rate_events+gripper_events)/(N×H×7)`。事件可重叠，它是事件率不是“独立坏动作概率”。保留冻结sample mean门，episode等权与聚类CI独立展示。高分辨率H变更时用对应H并明确新动作合同。

### 17.3 统计规范

按task family配对，bootstrap重采样整个family，不把重叠chunk或同任务seed当独立任务。训练seed可用分层bootstrap或分别给结果再汇总，不能只显示最优seed。

窗口内没有闭合的样本报告未发生/右删失，不从延迟均值中消失。1.6s窗口不足1s持续保持时报告观察不足。unknown接触缺失可能非随机，应给覆盖率及支持范围，不能只用已知子集推断总体。

宏微观交互用差中差描述，置信区间同样基于任务家族。多消融先筛选再用新测试集确认，不在同一小集上无限选择阈值。

### 17.4 最小报告文件（拟议）

```text
run_manifest.json
resolved_system_contract.json
sample_families.json
per_episode_results.jsonl
per_query_metrics.jsonl
per_control_tick_events.jsonl
aggregate_report.json
source_and_artifact_checksums.sha256
interpretation.md
```

`interpretation.md`必须分已有证据、未验证假设、oracle限制、失败原因和下一步。没有原始视频不自动使数值无效，但未保留必要观察/控制证据就不能提出对应因果解释。

## 18. 全部修改的理论与证据追踪表

这是“为什么改—依据在哪里—如何验证”的总索引。标注ENG的内容不强行寻找一篇论文为工程规则背书。

| 修改ID | 拟议修改 | 理论/论文与代码来源 | 来源支持什么 | 尚待本项目证明 | 验证 |
|---|---|---|---|---|---|
| M01 | 完整episode为主、任务/动作双视图 | R03 π0.5；R06 MimicGen；U01 | 分层任务条件、结构化示范的相关先例 | 双视图是否提升本任务学习 | D-03/V-03 |
| M02 | 正常/边界/恢复覆盖 | R06；R07 DAgger | 不同初态与学生访问状态的监督价值 | 180/60/60配比和恢复收益 | D-04/V-03 |
| M03 | 生效命令、reset隔离、未知来源mask | C02/C03/C14；ENG | 直接修正已发现来源问题 | 新采集是否彻底阻断该路径 | ACC-01/02 |
| M04 | 50Hz事实层与模型秒级派生 | C02/C04/C11；R15 | 时间窗口与动作表示需明确 | 低频保持是否保留任务行为 | D-02/ACC-06 |
| M05 | 当前计划、事件与可撤销事实 | U01；R03；R09 ReKep | 高层条件、反馈/回退的设计依据 | 避免planning drift的真实收益 | V-01/ACC-08 |
| M06 | H1固定骨架强基线 | ENG；R03 | 隔离计划状态与自由编辑两个因素 | H2是否真的优于简单H1 | V-01 |
| M07 | plan_version与active_task_epoch分离 | ENG | 并发和语义失效的工程约束 | 是否避免过时接管而不过度停止 | ACC-09/10 |
| M08 | 推理期RTC/VJP软重叠 | R01；K01 | 连续动作的inpainting与异步接续 | M0适配和闭合推迟问题能否改善 | ACC-13/14/V-02 |
| M09 | 训练期动作前缀条件化 | R02；K01 | 模拟延迟、干净前缀、后缀训练 | 本模型time改造与分布覆盖 | ACC-15/V-04c |
| M10 | NAV与Mani分域、NAV目标主输出 | C08/C13；R08；ENG | 动作域不同、下游站位的重要性 | endpoint head是否足够/更有效 | V-04a/V-06 |
| M11 | Mani 25Hz×50、保持2s时域 | R13；R01；ENG | 预测与执行时域可分离 | 25Hz是否值得计算/数据成本 | V-04b |
| M12 | 持续语义+实时条件耦合 | R03/R04/R05；K02 | 多种条件耦合与KV复用先例 | 本任务何种耦合更好 | V-05/V-07 |
| M13 | RGB-D几何分支 | R10/R11/R04 | 3D表示与动作生成的实证动机 | 本相机/物体/噪声条件收益 | V-05/ACC-19 |
| M14 | FK一致性辅助监督 | R14；ENG | 关节到任务空间的运动学关系 | 是否比单纯joint损失有效 | V-04e |
| M15 | 有界clean-gripper输出 | C09；ENG | 控制变量本身有界、clean输出接口可用 | 是否改善物理动作而非仅计数 | V-04d |
| M16 | 导航ReachConfig与连续末段在线检查 | C13/C16；U02；ENG | 直接接口证据、几何约束 | 实际控制能否持续到达安全C | ACC-16—18 |
| M17 | evaluator与assistance分离、三态评分 | C15；R16；ENG | 当前覆盖缺陷和传感接口限制 | 物理误报/漏报及覆盖范围 | ACC-20/21 |
| M18 | H0/H1/H2×L0/L1因子设计与配对统计 | ENG | 因果因素拆分、统计单位明确 | 整体收益和交互是否存在 | V-01/V-08 |
| M19 | 共享codec、身份、数据家族切分 | C10/C11；R15；ENG | 可复现和泄漏防护 | 全链一致性是否真实实现 | ACC-04—07/11/22 |
| M20 | 双服务原型→共享VLM、实时延迟验收 | R01/R03/R05；K02；ENG | 层次频率、缓存与异步的相关依据 | 最终单体系成本和实时表现 | V-07/V-08 |

M01—M20覆盖本套文档所有主要方法和接口修改。日志文件名、数量预算、schema字段、样本比例、验收决策阈值是ENG，不是额外未引用的算法创新。

## 19. 论文与官方代码的适用范围核对

| 参考 | 实际采用/借鉴内容 | 不声称复现的内容 |
|---|---|---|
| RTC / Training-time RTC | 动作重叠条件、延迟、后缀训练 | 宏观task rolling plan；对本机器人的成功或安全保证 |
| π0.5 / openpi | 语义subtask条件、动作prefix/suffix结构 | openpi等于论文全部高层系统，或直接获得本任务planner |
| ABot-M0 | clean-action接口、最后层条件、可选几何融合 | clean-action必然物理可行，或升级到ABot-M0.5 |
| SmolVLA | cross/self耦合与分离观察/动作计算的相关设计 | 当前模型缺少self-attention |
| MimicGen | 任务/对象结构与示范扩展思路 | 已移植MimicGen数据生成器 |
| DAgger | 学生访问状态上的专家纠正 | 任意失败数据直接当正确标签、无条件理论保证 |
| Mobi-π | 策略依赖的站位/视角适配 | 必須引入3DGS优化或已经确认handoff是最大根因 |
| ReKep | 带反馈的阶段、约束与回退 | 本模型使用约束优化器或等价的形式化保证 |
| 3D Diffuser Actor / DP3 | 校准几何与扩散动作的结合 | 直接证明depth优于当前RGB系统 |
| Modern Robotics / FM | 坐标、FK/雅可比、条件流基本公式 | 工程控制安全或特定网络最优性 |

第三方仓库链接以本次阅读日期为核查点；开工时需锁定实际依赖commit和license。本文核对论文/公开代码接口，不声称运行了这些外部项目或复现了其结果。

## 20. 最终交付与发布清单

采集侧交付可信raw、双训练视图、来源与split、复现回放和版本；模型侧交付checkpoint、配置、normalizer、允许重置清单、训练日志和消融；执行侧交付共享合同、队列、导航/安全、评价/辅助独立配置；实验侧交付全部尝试、配对清单、指标、成本和限制。

发布前由采集负责人、模型负责人、执行负责人和实验负责人分别签认对应合同。角色是职责划分，不假设项目已经有四名独立成员。

最后应能够回答四个问题：

**这个动作从哪里来？现在执行的任务与计划是哪一版？新观察如何修改正在推进的动作？完整任务的成功由哪些独立证据支撑？**

四者清楚且实验支持时，才将候选组合定为最终体系。任务记忆、RTC、depth或更多参数本身，都不是验收通过的证据。


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

**C09｜当前M0 DiT实现**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/dit.py)。clean-action训练/采样、scalar time、self/cross-attention与future tokens。

**C10｜当前数据派生与normalizer**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory_data.py)。旧sampled5Hz数据合同、future目标、route映射、训练输入边界。

**C11｜当前模型、动作与语言合同**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory.py)。H=10、0.2s间隔/历史、四route、canonical subtask、无DONE。

**C12｜动作解码和运行时**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory_runtime.py)。DirectJointTrajectoryExecutor及位置/速率/夹爪限制。

**C13｜Isaac/PCT/DWA执行连接**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory_system.py)。NAV端点消费、实际执行时钟、joint顺序执行与到达配置。

**C14｜已发布执行接口v2实验文档**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/docs/execution_interfaces_v2_20260906.md)。reset缓存反例、源/部署回放、条件PICK及接触未知；不含用户后续新实验的完整工件。

**C15｜独立物理事件评价器**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/physical_events.py)。relative geometry与三态contact，独立于辅助控制。

**C16｜连续导航端点候选与已发布解释**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/docs/execution_consistency_validation_20260905.md)。量化、G/A/B/C和DWA退化根因；末段可部署性不能由文档自动推定。

**C17｜采集state machine**  
[来源](https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/pipeline/state_machine.py)。reset、阶段切换、物体prepare与可选固定抓取约束调用。

**C18｜采集quota wrapper**  
[来源](https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/scripts/pipeline/run_full_physics_success_quota.py)。成功接受条件、分片与恢复身份；用于检查续采合同。

**C19｜采集Sim6与相机适配**  
[来源](https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/simulation/isaaclab_runtime.py)。相机四元数边界、标定及runtime接口；具体部署仍需锁版本。

**C20｜当前正式指标**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/formal_metrics.py)。saturation事件口径、样本/episode统计与现有运行限制。

### 理论、论文与官方文档

**R01｜Black, Galliker, Levine. Real-Time Execution of Action Chunking Flow Policies (2025)**  
[来源](https://arxiv.org/abs/2506.07339)。采用动作重叠inpainting/异步接续思想；不提供任务级rolling plan或本任务安全保证。官方代码见K01。

**R02｜Black et al. Training-Time Action Conditioning for Efficient Real-Time Chunking (2025)**  
[来源](https://arxiv.org/abs/2512.05964)。采用训练期干净动作前缀与模拟延迟思路；当前M0适配仍需实现。官方代码见K01。

**R03｜Physical Intelligence et al. π0.5: a Vision-Language-Action Model with Open-World Generalization (2025)**  
[来源](https://arxiv.org/abs/2504.16054)。高层subtask与动作条件的相关先例。openpi见K02；其公开组件范围不等于论文完整高层系统。

**R04｜Yang et al. ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning (2026)**  
[来源](https://arxiv.org/abs/2602.11236)。clean-action与可选几何融合依据；不是输出物理可行性的定理。 [官方代码ABot-M0分支](https://github.com/amap-cvlab/ABot-Manipulation/tree/ABot-M0)。

**R05｜Shukor et al. SmolVLA: A vision-language-action model for affordable and efficient robotics (2025)**  
[来源](https://arxiv.org/abs/2506.01844)。参考交替cross/self-attention及系统分工，不代表当前网络缺少该结构。 [官方项目/代码说明](https://huggingface.co/blog/smolvla)；[LeRobot代码](https://github.com/huggingface/lerobot)。

**R06｜Mandlekar et al. MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations (2023)**  
[来源](https://proceedings.mlr.press/v229/mandlekar23a.html)。任务/对象结构化示范扩展的相关依据。 [官方代码](https://github.com/NVlabs/mimicgen)。

**R07｜Ross, Gordon, Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (2011)**  
[来源](https://proceedings.mlr.press/v15/ross11a.html)。DAgger：面向学习策略访问状态的监督；理论依赖专家/在线学习条件，不是任意恢复数据的收益保证。工程上由本项目教师查询实现，不声称移植原作者机器人代码。

**R08｜Yang et al. Mobi-π: Mobilizing Your Robot Learning Policy (2025)**  
[来源](https://arxiv.org/abs/2505.23692)。操作策略与初始底盘/视角适配。 [官方代码](https://github.com/yjy0625/mobipi)。

**R09｜Huang et al. ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation (2024)**  
[来源](https://arxiv.org/abs/2409.01652)。阶段、反馈与回退的结构参考；本方案不采用其整套约束优化。 [官方代码](https://github.com/huangwl18/ReKep)，关注`main.py`。

**R10｜Ke, Gkanatsios, Fragkiadaki. 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations (2024)**  
[来源](https://arxiv.org/abs/2402.10885)。RGB-D空间特征与动作生成依据；本文不复用其性能数字。 [官方代码](https://github.com/nickgkan/3d_diffuser_actor)。

**R11｜Ze et al. 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations (2024)**  
[来源](https://arxiv.org/abs/2403.03954)。紧凑3D表示与diffusion policy的相关依据。 [官方代码](https://github.com/YanjieZe/3D-Diffusion-Policy)。

**R12｜Lipman et al. Flow Matching for Generative Modeling (ICLR 2023)**  
[来源](https://arxiv.org/abs/2210.02747)。条件概率路径与向量场回归的理论基础，不保证机器人控制可行。

**R13｜Chi et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (RSS 2023 / IJRR 2024)**  
[来源](https://diffusion-policy.cs.columbia.edu/)。动作序列生成与观察/预测/执行时域设计参考。 [官方代码](https://github.com/real-stanford/diffusion_policy)。

**R14｜Lynch, Park. Modern Robotics: Mechanics, Planning, and Control (2017)**  
[来源](https://modernrobotics.northwestern.edu/)。第3—5章坐标、FK和Jacobian；第13章移动操作。 [官方配套代码](https://github.com/NxRLab/ModernRobotics)。

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
[C09]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/dit.py
[C10]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory_data.py
[C11]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory.py
[C12]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory_runtime.py
[C13]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory_system.py
[C14]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/docs/execution_interfaces_v2_20260906.md
[C15]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/physical_events.py
[C16]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/docs/execution_consistency_validation_20260905.md
[C17]: https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/pipeline/state_machine.py
[C18]: https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/scripts/pipeline/run_full_physics_success_quota.py
[C19]: https://github.com/lemonoscar/arm-vla-grasp-sim/blob/6e1b3856614862dd9eca88fbf8d2e3e6d7941d67/source/simulation/isaaclab_runtime.py
[C20]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/formal_metrics.py
[R01]: https://arxiv.org/abs/2506.07339
[R02]: https://arxiv.org/abs/2512.05964
[R03]: https://arxiv.org/abs/2504.16054
[R04]: https://arxiv.org/abs/2602.11236
[R05]: https://arxiv.org/abs/2506.01844
[R06]: https://proceedings.mlr.press/v229/mandlekar23a.html
[R07]: https://proceedings.mlr.press/v15/ross11a.html
[R08]: https://arxiv.org/abs/2505.23692
[R09]: https://arxiv.org/abs/2409.01652
[R10]: https://arxiv.org/abs/2402.10885
[R11]: https://arxiv.org/abs/2403.03954
[R12]: https://arxiv.org/abs/2210.02747
[R13]: https://diffusion-policy.cs.columbia.edu/
[R14]: https://modernrobotics.northwestern.edu/
[R15]: https://huggingface.co/docs/lerobot/lerobot-dataset-v3
[R16]: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html
[K01]: https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py
[K02]: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
[U01]: #source-u01
[U02]: #source-u02
[ENG]: #source-eng

