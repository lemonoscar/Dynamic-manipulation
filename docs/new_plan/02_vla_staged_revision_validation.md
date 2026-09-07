# 文档二｜层次化 VLA 的分阶段修正、架构实验与验证方案

**版本：** 1.0 · 2026-09-06  
**状态：** 待实施研究计划；所有“预期”“候选”“验收目标”均不是实验结果。  
**代码基线：** `lemonoscar/Dynamic-manipulation`，`fix/grasp-evaluation-execution-v2@bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31`。  
**采集基线：** `arm-vla-grasp-sim@6e1b3856614862dd9eca88fbf8d2e3e6d7941d67`。  
**配套文档：** [采集重构](01_collection_reconstruction_plan.md)；[开发体系与验收](03_developer_system_acceptance_theory.md)。

> **研究主线：把“逐次独立判断 route 并生成动作块”升级为“持续任务状态＋局部计划修复＋具有跨块连续性的动作生成”。不以更大 backbone 或 World Model 为默认答案；也不把全部架构研究推迟到所有历史故障排查完之后。**

## 1. 当前基线与研究边界

### 1.1 已实现结构

当前模型使用 ABot-M0 初始化 Qwen3-VL-4B 与两套参数独立的 M0 DiT。VLM 条件为最后层的 token 序列，不是单个最终 token；16个 DiT block 交替 cross-attention/self-attention，4次动作采样，Mani接收13D q/dq/夹爪状态。NAV与Mani均预测十点、间隔0.2s。progress无可用监督，权重为0。[C08] [C09] [C11]

已有 self-attention、独立专家和本体状态不能重复当作新增贡献。ABot-M0本身也采用clean-action与几何融合研究，不能仅凭“读取最后层”就宣布结构无效。[R04]

### 1.2 已有实验怎样约束设计

发布文档中，固定PICK、标准起始状态下长块在两个示范出现持续几何夹持；000024的源动作可以抬升，模型长短块均失败。严格接触评分未知，几何代理不是完整任务成功。[C14]

用户最新汇报进一步给出：train存在2,022个含失效缓存的MANI chunk；合并未知来源的隔离集合为11,819/27,871；18个局部A/B/C窗口中000006替换计划推迟闭合并降低局部抬升，但000024旧计划也失败、000030替换可抬升；源NAV出现15cm停止/12cm到达冲突及预检覆盖外运动。这些是REPORT级事实，未独立读取完整新工件。[U02]

这些证据支持三条并行路线：可信数据；任务/动作计划持续性；独立操作与交接能力。它们不支持一个解释所有失败的总根因。

## 2. 七个问题对应的研究决策

| 问题 | 主决策 | 必须比较的对照 | 不提前作出的判断 |
|---|---|---|---|
| 数据采集 | 完整episode、任务/动作双视图、边界与恢复覆盖 | 同量可信正常数据 vs 任务化数据 | 50Hz自然提高成功率 |
| Pipeline | Task-level rolling plan＋Action-level RTC | 无计划、固定计划、可修复计划 × 无RTC/有RTC | 宏观rolling plan就是原RTC算法 |
| Action expert | 保留分域，NAV语义贴合规划器，Mani带动作上下文 | NAV目标/轨迹；5Hz/25Hz；推理期/训练期RTC | 四个专家或更多head一定更好 |
| VLM耦合 | 稳定任务语义与实时几何/本体条件分工 | 单流最后层 vs 双流条件；固定视觉预算 | 最后一层必然信息不足 |
| 输入 | 动作历史与任务事实优先；深度独立验证 | RGB/RGB-D × 耦合方式 | 深度能替代任务记忆和接触判断 |
| 评估 | 完整episode为主，开环和局部反事实用于解释 | 正常、扰动、恢复、实时延迟分层 | 低MAE等于闭环成功 |
| 全链对齐 | 共享codec/schema/时间与评价合同 | 同一黄金样本跨训练、服务、执行 | 教师与学生必须用同一控制算法 |

## 3. 推荐的目标架构

```text
原始指令 + 当前可用观察 + 实际执行反馈
                         │
            Task-level rolling planner
         原始目标 / 已发生事件 / 当前事实
           当前任务 / 剩余计划 / 局部修复
                         │
           active_task + semantic condition
                ┌────────┴────────┐
                │                 │
           NAV expert         Mani expert
      局部目标或路径条件     当前RGB、q/dq/gripper
                │          旧动作重叠；可选RGB-D
      统一合同的PCT/DWA           │
                │           Action-level RTC
                └────────┬────────┘
                       执行器
                         │
                  新观察与执行反馈
```

高层保持原始意图与剩余任务，不直接发关节命令。低层执行当前任务，不自行改变目标物体、目的地或宣布完整任务成功。独立安全监控可以中止动作，但不能以不可见的真值自动完成任务决策。

本轮默认是分阶段移动操作：NAV期间保持机械臂，PICK/PLACE期间底盘采用明确冻结的控制条件。若以后需要移动中抓取，必须另做联合动作域和控制能力研究，不能只改任务列表。

## 4. 宏观层：Task-level rolling plan 的精确定义

### 4.1 数据结构与任务语义

维护：

$$M_n=(P_n,i_n,\mathcal E_n,\hat F_n),$$

其中P是计划、i是当前task ID、E是历史事件、F是当前事实估计。新观察只引发必要更新：

$$\Delta P_n\sim p_\phi(\Delta P\mid\ell,o_n,M_n,\text{execution feedback}).$$

支持 `CONTINUE / ADVANCE / REPAIR_SUFFIX / FINISH / UNRESOLVED`。任务可以回退或插入retry，但必须生成新attempt ID；历史完成事件不删除，`carrying`等持续事实可被新证据撤销。

主实验给H1/H2相同的四步初始骨架；H0仅获得原始目标、当前观察与当前反馈，不读取计划列表。自由生成初始计划另测，避免把一个简单模板问题伪装成核心创新。[U01] [R03] [R09]

### 4.2 三种高层对照

| 高层 | 上下文与允许操作 | 要验证什么 |
|---|---|---|
| H0 | 当前观察、原指令和与其他组同源的当前执行反馈；不维护任务列表 | 响应式高层基线 |
| H1 | 固定四阶段骨架、相同事实估计、可推进但不编辑后缀 | 显式计划/进度本身的价值 |
| H2 | 与H1相同初始骨架，可基于证据局部编辑后缀和恢复 | rolling repair的额外价值 |

H1不能偷用GT成功来推进；H2也不能额外获得目标真值、完整接触真值或更多传感器。H0可读当前反馈，不读历史计划；这是有意消融，不称为完全等信息量实验。

H0采用`current_task_candidate`（当前route/subtask候选）输出；统一服务适配器只补齐执行所需的版本和任务引用，不向H0模型偷偷加入历史计划。H1/H2采用计划编辑输出。三组共用响应封装和控制接口，不要求无计划的H0虚构一份剩余计划。

### 4.3 `previous_prediction` 语义歧义的禁止规则

`last_model_output`只保存上一次模型回答；`last_completed_task_id`只由完成事件更新；`active_task_id`表示当前意图；`current_facts`有值、来源、时间和unknown。训练与推理严禁替换字段含义。

上次模型输出“抓好了”不构成新的物理完成事件。评估真值只能作为oracle诊断组，不能暗中返回普通planner。

### 4.4 先原型、后共享backbone，避免“固定模型”含义混乱

已有action checkpoint没有被证明会理解新JSON计划。研究原型采用：

- **低层服务θL：** 原Qwen条件路径和双DiT整体冻结；输入仍是旧canonical active subtask。
- **高层服务φP：** 使用同一Qwen基座或明确指定的相同模型，先用任务级样本做一次共同的planner适配；然后H0/H1/H2共享同一个φP，差别仅是上下文与编辑权限。
- 若零训练prompt-only原型能稳定通过schema与小样本语义检查，可先用它，但不能假设旧route-only输出自然具备新能力。

因此应表述为“固定低层action checkpoint比较高层机制”，不能称“全系统没有训练”。两服务的显存、调用量与墙钟成本全部报告。迁移到目标的共享Qwen是后续单独阶段，需重新验证，不把原型成本冒充最终成本。[C08] [R03]

## 5. 微观层：把RTC作为动作条件化，而不是尾部硬复制

### 5.1 三段语义

新请求发生后，将旧计划划为：实际已执行部分、预计新推理返回前必然执行的承诺前缀、允许修正的剩余重叠/尾部。安全中止优先级高于“承诺”；承诺不是无条件继续危险动作。[R01]

原RTC通过推理期inpainting guidance将新生成与旧重叠关联；训练期RTC则在训练中显式提供干净前缀和模拟延迟。[R01] [R02]

本项目的“0延迟暂停仿真下的重叠条件化”首先是机制诊断，不称完整实时RTC部署。真实RTC需要物理在推理期间继续推进、动作队列和观测时钟都正确。

### 5.2 当前M0的适配边界

M0头返回clean-action预测f，而非直接速度v；需在当前时间约定下转换，再实现官方式VJP guidance。4次采样是否足够、额外梯度开销、指导权重和重叠长度都作为验证对象。不是把输出前缀替换后就声称复现RTC。[C09] [R01] [R02] [R04]

训练期前缀条件化需要将旧标量time接口拓展到动作token级噪声时间/条件。前缀为干净输入，loss只训练合法后缀；仅mask loss却不给模型前缀不是training-time RTC。

### 5.3 必须允许合理改计划

模型不能被迫按照固定时间闭合。新观察发现接近几何失效时，可以取消旧计划；取消必须有任务/执行理由，不由计划FIFO决定语义。

关节与夹爪必须作为协调序列，不把旧闭合拼接到无关新轨迹。旧query-relative动作换到新锚点：

$$\widetilde{\Delta q}_j=q_{old}+\Delta q^{old}_{j+m}-q_{new}.$$

宏观层改变当前任务或目标时，递增`active_task_epoch`；旧epoch动作不得继续被新任务消费。只编辑未来任务时不必取消当前有效动作，使用单独`plan_version`避免无谓中断。[ENG]

## 6. Action expert与时间尺度的候选

### 6.1 NAV：让学习输出与规划器消费对象一致

第一候选为局部目标`(dx,dy,dyaw)`，保持query-body参考系，并记录对应期望时域。当前十点头仍作基线；当PCT只消费第十点时，前九点可以是辅助监督但不是实际via-point执行。[C13]

保留三种解释：源/模型选择的目标A、规划端点B、真实到达C；下游适合操作的名义站位G仅用于评价。若连续末段完成验证，可形成B_exec，同时保留B_raw和原snap结果。原0.10m检查的失败不能通过把B改写为A隐藏；新修复执行必须独立版本化。

面向下游操作能力的站位选择值得研究，但只有实际C→Mani对照后才能判定其贡献。Mobi-π提供政策站位适应性的理论动机与开源实验框架，不证明当前任务必须采用其3DGS优化器。[R08]

### 6.2 Mani：先保留joint+gripper，不拆更多专家

PICK/PLACE共享Mani，以当前任务条件区分。比较5Hz×10与25Hz×50，保持2秒物理预测时域；固定0.4秒执行更新时，分别消费2点与10点。若高分辨率改变计算量，报告真实耗时，不称等算力。

clip-safe的关节边界不直接保证碰撞安全。夹爪有界clean输出和FK监督是两个分别消融的候选，不与主架构首轮打包。

$$\mathcal L_{FK}=\sum_k\|p(FK(q_t+\hat a_k))-p(FK(q_t+a_k^*))\|^2_{W_p}+\lambda_R\,d_{SO(3)}^2(\hat R_k,R_k^*).$$

这是基于机器人运动学的项目设计，不是要求另设EEF action head，也不是把物体中心当作所有时刻的TCP目标。[R14]

## 7. VLM—action耦合与depth候选

### 7.1 稳定语义与实时信息分离

建议条件为：

$$C_t=[Z^{task}_n,Z^{live}_t,E(s_t),E(A^{overlap}_t)].$$

高层语义随任务更新；当前RGB/可选depth及本体状态随观察更新。动作token对C做cross-attention，保留动作内部self-attention。

初版仍可由同一Qwen读取当前图像，只是不每次重新生成全计划。更快的目标架构才考虑共享视觉主干/轻量几何分支直接给action expert。缓存任务文本不等于缓存旧图像；旧观察的KV不可无条件视为当前视觉。[R03] [R04] [R05] [R13]

### 7.2 2×2输入/耦合实验的准确定义

|  | C0：单一条件流 | C1：任务语义+实时条件分流 |
|---|---|---|
| RGB | Qwen最后层token作动作条件 | 稳定任务token加当前RGB特征；无depth |
| RGB-D | depth经相同几何编码器投影后，在动作前合并成一个条件bank | 相同depth编码器与RGB当前特征单独作为live条件bank |

四组共享合法本体状态和动作历史。C0的RGB-D不是把伪彩depth直接交给未经适配的RGB模型。额外token数和编码器容量尽量匹配，剩余差别明确报告。

深度实验优先验证wrist局部几何；head depth对近场NAV另作后续对照。理想深度、部署合理噪声/孔洞/标定扰动和缺失深度分别评价。单目估计depth单列，不当作新增独立传感。[R10] [R11]

## 8. 阶段P0：建立可研究基线，而不是无限等待无故障系统

### 8.1 两个不同基线名称

`B-old`：旧checkpoint+旧runtime，作为历史记录。

`B0-runtime`：旧checkpoint和原normalizer不变，修正确定性的输入/执行语义，统一导航停止与到达及安全监控。其权重仍受历史训练数据影响，不能称干净重训模型。

`B0-learned`：在新可信数据上按冻结训练recipe训练的共同架构，用于数据、输入和expert实验。不能把它与B0-runtime混为同一权重。

### 8.2 最低入场条件

源目标能够被合理执行；观测/动作码本及时间合同明确；真值不进入普通策略；硬安全错误被拦截；评分unknown不伪造成功。已知停止间隙必须修正后再用于比较NAV架构。

0.5% saturation失败可在限制生效、证据完整的研究基线中继续报告诊断；它仍阻止最终部署验收。不能要求先达到完整任务高成功率才允许研究架构，也不能把研究许可写成门槛通过。

## 9. 阶段P1 / V-01：高层与低层主架构因子实验

### 假设与矩阵

主要假设：H2减少任务漂移/恢复失败，L1改善跨块接续；两者在完整episode中可能互补。采用H0/H1/H2 × L0/L1，共六组。

L0与L1都以H=10、Δ=0.2s、同一0.4s期望更新周期为主比较。另加完整十点2s同步执行为参考，不冒充L0同频对照。

H0/H1/H2使用同一planner权重φP和相同反馈估计器；低层θL固定。H2初始计划固定与H1相同。已知案例仅回归，不参与总体能力估计。

### 样本与预算

20个开发任务家族，每个正常+一种预先确定扰动，共40条件。六组共240次rollout；若加长块参考则增加40次。所有数目是DESIGN，不是完成记录。

扰动按家族预分层，包含小幅目标变化、一次可恢复空抓、携物事实失效、局部路径暂时受阻。外部干预日志进入评价，不直接进入策略输入；只有真实传感可看到的后果才能被普通planner使用。

### 指标

主指标为完整transfer成功（几何/严格物理分开）。次指标为错误ADVANCE、遗漏/重复任务、修复次数、恢复完成率、闭合实际执行时间、跨块关节变化、任务时长、调用数和成本。

高层与低层交互以配对任务计算：

$$\Delta_{int}=(p_{H2,L1}-p_{H2,L0})-(p_{H0,L1}-p_{H0,L0}).$$

这是概率差的描述，不是天然显著性。按task family聚类bootstrap，正常和扰动分层报告。若H2仅优于H0而不优于H1，主要收益可能是显式任务状态而非后缀编辑。

### 停止/晋级

出现跨任务旧chunk执行、GT反馈泄漏、未知被标成功或安全监控漏拦时，停止该配置并保留失败。若仅局部轨迹变平滑但完整任务不改善，RTC不能作为已验证能力贡献。若差异不确定，增加新家族覆盖，不通过不断修改同一批案例阈值获得显著结果。

## 10. 阶段P1补充 / V-02：共享前缀与任务事实反例

已有A/B/C结果不重复包装为新实验。在同一首块及相同前两点后增加D：真实重叠条件化。A旧尾部继续、B新预测只记录、C独立替换作为对照。

主观察窗为剩余八点1.6s。统一追加2s末目标保持时，仅观察已发生夹持的持续性，作为单独指标；不能混入主窗或允许某一组继续规划追赶。

逐query记录闭合索引、绝对预计闭合时间、实际闭合时间及闭合前几何。000006检验无效延迟；000024防止把旧计划当永远正确；000030检验有效替换是否被新机制破坏。

宏观反例：PICK曾完成但物体后续掉落；上一回答声称完成但无执行证据；未来任务编辑不改变当前target；当前target改变必须取消旧epoch。先做合成schema回归，再做物理反例，不能将合成通过算真实恢复成功。

## 11. 阶段P2 / V-03：数据覆盖与训练修正实验

### 数据对照

在文档一D-normal-300与D-task-300上比较相同B0-learned结构。两个数据集都排除明确失效监督；无需为了科学对照主动重建一个大规模已知污染数据集。

单独需要量化“旧污染”的贡献时，可保留B-old作为历史诊断；它与新训练的初始化、样本和runtime变化必须如实列出，不把整体差值全部归因于清洗。

### 训练日程

初期冻结Qwen，适配新动作输入/输出与任务adapter；之后采用小学习率分阶段解冻。高层CE与低层动作损失使用各自有效样本mask，不用无标签progress头制造监督。

已正确标注的自产高层prefix可混入action训练；错误route不能配给原expert动作标签。历史计划扰动、遗漏和stale facts需要合法的新目标标签，否则只作检测/拒绝样本。

保持固定优化步数、effective batch、动作事件采样预算，保存验证轨迹。旧“2个数据等效epoch”是旧实验配置，不是所有新数据的收敛定理。[C08]

先单训练种子筛选；候选再扩三训练种子。采用相同解码时钟和runtime，完整任务/恢复为主指标。若恢复提高但正常任务明显回退，分析覆盖和采样权重，不直接宣布整体胜出。

## 12. 阶段P3 / V-04：Action expert消融

| 子实验 | 只改变什么 | 固定什么 | 主要结果 |
|---|---|---|---|
| V-04a | NAV十点主输出 vs endpoint主输出 | 同一任务条件、可信数据、PCT/DWA | 实际到达C、完整handoff成功 |
| V-04b | 5Hz×10 vs 25Hz×50 | 2s时域、0.4s更新、同任务数据 | 接近/闭合/释放与计算代价 |
| V-04c | 推理期RTC vs训练期前缀条件化 | 网络规模、训练数据、延迟分布 | 闭环成功、延迟、承诺一致性 |
| V-04d | 有界clean-gripper | 其他输出与损失 | 裁剪幅度和实际抓取，不只事件率 |
| V-04e | 单独FK一致性损失 | joint输出与主FM | 任务空间误差、完整操作成功 |

高分辨率H变化会影响位置embedding、动作长度和strict-load，应明确允许重置的边界，不以静默部分加载混作同一checkpoint。

训练期RTC与推理期RTC不必同时启用。原论文提供机制，但M0的scalar-to-token time改造是本项目实现工作；需要解析目标和梯度回归。[R01] [R02] [C09]

## 13. 阶段P4 / V-05：耦合与depth的2×2

采用第7节四组，使用在V-01中选定且冻结的宏微观pipeline。相同训练数据的RGB-D记录允许为RGB组屏蔽depth，从而匹配任务分布。

验证集以物体相对位置、桌面高度、相机视角、近场遮挡和depth退化分层。诊断性oracle几何可以估计信息上界，但只能作为额外oracle组，不能替代RGB-D部署组。

至少报告：三维接近误差、释放位置、完整episode、depth失效时退化、推理p50/p95、参数/token数量。depth-shuffle或全无效mask只作“是否利用正确几何”的检查，不新增复杂方法。

若仅完美仿真depth改善而退化depth无改善，不宣布解决真实空间感知；若仅开环改善，需继续解释闭环差异。

## 14. 阶段P5 / V-06：实际交接与自主route分解

前置条件：源NAV实际到达验证已经通过，能够生成可信C。比较标准源PICK初态、学习NAV实际完整交接状态、同状态下自主route三组。底盘、关节、物体、速度和相机历史都按真实状态处理，不把A/B/G当成C。

先10个任务家族；未到达的任务记在系统分母，不能为了条件PICK统计而从整任务结果消失。

若标准状态表现好而真实交接差，再研究面向Mani成功域的目标选择、导航末段对齐与handoff扰动数据。[R08] 若固定route好而自主route差，再优化高层事实估计和prefix训练，不把action expert一起推翻。

## 15. 阶段P6 / V-07：从双服务原型迁移到共享VLM

选出最小有效组合后，才将高层planner和低层视觉条件迁移到共享Qwen或共享视觉编码器。

保持task schema和action codec不变；高层使用低频任务token，动作分支读取当前live特征。共享参数更新可能影响原action条件分布，必须单独fine-tune并重新验证，不把双服务结果直接当共享网络结果。

比较双服务原型、共享VLM目标系统、以及必要时的轻量live encoder系统。报告完整任务表现、显存、平均与p95延迟、高层排队对动作队列的影响。低频高层不得饿死动作推理；缓存键必须包含模型/adapter、观察、任务epoch与标定身份。

## 16. 阶段P7 / V-08：完整验收、实时和泛化

先完成无模型完整源命令/派生命令回放，再完成选定系统的自主episode。固定底座锁、抓取辅助、相机和物理版本；无辅助、辅助、去底座锁分别报告。

最终建议100个新任务家族×3个有效环境随机种子×2个主要方法=600次/训练种子。若重复三训练种子，总计1800次。只有确实引入独立随机性时才把重复视为随机种子覆盖；相同确定性运行不增加有效样本。预算不足时缩减并如实报告不确定性，不使用旧开发案例冒充最终测试。

先正常/扰动完整任务，再让物理在推理期间持续推进，叠加预注册延迟和输入退化。检查剩余动作队列覆盖是否足以跨越实际延迟；p95不是硬实时保证，长尾超时要有安全行为。

移动目标与更多物体/场景作为后续独立任务版本。当前良渚静态可乐通过不自动证明动态抓取、whole-body并行控制或跨场景泛化。

## 17. 统一评价规范

### 17.1 三层结果，不能混淆

`geometry_transfer_success`：按冻结几何任务合同完成搬运和放置。  
`strict_full_success`：在几何任务完成基础上，必要接触/支撑证据充分且满足严格合同。缺测为unknown。  
`deployment_gate_passed`：任务、接口、安全、来源及saturation等发布门整体通过。

高层`FINISH`是模型决策，不是评价真值；模型提前结束属于错误完成。评价器在任务完成但planner不结束时也要记录额外动作和损伤风险。

### 17.2 开环评价

保留专家状态、关键边界、学生访问状态三类。joint MAE/NAV ADE只作诊断；同时报告错误route导致的动作覆盖、FK误差、闭合事件时间和计划编辑正确性。hold-tail与真实future分别统计。

### 17.3 统计与有效性

配对任务、固定预算、按任务家族聚类bootstrap；训练种子另报方差。18个局部窗口、同episode不同prefix和50Hz相邻帧不当独立任务。

未发生闭合的窗口报告右删失/未发生比例，不只对闭合成功子集求平均。初始窗口不足以观察1s持有时标“观察不足”，不当作严格失败。

系统分母包含启动失败、安全中止、无合法控制；条件策略分母可另报，但不可替代系统成功率。strict unknown保留覆盖率和可判定率，不能直接记0或删掉后报告成功率。

建议在初筛前预注册最小有意义改善，例如完整任务+5个百分点、正常任务允许回退不超过5个百分点；这些是候选决策尺度，不是安全阈值，也不是保证样本足够检出。正式采用同时要求机制、整体效用和成本均可解释。

## 18. 推荐排程与研发工作包

| 阶段 | 首要交付 | 并行项 | 不等待什么 |
|---|---|---|---|
| P0 | B0-runtime、数据合同、统一目标到达和评分层次 | planner schema、深度留存 | 不等待全部历史失败解释 |
| P1 | V-01六组主实验、V-02衔接反例 | 高层任务适配与来源审计后处理 | 不等待大规模重训 |
| P2 | V-03数据对照 | V-04a NAV接口候选 | 不引入新backbone |
| P3/P4 | V-04动作、V-05输入/耦合 | 接触覆盖、迁移回放 | 不同时堆叠所有候选 |
| P5/P6 | V-06真实handoff、V-07共享系统 | 新验收集准备 | 不用局部成功替代组合 |
| P7 | V-08完整/实时验收 | 外部复现 | 不先承诺真机能力 |

资源先测10条rollout的真实墙钟，再估算批次时间。表中不是GPU运行承诺，也不授权后台启动或改变现有采集进程。

## 19. 怎样决定最终保留的架构

若H1已覆盖大部分收益而H2无额外收益，采用固定任务骨架而不是为了论文叙述保留复杂planner。若RTC只改善局部平滑而损害恢复，缩小作用范围或不采用。若RGB-D在部署退化下仍有效，再保留几何分支。若局部目标head已经足够，不保留没有执行价值的路径输出。

主候选始终是：**持续任务状态、可修复剩余计划、与已承诺动作相容的生成、实时可观测条件。**完整episode收益与失败恢复决定是否采用，参数量或方法数量不决定研究价值。


## 来源与引用索引

来源核查日期：2026-09-06。用户仓库链接锁定上述提交；外部代码如链接到main/master，开工时仍需锁定SHA。本文不分发代码权重、场景、原始数据或外部图片。

### 项目代码与已发布实验

**C08｜当前VLA生效配置**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/configs/manipulation_navi_v1.json)。ABot初始化、Qwen、双DiT、5Hz目标、损失、关闭项与saturation门。

**C09｜当前M0 DiT实现**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/dit.py)。clean-action训练/采样、scalar time、self/cross-attention与future tokens。

**C11｜当前模型、动作与语言合同**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory.py)。H=10、0.2s间隔/历史、四route、canonical subtask、无DONE。

**C13｜Isaac/PCT/DWA执行连接**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory_system.py)。NAV端点消费、实际执行时钟、joint顺序执行与到达配置。

**C14｜已发布执行接口v2实验文档**  
[来源](https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/docs/execution_interfaces_v2_20260906.md)。reset缓存反例、源/部署回放、条件PICK及接触未知；不含用户后续新实验的完整工件。

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

**R08｜Yang et al. Mobi-π: Mobilizing Your Robot Learning Policy (2025)**  
[来源](https://arxiv.org/abs/2505.23692)。操作策略与初始底盘/视角适配。 [官方代码](https://github.com/yjy0625/mobipi)。

**R09｜Huang et al. ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation (2024)**  
[来源](https://arxiv.org/abs/2409.01652)。阶段、反馈与回退的结构参考；本方案不采用其整套约束优化。 [官方代码](https://github.com/huangwl18/ReKep)，关注`main.py`。

**R10｜Ke, Gkanatsios, Fragkiadaki. 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations (2024)**  
[来源](https://arxiv.org/abs/2402.10885)。RGB-D空间特征与动作生成依据；本文不复用其性能数字。 [官方代码](https://github.com/nickgkan/3d_diffuser_actor)。

**R11｜Ze et al. 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations (2024)**  
[来源](https://arxiv.org/abs/2403.03954)。紧凑3D表示与diffusion policy的相关依据。 [官方代码](https://github.com/YanjieZe/3D-Diffusion-Policy)。

**R13｜Chi et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (RSS 2023 / IJRR 2024)**  
[来源](https://diffusion-policy.cs.columbia.edu/)。动作序列生成与观察/预测/执行时域设计参考。 [官方代码](https://github.com/real-stanford/diffusion_policy)。

**R14｜Lynch, Park. Modern Robotics: Mechanics, Planning, and Control (2017)**  
[来源](https://modernrobotics.northwestern.edu/)。第3—5章坐标、FK和Jacobian；第13章移动操作。 [官方配套代码](https://github.com/NxRLab/ModernRobotics)。

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


[C08]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/configs/manipulation_navi_v1.json
[C09]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/dit.py
[C11]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory.py
[C13]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/src/conveyor_bench/conveyorvla/joint_trajectory_system.py
[C14]: https://github.com/lemonoscar/Dynamic-manipulation/blob/bf5d5ab1130cc4a91f9cf4396b275d3b52e1da31/docs/execution_interfaces_v2_20260906.md
[R01]: https://arxiv.org/abs/2506.07339
[R02]: https://arxiv.org/abs/2512.05964
[R03]: https://arxiv.org/abs/2504.16054
[R04]: https://arxiv.org/abs/2602.11236
[R05]: https://arxiv.org/abs/2506.01844
[R08]: https://arxiv.org/abs/2505.23692
[R09]: https://arxiv.org/abs/2409.01652
[R10]: https://arxiv.org/abs/2402.10885
[R11]: https://arxiv.org/abs/2403.03954
[R13]: https://diffusion-policy.cs.columbia.edu/
[R14]: https://modernrobotics.northwestern.edu/
[K01]: https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py
[K02]: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
[U01]: #source-u01
[U02]: #source-u02
[ENG]: #source-eng

