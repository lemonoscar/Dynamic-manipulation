# Agent 交接：接近可乐并抓起

交接时间：2026-09-09，Asia/Shanghai。现场核查时间：14:09。
用户已要求本 Agent 完成交接，后续工作由其他 Agent 接手。

## 先读：当前目标与授权

1. 当前任务是 **接近可乐 → 抓住并稳定抬起**，只保留 `NAV_TO_SOURCE → PICK`。
   用户明确要求为此新建分支；不再要求携物导航、放置才能算本任务成功。
   原完整搬运代码和数据保留，不能把简化任务结果当成完整搬运能力。
2. 用户暂时不要 depth：不采 depth、不把 depth 加入训练输入。本轮也未启动采集。
3. **工作服务器仅为 10.130.130.37（node04）**。4×H20 已弃用，不再连接、
   写入、训练或依赖其在线服务。用户说“废弃”没有被执行为删除旧服务器数据。
4. 用户支持 Agent 自主协作和完整访问权限；不要反复申请已有授权。
   但 H20 的 GPU2/3 授权不等于 node04 的 GPU2/3 授权。新的训练/物理测试
   需要明确 node04 的可用 GPU、并发和有限预算，不能占用他人资源。
5. 旧 8 小时训练预算、24 条/6 小时 demo 搜索预算均是历史运行预算，不能复用。
   用户曾明确终止训练；本次仅创建新任务分支、核查数据及交接，**没有启动新训练**。
6. 推理此前要求只有 60 秒仿真任务时间作为正常终止媒介，加载和暂停推理不计入。
   新分支沿用该规则；抓起时记录首次达标时间，继续 PICK 到 60 秒。
   因而评分是“曾稳定抓起”，不是“第 60 秒仍持物”。变更需明确版本和语义。

## 唯一工作入口

- 连接：`ssh 10.130.130.37`，沿用已有配置；实际为
  `dzb_xhq@10.130.130.37:23698`，hostname `node04`。
- 可写工作区域：`/hdd1/dzb_xhq/`，不得写他人目录或修改系统环境。
- 当前仓库：`/hdd1/dzb_xhq/VLA/ConveyorVLA-approach-grasp-20260909`。
- 分支：`feat/approach-grasp-task-20260909`。
- 交接文档前的准确 HEAD：`ad061d5aa42e9eb48568a2c2911bbc77d3476ea3`。
  此提交已同步本地、GitHub 和 node04；本 HANDOFF 会另产生文档提交。
- GitHub：<https://github.com/lemonoscar/Dynamic-manipulation/tree/feat/approach-grasp-task-20260909>。
- 该仓库为隔离仓库，迁移快照没有被覆盖。Git 保留本地主控的浅历史边界
  `92dfddc5d12b345daa5e8a88964a79153841825d`；不是完整历史克隆。
  首次 bundle 导入因浅历史父对象缺失失败，随后传入原 shallow 边界并重试成功；
  工作文件和 HEAD 已核对。不要用 reset/clean 或修改历史来“修复”这个已知浅克隆。
- 同一工作树只允许一个写入负责人；并行代码工作使用隔离 worktree，统一集成。

## 已实现什么，尚未证明什么

主要实现：`scripts/run_full_episode_diagnostic.py`。
新增 `--mode approach_grasp --timer-only`，上下文文件
`configs/approach_grasp/task_context.json`，默认指令：
`Approach the cola can, grasp it, and lift it stably.`

- NAV 阶段允许模型提出 NAV→PICK 切换；PICK 为末任务，不再请求后续切换。
  服务错误返回终端 ADVANCE 时记录并忽略该切换，合法低层动作仍可消费。
  NAV 完成依然是模型声明，`completion_verified=False`，不能伪称物理证据。
- 成功独立只读评分：抬起 ≥ 4 cm、TCP 距离 ≤ 8 cm、连续 ≥ 1 s、
  相对位置漂移 ≤ 1 cm、相对旋转漂移 ≤ 0.15 rad、最大采样间隔 0.05 s。
  同一窗口内实际主夹爪和生效命令均需 ≤ 0.5 open fraction，且有原 physical pick 事件。
  要求评分来源有效、60 秒正常闭合、无固定物体约束或中途物体重置。
- `summary.task_contract=approach-grasp-v1`，简化任务结果在 `approach_grasp`/`success`；
  `full_task_success=null`。严格双指接触证据缺失为 unknown。
  已声明的 MANI 底盘/支持关节锁仍存在，不声称完全无辅助。
- 实现提交 `41d5887`，15 项 CPU 回归在 **旧 H20** 通过；本轮未在 node04
  重跑这些测试，未做新任务 Isaac 物理测试，也未训练该任务。
  测试入口及旧日志路径见 `docs/approach_grasp_20260909.md`。

**当前 runner 还不能直接在 node04 启动**：仍有 H20 的 GPU3、RTX ordinal、
`/diff/wallx_workspace/dzb` 资产缓存限制等冻结绑定。不要只替换 CUDA_VISIBLE_DEVICES
就当完成部署迁移。源码已同步不等于环境、GPU 或仿真路径已经验收。

## 数据：最短可用路径

详细统计见 `docs/approach_grasp_data_20260909.md`。
审计目录：`/hdd1/dzb_xhq/master_integration_20260908/approach_grasp_data_audit_20260909/`。

### 当前 116 条 RGB，优先构造专用视图

已迁移的派生 release：
`/hdd1/dzb_xhq/migration-4xH20-20260908/dzb/integration_runs/rgb_full_episode_20260908_v1/release`。
原始 episode 位于：
`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/`。

| 划分 | 原 episode | NAV/PICK 动作 query | NAV→PICK 标签 | 成功抓起前段候选 |
|---|---:|---:|---:|---:|
| train | 90 | 2755 | 176 | 71 |
| validation | 13 | 366 | 26 | 10 |
| test | 13 | 415 | 26 | 11 |

92 条候选都具有接近/抓起教师事件、两阶段动作及一对 CONTINUE/ADVANCE。
其中 15 条后来搬运失败，成功抓起前段仍有价值。N/B/R 候选为 71/12/9。
若仅取这 92 条，预计 train 为 2176 个动作 query + 142 个切换标签，
validation 289+20，test 355+22。这是待构造视图的数量，不是现成新任务发布。

**不能直接按 route 过滤就开训**：

- 92 条均在 pick_success 之后才创建携物固定约束，按墙钟顺序已核实；
  两个事件可能在同一仿真 tick。没有所选 query 落在约束创建后，
  但 35 个原有效未来应用区间跨过边界（train 30、validation 5），要重新 mask。
- 旧教师成功事件不等于新分支连续 1 秒门；需逐帧重新评分，不能把候选写成
  已通过新门的无辅助示范。接触标定在所有 116 条上均为 not_calibrated。
- 缩短指令和任务后缀，更新 parent memory identity/hash；保留 NAV→PICK 标签，
  排除 PICK→NAV_TO_TARGET，不能编造 FINISH。输入不可泄漏未来教师阶段/真值。
- family 划分保持不交叉；normalizer 继承或 train-only 重拟合须独立版本化。
- 原 release 仍含 `/diff/...` 绝对路径，不要修改原发布或伪装哈希未变。
  使用新派生目录和显式映射。

目标机已实际验证：5 个派生文件哈希匹配、116 个 UUID/family 匹配、232 个原
manifest/事件文件哈希匹配；14,304 张所需 RGB 全部存在；抽样解码 24 张通过。
没有重新全量解码或重验所有 raw 控制。
`node04_verification.json` 包含逐 episode 目标路径映射；
`inventory.json` 是切换服务器前已完成的历史只读统计，其旧路径不是当前入口。
release manifest SHA256：`4e5aaff4a65878e873add2d153f447f6e639bbc9537498e66ca98cd2a0856391`。

### 350 条原始 50 Hz，可作为扩充来源

`/hdd1/dzb_xhq/modelscope_uploads/liangzhuNeW_350_50fps`。
实际 manifest 350 条、35 个 tar、170,625,382,400 字节。
快照统计 734,668 个样本，双 RGB、11 维控制、frames 状态、任务事件与 summary。
快照记载旧成功门与 50 Hz 审计通过；本轮只重新核查清单、归档存在和首条元数据。
不是对全部 350 条重新验收。

用户之前提到的 350 条就是这套；之前仅查 H20 派生目录而遗漏，现已纠正。
优先适配它扩充数据，但不要直接伪装成 raw-control-v2：需要根据 frames 中实际
控制报告确认 issued/held 可恢复范围、按名字/冻结标定绑定关节、保留未知事实。
原录制 samples 不是每通道事务合同。不得用双夹爪平均替代 joint7 命令。

另有 `/hdd1/dzb_xhq/modelscope_uploads/liangzhuNeW_500`，实际 500 条、20 个 tar。
这是独立旧批次，旧派生模型视图含双指平均和 legacy 时间语义，不能直接混训。

## 权重、环境和此前结论

迁移根目录：`/hdd1/dzb_xhq/migration-4xH20-20260908/dzb`。

- `integration_runs/rgb_full_episode_20260908_v1/trained/best.pt`：step1700，
  本轮现场确认 19,061,369,297 字节。
  已知源 SHA256 `474d03ff22da939f763574bf434c75226a2a2533ae984ced80e25e34a1df7fa0`。
- 同目录 `last.pt`：迁移/停止记录为 step2600，57,184,124,843 字节。
  不要把 best1700 和 last2600 混称，也不要把 optimizer step 当 epoch。
- normalizer ID：`eb5b5fa3116b595e0e65539b7b9d482a84882971c5b78d31a9cc8f7b05a10ae1`。
- 老模型架构/tokenizer：迁移根目录下
  `training_runs/conveyorvla-abot-m0-liangzhunew500-5hz-formal-20260905-r1/checkpoints/step_002414`。
- 源训练从 ABot-M0 系初始化，Qwen 和两个动作专家全量微调；FP32 主参数、BF16 autocast，
  不是 LoRA。step1700 本轮数据 90 train episode，7896 个 action/transition query，
  batch8，约 1.72 次数据遍历。尚无新简化任务权重。
- 迁移 Python 文件存在：
  `ConveyorVLA/artifacts/.conda-envs/conveyorvla-al0-lerobot044/bin/python`。
  复制环境的 shebang/editable 安装和旧绝对路径尚未全面验证；不要全局安装或盲目改包。
- 目标机既有 `/hdd1/dzb_xhq/VLA/dynamic-isaaclab-5.1-20260804` 存在；
  本轮未验证其与旧模型/仿真版本一致。源采集 Sim6 与旧诊断 Sim5.1 不是相同初态复现。
- 完整迁移报告：`/hdd1/dzb_xhq/migration-4xH20-20260908/MIGRATION_REPORT_20260909.md`。
  报告记录关键 best/last ZIP CRC 校验通过；本轮未重新计算两个大 checkpoint 的完整哈希。

此前 24 条训练 seed、每条 60 秒仿真的旧 step1700 搜索：0 个合格抓起 demo，
多数有模型任务切换但无实际抓起。seed16100021 是失败视频，不是成功示范。
权重、条件与失败归档见 `docs/demo_search_20260909/`。

用户怀疑 qpos 压过视觉、双相机权重失衡：已做真实视觉诊断，但未证实该根因。
小样本 PICK 中 RGB 换图影响大于换 q；front/wrist 各 300 visual token，
影响随阶段和输出维度变化，不能据此定固定权重。
实现了默认关闭的 `mani_state_dropout` 训练选项；0.3 只是待试候选，未训练验证。
不要默认启用或宣称视觉问题已修复。见 `docs/mani_visual_conditioning_20260909/`
和 `docs/mani_camera_conditioning_20260909/`。

## 接手顺序与并行边界

以下是工作分工建议，尚未派发或启动任何子任务：

1. **数据 Agent**：只写新的数据派生工具/发布目录。先处理 116 条中的 92 条候选，
   完成边界 mask、成功门重评分、二任务上下文/标签、路径重绑定和黄金样本。
   输出真实 eligible 数量、排除原因、文件哈希；350 条适配随后独立版本化。
2. **运行环境 Agent**：独立 worktree，仅处理 node04 Python、模型加载、Sim 和设备配置。
   保留旧实验身份，不热改运行 worker。先 CPU preflight；GPU 测试前明确资源/预算。
   当前 runner 校验服务 global_step==1700；未来新模型需要明确更新身份合同。
3. **训练/集成 Agent**：在数据与环境明确后，统一输入/标签/normalizer/checkpoint
   合同，实际真实数据前后向、保存/重载，再按新授权启动有界训练和验证。
   不必等所有 350 条适配才开展小闭环，也不能把数据读取通过当成策略有效。

优先阅读文件：本 HANDOFF → `docs/approach_grasp_data_20260909.md` →
`docs/approach_grasp_20260909.md` → `scripts/prepare_full_episode_rgb.py` →
`scripts/train_full_episode_vla.py` → `scripts/run_full_episode_diagnostic.py`。
所有具体实现仍应核对当前 HEAD 和 dirty 状态；不要覆盖接手后其他 Agent 的新证据。

## 交接时运行状态

本 Agent 未在 node04 启动 GPU 任务、训练、模型服务或物理推理。
14:09 检查当前用户进程，未发现本任务 train_full_episode / serve_full_episode /
run_full_episode / approach_grasp 进程。该检查不表示整台服务器没有其他任务。
此前资源盘点看到 A40/3090 混合卡且均有计算进程；接手时必须重新核查归属。
没有本 Agent 留给接手者的待续训练 PID、采集预算或无限自动重试任务。
交接后本 Agent 停止推进训练和实现，等待用户新的明确指令。
