# ConveyorVLA 文档索引

更新时间：2026-08-28。Waypoint v1/v2 的旧数据、checkpoint 和合同继续作为冻结复现基线；
新的 direct-joint successor 已在 `Manipulation_Navi_v1` 分支完成代码候选、Isaac 接口接线
和启动级门禁，并审计了 4 条 Gate-A review episode，但尚无冻结的正式数据 release、overfit、
正式训练或真实控制闭环证据。review 样本只证明 schema/时序可审计性，不代替正式规模与覆盖
门禁。
远端实时状态必须在任何操作前重新核验，本文档索引中的历史 GPU 状态不得用于资源判断。

## 权威性顺序

出现冲突时按以下顺序判断：

1. [Waypoint Policy v1](conveyorvla_waypoint_policy_contract_v1.md) 与
   [Waypoint Policy v2](conveyorvla_waypoint_policy_contract_v2.md) 分别决定其冻结数据、
   checkpoint 和 runtime 的复现语义；新方案不得原地改写它们。
2. [Joint-Trajectory 训练改进方案](conveyorvla_joint_trajectory_training_improvement_plan.md)
   决定 `Manipulation_Navi_v1` successor 的目标模型、训练和 runtime 语义。
3. [Joint-Trajectory 数据采集规范](conveyorvla_joint_trajectory_fresh_data_collection_spec.md)
   决定 successor 的 raw/derived 数据、采集速度、随机化和质量门禁。
4. 当前 Git 代码和 resolved config 只证明已经实现的行为；目标方案在代码、测试和 manifest
   完成前不得宣称已落地。
5. [当前状态](status.md) 只记录已有证据和未通过门禁，不修改合同。
6. 本索引中标为“历史”的页面只用于解释来源和失败，不得作为现行启动命令或接口规范。

合同中实施前写下的“尚未实现”等状态句应按第 5 项读取；冻结合同的动作、
输入和执行决议本身没有被更改。

## 已批准的 successor 目标

- [Joint-Trajectory 训练改进方案](conveyorvla_joint_trajectory_training_improvement_plan.md)：
  四 route、无 DONE、NAV `[10,3]@0.20s`、Mani direct-joint `[10,7]@0.04s`、Mani-only
  13D state、M=1/10-step inference、分层 global batch 64 和约 2 个数据等效 epoch。
- [Joint-Trajectory 数据采集规范](conveyorvla_joint_trajectory_fresh_data_collection_spec.md)：
  1,600 条首版成功 episode、全新 immutable schema/manifest、applied joint target、采集
  随机化、速度与质量门禁；同时记录 2026-08-28 Gate-A 审计结果，并冻结由 raw 时序推导的
  `K=0 boundary/success_tail` 完整 hold 规则。
- [Manipulation_Navi_v1 代码实施报告](manipulation_navi_v1_code_implementation_20260827.md)：
  已实现文件、合成/回归验证、明确未完成门禁和 fresh data 到达后的执行顺序。
- [Manipulation_Navi_v1 系统接线与启动级验证](manipulation_navi_v1_system_wiring_20260827.md)：
  NAV→PCT/DWA、direct-joint/continuous-gripper、raw recorder、远端旧 run 停止和真实
  stage/reset smoke；明确 Vulkan/RTX 与真实 control loop 尚未通过。

## 现行文档

- [Waypoint v2 阶段切换执行与长训计划](waypoint_v2_stage_transition_execution_plan.md)：
  已批准执行；冻结 v1 基线，从全新 v2 schema 开始，依次验证 train/inference suffix
  语义、边界进度、动态 prefix、PRTS 方法启发的局部 CRL、训练 FM Monte Carlo sample
  `1→4` 和 on-policy correction，并以证据选择正式长训组合。PRTS 权重不在范围内。
- [架构说明](architecture.md)：模型、协议、planner/executor 与旧采集 runtime 的边界。
- [数据格式与质量门禁](data.md)：无 state waypoint schema、冻结数据身份、split、hash
  和 legacy canonical 数据边界。
- [操作手册](operations.md)：数据构建、四卡训练、checkpoint/open-loop、单卡服务、
  PCT/DWA、cuRobo 和 rollout 命令。
- [当前状态](status.md)：lookahead 闭环复测、训练停止状态和未完成门禁。
- [step 002000 lookahead 选点策略完整自主闭环复测](checkpoint_step2000_lookahead_evaluation_20260821.md)：
  可信前缀/目标 lookahead/PCT 选点、正面可见 seed、18 次真实导航、ARM target 2 failure
  和三路 74.2 s 视频。
- [step 002000 原始 arm-vla 规则闭环复测](checkpoint_step2000_arm_vla_reference_evaluation_20260821.md)：
  历史首点基线；额外导航门控与原始 reference 规则的边界、首点容差问题和短视频。
- [step 001000 开环与真实 Isaac 闭环评测](checkpoint_step1000_evaluation_20260821.md)：
  四卡 load、动作质量、StarVLA/动力学复核、executor 修复及两组三路视频证据。
- [Benchmark 规范](benchmark.md)：任务、场景、时钟与成功定义；不等同于模型闭环结果。
- [版本迁移与兼容策略](history.md)：旧 `state28 + direct action` 到 Waypoint v1 的
  不兼容边界和 Git 历史。

## 历史合同与诊断

- [Liangzhu seen dense-transition 合同](liangzhu_seen_dense_transition_contract.md)：
  Waypoint v1 之前的 `state28 + velocity/TCP-delta` 数据/训练合同。
- [Seen 子任务数据分析](seen_subtask_data_analysis_and_remediation.md)：旧 step 7000
  数据、phase boundary、history 泄漏和导航姿态问题。
- [step 002500 闭环失败分析](step_002500_closed_loop_failure_analysis_and_remediation.md)：
  旧直接动作策略的真实失败证据。
- [step 003000 中间检查](checkpoint_step3000_intermediate_20260815.md)：更早的旧
  checkpoint 与 scheduler 诊断。
- [开环动作质量标准 basic-v1](open_loop_action_quality_standard.md)：旧动作空间标准；
  不可直接用于 Waypoint v1。
- [VLM 路由双 DiT 提案](vlm_routed_dual_dit_proposal.md)：被已批准 Waypoint 合同取代的
  早期设计。

`handoff_private/`、`artifacts/`、数据、checkpoint、日志和视频是本地/远端运行
材料，不属于公开文档，不得提交或从公开页面链接为必需依赖。
