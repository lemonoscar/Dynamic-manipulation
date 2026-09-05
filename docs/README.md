# ConveyorVLA 文档索引

更新：2026-09-05。当前分支 `Manipulation_Navi_v1` 已完成 ABot-M0 Joint-Trajectory 5 Hz
正式训练与全量 validation/test 开环。裁剪事件率未通过 ≤0.5% 门槛，迁移闭环仍在运行。
历史文档中的“尚未训练”、Mani 0.04s 或不同动作头语义仅适用于其注明版本。

## 当前开发与实验入口

- [开发指南](joint_trajectory_development_guide.md)：模块地图、输入/动作/时钟、安装、训练、测试、外部工件与修改边界。
- [正式训练与评估实验卡](experiments/formal_5hz_20260905.md)：固定模型、完整开环 95% 区间和带时间戳的闭环快照。
- [正式评估操作](formal_joint_trajectory_evaluation.md)：验证集冻结、test 门禁、服务、任务准备、辅助物理与失败归因。
- [架构分析](joint_trajectory_architecture_analysis_20260905.md)：源码和轨迹证据、待验证原因、下一轮对照实验。
- [贡献与提交规则](../CONTRIBUTING.md)：测试、工件排除、不可变实验、分支与提交边界。
- [SPCGVLA 设计草案](SPCGVLA/README.md)：独立未来模型合同；感知探针不等于点云已经接入正式模型。
- [停止与保存设计草案](joint_trajectory_stop_and_save_pending_design.md)：pending design，不能当现有能力调用。
- [2026-09-04 数据/训练准备记录](manipulation_navi_liangzhunew500_readiness_20260904.md)：此次 5 Hz 适配来源，后续完成状态以实验卡为准。

- [开发快照发布记录](repository_release_20260905.md)：变更范围、兼容边界、验证、工件及仓库维护缺口。

## 合同和证据冲突时

当前生效实现由 [5 Hz 配置](../configs/manipulation_navi_v1.json)、对应 schema、实际模型构建路径
和 checkpoint/dataset/source manifest 共同确定。开发指南解释其落点，实验卡只报告已有证据。
旧 Waypoint v1/v2、早期 0.04s direct-joint 计划和 SPCGVLA 草案各自保留其语义，不能混用。
任何新设计在代码、测试和绑定证据齐备前均不宣称已实现；改合同需新配置/schema/协议和迁移说明。
状态页不修改冻结合同，历史 GPU 占用不用于判断当前资源。

以下保留旧索引以便追溯。各页面标题中的“当前”“正式”应结合该页版本和日期阅读。

## 历史：2026-08-28 successor 目标

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

## 历史与其他合同文档

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
