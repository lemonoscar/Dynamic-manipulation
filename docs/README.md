# ConveyorVLA 文档索引

更新时间：2026-08-20。

## 当前权威合同

- [Waypoint Policy v1](conveyorvla_waypoint_policy_contract_v1.md)：已审核通过，是下一轮
  数据、模型、训练和推理实现的唯一目标合同；当前代码尚未实现。
- [Benchmark 规范](benchmark.md)：任务、场景和成功语义。
- [数据格式与质量门禁](data.md)：现有 canonical/LeRobot 数据背景；waypoint 新 schema
  以 Waypoint Policy v1 第 9 节为准。

## 实施依据与历史诊断

- [step 002500 闭环失败分析](step_002500_closed_loop_failure_analysis_and_remediation.md)：
  说明为何直接速度/TCP-delta、提示词和闭环链路需要更换。
- [Seen 子任务数据分析](seen_subtask_data_analysis_and_remediation.md)：记录 0.20 秒视觉、
  phase boundary、history 泄漏和导航姿态整改证据。
- [开环动作质量标准](open_loop_action_quality_standard.md)：旧动作合同的诊断基线；
  waypoint v1 实现后需按新动作空间扩展，不得原样套用阈值。
- [双 DiT 路由提案](vlm_routed_dual_dit_proposal.md)：历史架构讨论；若与已批准的
  Waypoint Policy v1 冲突，以后者为准。

## 仓库与操作背景

- [架构说明](architecture.md)：采集/runtime 与旧模型架构，已标注版本范围。
- [采集、训练与测评操作](operations.md)：旧合同复现命令，不能启动 waypoint v1。
- [当前状态与下一步](status.md)：2026-08-13 采集阶段快照，不是 waypoint 实施状态。
- [版本迁移与兼容策略](history.md)：单一 live tree 与 schema 升级原则。

`handoff_private/` 保存机器路径、进程和 Agent 交接信息，严格本地私有并被 Git 忽略；
公开文档不得链接或复制其中内容。
