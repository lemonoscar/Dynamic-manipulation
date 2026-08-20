# ConveyorVLA 文档索引

更新时间：2026-08-21。现行 runtime/eval 实现基线：
`feature/conveyorvla-waypoint-v1@121512903667e16578525ec22dcfb2d0deca92e5`；正式
step 1000 checkpoint 的训练 source 为 `724ead21be2c27d9b40c200375ee4ab49ccedc84`。

## 权威性顺序

出现冲突时按以下顺序判断：

1. [Waypoint Policy v1](conveyorvla_waypoint_policy_contract_v1.md) 决定模型输入、两次
   Qwen forward、route 语法、动作 shape/坐标、planner 边界和门禁语义；它是已批准且
   冻结的规范正文。
2. `configs/waypoint_v1.json`、`src/conveyor_bench/conveyorvla/waypoint*.py` 和
   `scripts/*waypoint*.py` 是现行可执行实现。
3. [当前状态](status.md) 只记录哪些实现和门禁已有证据、哪些仍未通过，不修改合同。
4. 本索引中标为“历史”的页面只用于解释来源和失败，不得作为现行启动命令或接口规范。

合同中 2026-08-20 实施前写下的“尚未实现”等状态句应按第 3 项读取；合同的动作、
输入和执行决议本身没有被更改。

## 现行文档

- [架构说明](architecture.md)：模型、协议、planner/executor 与旧采集 runtime 的边界。
- [数据格式与质量门禁](data.md)：无 state waypoint schema、冻结数据身份、split、hash
  和 legacy canonical 数据边界。
- [操作手册](operations.md)：数据构建、四卡训练、checkpoint/open-loop、单卡服务、
  PCT/DWA、cuRobo 和 rollout 命令。
- [当前状态](status.md)：训练暂停点、step 1000 开环/闭环证据和未完成门禁。
- [step 001000 开环与真实 Isaac 闭环评测](checkpoint_step1000_evaluation_20260821.md)：
  四卡 load、动作质量、真实 cuRobo、闭环失败原因及三路视频 hash。
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
