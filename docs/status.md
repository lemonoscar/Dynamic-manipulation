# 当前状态、证据与剩余门禁

最后复核：2026-08-21 00:36 CST。现行 runtime/eval 代码基线：
`feature/conveyorvla-waypoint-v1@121512903667e16578525ec22dcfb2d0deca92e5`；正式
checkpoint 的训练 source 仍为 `724ead21be2c27d9b40c200375ee4ab49ccedc84`。

## 1. 总结

Waypoint Policy v1 的无 state 数据、两次完整 Qwen、双 Layerwise FM head、训练、
checkpoint、开环、单卡 inference、PCT/DWA、真实 cuRobo 和模型自主管理的 Isaac rollout
均已有可执行实现与分层证据。

正式 4×H20 训练已按用户指令主动暂停：最后有效 optimizer step 为 1181，最后完整
checkpoint 为 step 1000。以后新启动的训练默认每 500 effective optimizer steps 保存，
而不是 1,000 steps。

step 1000 已完成四卡 load 和完整 diagnostic 开环。route/格式表现通过，但 NAV/ARM
动作质量明显不足。真实无辅助 Isaac 闭环已启动并发出一条严格无 state 的模型请求；
模型选择 `NAV_TO_SOURCE` 后，其第 18 个 navigation segment 超出 yaw 安全限值，执行器
按合同 fail-closed。因此本轮闭环是“测试链路完成、模型能力未通过”，不能声明 episode
成功。三路视频已经校验并下载到本地 Git 忽略目录。

详细证据见 [step 001000 开环与真实 Isaac 闭环评测](checkpoint_step1000_evaluation_20260821.md)。

## 2. 门禁总表

| 门禁 | 状态 | 证据/边界 |
|---|---|---|
| 无 state 数据构建与 audit | 通过 | 522 episodes、119,700 rows；state field/tensor=0；manifest/normalizer hash 冻结 |
| Pass 1/Pass 2 与双 Layerwise FM | 通过（静态/训练） | 两次完整 Qwen、模型自产 prefix、NAV/ARM 均有非零梯度 |
| 80-step 小样本 overfit | **未通过** | route 未过旧诊断的置信度门禁；不能被正式训练替代为“通过” |
| 正式长训健康启动 | 通过后暂停 | step 1–1181 有效；用户授权 `SIGINT`，当前无训练进程 |
| 后续 checkpoint 间隔 | 已修改 | 默认和操作命令均为 500 step，commit `aa06479` |
| step 1000 四卡 checkpoint load | 通过 | world size=4，5,021,782,540 partition values，non-finite=0 |
| step 1000 route/格式开环 | 通过 | val 40 rows；5 类各 8；accuracy=1.0，RECOVER/invalid=0 |
| step 1000 action 开环质量 | **未通过可用性判断** | NAV/ARM OOB、segment/step violation 和 pose error 偏高；diagnostic profile 没有严格质量阈值 |
| inference export + 实际服务 | 通过 | tied Qwen export 已修；约 21 GB export；单卡四图 request 成功 |
| PCT/DWA reference navigation probe | 通过（已知 waypoint） | 无 fallback、有限有界 DWA、首目标后 requery |
| 真实 cuRobo known-pose | 通过 | reachable/collision-free，41 点 path，position/orientation error 约 `6e-8` |
| 完整自主 Isaac 测试链 | 已执行 | 无 GT phase/FSM/route gate；真实模型 query 和三路视频齐全 |
| 完整自主 Isaac 成功 | **未通过** | 首个预测 NAV chunk 的 segment 18 超出 yaw limit，被安全拒绝 |
| oracle-route / 四阶段 staged rollout | 未完成 | 本轮优先执行用户要求的正式开环和完整自主闭环，不外推未跑门禁 |

“通过（结构/接线）”只覆盖表中明确层级，不向 waypoint 数值质量或完整 episode 成功外推。

## 3. 正式训练与暂停点

| 项目 | 值 |
|---|---|
| host | `4xH20`，实际 `VM-0-3-ubuntu` |
| run | `/diff/wallx_workspace/dzb/runs/conveyorvla-waypoint-v1-formal-724ead2-s10000-20260820T1813` |
| source | clean `724ead21be2c27d9b40c200375ee4ab49ccedc84` |
| train rows | 108,603，全量、无 subset |
| batch | micro 3/GPU × 4 GPU × accumulation 2 = global 24 |
| precision / sharding | bf16 / DeepSpeed ZeRO-3，无 CPU offload |
| last valid event | step 1181，2026-08-20 23:07:20 CST |
| durable checkpoints | step 20、step 1000 |
| current process state | 已停止；无训练 tmux/process/GPU allocation |

step 1181 仍为有限有效 event：total loss `0.6851`，route loss `0.1662`，NAV/ARM loss
`0.1195/0.3433`，VLM/NAV/ARM gradient norm `438.57/20.25/10.59`，
`valid_optimizer_step=true`。尾部 traceback 是用户授权 Ctrl-C 引起的 `KeyboardInterrupt`。

step 1001–1181 共 181 个计算 step 未写入 checkpoint。`train_waypoint.py` 当前没有
optimizer-resume CLI；step 1000 已验证可加载做评测，但不能称为已经验证的无损续训点。

## 4. step 1000 开环

四卡评测使用 val 40 rows、batch size 2、diffusion seeds 17/29/43/71：

| 指标 | 结果 |
|---|---:|
| route accuracy / RECOVER / invalid | 1.000 / 0 / 0 |
| NAV ADE / FDE | 0.3959 m / 0.7522 m |
| NAV direction accuracy / segment violation | 0.625 / 0.8125 |
| NAV normalization OOB | 0.671875 |
| ARM position / orientation error | 0.1195 m / 2.1501 rad |
| ARM gripper accuracy | 0.87265625 |
| ARM normalization OOB / step violation | 0.953125 / 1.0 |
| missing action / non-finite | 0 / 0 |

报告是 `diagnostic` profile；它的 `quality_pass=true` 仅表示该 profile 没有启用 overfit
质量阈值。不得用这个布尔值掩盖动作误差和违规率。

## 5. planner、闭环与视频

真实 cuRobo 服务绑定 arm-vla `388b6818` 和 cuRobo `8726021`，输入
`query-base-B_t`，planner frame 为 `curobo-planner-base`，orientation fallback 关闭。
known-pose 真实规划通过，说明 cuRobo code、CUDA runtime、frame transform、IK 和 joint
path 返回链路可用；它不证明模型预测的任意 ARM target 可达。

最终闭环 `autonomous_seed861_r3` 使用 Liangzhu full visual、PCT、DWA、
`pct_multifloor` locomotion、真实模型服务与三路录像。首条 request 的输入为完整任务和
head/wrist 两时刻共四图，`model_state_fields=0`。Qwen 返回 `NAV_TO_SOURCE`，置信度
0.98594；执行器在 PCT 前拒绝超限 yaw segment。最终：

- `query_count=1`，`control_steps=58`；
- `state_trace=[NAV_TO_SOURCE]`；
- `failure_reason=navigation_waypoint_rejected:navigation segment 18 exceeds yaw limit`；
- `success=false`，`external_fsm_used=false`；
- overview/front/wrist 分别为 28/29/29 帧，全部通过解码、hash 和抽帧检查。

视频、trace、report、日志与派生场景存放在本地
`artifacts/evaluation/waypoint_step001000_20260820T231424/`，由 `.gitignore` 排除。

## 6. 本轮实现修复

- `aa06479`：默认 checkpoint interval 1,000 → 500；
- `23afff4`：cuRobo source root 与 arm-vla workspace/assets root 分离；
- `13f6e87`：按 Transformers tied-weight 声明导出 Qwen safetensors；
- `1215129`：rollout 复用并 capability-gate 已启动的 Waypoint cuRobo 服务，禁止回退到
  legacy `8765` 服务。

第一次闭环因 scene fallback arc 身份不匹配而停止；第二次因 Omniverse Vulkan 与
`CUDA_VISIBLE_DEVICES` 映射不一致而停止；第三次分别通过派生 scene binding 和取消该
mask 解决启动问题，最终暴露的是模型 waypoint 质量问题。所有失败尝试均保留，没有覆盖
旧证据。

## 7. 可声明与不可声明

可以声明：训练已按用户要求暂停；后续默认每 500 step 保存；step 1000 四卡 load、
route/格式 diagnostic、inference export/service 和真实 cuRobo known-pose 通过；真实
无辅助 Isaac 测试及三路录像已完成。

不可声明：step 1000 动作质量通过、模型已收敛、oracle-route 或四阶段 staged rollout
通过、模型 ARM target 已通过 cuRobo、或完整自主 episode 成功。安全门拒绝是正确行为，
不能通过放宽 yaw/workspace/rate 限制来改写结论。

## 8. 若继续推进

1. 明确选择重新长训还是实现并验证同源 checkpoint resume；无论哪种路径都保持 500-step
   保存间隔；
2. 先用同一 40-row/四 seed 协议确认 NAV/ARM 动作质量改善；
3. 再补 oracle-route planner 和四阶段 staged rollout；
4. 最后重跑完整自主 episode，要求逐 query trace 覆盖 PCT/DWA 与 cuRobo/IK，并取得
   物理成功，而不只是测试链可启动。
