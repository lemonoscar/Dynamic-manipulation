# 当前状态、证据与剩余门禁

最后复核：2026-08-20 18:54 CST。现行实现：
`feature/conveyorvla-waypoint-v1@724ead21be2c27d9b40c200375ee4ab49ccedc84`。

## 1. 总结

Waypoint Policy v1 的数据 schema、无 state 模型、训练、checkpoint、开环评测、
runtime protocol、PCT/DWA 与 cuRobo adapter、单卡服务和模型自主管理的 rollout loop
均已进入公开分支。4×H20 正式 10,000-step 长训已从干净提交启动，并通过前 20 个连续
有效 step、四条非零梯度路径、四卡利用率和 step 20 checkpoint 提交门禁。

本轮实施里程碑按“健康启动正式长训”判据关闭。该里程碑不改变批准合同中的质量门禁：
目前不能声明 overfit 通过、模型已收敛、正式 checkpoint 已通过开环、真实 cuRobo
已通过、或 Isaac 无辅助闭环成功。

## 2. 门禁总表

| 门禁 | 状态 | 证据/边界 |
|---|---|---|
| 无 state 数据构建与 audit | 通过 | 522 episodes、119,700 rows；state field/tensor=0；manifest/normalizer hash 已冻结 |
| 模型与协议静态测试 | 通过 | 9 个 Waypoint 测试文件、49 tests passed，commit `724ead2` |
| Pass 1/Pass 2 与双 Layerwise FM 接线 | 通过（静态/训练） | 两次完整 Qwen、模型自产 prefix、NAV/ARM 均有反向梯度 |
| 4GPU checkpoint load | 通过（诊断 checkpoint） | 早期 step 3 与 80 的 ZeRO binding/load gate 通过 |
| 80-step 小样本 overfit | **未通过** | route 未过置信度门禁；动作 loss 有改善但不足以替代 route |
| PCT/DWA reference navigation probe | 通过 | 已知 0.60 m body waypoint、无 fallback、有限有界 DWA、首目标后 requery |
| 正式长训健康启动 | 通过 | step 1–20 连续有效，四条 loss/gradient/LR 有限，step 20 checkpoint 完整提交 |
| 正式 checkpoint load/open-loop | 未执行 | step 20 已落盘，但不能沿用诊断 checkpoint 的结论 |
| 单 GPU train smoke | 未执行 | 四卡训练已验证，不等于单卡路径验证 |
| inference export + 实际 checkpoint 服务 | 未执行 | 代码/静态测试通过；未完成真实 consolidation/request |
| 真实 cuRobo known-pose smoke | 未执行 | adapter/service 静态测试通过；不能声明实际 planner 可用 |
| oracle-route planner rollout | 未执行 | rollout loop 已实现，尚无真实 episode evidence |
| 分阶段/完整自主 Isaac 闭环 | 未执行 | 无三视角视频/trace/success evidence |

“通过（静态/训练）”只覆盖表中明确写出的层级，不向真实 planner 或仿真闭环外推。

## 3. 正式训练

| 项目 | 值 |
|---|---|
| host | `4xH20`，实际 `VM-0-3-ubuntu` |
| tmux | `cvla-wp-formal-724ead2-s10000` |
| run | `/diff/wallx_workspace/dzb/runs/conveyorvla-waypoint-v1-formal-724ead2-s10000-20260820T1813` |
| source | clean `724ead21be2c27d9b40c200375ee4ab49ccedc84` |
| train rows | 108,603，全量、无 subset、无 resume |
| schedule | 10,000 steps，warmup 200 |
| batch | micro 3/GPU × 4 GPU × accumulation 2 = global 24 |
| precision / sharding | bf16 / DeepSpeed ZeRO-3，无 CPU offload |
| checkpoint | step 20，之后每 1,000 step |

step 20 的代表事件：

| 指标 | 值 |
|---|---:|
| total / answer / decision / active-route loss | 1795.844 / 9.576 / 0.688 / 1.386 |
| NAV / ARM loss | 730.274 / 1053.920 |
| Qwen / NAV / ARM gradient norm | 9,959,407 / 5,404,630.5 / 1,316,593.75 |
| optimizer step | `valid_optimizer_step=true` |

step 20 checkpoint 约 60 GB，四个 model shard、四个 optimizer shard、trainer state、
scheduler、random state、ZeRO consolidation script 和 checkpoint manifest 均已提交。
训练在 checkpoint 后继续；18:54 CST 只读复核时已到 step 453，四个训练进程仍在，
四张 H20 各约 49–50 GiB 显存，最近 event 仍为有限非零梯度。瞬时 GPU utilization
为 33–100%，step 持续推进。

该快照是运行时证据，不是持续监控承诺。训练 worktree 固定在 `724ead2`；后续文档
commit 不同步进正在运行的 worktree。

## 4. 数据与可复现性

正式 run 绑定：

- dataset：
  `/diff/wallx_workspace/dzb/datasets/derived/conveyorvla-waypoint-v1-full-8fcccd9`；
- dataset manifest：
  `0db6169d726b2165a90ec6e833403666179eb68135248af5681de92a400ec957`；
- normalizer：
  `75a60ba125a83383f1d00ef4151933a77c796faee5d5c559364310cb64acfca0`；
- policy config：
  `bbf5ab2cf44391e73c98ace3c2ef990aab4076244c41bec2ba260c58403827ce`；
- Qwen base 文件逐项 SHA-256、token IDs、完整 argv、环境、hostname 和 clean dirty-state
  记录在 `resolved_run.json`。

raw n200+n400 保持只读；dataset、run、checkpoint、日志和视频均不进入 Git。

## 5. 80-step overfit 结果

32-sample、80-step 诊断没有通过严格 overfit profile：

- step 80 decision loss 约 0.624；
- active-route loss 约 1.346；
- self-conditioned 样本因 route confidence 低于 0.55 全部进入 `RECOVER`；
- NAV/ARM 动作 loss 相比初始化明显改善。

这说明训练/梯度/动作 head plumbing 可运行，但不能证明模型在小样本上学会自主 route。
正式长训是按明确决定直接启动的，不应把“给更多训练时间”写成“overfit 已通过”。
正式 run 的 `lambda_self` 在 5% 进度（step 500）前为 0，早期健康 step 也不能回答
self-conditioned route 是否可用。

## 6. 已知问题与警告

- `scripts/audit_training_events.py` 仍解析 legacy event 字段，不能直接验证 Waypoint
  v1 的 `answer_loss/decision_loss/active_route_loss/lambda_self`；正式前 20 step 使用
  独立 JSONL 检查。
- Torch 无法创建 run-local kernel cache 时禁用了 kernel caching；这是非致命性能警告，
  未造成训练中断。
- 0815 source 只有 head/front 与 wrist，没有第三视角；数据 review clip 不满足合同的
  三视角闭环视频要求。
- PCT/DWA 已跑真实 reference code 的合成地图 probe；cuRobo 目前只有接口与静态测试，
  两者证据层级不同。
- 代码完成不代表 arm-vla/curobo 环境、场景碰撞、frame calibration 和 joint controller
  已在完整 episode 中联调。

## 7. 若继续推进

建议严格按以下顺序补齐证据：

1. 在 step 500 之后审计 self-conditioned route/RECOVER 计数，并对正式 checkpoint
   运行四卡 load gate；
2. 运行正式 checkpoint 的 diagnostic 与 overfit/open-loop profile；
3. 完成 inference export 和单卡服务真实 request smoke；
4. 完成 cuRobo known-pose、frame transform、碰撞与误差 gate；
5. 运行 oracle-route NAV/ARM planner rollout；
6. 运行四阶段 staged rollout；
7. 最后运行无 GT phase、无外部 FSM 的完整自主 Isaac episode，并保存 head、wrist、
   第三视角视频及逐 query trace。

## 8. 可声明与不可声明

可以声明：Waypoint v1 已实现；数据 audit 和 49 项静态测试通过；4×H20 正式长训从
干净、绑定的数据/配置/提交健康启动，前 20 step 与首 checkpoint 有证据。

不可声明：80-step overfit 通过、正式模型收敛、正式 checkpoint 开环通过、cuRobo
实际规划通过、planner rollout 通过、或真实仿真闭环成功。旧教师 episode 和旧
state28 checkpoint 的成功/失败也不能作为 Waypoint v1 的结果。
