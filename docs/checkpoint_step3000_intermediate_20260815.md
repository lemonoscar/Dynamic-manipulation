# ConveyorVLA AL0 `step_003000` 中间检查

> 历史 checkpoint 记录：本文对应 2026-08-15 的旧 state28/direct-action 训练，不能
> resume、转换或用于评估 Waypoint v1。现行正式训练见 [status.md](status.md)。

## 结论

`step_003000` 是本轮训练最后一个完整 checkpoint。训练损失持续下降，未出现 NaN、OOM、NCCL 或 checkpoint 损坏；四阶段验证集自由生成达到 64/64 正确。旧训练已停止，3001–3080 仅存在于日志而未保存。

续训从该 checkpoint 开始，保持有效全局 batch 为 16，将单卡 micro-batch 从 1 提至 2、梯度累积从 4 降至 2。学习率仍使用 200 步 warmup、余弦衰减和 10% 下限，但修复 Accelerate 在四卡下每次全局更新推进 scheduler 四次的问题。

## 工件与数据

- 训练 checkpoint：`runs/conveyorvla-al0-seen-two-pass-20260815-r3-4gpu/output/checkpoints/step_003000`
- 训练数据：`datasets/derived/conveyorvla-pct-seen-subtask-20260815-r1`
- 数据规模：373 episodes，74,168 rows；train/val/test 为 65,930/2,947/5,291 rows
- 切分单位：episode，验证帧不会与训练帧来自同一 episode
- 评测：验证集四阶段等量抽样，每阶段 16 rows，共 64 rows

## 训练学习情况

以下数值是 `events.jsonl` 中每 10 个 optimizer steps 记录一次后，按 500-step 窗口求均值；action loss 含 diffusion 随机噪声，因此观察趋势而不是单点。

| optimizer steps | subtask loss | action loss | gradient norm |
|---|---:|---:|---:|
| 1–500 | 0.799406 | 0.445455 | 4.10912 |
| 501–1000 | 2.86978e-5 | 0.233286 | 2.83611 |
| 1001–1500 | 8.06665e-6 | 0.139434 | 1.62403 |
| 1501–2000 | 3.31852e-6 | 0.0794635 | 1.24834 |
| 2001–2500 | 1.72580e-6 | 0.0674778 | 1.04948 |
| 2501–3000 | 1.57360e-6 | 0.0647282 | 0.713112 |

这说明四阶段语言监督已经收敛，动作拟合仍在改善但明显更难；仅凭训练损失不能代表闭环成功率。

## Held-out 离线评测

四卡在验证集上按阶段各取 16 rows，结果如下：

| 阶段 | rows | 自由生成正确 | subtask loss | action loss |
|---|---:|---:|---:|---:|
| NAV_TO_SOURCE | 16 | 16 | 1.46217e-6 | 0.247607 |
| PICK | 16 | 16 | 9.55865e-7 | 0.00903043 |
| NAV_TO_TARGET | 16 | 16 | 1.65403e-6 | 0.108914 |
| PLACE | 16 | 16 | 2.07126e-6 | 0.0114844 |
| **总计/均值** | **64** | **64（100%）** | **1.53583e-6** | **0.0942590** |

四阶段 confusion matrix 为严格对角矩阵，invalid rate 为 0。PICK/PLACE 的离线动作损失较低，两个导航阶段仍明显更难，其中 NAV_TO_SOURCE 是下一阶段应重点关注的动作域。

第一次评测曾显示 64/64 生成无效。原始文本诊断证明每条答案的第一段阶段文本都正确，但模型在 `<|end_subtask|>` 后继续重复生成；原因是推理没有把已监督的结束分隔符注册为停止 token。将 `<|end_subtask|>` 设置为 generation EOS 后，同一 checkpoint、同一 64-row 样本得到 100% 正确，严格 parser 本身没有放宽。修复还将评测主体耗时从 206.24 秒降至 87.27 秒。

该评测检查两次 VLM 前向中的子任务生成和 teacher-forced action objective，不是 Isaac Sim 无辅助闭环成功率。它适合在改 batch 和 scheduler 前确认 checkpoint 可载入、阶段路由已学到且动作目标没有退化。

## Scheduler 缺陷与修复

checkpoint 的 `trainer_state.json` 是 global step 3,000，但 `scheduler.bin` 为：

- `last_epoch = 12000`
- `_step_count = 12001`
- 六组学习率已经全部到达各自 base LR 的 10% 下限

原因是 Accelerate 默认在未 split batch 的四卡训练中，将一次 scheduler 调用推进 `world_size` 次。修复包括：

1. 关闭 Accelerate 的隐式多进程 scheduler 补偿；
2. 仅在 `sync_gradients` 时推进一次 scheduler；
3. resume 时以 `trainer_state.json` 的 global step 为真值校正 scheduler 状态和 optimizer learning rates；
4. 将修复前后的步数及学习率写入 `scheduler_resume_alignment` 事件。

在 step 3,000，正确的余弦 scale 是 `0.8305704108`；校正后六组学习率为 `[1.66114e-6, 8.30570e-6, 1.66114e-5, 8.30570e-5, 1.66114e-5, 8.30570e-5]`。这是原配置范围内的恢复，但相对错误状态会出现上调，所以续训启动后必须监控 loss、gradient norm 与有限值。

## 续训合同

- GPU：4 × H20，ZeRO-3，bf16，无 CPU optimizer/parameter offload
- micro-batch：2/GPU
- gradient accumulation：2
- effective global batch：`2 × 2 × 4 = 16`（与旧训练相同）
- resume global step：3,000
- target global step：10,000
- 首次验收：至少连续完成 20 个新 optimizer steps，无 OOM/NaN/NCCL，四卡均持续参与，记录显存与吞吐

micro-batch 2 是基于旧训练每卡约 44–45 GiB、H20 总显存约 98 GiB 的保守提升。若实际峰值 OOM，将回退到 micro-batch 1，而不会修改有效全局 batch 或模型结构。
