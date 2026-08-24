# Waypoint v2 self1500 修正与替代训练健康启动

日期：2026-08-24 CST

状态：替代 run 已连续通过 10 个有效 optimizer step，训练保持运行

## 1. 结论

用户要求停止在 step 152 就激活 self-conditioned assistant-prefix 分支的旧训练，把该分支
整体移到 step 1500 之后，再从头训练。实现已改为绝对 optimizer-step 调度：steps 1–1500
权重严格为 0，step 1501–2550 线性升至 0.5，之后保持 0.5。权重为 0 时，训练循环不会
执行自产 prefix decode、额外 Qwen forward 或 self-conditioned S4 FM forward。

这里的 assistant-prefix 分支不是 learned prefix `K*` head。learned prefix、local CRL 和
B5 on-policy correction 混合采样仍全部关闭；NAV runtime 的 trusted-prefix 10 上限也没有
改变。

## 2. 发现问题与修正

首轮 command-gripper S4 run 沿用按总进度 5% 开始的旧调度，在 max steps=3000 时于 step
152 激活。step 时间由约 60 s 增至约 305 s，而 step 152–160 尚无有效自产 route match。
用户在有效 step 238 终止该 run；没有 checkpoint，因此不得 resume。

公开修正 source 为 `4fb50ffa8f0a05eeda5d9dcc34a898658ba8d9f3`：

- 新增 immutable config `configs/waypoint_v2_b2_s4_command_gripper_self1500.json`；
- 训练入口支持绝对 step 的 `zero_until_step/linear_to_step/maximum` 调度；
- 用即将提交的 optimizer step 计算权重，避免边界错一位；
- 保留 legacy progress 调度兼容性，但本 run 只绑定 self1500 config；
- 增加 step 1、1500、1501、2025、2550 的边界回归与非法 max-step 校验。

配置 SHA-256 为
`f914462a34b210bc969386669594c3f23d07313c1cc31f8477f938b17bbf1401`。正式训练解释器缺少
`pytest`，没有临时安装或改动环境；改用同一正式 Python/代码入口完成调度、config delta 和
非法配置断言，并完成 522 episode、119,700 row 全量数据审计，问题列表为空。

## 3. 启动身份

| 项目 | 冻结值 |
|---|---|
| source | `waypoint-v2@4fb50ffa8f0a05eeda5d9dcc34a898658ba8d9f3`，clean |
| run ID | `conveyorvla-waypoint-v2-b2-s4-command-gripper-self1500-full522-2gpu23-zero3-gbs128-4fb50ff-s3000-20260824T113345CST` |
| initialization | 官方标准 Qwen3-VL-4B-Instruct fresh；无 resume |
| data | command-gripper v2，522 episode / 119,700 row |
| dataset manifest | `6f534e1b7ed456ab6595985d7148eea5e9ff214d4e6a308c5e34baa93fa2506f` |
| GPU | 物理 GPU 2/3，world size 2；GPU 0/1 无关任务未触碰 |
| batch | micro 2/GPU × 2 × accumulation 32 = global 128 |
| precision / sharding | bf16 / ZeRO-3，无 offload |
| FM | repeated diffusion steps 4；inference timesteps 4 |
| length / save | 3,000 effective optimizer step；每 250 step 保存 |

resolved run 已在进程启动前写入代码快照、Conda、`CUDA_VISIBLE_DEVICES=2,3`、两张卡的
精确 UUID、数据/config 哈希和独立 output identity。

## 4. 启动尝试审计

正式 run 前有两次未晋级尝试，均保留私有证据且未被伪装成成功：

1. 第一次启动命令组装错误，在创建训练 output 和占用 GPU 前退出；
2. 第二次数值训练到 step 11，但 resolved run 缺少代码快照、Conda 和 GPU UUID 字段。该
   run 被正常停止且无 checkpoint；没有事后篡改 artifact 来补身份。

最终 run 修正了启动前环境注入，resolved identity 完整后才进入健康门禁。

## 5. steps 1–10 健康证据

自定义审计精确覆盖 steps 1–10；通用连续窗口审计在训练继续运行时覆盖 steps 2–11，二者
均通过。

| 指标 | 结果 |
|---|---:|
| 有效 optimizer step | 10 / 10 |
| `lambda_self` | 全部 `0.0` |
| `self_conditioned_loss` | 全部 `0.0` |
| self NAV/MANI/match/mismatch/recover 计数 | 全部 `0` |
| optimizer step time，中位 | 58.17 s |
| throughput，中位 | 2.20 samples/s |
| total loss，中位 | 3,866.11 |
| gradient norm，中位 | 7,364,383.5 |
| reserved 显存峰值 | 64,408 MiB |
| 显存容量占比 | 65.8% |

每步 total、answer、route、NAV/ARM 四组 FM draw、boundary/progress、各梯度、LR、时间、
吞吐和显存均为有限值，且两 rank 分别绑定 GPU 2/3。未发现 failed event、OOM、NaN/Inf、
NCCL error、traceback 或进程退出。早期 loss 和 gradient 的 batch 波动只用于健康检查，不能
外推模型已经收敛。

## 6. 当前边界

达到 10-step 门槛后没有停止训练。首个计划 checkpoint 是 step 250；模型动作质量、阶段
切换、MANI 对准—闭合—抬升、planner 和真实三视角闭环仍须按 checkpoint 门禁验证。此次
结果只证明代码、身份、双卡分布式训练和 self1500 延后逻辑健康，不证明完整抓取成功。
