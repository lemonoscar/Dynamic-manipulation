# ConveyorVLA Waypoint Policy v2 合同

- 合同版本：`conveyorvla-waypoint-policy-contract-v2-command-gripper-s4`
- 分支：`waypoint-v2`
- 状态：正式双卡训练运行中；source commit 与 run identity 已冻结
- 基线：Waypoint v1 和 legacy v2 数据/checkpoint 均保持只读

## 1. 模型输入与 route 所有权

模型输入只包含完整任务文本和 `head/wrist[t-0.20s,t]` 四张图。模型 request、loader batch
和 runtime 控制链均不得包含 robot state、GT phase、operation、previous subtask、物体 truth
或 simulator target。

推理保持两次完整 Qwen forward：

1. Pass 1 受约束生成 `ACTION/DONE`、route 和 subtask；
2. Pass 2 使用模型自己的完整 assistant prefix 和相同视觉重新运行 Qwen；
3. route 只来自 Pass 1。辅助 boundary/progress、prefix、CRL 或评测 truth 均不得覆盖 route。

## 2. 动作合同

| domain | 输出 | stride | frame / semantics |
|---|---|---:|---|
| NAV | `[20,3]` | 0.60 s | query-body waypoint `[dx,dy,dyaw]` |
| MANI | `[20,7]` | 0.20 s | query-base absolute TCP `[x,y,z,r,p,y,gripper]` |

MANI gripper 使用专家命令语义：raw `0=close,1=open`，normalizer 中为 `-1/+1`。测得手指
开度只可用于 episode 起点 held-command 初始化和评测，不得作为 action target。

NAV runtime 只在 `min(predicted_K,10)` 内按现有几何规则选取 waypoint，再运行 PCT/DWA。
MANI runtime 每次 query 只按时间顺序规划并执行 `target0`；后续 19 点只作 FM 训练和开环
审计。当前 target 无可用规划或 chunk timeout 时，底盘/机械臂保持安全零动作并重新观察。
协议、shape、有限值、workspace、gripper 范围、collision/IK 和关节步长检查继续 fail-closed。

MANI route 下底盘速度始终强制为零。不得恢复已删除的 local fatal navigation stall，也不得
增加外部 phase/FSM、距离门控或 truth gate。

## 3. 数据合同

| 项目 | 冻结值 |
|---|---|
| schema | `conveyorvla-waypoint-dense-transition-v2-command-gripper-v1` |
| transform | `conveyorvla-waypoint-v1-to-v2-terminal-hold-command-gripper-v2` |
| episode / row | 522 / 119,700 |
| train / val / test | 108,603 / 5,771 / 5,326 |
| manifest SHA-256 | `6f534e1b7ed456ab6595985d7148eea5e9ff214d4e6a308c5e34baa93fa2506f` |
| normalizer SHA-256 | `e781bfed2661befa77dc13cdc3d4a7b88a77ee2678562fc952089f6cc307dc4a` |
| state field / tensor | 0 / 0 |

真实 boundary suffix 使用 full-horizon terminal-hold，保留 original `K*`；source-tail 和
episode-tail 不伪装成 boundary。旧/new 119,700 row 已确认除 schema/provenance 和 ARM
gripper channel 外无差异。旧 v2 仍可审计，但不得和本合同 checkpoint 互相 resume。

## 4. 晋级组件与 loss

本 run 使用最小组合：

- terminal-hold；
- corrected B2 soft route/boundary/progress；
- command-gripper 标签；
- FM Monte Carlo `S=4`。

learned prefix、local CRL 和 on-policy correction 关闭。NAV runtime 的 trusted-prefix 10
是固定执行上限，不是启用 learned prefix head。

S4 定义为一次 Qwen forward 后，对每个真实 action chunk 采四组相互独立的
`(noise_m, flow_time_m)`：

```text
L_FM^(4) = (L_FM^1 + L_FM^2 + L_FM^3 + L_FM^4) / 4
```

四组梯度共同进入对应 NAV/ARM FM head 和共享 Qwen 表征。推理
`action_model.num_inference_timesteps=4` 保持不变。用户已明确选择 S4 用于本 run；这不等于
已有证据证明 S4 优于 S1，后续仍须报告动作质量、gradient CV、吞吐和 GPU-hour 成本。

## 5. 训练合同

旧 `step_000500@7ec8424` 绑定 legacy measured-opening schema 和 optimizer moments，不是
本合同的合法 resume parent。本 run 从官方标准 `Qwen3-VL-4B-Instruct` fresh 初始化，使用
全新 run ID、output、resolved config、manifest 和 checkpoint identity。

| 项目 | 冻结值 |
|---|---|
| GPU | 物理 GPU 2/3，共 2 × H20；启动前逐卡确认无外部进程 |
| precision / sharding | bf16 / ZeRO-3，无 offload |
| micro batch | 2 / GPU |
| accumulation | 32 |
| global batch | `2 × 2 × 32 = 128` |
| max steps | 3,000 effective optimizer step |
| equivalent sampling epochs | `3000 × 128 / 108603 ≈ 3.5358` |
| warmup | 200 step |
| checkpoint | 每 250 effective optimizer step |
| config | `configs/waypoint_v2_b2_s4_command_gripper.json` |
| config SHA-256 | `a173ef0ec0ddb5e605f313f6759bf61ce0b26e2b214cdb105e5303e3771c043e` |
| distributed config | `configs/accelerate_zero3_2gpu_waypoint_gbs128_s4.yaml` |
| distributed config SHA-256 | `075ea150b7272cd94b44a4a0468047dbfc0bdac6b71b651f46e0383939d55f57` |
| training source | `waypoint-v2@571f306154aa30d0544bb14cd2754b0c6e6e6637` |
| run ID | `conveyorvla-waypoint-v2-b2-s4-command-gripper-full522-2gpu23-zero3-gbs128-571f306-s3000-20260824T002235CST` |

选择 micro 2 是为了给 S4 的四倍 action-head activation 留出显存；更大的 global batch 通过
accumulation 获得，不用冒险提高单卡峰值。GPU 0/1 的无关 StarVLA 保持不动，训练只暴露
物理 GPU 2/3。

## 6. 健康启动与后续门禁

最新用户指令要求直接启动 full-data run，因此 corrected-data 8–16 episode overfit 仍列为
未完成的科学门禁，不能因长训已启动而标记为通过。

启动后至少连续观察 10 个有效 optimizer step，要求：两 rank/GPU 2/3 参与；step 连续；
total、answer、route、NAV/ARM 四个 draw、boundary/progress loss、gradient、LR、吞吐和显存
均有限；无 OOM、NaN/Inf、NCCL error、traceback 或输出路径错误。达到门槛后停止监视，
训练进程保持运行。

2026-08-24 启动后，steps 1–10 连续健康审计通过，额外观察的 step 11 也正常。双 rank 分别
绑定物理 GPU 2/3；10 步 loss、梯度、LR、吞吐、显存和所有四组 FM draw 均为有限值，未见
failed event、OOM、NaN/Inf、NCCL error 或 traceback。详细证据见
[Waypoint v2 command-gripper S4 修正与正式训练启动](waypoint_v2_command_gripper_s4_launch_20260824.md)。

健康启动只证明训练系统正常。step 250 起仍需依次执行 checkpoint load、严格开环、完整
transition lag/crossover/flicker、NAV/ARM 20-step 图片、planner 和真实三视角闭环门禁。
