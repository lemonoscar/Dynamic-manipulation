# Waypoint v2 操作时序与夹爪监督修正

- 复核日期：2026-08-23 CST
- 实现提交：`8970dea82ca163c73e21ea272722993b04499898`
- 状态：代码已推送；新 immutable 数据已构建并通过全量审计；训练尚未启动
- 适用限制：后续所有 GPU 训练和评测只允许使用两张 GPU

## 1. 结论

“到达第一个 ARM target 后立刻闭合”的现象不是单纯训练不足，而是两个相互独立的问题
叠加：

1. 旧诊断执行器会跳过不可规划的 `target0`，直接执行同一 20-step chunk 中第一个可规划
   的后续 target；seed 147 实际跳到了 `target1`，因此一次 query 跨过了本应重新观察的
   时序边界。
2. 冻结的旧 v2 数据把 ARM 第 7 维标成测得的手指开度，但 runtime 把它解释为夹爪命令。
   抓住可乐后，物体会把手指撑开，所以“测得开度较大”并不等于专家仍在命令张开。

两处均已修正。runtime 现在每次 MANI query 只按时间顺序尝试 `target0`，规划不可用或本次
chunk 超时就安全停车并重新观察；数据则从 raw 中显式的专家 `gripper_command` 生成
`0=close, 1=open` 标签。未执行的 `target1..19` 仍保留作 FM 训练和开环质量审计，但不能
再在一次 query 中被跳选执行。

新数据的结构、来源、split、NAV、TCP pose、boundary、terminal-hold、`K*` 和 normalizer
连续量均已审计通过。由于夹爪监督语义和数据身份发生变化，旧 `step_000500` 不能使用
`--resume-from` 恢复 optimizer；正确下一步是 corrected-data overfit 门禁后启动一个全新
双卡全量训练。

## 2. 失败证据与运行时修正

seed 147 的历史 trace 中：

- `target0` 的预测夹爪值约为 `0.813`，但该 pose 无可行 cuRobo 规划；
- 旧诊断 selector 继续扫描 suffix，选择了可规划的 `target1`；
- `target1` 的夹爪值约为 `0.430`，被 runtime 解释为闭合；
- 因而第一次 MANI query 同时表现为“越过接近点并开始闭合”。

这不是正确的 receding-horizon 语义。每次 5 Hz query 应只执行当前时序上的第一个动作，
然后用新视觉重新判断是否继续 PICK、何时闭合以及何时抬升。

| 项目 | 旧诊断行为 | 当前行为 |
|---|---|---|
| MANI 候选 | 扫描 `target0..K-1` | 只尝试 `target0` |
| `target0` 无规划 | 跳到后续 target | 零动作并重新 query |
| 未执行 suffix | 可被同 query 跳选 | 只作训练/开环证据 |
| chunk timeout | 终止 episode | 安全停车并重新 query |
| 协议/schema/本地安全错误 | fail-closed | 仍然 fail-closed |
| MANI 底盘速度 | 零 | 仍强制为零 |
| trace identity | first-plannable diagnostic | `chronological_target0` |

已删除 `select_first_plannable_model_target` 配置和对应 CLI。新
`ArmPlanUnavailableError` 只把服务明确返回的当前 target 无可用规划转换为安全重询；格式、
有限值、workspace、gripper 范围、collision/IK 返回和执行期关节步长等既有安全检查没有被
放宽。此前已删除的 local fatal navigation stall 也没有恢复。

## 3. 原始数据审计发现

Liangzhu 0815 raw 同时包含：

- 物理测量值：手指实际开度；
- 专家控制值：`gripper_command=open|close|hold`；
- 对应 pipeline state 和 source row provenance。

旧 waypoint 派生器使用了第一项。全量 raw 审计发现，在专家已经命令闭合后，测量开度仍
可能较大：`exec_pick` 有 8,981 frame，`exec_nav_to_place` 有 53,411 frame。成功抓住物体
时这种差异是正常物理结果，却会把训练监督反转成“继续张开”。

522 个 episode 的显式命令转换模式完全一致：

```text
PICK/open_gripper:  close -> open
PICK/close_gripper: open  -> close
PLACE/open_gripper: close -> open
```

每一种转换均出现 522 次。专家 `PICK open→close` 时的相邻 TCP 位移为：median
`0.001390 m`、p95 `0.008470 m`、max `0.011140 m`。这直接证明专家是在 TCP 已基本到达
抓取 pose 后才闭合，而不是到达第一个远端接近点就闭合。

## 4. 新 immutable 数据身份

旧 v2 schema 继续冻结、可加载和可审计，只用于历史 checkpoint 对照。新训练必须使用：

| 项目 | 冻结值 |
|---|---|
| dataset ID | `conveyorvla-waypoint-v2-command-gripper-full-8970dea-20260823T144103Z` |
| schema | `conveyorvla-waypoint-dense-transition-v2-command-gripper-v1` |
| transform | `conveyorvla-waypoint-v1-to-v2-terminal-hold-command-gripper-v2` |
| config | `configs/waypoint_v2_b2_s1_command_gripper.json` |
| episode / row | 522 / 119,700 |
| train / val / test row | 108,603 / 5,771 / 5,326 |
| manifest SHA-256 | `6f534e1b7ed456ab6595985d7148eea5e9ff214d4e6a308c5e34baa93fa2506f` |
| normalizer SHA-256 | `e781bfed2661befa77dc13cdc3d4a7b88a77ee2678562fc952089f6cc307dc4a` |
| ARM gripper raw语义 | absolute expert command，`0=close, 1=open` |
| ARM gripper normalized | `-1=close, +1=open` |

命令生成规则为：显式 `open/close` 更新 held command，`hold` 或空值沿用上一条显式命令；
episode 起点只用首帧测得开度初始化 held state。测得开度之后不再作为 action target。
boundary terminal-hold 重复最后一个合法 pose 和命令，`original_valid_prefix_k` 不变。

## 5. 审计结果

| 门禁 | 结果 |
|---|---|
| 新数据完整 audit | `ok=true`，`problems=[]` |
| raw source command 可追溯 | 522/522 episode |
| state field / tensor | 0 / 0 |
| NAV round-trip max | `1.3322676295501878e-15` |
| ARM round-trip max | `4.2146848510894035e-08` |
| train max one-sided normalizer clip | `0.009977296468558237` |
| 旧数据后向兼容 audit | 522 episode、119,700 row、`ok=true` |
| 旧/新逐行差分 | 119,700 row，无非夹爪差异 |
| normalizer 差分 | 只改变 dataset schema 绑定 |
| 新真实 ARM target command | close 378,777；open 192,504 |
| 被纠正的真实 ARM gripper target | 415,290 |
| 新 MANI target0 command | close 26,814；open 12,768 |
| 训练 loader | NAV `[20,3]`、ARM `[20,7]`，精确 schema 绑定通过 |

本地 contract/runtime/planner/rollout/data/video 定向回归为 60 passed、1 skipped；skip 是
本地环境缺少 `accelerate`，不是测试断言失败。远端正式环境没有 pytest 包，因此没有伪称
远端 pytest 通过；远端使用正式 Python 环境完成了旧/新 config import、完整数据 audit、
逐行差分和真实 loader 样本加载。

## 6. 对 step 500 的影响

旧 full-data B2 run 的最后有效内存 step 是 568；用户停止后没有本任务进程，最后 durable
checkpoint 是 `step_000500@7ec8424`。该 checkpoint 绑定：

- schema `conveyorvla-waypoint-dense-transition-v2`；
- manifest `5361ed00f808d56537503cb2bfde25ee0ba8cbf9e7e85d7c6e1c35924c3ba56d`；
- 旧 measured-opening 夹爪监督和对应 Adam moments。

新数据使用不同 schema 和 manifest。训练入口会按设计拒绝 strict resume；不能改 manifest、
绕过 binding 或把旧 optimizer state 伪装成同一训练合同。旧 checkpoint 仍可用于历史开环、
闭环和新旧监督对比，但不是新数据的合法父 checkpoint。

首个 corrected baseline 也不引入未经配对验证的“部分权重迁移”。那会同时增加初始化策略、
选择性 reinit 和 optimizer reset 三个变量，削弱本轮对夹爪标签修正的归因。若以后确有节省
算力需要，应把 weight-only warm-start 作为独立配对 pilot，而不是称作 resume。

## 7. 冻结的双卡训练计划

训练前先在覆盖全部 route 和四种 boundary 的 8–16 个 episode 上完成 corrected-data
overfit 与严格开环，重点检查：

1. PICK target0 在接近阶段保持 open；
2. close crossover 与 raw 专家 close 事件对齐；
3. close 前 TCP 已接近 grasp pose，随后 query 继续闭合和抬升；
4. MANI 每次只执行 target0，无 suffix 跳选；
5. route/action、terminal-hold、ARM 连续性和 cuRobo 可执行性通过。

门禁通过后启动全量新 run：

| 项目 | 计划值 |
|---|---|
| initialization | 官方标准 `Qwen3-VL-4B-Instruct`，fresh |
| components | terminal-hold + corrected B2 boundary/progress + S1 |
| disabled | learned prefix、CRL、on-policy correction |
| GPUs | 仅两张；启动前按实时 UUID 选定并冻结 |
| sharding | ZeRO-3，无 optimizer offload |
| batch | micro 8/GPU × 2 × accumulation 4 = global 64 |
| length | 2,000 effective optimizer step |
| warmup | 200 step |
| save | 每 500 effective optimizer step |
| training subset | false，完整 108,603 train row |
| runtime | MANI chronological target0 + requery；NAV trusted prefix cap 10 |

启动仍须重新核验 host/user、目录、Git HEAD/upstream/clean、Conda、两张 GPU UUID、tmux 和
精确 PID，并使用全新 run ID。启动后连续审计至少 20 个有效 optimizer step；loss、gradient、
LR、吞吐、显存、双 rank 和输出路径均健康后才算正式健康启动。当前文档冻结的是计划，
没有启动训练。
