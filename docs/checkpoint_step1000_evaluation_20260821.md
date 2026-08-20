# step 001000 开环与真实 Isaac 闭环评测

首次复核：2026-08-21 00:36 CST；追加复核截至 02:13 CST。被测 checkpoint 来自训练提交
`724ead21be2c27d9b40c200375ee4ab49ccedc84`；评测与 runtime 修复截至
`0deec5ec60f771826b4c5d2ff47fe731dfa7e477`。

## 1. 结论

step 001000 已完成四卡绑定/load、40-row 多 diffusion seed 开环、真实 cuRobo known-pose
和一次无 GT phase、无外部 FSM、无 route gate 的 Liangzhu Isaac 自主闭环尝试。

- checkpoint 绑定和四卡 load：通过；
- route/格式开环：通过，40/40 route 正确，`RECOVER=0`，format invalid=`0`；
- waypoint/TCP 动作质量：未达到可闭环水平；
- 真实 cuRobo known-pose：通过；
- 完整自主闭环：已真实执行，但首个 NAV chunk 被 yaw segment 安全门拒绝，episode 失败；
- 三路视频：成功封装、校验并下载到本地 Git 忽略目录。

因此不能声明模型已通过开环质量门禁或 Isaac 完整闭环。失败发生在模型 waypoint 的
数值质量，不是 Qwen route、runtime protocol、scene 启动、录像或 cuRobo 服务启动。

## 2. 训练暂停与 checkpoint

正式训练由用户授权的 `SIGINT` 停止。最后一个完整有效 optimizer event 是 step 1181；
日志尾部的 `KeyboardInterrupt` 是主动暂停证据，不是训练异常。磁盘上完整 checkpoint 为
step 20 和 step 1000，因此 1001–1181 共 181 个已计算 step 没有 checkpoint。

step 1000 四卡 load 报告：

| 项目 | 结果 |
|---|---:|
| world size | 4 |
| global/scheduler step | 1000 / 1000 |
| parameter partition values | 5,021,782,540 |
| non-finite parameter partitions | 0 |
| dataset manifest | `0db6169d726b2165a90ec6e833403666179eb68135248af5681de92a400ec957` |
| normalizer | `75a60ba125a83383f1d00ef4151933a77c796faee5d5c559364310cb64acfca0` |
| status | pass |

`aa064794e9352a855a558732b114a8e78dabb8ef` 已把 `train_waypoint.py` 的默认保存间隔和
操作手册统一改为 500 effective optimizer steps。这个修改只影响后续新启动的命令，
不会追溯生成 step 1181 checkpoint。`22b186c35e36122a6fc8d876d4a45355ebf42172`
随后实现了严格的同合同 ZeRO resume：父数据、配置、Qwen root、world size、batch、
accumulation、max steps、warmup、seed、attention 和 subset 任一变化都拒绝；新 run 必须
使用新输出目录并记录父 manifest/hash。第 8 节记录实际四卡恢复证据。

## 3. 四卡开环

评测使用 val split 的 40 个平衡样本，每个 route 与 DONE 各 8 个；batch size=2，
diffusion seed 为 17、29、43、71。自主 route 与 oracle-prefix action 分开评估，动作
采样没有使用 online route。

| 指标 | 结果 |
|---|---:|
| route accuracy | 1.000 |
| RECOVER / format invalid | 0 / 0 |
| missing action / non-finite metric | 0 / 0 |
| NAV ADE / FDE | 0.3959 m / 0.7522 m |
| NAV first-waypoint direction accuracy | 0.625 |
| NAV yaw error | seed 均值约 0.526 rad |
| NAV normalization OOB | 0.671875 |
| NAV segment violation | 0.8125 |
| ARM TCP position / orientation error | 0.1195 m / 2.1501 rad |
| ARM gripper accuracy | 0.87265625 |
| ARM normalization OOB | 0.953125 |
| ARM inter-target step violation | 1.0 |
| ARM workspace violation | 0.0 |

报告的 `profile=diagnostic` 只要求结构、有限值和缺失动作门禁，因此其
`structural_pass=true`、`quality_pass=true` 不能解释为动作质量通过。上述 NAV/ARM
误差、OOB 和 segment/step violation 明确说明 step 1000 尚不具备可靠闭环能力。

## 4. inference export 与 cuRobo

第一次 consolidation 暴露 Qwen tied weights 在 safetensors 分片导出时重复引用；
`13f6e87a3496bd840df83cc60692093e313a2914` 改为通过 Transformers 的 tied-weight
声明导出，第二次成功生成约 21 GB 的绑定 inference export，并在单张 H20 上完成真实
四图 request。

cuRobo 使用相互独立的代码根与运行资产根：

- arm-vla：`388b6818f4c605a707d13c519fbb58b1d07acd92`；
- cuRobo：`87260212b9ad5ebe486427cbf168611145232884`；
- service port：loopback `8766`；
- 输入 frame：`query-base-B_t`；planner frame：`curobo-planner-base`；
- orientation fallback：关闭；world collision：开启。

真实 known-pose 结果为 reachable、collision-free，返回 41 点 joint path；TCP position
error 为 `6.14e-8 m`，orientation error 为 `5.96e-8 rad`，未使用 orientation
fallback。`23afff481498d846581f23ac2c7165b64c7af675` 分离 code/assets roots，
`121512903667e16578525ec22dcfb2d0deca92e5` 让 rollout 只复用并严格核验这个 Waypoint
cuRobo 服务，拒绝 reference pipeline 启动旧的 legacy 服务。

## 5. 真实自主闭环

最终有效尝试为 `autonomous_seed861_r3`：Liangzhu full visual、真实 PCT 配置、
`pct_multifloor` locomotion、三路录像、最多 400 query/24,000 control step。未使用
`--required-first-route`、`--stop-after-route`、GT phase、外部 route gate 或 FSM。

启动过程中保留了两次失败尝试：

1. r1 因 source USDA 的 fallback arc 不符合批准 scene profile 身份而 fail-closed；通过
   派生、不可提交的 USDA 把 visual/collision fallback 精确绑定到批准 reference 后解决；
2. r2 因 `CUDA_VISIBLE_DEVICES=2` 与 Omniverse Vulkan physical-device 枚举不一致而无法
   创建 device；r3 取消该 mask，并显式关闭 renderer multi-GPU 后解决；
3. r3 完成 scene materialization、Isaac app、91 万面导航网格、robot articulation、
   locomotion policy、对象 settle、四帧采集和真实模型请求。

r3 的第一个模型 request 只含任务、head/wrist 双时刻共四图和 calibration ID，
`model_state_fields=0`。Qwen 返回：

| 字段 | 值 |
|---|---|
| route | `NAV_TO_SOURCE` |
| confidence | 0.98594 |
| subtask | `Walk toward the box holding the Coke can.` |
| action | `[20,3]` body waypoint |
| model latency | 3123.6 ms |
| normalization clip rate | 0.0333 |

执行器在进入 PCT 前验证整个 NAV chunk，发现 `navigation segment 18 exceeds yaw limit`，
按合同零速 fail-closed。最终 `query_count=1`、`control_steps=58`、
`state_trace=[NAV_TO_SOURCE]`、`success=false`。安全门没有把 route 改写成其他阶段，也没有
用 GT phase 或旧 state28 覆盖模型。

### 5.1 waypoint 与 reference 实现复核

训练集 63,350 个 NAV row 的原始标签在 0.8 m/45° 合同下 sample/segment violation 都为
0；最大相邻平移为 0.2652 m，最大偏航为 19.881°，因此批准阈值与 GT 并不冲突。
但只有 20,125 row 具有完整 20 点有效 horizon，前部位置有 60,506 row 有监督，尾部监督
显著更少。q01/q99 clip 对环形 yaw 的极少数样本还会制造约 0.126% 的边界跳变，这是次要
数据问题，不足以解释主要失败。

对 64 个 NAV oracle row、4 个 diffusion seed 的逐段审计显示：首点 violation rate=0，
完整 20 点 violation rate=0.96875；最早坏点为 index 4，累计 123 个 yaw 与 35 个 translation
原因。即使把 yaw 临时放到 180°，完整 horizon 仍有 0.6875 失败。因此简单放宽安全阈值
既不能解决主要质量问题，也会掩盖尾部欠训练。

官方 StarVLA 的 QwenPI/Layerwise FM 路线同样使用 Qwen3-VL、逐层 cross-attention、
Beta(1.5,1)、`noise_s=0.999` 和 4 个 Euler inference step；当前实现没有显著的 diffusion
步数或噪声日程偏差。参考：
[StarVLA](https://github.com/starVLA/starVLA)、
[QwenPI_v3](https://github.com/starVLA/starVLA/blob/starVLA_dev/starVLA/model/framework/VLM4A/QwenPI_v3.py)、
[LayerwiseFM_ActionHeader](https://github.com/starVLA/starVLA/blob/starVLA_dev/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py)。
本地 `arm-vla-grasp-sim` 的 `pct_multifloor` 初始姿态、腿/臂/夹爪 actuator、DWA 速度/
加速度和 0.02 s control dt 也与 rollout 实际配置一致，没有发现动力学配置漂移。

初始化同样不是首要原因：strict 与 staged run 都先完成对象 settle 和机器人初始化，模型
query 在 control step 58；query 时 root z 约 0.1914 m，训练 query 约 0.1897 m，速度量级
相当。不能把失败归因于“机器狗尚未落地就开始预测”。

### 5.2 可执行 prefix 诊断与 executor 修复

`22b186c` 增加显式 `executable-prefix-diagnostic` profile。它仍先检查完整 horizon 并记录
原始 violation，只有选中的第一个非退化 prefix 自身合法时才执行；默认 `contract` 行为
不变，0.8 m/45° 等阈值也没有改变。

前两次 staged run 依次暴露了执行器语义问题，而不是新的模型非法点：

1. r1 的首点和 PCT plan 合法，但目标平移已在 0.12 m 到达容差内，DWA 返回零平移；旧逻辑
   仍等待距离进展并触发 stall；
2. r2 的首点平移约 0.033 m，却仍送进 0.2 m PCT 栅格，最近端点 snap 因数值波动越过
   0.10 m 门禁；放宽 snap 会破坏合同，正确处理是跳过不需要的 PCT；
3. `92ba25f` 让 PCT/DWA 行进后进入位置容差的终段使用限幅 terminal-yaw；`a8d57a2`
   让诊断 run 中一开始就位于 0.12 m 到达容差内的目标绕过不需要的栅格 snap。最终合同
   复核 `0deec5e` 把后一旁路限定为 diagnostic；production 仍只在平移 `<0.03 m` 时
   绕过 PCT。

修复后的 r3 在完整空间指令、无 state/phase/FSM、同一 step 1000 模型下成功完成单 route
staged gate。模型首点为 `[-0.03717,-0.00095,0.23649]`，完整 horizon 仍明确记录
`segment 18 exceeds yaw limit`；executor 选择 `planner=terminal_yaw`，在 control step 75
达到 `first_waypoint_reached`。summary 为 `success=true`、`query_count=1`、
`state_trace=[NAV_TO_SOURCE]`。这是“合法首点可执行”的诊断证据，不是完整自主 episode 或
完整 horizon 通过。

## 6. 视频与证据隔离

| stream | 分辨率 | 帧数 | 时长 | SHA-256 |
|---|---:|---:|---:|---|
| overview | 1280×720 | 28 | 1.12 s | `16cfb5d9bc6412b2fba66228ac78cfa658435dbb073368b28dc8d39cd00033af` |
| front | 640×480 | 29 | 1.16 s | `9de898c47143bf67e55c38476bd0ee894842d13ce4031941755a352306947086` |
| wrist | 640×480 | 29 | 1.16 s | `742e376485d48e838aa667b3672d90deab370caca5ff5fd871c8cc38a12bba7d` |

短时长是因为对象 settle 后的第一条模型 waypoint 立即触发安全拒绝，不是录像提前崩溃。
三路文件均通过 `ffprobe`、远端/本地 SHA-256 比对和中间帧目检。它们与 checkpoint、
JSON report、trace、日志和派生场景只保存在
`artifacts/evaluation/waypoint_step001000_20260820T231424/` 等 Git 忽略目录，不加入仓库。

staged r3 新视频也已下载到同一忽略目录的 `prefix_diagnostic_r3/videos/`：overview/
front/wrist 为 37/38/38 帧，SHA-256 分别为 `650bcf51…`、`9b03e8f6…`、`19cef1a6…`；
summary 与完整 trace 同步保存，但不加入 Git。

## 7. 后续门禁

已选择从 step 1000 严格恢复正式长训。后续应在更新 checkpoint 上重跑同一 40-row/四
seed 开环；只有 NAV segment/方向/ADE/FDE 与 ARM pose/step 指标改善后，才依次重跑
oracle-route planner、ARM staged route 和完整自主 episode。诊断 profile 不支持绕过
yaw/workspace/rate gate 来宣称 production 通过。

## 8. 四卡恢复结果

resume run 为
`conveyorvla-waypoint-v1-resume-step1000-a8d57a2-s10000-20260821T015929`，source 固定
在 clean `a8d57a2`，父 checkpoint 为本报告的 step 1000。实际恢复了 Qwen、双 FM head、
AdamW、scheduler 和随机状态；scheduler `loaded_step=1000, repaired=false`，sampler 跳过
2,000 个已消费 micro-batch。

step 1001–1020 共 20 个连续 event 全部 `valid_optimizer_step=true`。total/NAV/ARM loss
均值为 `1.0809/0.3959/0.4396`；VLM/NAV/ARM gradient norm 最小值为
`88.02/8.01/6.54`，四组 learning rate 全部有限且为正。四 rank 存活，四卡均有真实计算，
日志没有 traceback、OOM、NCCL error 或 NaN/Inf。训练在完成该健康门禁后继续运行，下一
个新 checkpoint 按 500-step 合同写在 step 1500。
