# ConveyorVLA AL0：动态抓取架构与正式采集执行方案

状态：G0–G3 已通过，正式 production 正在运行。冻结日期：2026-08-06。
生产运行固定在提交 `d7b6f0963bef0864b8101571981b7d02e40c3122`；运行期间
不得用后续文档提交替换源码树。

## 交付终点

本阶段不是训练分类器，也不是继续扩大旧的 `0.06 m/s` 多物体分拣矩阵。终点是：

1. 保留可复现的 ConveyorVLA AL0 单帧兼容基线；
2. 增加不破坏旧 checkpoint/schema 的 `conveyorvla_al0_temporal_v1` profile；
3. 让新数据具备双帧运动观测、可跳过的未来动作和严格时钟身份；
4. 在 4xH20 的 GPU 2/3 上通过单回合和并行 pilot；
5. 启动 384 条低速单物体成功示教的正式、可恢复采集。

只有在时序模型相对 AL0 基线取得闭环提升后，才晋级命名为 ConveyorVLA AL1。
当前未验证的结构不能提前使用 AL1 名称。

## 架构合同

```text
head[t-2,t] + wrist[t-2,t] + state28[t]
                     │
              Qwen3-VL ordered clips
                     │
                DiT-B 20×10
                     │
     episode/generation/observation tick
                     │
        exact-target-tick streaming buffer
                     │
              50 Hz Cartesian servo
```

### 观测

- 相机/模型频率为 25 Hz；控制频率为 50 Hz；
- 狗头、腕部相机分别读取 `[t-2,t]`，历史跨度严格为 80 ms；
- 两个相机是两个有序短 clip，不把四张图当作无序图片；
- 只使用当前 `state28`；overview 和物体真值均不是模型输入；
- 每条记录保存 capture/model/control tick，禁止依赖文件顺序猜时间。

### 动作

- 输出 `20×10` 的 25 Hz 动作，覆盖未来 0.8 s；
- base 三维是对应未来 model tick 内两条 50 Hz 控制命令的均值；
- TCP 六维不是逐步增量，而是每一行独立描述未来 TCP 相对观测时刻
  root/TCP 的目标；移动底盘产生的世界位姿变化也包含在该目标中；
- 旋转通过四元数组合后转 rotation vector，禁止直接跨姿态相减；
- gripper 为绝对 `0=close, 1=open`，base-y 继续按机器人合同屏蔽。

因此即使模型迟到，丢弃前若干行不会改变剩余行的参考系。

### 流式执行

每个 chunk 必须携带：

- `episode_id`；
- 每次 reset 唯一的 `generation_id`；
- observation model/control tick；
- 推理开始和结束时间。

控制端按目标 control tick 合并动作，不使用负切片推测重叠关系。目标 tick
小于等于当前 tick 的行全部过期；整块过期或有效后缀少于两行时 fail-closed。
旧 episode、旧 generation、倒序观测一律拒绝。size-one 输出队列永远替换成最新
结果，不能因为队列满而保留旧结果。

## 数据课程

### 原始 episode

正式采集继续运行物理上完整的单目标抓取—安全携带—固定蓝盘投放，以复用经过
验证的 canonical recorder、成功判据和质量门禁。每个 episode 只有一个活动物体，
没有分类或干扰物选择。derived temporal profile 只保留以下抓取相关阶段：

```text
mobile_settle → mobile_approach → mobile_stabilize → arm_preposition
→ settle → select → pregrasp → track → descend → close → lift
→ carry_retract
```

训练指令统一为“抓取传送带零件并安全抬升”，不会把目标分拣盘作为模型条件。
投放部分仍在 canonical episode 中供未来扩展，但不进入当前 grasp-only profile。

### 矩阵

| 维度 | 设置 |
|---|---|
| 方向 | 左到右 |
| 速度 | `0.01 / 0.02 m/s` |
| 物体 | red block、blue bar、yellow bushing、green shaft |
| 活动物体数 | 1 |
| 目标盘 | 固定 blue，仅用于完成 canonical 成功判定 |
| 拦截提前量 | 5 s |
| pilot | 8 条：每个速度×物体一条，必须 8/8 成功 |
| production | 每 cell 48 条通过全部门禁的成功轨迹，共 384 条 |

每个 cell 冻结 72 个互不重叠的 production seed 作为预留池。采集器达到 48 条
成功且通过门禁的轨迹就停止该 cell；任务失败保留作诊断但不计入配额，并自动使用
下一个 seed 补采。若 72 次尝试仍不足 48 条，采集器 fail-closed，不会把“尝试数”
冒充“成功数据数”。

静止示教不重复大规模采集：已有 3 条 train、1 条 val、1 条 test 全部通过门禁。
`0.03 / 0.06 m/s`、多物体、分类和远端投放均延后到低速闭环通过之后。

已有 0.06 m/s 矩阵 128 条数据在先前并行配置下约 2.02 小时发布完成。新任务
降低了物体速度，但用 5 秒提前量避免长距离等待。4xH20 实测 8 条 pilot（含完整
门禁）约 8 分钟；首个 production 双 worker 批次的 16 条 episode 加完整门禁约
11–12 分钟。按每 16 条一个批次投影，384 条的无失败基准约 4.8 小时，保守运行
窗口为 5–6 小时；若出现任务失败并消耗预留 seed，可能延长到约 7.5 小时。按
已发布目录的均值估计，pilot 加 production 最终约占 55–60 GB。以上是吞吐预测，
完成条件仍以 384 条 training-eligible 总账为准。

## 晋级门槛

### G0：纯逻辑

- 全仓测试通过；
- SE(3) 目标可逆；
- 历史跨度、action horizon 和 tick 严格一致；
- 注入 1–40 control tick 延迟不会执行任何过期 action；
- 旧 AL0 schema、health payload 和 checkpoint key 继续可读。

### G1：单回合

- GPU 3 上完成一个 `0.01 m/s` episode；
- success、strict validator、quality audit、temporal camera gate 全部通过；
- legacy 三 profile 与 temporal profile 均能加载；
- head/wrist 各抽查两帧，动作与事件时间轴一致。

### G2：并行 pilot

- GPU 2/3 各一 worker；
- GPU 0/1 不可见且无本项目进程；
- 8/8 物理成功、8/8 数据门禁通过；
- 没有 `.inprogress`、runtime_error、CUDA/OOM、相机丢帧或空 temporal export；
- 采集锁、断点续跑和每 8 条进程边界有效。

### G3：正式采集

- 新建唯一 production 输出目录和 tmux 会话；
- 仅暴露物理 GPU 2/3；
- 至少连续观察每个 worker 的有效 episode 发布与全部门禁；
- 报告、成功根列表、stdout/Kit 日志持续更新；production 报告的
  `training_eligible_episodes` 目标为 384；
- 达到上述条件后只能声明“正式采集已正常启动”，不能声称 384 条已经完成。

## 2026-08-06 正式启动证据

- 生产源码：`d7b6f0963bef0864b8101571981b7d02e40c3122`，远端 worktree clean；
- 源码树指纹：`4b353a1bd247c913daa096a762c2e54ad5a0a3af168f65151c44948cc0245a2e`；
- 资产锁：`3351a6cf3ef7bb65fcd44245541c8cd044d5fb3e65434b18ebfb9ee488b2e075`；
- G1 单回合：GPU 3，976 control steps、1464 张 PNG、300 条 temporal record，
  success/strict/quality/camera/four-profile 全部通过；
- G2 pilot：8/8 success、8/8 fully gated、2438 条 temporal record，无失败 seed；
- G3 首批：16/16 success、16/16 fully gated，64 个 profile 文件哈希全部复核；
  独立加载 9 个首/中/末真实 sample，均为 `state28`、`20×10` action、四张
  `224×224 RGB` 图；
- `2026-08-06T12:01:22Z` 快照已有 26 条 production episode，26/26 success、
  0 failure；首批门禁完成后第二批已自动在物理 GPU 2/3 各启动一个 worker；
- GPU 2/3 对应的进程分别传入 `activeGpu=2/3`，GPU 0/1 上没有本项目进程；
  coordinator 持有唯一采集锁，未出现 traceback、OOM、runtime fatal 或退出文件。

生产总账在一个双 cell wave 完整达到 cell 配额后原子更新，因此 cell 尚在采集时，
`collection_report.json` 可能暂时仍显示旧快照。运行中进度应同时检查 canonical
`summary.json`、四个门禁文件、`.inprogress` 和精确进程；不得据此手工改写总账。
机器可读证据见 `docs/conveyorvla_al0_collection_launch_20260806.json`。

## 运行入口

纯预检：

```bash
python scripts/collect_conveyorvla_al0_grasp.py \
  --phase pilot \
  --output-root NEW_OUTPUT_ROOT \
  --python ISAAC_PYTHON \
  --isaaclab-source ISAACLAB_CORE_SOURCE \
  --kit-cache-root PREWARMED_KIT_CACHE \
  --runtime-library-dir ISAAC_RUNTIME_LIBRARY_DIR \
  --physical-gpu 2 \
  --physical-gpu 3 \
  --workers 2 \
  --dry-run
```

正式运行必须使用同一命令去掉 `--dry-run`，pilot 通过后将 phase 改为
`production`。脚本只接受 GPU 2/3，并把 CUDA、临时目录、XDG、日志和 Kit 数据
限制到输出根。每批最多 8 条，任务中断后根据已发布 seed 精确续跑。

## 明确不做

- 不把失败 policy rollout 伪装成成功示教；
- 不把 assisted diagnostic 回合写入标准训练集；
- 不改写 `m0_mobile_v1` 或历史 checkpoint tensor key；
- 不让 overview 或物体真值进入模型；
- 不在 pilot 失败时通过增加数据量掩盖结构问题；
- 不使用 GPU 0/1，不控制非本任务进程。
