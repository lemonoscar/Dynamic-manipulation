# ConveyorBench V1 数据采集交付手册

本文给出从环境预检、物理任务门禁到 M0/DynamicVLA 离线投影的可执行流程。
V1 的主任务是 Go2-X5 在横向传送带前移动、动态抓取指定零件、携带并释放到
指定分拣盘；`fixed_base` 只作为机械臂消融对照。当前交付是可开启小批量采集
的框架，但必须先关闭本文列出的全部门禁；它不包含训练好的 VLA，也没有进行
大规模轨迹采集。

规范与机器可读配置分别见
[BENCHMARK_V1_SPEC.md](BENCHMARK_V1_SPEC.md) 和
[configs/v1.json](configs/v1.json)。

## 1. 已实现的采集对象

### 1.1 真实工位语义

布局 `transverse_dynamic_sort_station_v1` 使用顶面高 `0.50 m` 的横向低位
传送带。皮带几何中心为 `X=0.70 m`，零件使用靠机器人一侧但仍位于带内的
`X=0.65 m` 车道，沿世界 `+Y → -Y` 运输，在机器人视角中从左向右移动。
工位包含：

- 独立的物理运输面，使用 `PhysxSurfaceVelocityAPI` 产生接触输送；
- 无碰撞视觉皮带、接缝标记、驱动/从动滚筒、机架、支腿和横撑；
- 电机护罩、远侧安全栏、上下游光电传感器、急停和出口标记；
- 蓝/黄两个计分分拣盘，以及不计分的下游漏件接料盘。

视觉细节与运输碰撞体分离，避免滚筒和机架外观改变 benchmark 的接触语义。
工位参数和来源记录在
[assets/workcells/conveyor_station_v1/ASSET_MANIFEST.json](assets/workcells/conveyor_station_v1/ASSET_MANIFEST.json)。

### 1.2 物体、划分与任务

8 个项目内程序化零件冻结为三个互斥 curriculum split：

| split | 数量 | 物体 |
| --- | ---: | --- |
| `train` | 4 | red block、blue bar、yellow bushing、green shaft |
| `val` | 2 | silver hex、orange flange |
| `unseen` | 2 | purple bracket、cyan gear |

每个零件都在
[assets/objects/registry.json](assets/objects/registry.json)
中记录几何、质量、摩擦、稳定姿态、双指抓取 affordance、中英文别名和来源。
运行时只从所选 split 采样，禁止跨 split 混用。`single_target` 适合最小闭环；
`language_conditioned` 同时激活目标和干扰物，并冻结中英双语指令、目标 ID 与
目的分拣盘。

### 1.3 时钟、动作与相机权限

- physics/control/camera-model 固定为 `400/50/25 Hz`；
- canonical action 为 body-frame base 3D、robot-root/base-frame TCP delta
  6D 和 gripper 1D，共 10D；
- `head_rgb`：狗头前缘、沿机身 `+X` 水平前视，策略可见；
- `wrist_rgb`：夹爪正上方、向下俯视抓取区，策略可见；
- `overview_rgb`：远处第三视角，只供人工观察、回放与质量检查。

相机采集必须保持 Fabric 开启。不要向相机任务传入 `--disable_fabric`：
物理仍可能正常运动，但渲染几何可能停留在初始位姿。V1 benchmark 运行时把
`use_fabric=true` 写入 episode manifest；无相机的独立 locomotion probe
使用另一套只验证浮动根运动的配置，不能替代相机门禁。

## 2. 自包含与资产完整性

运行时不联网，也不从 `Dynamic/` 外读取其他项目、权重或数字资产。宿主机仍
需预装 Python 3.11、Isaac Sim、Isaac Lab、PyTorch、NumPy 和 OpenCV。

[assets/asset_lock.json](assets/asset_lock.json) 锁定机器人 USD/URDF、URDF
引用的全部唯一 mesh、移动策略及 contract、物体注册表、分拣盘和工位 manifest。
运行启动会校验这些哈希；episode manifest 还记录 asset-lock 哈希和项目源码树
指纹。权重来源和再分发边界见
[assets/policies/go2_x5_pct_dog_only/PROVENANCE.md](assets/policies/go2_x5_pct_dog_only/PROVENANCE.md)。

夹爪保留原 FinRay 可视 mesh，但使用在可用平行接触垫处测量得到的薄 compound
collider，避免完整 convex hull 填满弯曲指间空间。TCP 冻结在
`arm_link6 + (0.125, 0, 0) m`；代理 model ID、尺寸、摩擦、contact/rest
offset 和“未新增刚体/质量”的拓扑检查均写入 episode manifest。

从项目根目录执行：

```bash
cd Dynamic/conveyor_bench
conda activate env_isaaclab
python -m pip install -e .
python scripts/check_environment.py
PYTHONPATH=src python -c "from conveyor_bench.v1.assets import verify_asset_lock; print(f'locked assets: {len(verify_asset_lock())}')"
```

随后运行不启动 Isaac Sim 的回归测试：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider
```

## 3. 物理与视觉门禁

### 3.1 工位、运输方向与三相机

```bash
python scripts/probe_v1_scene.py \
  --output-dir outputs/gate/v1_scene \
  --belt-speed 0.04 \
  --settle-seconds 1.0 \
  --enable_cameras \
  --headless \
  --device cpu
```

该探针检查工位构成、落带接触、`+Y → -Y` 输送和三路静态图像，不证明动态
抓取成功，也不证明相机随物理状态实时变化。

### 3.2 浮动根移动策略

```bash
python scripts/probe_mobile_locomotion.py \
  --output outputs/gate/mobile_locomotion/report.json \
  --vx 0.20 \
  --wz 0.0 \
  --settle-seconds 1.0 \
  --hold-seconds 2.0 \
  --stop-seconds 1.0 \
  --warmup-steps 50 \
  --headless \
  --device cpu
```

该 probe 只验证本地 locomotion actor、浮动根稳定性、速度响应与停止行为。

### 3.3 固定机身消融

```bash
python scripts/run_benchmark_v1.py \
  --robot-mode fixed_base \
  --episodes 1 \
  --seed 0 \
  --split train \
  --task-family single_target \
  --belt-speed 0.06 \
  --max-duration 20 \
  --active-objects 1 \
  --target-asset part_red_block \
  --destination sort_bin_blue \
  --output-dir outputs/gate/v1_fixed \
  --enable_cameras \
  --save-camera-frames \
  --require-all-success \
  --headless \
  --device cpu
```

### 3.4 全身移动单目标

全身闭环当前约需 `21.6 s`，因此门禁显式使用 `35 s`，不能沿用 CLI 的
`20 s` 默认值：

```bash
python scripts/run_benchmark_v1.py \
  --robot-mode whole_body_policy \
  --episodes 1 \
  --seed 0 \
  --split train \
  --task-family single_target \
  --belt-speed 0.06 \
  --max-duration 35 \
  --active-objects 1 \
  --target-asset part_red_block \
  --destination sort_bin_blue \
  --output-dir outputs/gate/v1_whole_body \
  --enable_cameras \
  --save-camera-frames \
  --require-all-success \
  --headless \
  --device cpu
```

### 3.5 三物体双语选择

```bash
python scripts/run_benchmark_v1.py \
  --robot-mode whole_body_policy \
  --episodes 1 \
  --seed 0 \
  --split train \
  --task-family language_conditioned \
  --instruction-language en_zh \
  --belt-speed 0.06 \
  --max-duration 35 \
  --active-objects 3 \
  --target-asset part_red_block \
  --destination sort_bin_blue \
  --output-dir outputs/gate/v1_whole_body_language \
  --enable_cameras \
  --save-camera-frames \
  --require-all-success \
  --headless \
  --device cpu
```

`val` 和 `unseen` 各只有 2 个物体，因此这两个 split 的
`--active-objects` 最大为 2。普通采集应移除 `--require-all-success`：
具有明确任务失败原因且完整发布的 episode 仍是有效 benchmark 数据。

## 4. 数据、相机与导出验收

每条用于训练投影的 episode 按以下顺序验收。把示例目录替换为 run summary
中的实际 episode 路径：

```bash
python scripts/validate_v1_dataset.py outputs/gate/v1_whole_body

python scripts/audit_v1_episode.py \
  outputs/gate/v1_whole_body/episodes/EPISODE_ID

python scripts/check_v1_camera_gate.py \
  outputs/gate/v1_whole_body/episodes/EPISODE_ID

python scripts/export_v1.py \
  outputs/gate/v1_whole_body/episodes/EPISODE_ID \
  --profile all
```

四层检查不能互相替代：

1. strict validator 检查流结构、时间、引用、成功证据、PNG 数量和真实未来标签；
2. quality audit 把任务失败与数据损坏分开，并检查已有图像质量统计；
3. temporal camera gate 检查物理发生位移时三相机画面确实变化，并要求
   head/wrist 中存在目标变化证据；overview 不计入策略可见性；
4. exporter 再次 fail-closed 校验 canonical episode，训练投影必须实际包含
   `head_rgb` 和 `wrist_rgb`，然后才生成 DynamicVLA、legacy M0 与因果
   M0-Mobile 视图。

导出只在 episode 内创建：

```text
exports/
├── dynamicvla.jsonl
├── m0.jsonl
├── m0_mobile.jsonl
└── export_manifest.json
```

`export_manifest.json` 记录 canonical 文件哈希和派生文件哈希；导出前后不得
改变 canonical 源流。DynamicVLA 视图使用历史 `[-2,0]`、未来 TCP `+5`
model tick、20-step chunk 和 base-frame state/action；M0 视图使用 world-frame
state/arm delta、16-step chunk 和右臂填充的 14D action。两者都保留 canonical
10D、body-frame base 3D 与有效位 mask。

### 4.1 M0-Mobile 因果训练视图

运行时每个 canonical step 都是“动作已经执行、物理已经推进”后的状态，因此
legacy `m0.jsonl` 的同 tick action 只用于兼容既有分析消费端，不能直接训练
观测到动作的因果策略。`m0_mobile.jsonl` 使用以下闭合约定：

- 相机所在的 50 Hz row 为观测 `t`，标签是后续 `t+1 ... t+16`；
- 每条样本包含 `state28`、语言、head/wrist 两路图像和未来 `16×10` 动作；
- canonical 夹爪 `[-1,1]` 映射到 model 夹爪 `[0,1]`，`base_vy` 不参与 loss；
- 尾部不 padding，overview、phase、目标 ID、物体状态和未来真值不会进入样本。

只生成训练 profile 时使用：

```bash
python scripts/export_v1.py \
  outputs/gate/v1_whole_body/episodes/EPISODE_ID \
  --profile m0_mobile
```

如果同一 episode 已有导出 manifest，需明确使用 `--force` 重建派生清单；
canonical 六个事实源仍受到哈希保护，不会被覆盖。

### 4.2 静止传送带补充数据

静态诊断只接受 `part_red_block → sort_bin_blue` 和五个预注册 seed。训练集合并
1101–1103 三条完整成功轨迹；2101 与 3101 只做 val/test。导出记录同时包含：

- `source_task_type=stationary_sort`；
- `belt_speed_mps=0`；
- `split=train|val|test`，来自 scenario；
- `object_curriculum_split=train`，只说明红色方块在资产注册表中的归属。

训练加载器默认拒绝非 train 记录。静态专用烟测应显式 fail-closed：

```bash
python scripts/train_m0_mobile.py \
  --episode-root TRAIN_EPISODE_ROOT \
  --state-statistics TRAIN_STATE_STATISTICS \
  --initial-action-checkpoint ACCEPTED_M1_ACTION_MODEL \
  --model-root LOCAL_MODEL_ROOT \
  --output-dir NEW_STATIC_EXPERIMENT_ROOT \
  --belt-speed 0 \
  --task-type stationary_sort \
  --max-steps 1000 \
  --batch-size-per-device 2
```

混合静态与动态完整轨迹时使用 `--all-belt-speeds`，并分别重复
`--task-type stationary_sort --task-type dynamic_sort`。状态统计只能由最终训练
混合中的 train 记录计算；不得读取 val/test，也不得把失败的 online diagnostic
episode 当作成功示教。训练加载器会无条件核对 `source_task_type` 与带速语义，
且 `--all-belt-speeds` 必须配显式 `--task-type`。旧版动态导出若没有
`source_task_type` 会被拒绝，必须从 canonical episode 使用当前 exporter 重新
生成，不能在 JSONL 中手工补字段。

最小 AML action head 是纯 PyTorch 实现，复现 ABot-M0 的 clean-action
velocity-matching loss 与四步 Euler 方程。它使用轻量 Transformer decoder，
并不声称复现完整 M0 DiT-B 架构。先在 CPU 做可重复过拟合烟测：

```bash
PYTHONPATH=src python scripts/smoke_m0_aml.py \
  --device cpu \
  --dtype float32 \
  --steps 250 \
  --output-dir outputs/smoke/m0_aml_cpu_RUN_ID
```

脚本要求新输出目录，保存 `checkpoint.pt` 和 `report.json`，并检查 loss 下降、
有限梯度、`16×16×10` 四步采样以及 checkpoint 重载一致性。H20 仅在确认某张
卡独占空闲后运行单卡 BF16 烟测：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID PYTHONPATH=src \
python scripts/smoke_m0_aml.py \
  --device cuda \
  --dtype bfloat16 \
  --steps 250 \
  --output-dir outputs/smoke/m0_aml_h20_RUN_ID
```

该烟测只证明数据维度、AML 方程、反传、采样与 checkpoint 链路已接通，不代表
VLA 任务性能、视觉语言骨干训练或四卡扩展已经通过。

训练后不要直接扩大在线采集。先按
[M0_ONLINE_GUIDE.md](M0_ONLINE_GUIDE.md) 启动本地回环服务、核对模型与状态
统计 SHA、运行移动意图/抓取阶段门禁，再执行单 seed Isaac 闭环。门禁或闭环
失败都必须保留为负例，不得通过 oracle TCP 修正或强制闭爪改写成成功轨迹。

## 5. 当前本地烟测证据

以下结果用于说明闭环已经达到的范围，不等同于完整物体/seed/速度矩阵：

| 路径 | 已观察结果 |
| --- | --- |
| `outputs/gate/v1_fixed_single_current_v5` | fixed 单目标成功；`10.48 s`、524 control 样本；strict validator 通过、quality 为 clean；该条未保存相机 |
| `outputs/gate/v1_release_camera` | whole-body 单目标 release 成功；`21.60 s`、1080 control 样本、540 个同步相机 tick、1620 张 PNG；strict validator、quality 和 temporal camera gate 均通过；M0/DynamicVLA 各导出 540 条，canonical 哈希未改变 |
| `outputs/gate/v1_mobile_multi_current` | whole-body 三物体、双语目标选择成功；`21.58 s`、1079 control 样本、3237 条物体记录；strict validator 通过、quality 为 clean；该条未保存相机 |
| `outputs/gate/v1_stationary_train_oracle_final_1101_1103` | 零速 whole-body 静态诊断 3/3 成功；2906 control 样本、4356 张三相机 PNG、1428 条 M0-Mobile train 记录；strict validator 与 camera gate 均 3/3 通过 |
| `outputs/gate/v1_stationary_val_oracle_2101` / `v1_stationary_test_oracle_3101` | val/test 各 1/1 成功并通过 strict/camera gate；导出分别标记为 `val`/`test`，不能进入训练 |

最终 release 正例的 episode ID 为
`run-20260730T142415659352Z-6c097b79-ep0000-seed0-whole_body_policy`，记录的
源码树 SHA-256 为
`a5c2802447abd4e4c50365549b7b0cc83db313f01800cb26d734fc8fc695f39c`。
head/wrist/overview 最大结构变化率分别为
`0.704164/0.688824/0.039858`，head/wrist 目标证据为 `0.760409`，均由
`check_v1_camera_gate.py` 实际复核通过。Fabric 修复前的
`outputs/gate/v1_mobile_camera_current` 保留为负例：尽管物理任务、
strict validator 和 quality 通过，其三路图像仍被 camera gate 判为冻结，
说明 PNG 存在和图像统计正常不能替代时变门禁。

因此，whole-body 单目标的物理—三相机—canonical—双导出主链路已经具备一条
完整正证据。fixed 与三物体语言烟测只证明各自的物理/data 路径；若它们要进入
视觉训练矩阵，仍需分别生成相机 episode 并通过同一 temporal camera gate 和
双导出。

当前预览可用于人工检查场景构图：

- [release 三相机阶段拼图](outputs/previews/v1_release_contact_sheet.png)
- [release 三相机同步视频](outputs/previews/v1_release_three_camera.mp4)
- [release overview 视频](outputs/previews/v1_release_overview.mp4)

预览用于观察构图；时变合格性仍以对应 episode 的 camera-gate 报告为准。

## 6. 放量边界

正式第一批动态训练矩阵已经冻结为 4 个 train 目标 × 2 个分拣盘 × 2 种语言 ×
每格 8 个 seed，共 128 条。不要用裸 shell 循环放量；统一使用
`collect_v1_train_matrix.py`，它会从 canonical manifest/summary 重建恢复状态，
拒绝重复或冲突 seed、可见 orphan episode 和并发写入，并在每条 episode 后依次
执行 strict validator、quality audit、temporal camera gate 与三个 profile 导出。
runner 启动时还会冻结当前可执行源码树 SHA-256；dry-run 与 `matrix_report.json`
都会记录该指纹，任何恢复出的 episode 指纹不完全一致都会立即停止。源码发生
变化后，旧 output root 只保留作审计证据，必须换一个全新的 output root 重跑
pilot，不能混用、覆盖或手工改 manifest。

先查看不会启动 Isaac 的冻结命令：

```bash
python scripts/collect_v1_train_matrix.py \
  --phase pilot \
  --output-root /NEW/DATASET/v1-dynamic-train-128 \
  --renderer-active-gpu VALIDATED_KIT_ORDINAL \
  --dry-run
```

实际执行必须先通过远端 Compatibility Checker，并从其 `[gpu.foundation]` 表确认
`VALIDATED_KIT_ORDINAL` 对应获准使用的物理 GPU UUID。`CUDA_VISIBLE_DEVICES` 不
能替代这个 Vulkan/Kit 核对。pilot 每格只采 base seed，共 16 条，并且必须 16/16
物理成功且四层门禁全部通过：

```bash
python scripts/collect_v1_train_matrix.py \
  --phase pilot \
  --output-root /NEW/DATASET/v1-dynamic-train-128 \
  --renderer-active-gpu VALIDATED_KIT_ORDINAL
```

只有 runner 自己判定 pilot barrier 通过，才可在同一个 output root 启动 bulk；
bulk 只补每格剩余的 `base+1 ... base+7`，共 112 条：

```bash
python scripts/collect_v1_train_matrix.py \
  --phase bulk \
  --output-root /NEW/DATASET/v1-dynamic-train-128 \
  --renderer-active-gpu VALIDATED_KIT_ORDINAL \
  --workers 2
```

`--workers` 只对 bulk 生效。一个 coordinator 继续独占 `.matrix.lock`，每个 worker
只写自己的 cell 和隔离的 TMP/XDG 运行目录；同一 wave 全部退出后，主线程才扫描、
门禁并更新总账。首次在新服务器上固定从 2 开始，至少完成一个双 cell wave 且确认
相机、导出、GPU UUID、显存回落都正常后才能提高；不要同时启动多个 matrix runner。

中断后重复同一命令只补 canonical 中缺失的连续 seed 范围。若上次硬崩留下可见
orphan、重复语义 seed、合同冲突或陈旧 `.matrix.lock`，runner 会停止，
要求人工审计，绝不自动覆盖。`matrix_report.json` 是原子更新的总账；
`successful_episode_roots.txt` 只列出物理成功且完整门禁通过的训练候选。bulk 的
普通任务失败会保留为 benchmark 负例并标为 training-ineligible；runtime error、
数据损坏、相机或导出失败都会立即停止后续放量。

V1 非完整底盘的 carry navigation 采用 stop-turn-drive：目标方位误差超过
`0.21 rad` 时只转向，进入阈值后角速度归零再沿弦线前进。该值来自 seed 10202
的实测几何：导航起点方位误差为 `0.193 rad`，但横向误差仅 `0.030 m`，已经在
最终 `0.045 m` 位置门限内。该约束避免负向转弯后同时输入较大的前进和角速度，
使带载 locomotion 停滞；不得在没有新 canonical 回放证据时改动该合同。

不得把 smoke/debug 目录直接并入正式训练集，也不得绕过 16-cell pilot barrier。
