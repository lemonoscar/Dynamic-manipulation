# ConveyorBench

ConveyorBench 现在同时保留 V0 单目标固定机身基线、冻结的 V1 动态分拣采集
框架，并新增 V2 任务套件。V2 围绕“从横向传送带动态抓取零件，再近端分拣或
持物移动到远端投放”组织：共有 8 类本地程序化零件、2 个场景和 7 个允许的
scene/task/mode 组合，包含首版双目标连续分拣与强制 whole-body 的远端交付。

- V2 规范：[BENCHMARK_V2_SPEC.md](BENCHMARK_V2_SPEC.md)
- V2 可机读快照：[configs/v2.json](configs/v2.json)
- V2 采集与验收：[COLLECTION_V2_GUIDE.md](COLLECTION_V2_GUIDE.md)
- V2 场景俯视图：[docs/images/conveyorbench_v2_layout.svg](docs/images/conveyorbench_v2_layout.svg)
- V1 冻结规范：[BENCHMARK_V1_SPEC.md](BENCHMARK_V1_SPEC.md)
- V1 可机读快照：[configs/v1.json](configs/v1.json)
- V1 采集与验收手册：[COLLECTION_GUIDE.md](COLLECTION_GUIDE.md)
- M0-Mobile 在线闭环与验收：[M0_ONLINE_GUIDE.md](M0_ONLINE_GUIDE.md)
- V0 冻结规范：[BENCHMARK_SPEC.md](BENCHMARK_SPEC.md)
- V0 可机读快照：[configs/v0.json](configs/v0.json)
- 阶段状态：[ROADMAP.md](ROADMAP.md)

V2 是复用 `conveyor-bench-v1` canonical 数据协议的 suite 增量，不另造一套
原始 episode 格式。它不包含训练好的 VLA 模型，也不把尚未在目标 Isaac 环境
执行的物理门禁描述为已通过。冻结源码候选 `0a2fd7c…` 已完成 near fixed-base
双目标 continuous，以及 remote whole-body 蓝/黄双向单目标投放；连续持物位移
分别为 `0.735903 m` 和 `0.778166 m`。本机 RTX 4060 上的 near fixed single 与
remote whole-body 三相机正例均通过 V2 strict validator、temporal camera gate
和 DynamicVLA/M0 双导出。near whole-body、语言条件、near continuous 相机及
其余物体/速度/seed 矩阵仍必须按采集手册逐条验收。

所有项目代码、资产与本地策略权重都在 `Dynamic/conveyor_bench/`；采集运行不
联网，也不读取 `Dynamic/` 之外的项目文件。V1 资产由
[assets/asset_lock.json](assets/asset_lock.json) 冻结，V2 新场景与远端投放盒
由独立的 [assets/asset_lock_v2.json](assets/asset_lock_v2.json) 冻结。宿主机
仍需预装 Python、Isaac Sim、Isaac Lab、PyTorch、NumPy 和 OpenCV。

本地 whole-body `policy.pt` 的权重专属再分发许可证尚未确认；本仓库内运行与
研究验收不等于获得公开二进制再分发授权。对外发布前必须取得授权或替换为许可
明确的权重，具体审计边界见
[assets/policies/go2_x5_pct_dog_only/PROVENANCE.md](assets/policies/go2_x5_pct_dog_only/PROVENANCE.md)。

以下 V0 链路和使用说明继续保留：

```text
Isaac Sim 场景
  → 物理传送带与目标刚体
  → 特权状态动态抓取 oracle
  → Go2-X5 机械臂与双指夹爪控制
  → 200/50/25 Hz 多速率采样
  → C0/C1 自动判定
  → 原子发布 episode 数据
```

V0 只支持单环境、单目标和固定机身，不以批量生成轨迹为目标。

## V2 快速入口

不启动 Isaac 即可先运行纯逻辑回归和重建场景俯视图：

```bash
cd Dynamic/conveyor_bench
conda activate env_isaaclab
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider
python scripts/render_v2_layout.py
python scripts/run_benchmark_v2.py \
  --scene transverse_near_sort_v2 \
  --task-family continuous_multi_target \
  --robot-mode fixed_base \
  --split train \
  --seed 0 \
  --dry-run-task
```

V2 的两个 scene ID 为 `transverse_near_sort_v2` 和
`mobile_remote_delivery_v2`。near 支持单目标、语言条件和 fixed-base 双目标
连续分拣；remote 只支持 whole-body 单目标或语言条件交付，成功还要求连续持物
底盘位移至少 `0.65 m`。完整的 smoke、严格校验、相机门禁和 M0/DynamicVLA
导出顺序见 [COLLECTION_V2_GUIDE.md](COLLECTION_V2_GUIDE.md)。

## V0 使用说明

## 目录

```text
.
├── assets/robots/go2_x5/       # 项目内固化的机器人 URDF、USD 与资产哈希
├── assets/objects/             # V1 的 8 个本地程序化物品及 seen/unseen
├── assets/receptacles/         # V1 近端分拣盒、下游接料盒与 V2 远端投放盒
├── assets/workcells/           # V1 传送带与 V2 远端工位的程序化资产
├── assets/policies/            # V1 本地 whole-body 移动策略与 contract
├── configs/v0.json             # V0 协议与横向布局快照
├── configs/v1.json             # V1 冻结快照
├── configs/v2.json             # V2 场景、任务矩阵与门槛快照
├── COLLECTION_GUIDE.md         # V1 门禁、采集、验收与导出操作手册
├── M0_ONLINE_GUIDE.md          # M0 服务、离线阶段门禁与 Isaac 在线闭环
├── COLLECTION_V2_GUIDE.md      # V2 从 smoke 到正式采集的操作手册
├── BENCHMARK_V2_SPEC.md        # V2 benchmark 规范
├── scripts/render_v2_layout.py # 不启动 Isaac 的本地 SVG 场景预览
├── scripts/run_benchmark_v2.py # V2 任务预检与仿真采集入口
├── scripts/validate_v2_dataset.py # V1 canonical + V2 语义严格校验
├── scripts/export_v2.py        # V2 到 DynamicVLA/M0 的离线投影
├── scripts/check_environment.py # 本地资产与依赖预检
├── scripts/run_conveyor.py     # C0/C1 仿真与采集入口
├── scripts/run_benchmark_v1.py # V1 fixed/whole-body 采集入口
├── scripts/validate_dataset.py # run/episode 数据完整性校验
├── scripts/validate_v1_dataset.py # V1 严格数据校验
├── scripts/audit_v1_episode.py # V1 episode 数据质量审计
├── scripts/check_v1_camera_gate.py # V1 相机时变与策略可见性门禁
├── scripts/export_v1.py        # V1 到 DynamicVLA/M0 的离线投影
├── scripts/smoke_m0_aml.py     # M0-Mobile AML loss/采样/checkpoint 烟测
├── scripts/probe_kinematics.py # 机械臂位姿探针
├── scripts/probe_scene.py      # 传送带接触与运动探针
├── scripts/probe_v1_scene.py   # V1 工作站与三相机探针
├── scripts/probe_mobile_locomotion.py # V1 浮动根移动策略门禁
├── src/conveyor_bench/
│   ├── isaac/                  # 场景、机器人配置和单一物理主循环
│   ├── m0_aml.py               # 最小纯 PyTorch AML action head
│   ├── v1/                     # V1 配置、协议、记录、审计与导出
│   ├── v2/                     # V2 task context、连续协调、校验与导出注解
│   ├── config.py               # 运行时 V0 常量
│   ├── oracle.py               # 纯 Python 动态抓取 oracle
│   ├── protocol.py             # task、episode、sample、event 契约
│   ├── metrics.py              # C0/C1 自动判定
│   ├── recorder.py             # 原子 episode 记录器
│   └── video.py                # 三相机同步 MP4 记录
└── tests/                      # 不启动 Isaac Sim 的协议与 oracle 测试
```

## 环境准备

使用已经安装 Isaac Sim、Isaac Lab、PyTorch 和 OpenCV 的 Python 3.11 环境。以下命令都从仓库工作区执行：

```bash
cd Dynamic/conveyor_bench
conda activate env_isaaclab
python -m pip install -e .
```

`conveyor_bench` 的协议、判定和记录部分只依赖 Python 标准库；`src/conveyor_bench/isaac/` 和视频记录部分使用仿真环境中已有的 Isaac Lab、PyTorch、NumPy 与 OpenCV。

V0 默认使用 `--device cpu` 运行 PhysX，RTX 相机仍由显卡渲染。这是当前
Isaac Sim 5.1 中传送带 `PhysxSurfaceVelocityAPI` 的稳定配置；C1 会拒绝
GPU PhysX，以免生成目标物穿透传送带的无效轨迹。单环境烟测和采集不受此
限制影响，后续若升级并重新通过动态接触门禁，再开放 GPU 并行物理。

先检查依赖和本地资产引用；该命令不会启动仿真：

```bash
python scripts/check_environment.py
```

先运行不启动仿真的单元测试：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider
```

## 最小烟测

先用 C0、无相机、单 episode 验证物理循环、控制、判定和结构化记录：

```bash
python scripts/run_conveyor.py \
  --task c0 \
  --episodes 1 \
  --seed 0 \
  --max-duration 8 \
  --output-dir outputs/smoke/c0 \
  --no-video \
  --require-all-success \
  --headless \
  --device cpu
```

随后用 C1 验证传送带表面速度与动态拦截逻辑：

```bash
python scripts/run_conveyor.py \
  --task c1 \
  --episodes 1 \
  --seed 0 \
  --belt-speed 0.08 \
  --max-duration 15 \
  --output-dir outputs/smoke/c1 \
  --no-video \
  --require-all-success \
  --headless \
  --device cpu
```

该烟测的 `belt_speed_mps=0.08` 是沿 `transport_direction_xyz=(0,-1,0)` 的
正向速度幅值；落盘的测量速度同样是世界 surface velocity 在该方向上的投影。

需要检查相机链路时，去掉 `--no-video` 并增加 `--enable_cameras`：

```bash
python scripts/run_conveyor.py \
  --task c1 \
  --episodes 1 \
  --seed 1 \
  --belt-speed 0.08 \
  --max-duration 15 \
  --output-dir outputs/smoke/c1_camera \
  --enable_cameras \
  --require-all-success \
  --headless \
  --device cpu
```

默认采集命令只要完整发布全部请求的 episode 就返回 `0`；抓取失败属于有效
benchmark 结果，仍保留轨迹、事件和失败原因。`--require-all-success` 用于上述
烟测门禁，任一任务失败时返回 `3`。运行时、录制器或不完整运行返回 `1`，
命令行参数错误由 `argparse` 返回 `2`。

## 正式采集命令

只有在 C0、C1 和相机烟测均通过后，才开始小批量正式采集。以下命令生成
20 条带狗头、腕部和观察相机 RGB 的 C1 轨迹：

```bash
python scripts/run_conveyor.py \
  --task c1 \
  --episodes 20 \
  --seed 1000 \
  --belt-speed 0.08 \
  --max-duration 15 \
  --output-dir outputs/collection/v0/c1_speed_008_seed_1000 \
  --enable_cameras \
  --headless \
  --device cpu
```

第 `i` 个 episode 使用 `seed + i`。扩大采集时，应使用新的、不重叠 seed 区间和新的输出目录；不要覆盖已有目录，也不要把 `outputs/smoke/` 中的调试轨迹混入正式训练集合。

可选的机械臂标定探针：

```bash
python scripts/probe_kinematics.py --headless --device cuda:0
```

## 输出目录契约

一次运行会产生一个 run summary 和若干 episode：

```text
outputs/collection/v0/c1_speed_008_seed_1000/
├── run-<UTC>-summary.json
└── episodes/
    └── run-<UTC>-ep0000-seed1000/
        ├── manifest.json
        ├── steps.jsonl
        ├── events.jsonl
        ├── summary.json
        ├── head_rgb.mp4
        ├── wrist_rgb.mp4
        ├── overview_rgb.mp4
        └── camera_frames.jsonl
```

使用 `--no-video` 时不生成最后四个相机文件。

- `manifest.json`：协议版本、任务、随机种子、资产哈希、仿真版本与频率。
- `steps.jsonl`：50 Hz 状态、动作、接触、安全标志和六类观测到执行时间戳。
- `events.jsonl`：阶段变化、闭爪、抬升、越界、成功或失败等稀疏事件。
- `summary.json`：单 episode 成败、失败原因和指标。
- `camera_frames.jsonl`：25 Hz 视频帧与仿真步、仿真时间的对应关系。
- `run-<UTC>-summary.json`：本次命令的 episode 路径和汇总结果。

V0 的狗头相机位于头部前缘并严格沿机身 `+X` 水平前视；腕部相机位于夹爪
中线上方并下俯 25°，在闭合与抬升阶段同时保留双指和目标。当前传送带
顶面降低到 `0.50 m`，接近机器狗头部和背部的工作高度；狗头相机保留真实
水平前视，但近侧皮带边缘会遮挡顶面目标的大部分区域，因此不承担抓取区
全局相机职责。固定的
`overview_rgb` 从更远的场景斜上方覆盖完整机器人、横向传送带、目标和抓取过程，仅用于
人工观察与数据质检，不属于策略观测。head/wrist 两路策略视频为
224×224，overview 观察视频为 480×320；三路均为 25 Hz，并共享
`camera_frames.jsonl` 的帧索引。

新 episode 的 task manifest 使用 `transport_direction_xyz` 与
`exit_plane_point_xyz` 描述运输轴和出口平面。`target_crossed_exit`、出口
剩余距离、目标前向速度与未来标签均由同一投影坐标计算；旧纵向开发数据中的
`exit_x_m` 仍可由协议和校验器读取；新的横向 task 中该兼容字段为 `null`。

记录期间数据位于 `episodes/.<episode-id>.<uuid>.inprogress/`。只有 manifest、JSONL、summary 和相机文件全部关闭后，目录才会被一次原子重命名为最终 episode 路径。异常中止也会发布失败 summary，不会静默丢弃失败轨迹。

每次烟测或采集后校验整个输出根目录：

```bash
python scripts/validate_dataset.py outputs/smoke/c1_camera
```

校验器复核 run/episode 计数、JSON/JSONL、时间单调性、成功证据、事件
一致性，以及三路视频和相机帧索引；通过返回 `0`，发现问题返回 `1`。

## 开始采集前的门槛

必须同时满足：

1. 全部纯 Python 单元测试通过。
2. C0 单 episode 的 `summary.json` 中 `success` 为 `true`。
3. C1 单 episode 的 `summary.json` 中 `success` 为 `true`。
4. 相机烟测包含三个可读取的 MP4，且 `camera_frames.jsonl` 非空。
5. `steps.jsonl` 中 `sim_step` 和 `sim_time_s` 严格递增。
6. 输出根目录下没有本次运行遗留的 `.inprogress` 目录。
7. 使用同一 seed 重跑时，任务 manifest 中的目标、车道和任务参数一致。

未通过这些门槛时，保留失败 episode 继续调试，不应增加采集规模。

## 当前横向场景预览

- [V1 release 三相机阶段拼图](outputs/previews/v1_release_contact_sheet.png)
- [V1 release 三相机同步视频](outputs/previews/v1_release_three_camera.mp4)
- [V1 release overview 视频](outputs/previews/v1_release_overview.mp4)

## V1 快速开始

V1 固定三时钟为 physics `400 Hz`、control `50 Hz`、camera/model
`25 Hz`。原始动作是 10D canonical action：

```text
[base vx, base vy, base wz,
 tcp dx, tcp dy, tcp dz,
 tcp dRx, tcp dRy, tcp dRz,
 gripper]
```

base 速度在 body frame，TCP 平移和 rotation-vector 增量在
robot-root/base frame；固定机身模式前三维必须为零。`head_rgb` 与
`wrist_rgb` 是 `policy_observation`，`overview_rgb` 永远只可
`observer_only`。

8 个本地物品冻结为 6 seen（其中 4 train、2 val）和 2 unseen；完整 ID 与
划分见 V1 规范。采集 CLI 进一步冻结为互斥的 `train` 4 个、`val` 2 个和
`unseen` 2 个；运行任务时默认激活 3 个 train 物体，只计分一个目标，目的地
是 `sort_bin_blue` 或 `sort_bin_yellow`。

三相机任务必须保持 Fabric 开启，不能传入 `--disable_fabric`。否则 PhysX
可能仍在运动，但 RTX/Hydra 观测可能停留在初始几何。`head_rgb` 和
`wrist_rgb` 是训练所需策略视角；`overview_rgb` 只用于观察，不能替代任一
策略相机。

### 环境与无仿真测试

```bash
cd Dynamic/conveyor_bench
conda activate env_isaaclab
python -m pip install -e .
python scripts/check_environment.py
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider
```

### 物理门禁

先检查完整工作站、运输方向和三相机：

```bash
python scripts/probe_v1_scene.py \
  --output-dir outputs/gate/v1_scene \
  --belt-speed 0.04 \
  --settle-seconds 1.0 \
  --enable_cameras \
  --headless \
  --device cpu
```

全身模式还必须先检查本地浮动根移动策略：

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

### 任务门禁与一条采集链路

固定机身：

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

全身模式：

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

全身闭环当前约需 `21.6 s`，所以该门禁显式使用 `35 s`，不能依赖 CLI 的
`20 s` 默认值。三物体中英双语选择任务的完整命令见
[COLLECTION_GUIDE.md](COLLECTION_GUIDE.md)。

`--require-all-success` 只用于门禁：任一物理任务失败时返回 `3`。正式数据
采集应去掉它，因为有明确失败原因且完整发布的失败 episode 也是有效 benchmark
数据。

### 数据验收与模型视图

```bash
python scripts/validate_v1_dataset.py outputs/gate/v1_whole_body
python scripts/audit_v1_episode.py outputs/gate/v1_whole_body/episodes/EPISODE_ID
python scripts/check_v1_camera_gate.py outputs/gate/v1_whole_body/episodes/EPISODE_ID
python scripts/export_v1.py outputs/gate/v1_whole_body/episodes/EPISODE_ID --profile both
```

把 `EPISODE_ID` 换成 run summary 中的实际目录名。audit 默认在 episode 内
生成 `quality_report.json`。相机门禁额外检查物理运动期间画面是否真实时变，
并要求策略相机中有目标变化证据；overview 不计入策略证据。export 会再次执行
严格 canonical 校验并要求 head/wrist 帧，然后默认生成：

```text
exports/
├── dynamicvla.jsonl
├── m0.jsonl
└── export_manifest.json
```

DynamicVLA 视图为 25 Hz、历史 `[-2,0]`、未来 TCP 偏移 `+5` tick、20-step
chunk；M0 视图把 TCP delta 投影到 world frame，形成左臂 7 个零加右臂 7D
的 14D、16-step chunk。两者都单独保留 body-frame base 3D、原始 canonical
10D 和有效位 mask，且不会改写 canonical episode。

面向移动底盘策略训练，使用因果 `m0_mobile` profile，而不是上述 legacy M0
同 tick 视图：

```bash
python scripts/export_v1.py \
  outputs/gate/v1_release_camera/episodes/EPISODE_ID \
  --profile m0_mobile

PYTHONPATH=src python scripts/smoke_m0_aml.py \
  --device cpu \
  --steps 250 \
  --output-dir outputs/smoke/m0_aml_cpu_RUN_ID
```

该 profile 只暴露 head/wrist、语言、`state28` 和未来 `16×10` 50 Hz 动作，
并排除 overview 与仿真特权字段。完整契约、H20 单卡 BF16 命令和验收边界见
[COLLECTION_GUIDE.md](COLLECTION_GUIDE.md)。

训练后的 action head 可通过本仓库内的服务端接入同一 Go2-X5 任务；部署顺序、
SHA 身份校验、阶段 fail-closed 门禁、闭环命令和最新实测结果见
[M0_ONLINE_GUIDE.md](M0_ONLINE_GUIDE.md)。当前 checkpoint 已跑通在线传输与
Isaac 动作链路，但 seed 0 仍为 `target_missed`，不得据此宣称策略抓取成功。

当前本地物理烟测已经观察到 fixed 单目标约 `10.48 s` 成功、whole-body
单目标约 `21.60 s` 成功，以及 whole-body 三物体双语目标选择约 `21.58 s`
成功；对应 canonical 输出通过 strict validator 和 quality audit。Fabric
修复后的 whole-body 单目标 release `outputs/gate/v1_release_camera` 包含
540 个同步 tick、1620 张 PNG，temporal camera gate 已实际通过，并完成各
540 条的 M0/DynamicVLA 双导出且 canonical 哈希未改变；对应源码树 SHA-256
为 `a5c2802447abd4e4c50365549b7b0cc83db313f01800cb26d734fc8fc695f39c`。
fixed 和三物体语言烟测尚未保存相机，所以不能据此宣称这两个配置的视觉门禁
已通过；下一步仍是小批量回归，而不是大规模采集。证据路径、负例和逐项状态
见采集手册。
