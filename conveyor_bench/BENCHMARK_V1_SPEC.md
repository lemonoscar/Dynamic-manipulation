# ConveyorBench V1 冻结规范

本文定义 `conveyor-bench-v1` 的场景、时钟、动作、记录、导出和验收边界。
可机读快照为 `configs/v1.json`。V1 的交付物是可启动数据采集的仿真框架，
不是已经训练好的 VLA 模型或大规模轨迹集。物理成功、数据结构、图像质量、
相机时变和模型导出分别验收，任何一项都不得代替其他门禁。

## 1. 自包含边界

所有项目代码、机器人与物体资产、传送带、分拣盒、全身移动策略权重、配置和
校验脚本都位于 `Dynamic/conveyor_bench/`。运行时：

- 不访问网络，不执行资产或权重下载；
- 不读取 `Dynamic/` 之外的项目仓库、模型权重或数字资产；
- 不使用外部 URL 或远程资产引用；
- 只把本机已安装的 Python、Isaac Sim、Isaac Lab、PyTorch、NumPy 和
  OpenCV 视为宿主运行环境。

因此，“自包含”指项目文件自包含，不等于把 Isaac Sim 本身复制进仓库。
所有下文命令均从 `Dynamic/conveyor_bench/` 执行。

`assets/asset_lock.json` 锁定机器人 USD/URDF、URDF 引用的全部唯一 mesh、
移动策略及 contract、物体注册表、分拣盘和工位 manifest。运行时必须先验证
这些哈希，并把 asset-lock 哈希与项目源码树指纹写入 episode manifest。

## 2. 场景与坐标约定

冻结布局 ID 为 `transverse_dynamic_sort_station_v1`：

- Go2-X5 面向世界 `+X`，世界 `+Y` 是机器人左侧，`-Y` 是右侧。
- 低位传送带位于机器人正前方，顶面高度 `0.34 m`，低于 whole-body
  模式狗头相机名义光轴 `0.37 m`；尺寸为 `0.252 × 1.56 × 0.06 m`，即
  长度为上一版的 `1.3×`、宽度为上一版的 `60%`。
- 运输单位向量固定为 `[0, -1, 0]`。物体从上游 `+Y` 进入、经过
  `Y=0` 拦截区并向 `-Y` 出口移动；在机器人视角中必须呈现为左到右。
- 物体使用靠近机器人的带内车道 `X=0.65 m`；出口平面参考点为
  `[0.65, -0.75, 0.34] m`。传送带几何中心仍为 `X=0.70 m`。
- 接触关键路径使用本地程序化刚体和 `PhysxSurfaceVelocityAPI`；视觉皮带、
  滚筒和机架不得改变运输碰撞体语义。

真实工位外观由本地程序化组件组成：无碰撞视觉皮带与接缝、驱动/从动滚筒、
两侧机架、四支腿、横撑、电机护罩、远侧安全栏、上下游光电传感器、急停、
出口标记和漏件接料盘。物理运输面保持为简单、可审计的独立碰撞体。完整设计
冻结在 `assets/workcells/conveyor_station_v1/ASSET_MANIFEST.json`。

任务为 `dynamic_sort`：从同时激活的目标与干扰物中抓取唯一计分目标，将它
放入两个计分分拣盒之一：

- `sort_bin_blue`
- `sort_bin_yellow`

`reject_catch` 只接住下游漏件，不是计分目的地。CLI 默认激活 3 个物体、
带速 `0.01 m/s`、episode 时限 `20 s`；whole-body 门禁必须显式覆盖为
`35 s`。目标越过出口、错抓、掉落、错误分拣盒、机器人跌倒、禁区碰撞和
超时等均记录为明确失败原因。

oracle 示教使用 `overhead_slow_pick_place_v2`：夹爪局部进给轴相对水平向下
俯转 `75°`（距竖直 `15°`），并按零件注册表对齐 `x/y` 双指闭合轴。预抓取与
预放置各连续俯视观察 `0.5 s`；50 Hz 控制下普通 Cartesian、竖直下探和抬升
单步分别不超过 `3.0/1.5/2.0 mm`，对应最高 `0.15/0.075/0.10 m/s`。抓取从
零件上方缓慢跟踪下探，投放从分拣盘上方缓慢到达释放位，禁止使用水平正对
物体或框壁的单位姿态生成新训练数据。夹爪以实测开度为起点，使用 `0.7 s`
五次 smoothstep 加 `0.3 s` 保持；首次双指接触后，TCP 在闭爪期间继续追踪
运动零件，保持完成后才允许抬升。

### 2.1 零速传送带诊断

`stationary_sort` 是独立的能力诊断，不是把 `dynamic_sort` 的速度标签改成零，
也不计入动态 benchmark 分数。它仍要求完成抓取、携带、放入正确分拣盒以及
连续 `0.5 s` 稳定判据；不是只检查闭爪的简化任务。冻结合同为：

- 传送带命令和每步实测速度均为 `0.0 m/s`；
- `single_target`、一个激活物体、`part_red_block → sort_bin_blue`；
- 物体直接生成在可达拦截位，spawn、staging、oracle 和评测使用同一坐标；
- 只有下表五个 seed 合法，整条 episode 按场景切分，禁止窗口级混切；
- 根部位置和朝向扰动暂时为零。seed 1102 的初测证明当前非零 root yaw 会把
  下探目标推到标定工作空间外，因此根扰动推迟到单独的可达性扩展，不得暗中
  重试或筛 seed。

| scenario split | seed | 物体 `(dx, dy)` / m |
| --- | ---: | ---: |
| train | 1101 | `(0.000, 0.000)` |
| train | 1102 | `(+0.020, +0.020)` |
| train | 1103 | `(-0.020, -0.020)` |
| val | 2101 | `(+0.010, -0.025)` |
| test | 3101 | `(-0.010, +0.025)` |

物体所属的 `curriculum_split=train` 与诊断场景的 `scenario_split` 是两个不同
概念。AL0 的 legacy `m0_mobile` 导出字段 `split` 必须采用后者，并把前者另存为
`object_curriculum_split`，否则 val/test 会因红色方块属于 train 资产而泄漏。
strict validator 逐步核对实测带速，exporter 还会复核固定 target/destination
合同。

## 3. 本地物体与 seen/unseen

V1 冻结 8 个本地程序化物体，注册表为 `assets/objects/registry.json`：

| 划分 | 本地 ID |
| --- | --- |
| train/seen | `part_red_block`, `part_blue_bar`, `part_yellow_bushing`, `part_green_shaft` |
| val/seen | `part_silver_hex`, `part_orange_flange` |
| unseen | `part_purple_bracket`, `part_cyan_gear` |

当前单物体抓取课程采用确定性的治具对齐上料：蓝色长条和绿色轴绕竖直轴
旋转 90 度，使四种 train 物体都使用同一条、经过实机模型工作空间校准的
俯视平行夹抓腕部支路。该约束只固定初始上料朝向，不改变物体几何、质量或
后续运动。四种物体各自的抓取偏移还把有效夹持垫中心统一到
`z=0.376 m`；这是夹爪接触几何标注，并非抬高物体。未来的任意朝向抓取应
作为独立课程扩展，不得混入本阶段示教数据。

因此粗粒度划分是 6 个 seen、2 个 unseen；seen 内再冻结 4 个 train 与 2 个
val。manifest 必须记录每个实例的 `asset_id`、类别、seen/unseen 标记、seed、
目标 ID 和目的分拣盒。不得在导出阶段改变划分。

## 4. 机器人模式

正式 V1 采集入口只接受：

1. `fixed_base`：机器人根部固定，canonical action 的前三维必须严格为零；
2. `whole_body_policy`：根部浮动，使用仓库内
   `assets/policies/go2_x5_pct_dog_only/policy.pt` 以 50 Hz 控腿，同时执行
   机械臂和夹爪命令。

`whole_body_policy` 不是根节点运动学搬运。其本地权重哈希和 contract 必须写入
运行 manifest。协议枚举中的开发调试模式不属于 V1 正式采集门禁。
算法评测以 `whole_body_policy` 的移动—动态抓取—携带—放置闭环为主；
`fixed_base` 仅作为去除底盘移动能力后的机械臂消融。

X5 的 canonical TCP 与 PCT 一致，位于
`arm_link6 + (0.15757, 0, 0) m` 的 FinRay tip frame。原 FinRay 可视和碰撞
mesh 均保留；运行时在第一次 reset 前仅对 `arm_link7/8` 的原始碰撞 mesh
应用 `convexDecomposition`、2 mm contact offset 和零 rest offset，不创建
代理几何或额外刚体。双指 contact sensor 仍绑定原 finger link；patch 结果
必须写入 episode manifest。

## 5. 三时钟与同步

| 时钟 | 频率 | 语义 |
| --- | ---: | --- |
| physics | 400 Hz | PhysX、接触与机器人动力学 |
| control | 50 Hz | canonical 状态/动作、对象状态和事件采样 |
| camera/model | 25 Hz | 三相机采样与模型 tick |

每个 control 周期包含 8 个 physics step；每个 camera/model tick 对应 2 个
control 样本。导出器在每个模型 tick 选择最新的 control 样本，并合并该 tick
内最新的相机引用。`sim_step` 与 `sim_time_s` 必须严格递增，`model_tick`
不得递减。

冻结的监督时间参数：

- 视觉历史 model tick：`[-2, 0]`；
- 物体未来标签：`[0, 2, 5, 10, 20]` model tick；
- DynamicVLA 末端未来偏移：`+5` model tick，即 `0.20 s`。

尾部缺失数据必须以零填充并由有效位 mask 标记，不得伪造未来样本。

## 6. Canonical 10D action

`steps.jsonl` 和 `action_chunks.jsonl` 的唯一源动作是：

```text
0  base_vx_body_mps
1  base_vy_body_mps
2  base_wz_body_radps
3  tcp_dx_base_m
4  tcp_dy_base_m
5  tcp_dz_base_m
6  tcp_drx_base_rad
7  tcp_dry_base_rad
8  tcp_drz_base_rad
9  gripper
```

前三维是 body frame 平面速度；3–8 维是 robot-root/base frame 的 TCP 平移
增量与 rotation-vector 增量；夹爪范围为 `[-1, 1]`，`+1` 全开、`-1` 全闭，
中间值表示连续命令开度，不得重新二值化。
四元数在其他位姿字段中统一为 `wxyz`。任何模型适配都必须离线投影，不能把
world-frame 或模型专用动作回写成 canonical action。

## 7. 相机权限

| 相机 | 分辨率 | 频率 | 权限 |
| --- | ---: | ---: | --- |
| `head_rgb` | 640×480 | 25 Hz | `policy_observation` |
| `wrist_rgb` | 640×480 | 25 Hz | `policy_observation` |
| `overview_rgb` | 480×320 | 25 Hz | `observer_only` |

训练或在线策略只能读取 head、wrist、语言和允许的机器人本体状态。
`overview_rgb` 仅用于人工观察、回放和数据质量检查，不得进入策略输入。使用
`--save-camera-frames` 时，三路图像以无损 PNG 写入 `cameras/<camera_id>/`，
并由 `camera_frames.jsonl` 映射到 physics step 和采样时间。

冻结外参与 PCT `pct_scene@c7fe62c7` 一致：

- `head_rgb` 挂载在 `base` 的 `(0.28, 0.0, 0.07) m`，ROS `wxyz` 为
  `(0.5, -0.5, 0.5, -0.5)`，沿机身 `+X` 水平直面前方；
- `wrist_rgb` 挂载在 `arm_link6` 的
  `(0.0666580792, 0.0028071889, 0.0935779972) m`，ROS `wxyz` 为
  `(0.3377891849, -0.6214992221, 0.6185057335, -0.3421810063)`；这是
  `arm_link6_T_camera_color_optical` 手眼标定加 PCT v3 视觉对齐后的位姿；
- `overview_rgb` 固定在环境坐标 `(-2.10, -1.60, 2.40) m`，从拉远的
  斜上方覆盖机器人、传送带和两个分拣盘。

两路机器人相机共用 D436 640×480 OpenCV 内参：
`fx=383.44608095, fy=383.52724198, cx=324.33479864,
cy=238.90275478`，12 项畸变系数均为零。head 裁剪面为
`[0.1, 100000] m`，wrist 为 `[0.03, 5.0] m`。AL0 模型输入不是原始标定
图像：LeRobot 转换器会中心裁剪到 480×480 后缩放为 224×224，并记录等效
内参。

相机运行必须保持 Fabric 开启，不得传入 `--disable_fabric`。PhysX 在 Fabric
关闭时仍可能推进，但 RTX/Hydra 可能读取冻结的初始几何，产生看似有效却不
随物理状态变化的图像。manifest 必须记录 `use_fabric=true`；最终验收还必须
执行时变相机门禁，不能只检查 PNG 存在。

目标真值、未来状态、接触真值和成功判据同样只用于 teacher、监督标签或
evaluator，不是视觉策略输入。

## 8. 原始 episode 契约

发布后的 episode 至少包含：

```text
episodes/EPISODE_ID/
├── manifest.json
├── steps.jsonl
├── objects.jsonl
├── action_chunks.jsonl
├── events.jsonl
├── summary.json
├── camera_frames.jsonl            # 开启帧保存时
└── cameras/
    ├── head_rgb/*.png
    ├── wrist_rgb/*.png
    └── overview_rgb/*.png
```

- `manifest.json`：完整 task、seed、版本、频率、相机、资产与权重哈希。
- `steps.jsonl`：50 Hz canonical 状态/动作、时间戳、phase 与相机引用。
- `objects.jsonl`：各活动物体状态和冻结未来 horizon 标签。
- `action_chunks.jsonl`：chunk 的观测来源、生成/有效/执行窗口及丢弃数量。
- `events.jsonl`：spawn、phase、抓取、释放、放置、失败和 episode 结束事件。
- `summary.json`：任务结果、失败原因、计数和指标。

写入期间使用隐藏的 `.inprogress` 目录；所有流关闭并同步后才原子发布。
失败任务仍是可分析的 benchmark 数据并应保留。只有结构损坏、非有限数、
频率/时间错乱、引用越界或运行错误才是数据损坏。

成功要求目标释放后位于指定 goal zone，线速度不高于 `0.02 m/s`、角速度
不高于 `0.10 rad/s`，并连续保持 `0.50 s`。任务失败与数据损坏必须分开报告。

## 9. ConveyorVLA AL0 与 DynamicVLA 离线导出

canonical episode 永远是事实源。导出器只在 episode 的 `exports/` 新建文件，
并在 `export_manifest.json` 中记录 canonical 哈希；不得改写六个原始流。
训练导出前必须通过 episode 级 strict canonical validation，并实际记录
`head_rgb` 与 `wrist_rgb`；overview 可存在但不是训练资格条件。结构损坏、
真实未来标签不一致、`runtime_error` 或缺任一策略相机时必须 fail closed。
具有明确物理失败原因但结构完整的 benchmark episode 仍可保留其失败语义。

DynamicVLA 契约：

- 输出 `exports/dynamicvla.jsonl`，25 Hz，每条记录保留 `canonical_action10_chunk`；
- `state6` 和 7D TCP delta action 使用 robot-root/base frame；
- 历史为 `[-2, 0]`，未来 TCP 标签偏移 `+5` tick；
- action chunk 长度 20，base 3D body action 单独输出；
- 历史、canonical 与未来动作分别携带有效位 mask。

AL0 的 legacy `m0` 契约：

- 输出 `exports/m0.jsonl`，25 Hz，每条记录保留 `canonical_action10_chunk`；
- `state6_world` 与 7D arm delta action 使用 world frame；
- 用当前 `robot_root_world` 四元数把 canonical TCP 平移和旋转向量旋转到世界系；
- action chunk 长度 16；
- 14D 动作为左臂 7 个零加右臂 7D action，base 3D body action 单独输出；
- 尾部携带 `action_valid_mask`。

上述 `m0` profile 是为已有消费端保留的 25 Hz world-frame 投影。canonical
step 记录的是动作执行后的状态，因此同 tick action 不得作为该状态的因果训练
标签。移动底盘策略训练必须使用独立的 `m0_mobile` profile：

- 观测取带同步相机的原始 50 Hz control row `t`，监督严格取后续
  `t+1 ... t+16` 共 16 个 canonical action；尾部不足 16 步直接丢弃；
- 策略输入只含 `head_rgb`、`wrist_rgb`、语言和同一观测 row 的 `state28`；
  `overview_rgb`、phase、目标物 ID、物体真值及未来状态一律不导出；
- `state28` 依次为 body-frame 根部线/角速度 6、投影重力 3、机械臂前六关节
  位置/速度 12、base-frame TCP xyz/旋转向量 6、测量夹爪开度 1；
- 动作保持 canonical 10D 布局和 50 Hz 频率，同时给出 model 10D；model 夹爪
  使用 `0=close, 1=open`，不支持的 `base_vy` 维 mask 为 false；
- 导出 schema 为 `conveyor-bench-m0-mobile-v1`，输出及 canonical 哈希写入统一
  `export_manifest.json`。

导出命令：

```bash
python scripts/export_v1.py \
  outputs/gate/v1_whole_body/episodes/EPISODE_ID \
  --profile all
```

`both` 保留原 DynamicVLA/legacy `m0` 两个 profile；`all` 额外生成 AL0 因果
`m0_mobile.jsonl`。已有导出默认拒绝覆盖。只有明确需要重建派生文件时才加
`--force`；该选项仍不会覆盖 canonical 文件。

## 10. 验收命令

先激活已安装 Isaac 环境并进入项目：

```bash
cd Dynamic/conveyor_bench
conda activate env_isaaclab
python -m pip install -e .
python scripts/check_environment.py
```

### 10.1 协议与数据逻辑

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider
```

### 10.2 场景物理与视觉

```bash
python scripts/probe_v1_scene.py \
  --output-dir outputs/gate/v1_scene \
  --belt-speed 0.04 \
  --settle-seconds 1.0 \
  --enable_cameras \
  --headless \
  --device cpu
```

该探针应生成三路 PNG、拼图和 `report.json`；它是场景/传送带/相机检查，
不是抓取成功证明。

全身模式先单独通过浮动根部移动策略门禁：

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

### 10.3 固定机身任务门禁

```bash
python scripts/run_benchmark_v1.py \
  --robot-mode fixed_base \
  --episodes 1 \
  --seed 0 \
  --split train \
  --task-family single_target \
  --belt-speed 0.01 \
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

### 10.4 全身任务门禁

```bash
python scripts/run_benchmark_v1.py \
  --robot-mode whole_body_policy \
  --episodes 1 \
  --seed 0 \
  --split train \
  --task-family single_target \
  --belt-speed 0.01 \
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

全身闭环当前约需 `21.6 s`，因此门禁必须显式给出足够的
`--max-duration`；CLI 的 `20 s` 默认值主要保留给固定机身快速烟测。多物体
双语任务使用 `--task-family language_conditioned --instruction-language en_zh`，
train 可激活 3 个物体，val/unseen 各自最多激活 2 个 split-local 物体。

普通采集不加 `--require-all-success`：物理完成且成功或失败都应发布数据并返回
`0`。门禁加该参数后，任一任务失败返回 `3`；参数错误返回 `2`，运行/录制
异常返回 `1`。

### 10.5 零速诊断门禁

训练场景只需一次 Isaac 启动：

```bash
python scripts/run_benchmark_v1.py \
  --robot-mode whole_body_policy \
  --episodes 3 \
  --seed 1101 \
  --split train \
  --task-family single_target \
  --belt-speed 0 \
  --max-duration 30 \
  --active-objects 1 \
  --target-asset part_red_block \
  --destination sort_bin_blue \
  --output-dir outputs/gate/v1_stationary_train \
  --enable_cameras \
  --save-camera-frames \
  --require-all-success \
  --headless \
  --device cpu
```

val/test 必须分别使用 seed `2101` 与 `3101` 运行，不能通过重复随机 seed 挑选
成功轨迹。CLI 的 `--split train` 在这里仍表示资产注册表划分；真正的诊断数据
切分由 seed 对应的 `scenario_split` 冻结。三组输出各自通过 strict validator、
temporal camera gate 后再导出；只有 seeds 1101–1103 可以进入训练。

### 10.6 数据验收与导出

```bash
python scripts/validate_v1_dataset.py outputs/gate/v1_whole_body
python scripts/audit_v1_episode.py outputs/gate/v1_whole_body/episodes/EPISODE_ID
python scripts/check_v1_camera_gate.py outputs/gate/v1_whole_body/episodes/EPISODE_ID
python scripts/export_v1.py outputs/gate/v1_whole_body/episodes/EPISODE_ID --profile both
```

把 `EPISODE_ID` 替换为 run summary 中发布的实际目录名。validator 复核
run/episode/流/PNG/成功证据，audit 生成 `quality_report.json` 并把任务结果
与数据质量分开。audit 仅在数据损坏时返回 `2`；无损坏的告警仍返回 `0`。
camera gate 还要求物理发生位移时三相机图像确实随时间变化，并在
head/wrist 中找到目标变化证据；overview 永远不计入策略证据。exporter 会
再次执行严格 canonical 校验，拒绝真实未来标签不一致、运行错误或缺少
head/wrist 训练相机的 episode。

## 11. 放量条件与状态声明

开始小批量采集前，必须实际记录以下结果：

1. 环境预检与纯 Python 测试通过；
2. 场景探针的三路图像可读取，物体沿 `+Y → -Y` 运输；
3. 移动策略 probe 通过后，全身任务才可运行；
4. fixed 与 whole-body 各至少一个 `--require-all-success` episode 通过；
5. 对应输出根目录通过 V1 validator，episode 通过 quality audit；
6. 相机 episode 通过 temporal camera gate，且 overview 没有被当作策略证据；
7. AL0 与 DynamicVLA 导出生成且 canonical 哈希未变化；
8. 无遗留 `.inprogress`，同一 seed 的 task/资产选择可复现。

本规范冻结的是“可采集框架”的接口和验收方法。当前本地烟测已经观察到：

- fixed 单目标成功，约 `10.48 s`；
- whole-body 单目标成功，约 `21.60 s`；
- whole-body 三物体、中英双语目标选择成功，约 `21.58 s`。

上述 canonical 输出均通过 strict validator 和 quality audit。带三相机的
whole-body 单目标 release 输出 `v1_release_camera` 包含 540 个同步 tick 和
1620 张 PNG，已实际通过 temporal camera gate；head/wrist/overview 最大结构
变化率分别为 `0.704164/0.688824/0.039858`，策略相机目标证据为
`0.760409`。同一 episode 已生成各 540 条的 AL0/DynamicVLA 导出，
`export_manifest.json` 记录 `canonical_files_modified=false`。
该 release 的源码树 SHA-256 为
`a5c2802447abd4e4c50365549b7b0cc83db313f01800cb26d734fc8fc695f39c`。

fixed 与三物体语言烟测未保存相机，所以只能声明各自物理/data 闭环通过，不能
声明这两个配置的视觉门禁通过。Fabric 修复前的冻结相机 episode 继续保留为
camera-gate 负例。逐条证据与命令见 `COLLECTION_GUIDE.md`。当前也不得宣称
已经获得训练模型或大规模数据集。
