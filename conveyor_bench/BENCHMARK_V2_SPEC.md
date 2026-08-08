# ConveyorBench V2 Benchmark 规范

本文定义 ConveyorBench V2 的任务、场景、资产、记录、评价与放量边界。V2 的
目标是为“移动机器人在横向传送带上动态抓取零件，并按指令把零件投放到指定
位置”的新 VLA 策略提供可采集、可复现、可严格验收的仿真接口。它服务于
ConveyorVLA AL0（继承 ABot-Manipulation M0 checkpoint 结构）的动作建模与
DynamicVLA 风格的视觉—语言—动作学习，
但本交付不包含训练好的策略，也不把尚未执行的仿真门禁写成已通过结果。

机器可读的 suite 快照为 [configs/v2.json](configs/v2.json)，采集步骤见
[COLLECTION_V2_GUIDE.md](COLLECTION_V2_GUIDE.md)，场景俯视图见
[docs/images/conveyorbench_v2_layout.svg](docs/images/conveyorbench_v2_layout.svg)。

## 1. 版本边界：V1 canonical，V2 suite

V2 不复制也不改写 V1 的原始数据协议：

- `benchmark_suite_version = conveyor-bench-v2` 表示场景与任务套件版本；
- `canonical_protocol_version = conveyor-bench-v1` 表示 episode 的状态、10D
  action、对象流、事件流、相机同步和原子发布仍使用 V1 契约；
- `task_context_schema_version = conveyor-bench-v2-task-context-1` 只描述 V2
  新增的场景、目标序列、目的地区域和服务门控；
- V1 的冻结基线、既有数据和 `assets/asset_lock.json` 保持不变；V2 使用独立的
  [assets/asset_lock_v2.json](assets/asset_lock_v2.json)。

V2 suite 元数据的权威位置是：

```text
manifest.episode.task.metadata.benchmark_suite
```

其中至少记录 scene、task family、robot mode、顺序目标、每个目标的目的地、
spawn policy、service gates、目的地几何和持物移动距离门槛。episode metadata
可以镜像该对象；若存在镜像，必须与 task metadata 完全一致。实例 ID 与本地
asset ID 一一对应，并由 `instance_asset_map` 显式记录，避免导出时混淆实例和
几何类别。

## 2. 坐标、工位与离线资产

Go2-X5 初始面向世界 `+X`；世界 `+Y` 是机器人左侧，`-Y` 是右侧。两个 V2
场景都复用 V1 的横向低位真实工位语义：传送带顶面高 `0.34 m`，尺寸为
`0.252 × 1.56 × 0.06 m`，物体沿 `[0, -1, 0]` 从 `+Y → -Y` 运动，因此
从机器人视角看是左到右。

物理运输面与外观组件分离。皮带/滚筒、机架/支腿、护罩、安全栏、光电传感器、
急停、出口标记和接料盘均由项目内程序化几何生成；V2 地面也是本地静态 cuboid，
不组合 Isaac 的在线环境 USD。运行时不下载 mesh、纹理、USD 或权重，也不读取
`Dynamic/conveyor_bench/` 之外的项目文件。宿主机已安装的 Isaac Sim、
Isaac Lab、Python、PyTorch、NumPy 和 OpenCV 属于运行环境，不属于仓库资产。

V2 资产锁在 V1 机器人、mesh、移动策略、8 类零件、近端工位和接料盒基础上，
额外锁定远端投放盒与远端工位 manifest。程序化资产使场景可以先稳定验证物理
和任务语义；高保真视觉资产只能作为后续受控扩展，不能悄悄改变碰撞、质量、
目标区或相机契约。

## 3. 两个场景

### 3.1 `transverse_near_sort_v2`

该场景保留 V1 横向传送带和近端蓝/黄分拣盘，用于基线、语言选择和连续分拣：

| 目标区 | 中心 `(x,y,z)` m | 半尺寸 `(x,y,z)` m |
| --- | --- | --- |
| `sort_bin_blue` | `(0.34, 0.40, 0.40)` | `(0.105, 0.125, 0.075)` |
| `sort_bin_yellow` | `(0.34, -0.40, 0.40)` | `(0.105, 0.125, 0.075)` |

默认 episode 上限为 `45 s`。该 profile 不要求持物底盘位移；它的作用是把
“目标选择、动态抓取和正确投放”与远距离移动能力分开评测。

### 3.2 `mobile_remote_delivery_v2`

该场景保留传送带和下游漏件接料盘，关闭会阻挡转向走廊的两个近端分拣盘，
新增两座远端投放台：

| 目标区 | 盒中心 `(x,y,z)` m | 期望根部 `(x,y)` m | 期望 yaw | standoff |
| --- | --- | --- | ---: | ---: |
| `delivery_bin_blue` | `(-0.16, 1.20, 0.46)` | `(-0.16, 0.78)` | `+π/2` | `0.42 m` |
| `delivery_bin_yellow` | `(-0.16, -1.20, 0.46)` | `(-0.16, -0.78)` | `-π/2` | `0.42 m` |

地面上可见的导航走廊仅作观察标记，不产生碰撞。当前场景不加入需要感知和规划
绕行的障碍物，因为现有 teacher 的地形观测尚不足以定义可信的避障任务。
默认 episode 上限为 `60 s`。

远端成功除了满足 V1 的正确盒稳定放置判据，还必须在目标连续处于
`in_gripper=true` 的持物段内、释放前观察到机器人根部平面位移至少
`0.65 m`。该距离从该连续持物段的首个有效根部位置计算最大欧氏位移；空载
移动不能充当远端交付证据。

## 4. 冻结任务矩阵

V2 初始版只接受以下 7 个组合；其他组合必须在启动 Isaac 前 fail closed：

| 场景 | task family | robot mode | 用途 |
| --- | --- | --- | --- |
| near | `single_target` | `fixed_base` | 机械臂最小基线 |
| near | `single_target` | `whole_body_policy` | 移动动态抓取主基线 |
| near | `language_conditioned` | `fixed_base` | 语言选择消融 |
| near | `language_conditioned` | `whole_body_policy` | 语言条件全身抓取 |
| near | `continuous_multi_target` | `fixed_base` | 双目标连续服务首版 |
| remote | `single_target` | `whole_body_policy` | 强制持物移动交付 |
| remote | `language_conditioned` | `whole_body_policy` | 语言选择与移动交付 |

表中 `near` 和 `remote` 分别是
`transverse_near_sort_v2` 与 `mobile_remote_delivery_v2` 的简称。

以下组合有意不开放：

- remote + `fixed_base`：固定根部无法满足 `0.65 m` 持物移动门槛；
- remote + `continuous_multi_target`：尚未证明远端投放后的返程、重新对准和
  下一目标重抓闭环；
- near + continuous + `whole_body_policy`：尚未完成全身模式的回位/再武装
  物理门禁。

拒绝这些组合是 benchmark 定义的一部分，不能通过绕过 CLI 或改 manifest
启用。

## 5. 任务与物体

V2 继续使用 [assets/objects/registry.json](assets/objects/registry.json) 中的
8 类本地程序化零件：train/seen 4 类、val/seen 2 类、unseen 2 类。split、
几何、质量、摩擦、稳定姿态、抓取 affordance 和中英文别名继承 V1，禁止在
导出阶段重划分。

- `single_target`：一个计分目标，完成一次抓取—携带—投放。
- `language_conditioned`：目标和干扰物同时存在；中英文/双语指令决定唯一计分
  对象与目的地。
- `continuous_multi_target`：恰好两个不重复的计分目标，按
  `target_sequence_ids` 顺序服务，共享一个 episode 时间预算。

连续任务采用 `service_gated` 生成：第一个目标由 `episode_start` 门控，第二个
目标只有在前一目标完成后才由 `previous_target_completed` 门控。两个目标在
manifest 中预先注册，记录流中的对象注册表固定；尚未轮到的目标保持 inactive/
offstage，而不是中途改变 schema。第一个目标成功只推进子任务，不结束 episode；
任一目标失败立即使整个 episode 失败，只有两个目标按序稳定放置才算成功。
每次目标切换必须反映在 `steps.jsonl.selected_object_id` 和事件序列中。

## 6. 机器人、teacher 与 canonical action

V2 仍以 V1 canonical 10D action 为唯一事实源：

```text
[base_vx, base_vy, base_wz,
 tcp_dx, tcp_dy, tcp_dz, tcp_drx, tcp_dry, tcp_drz,
 gripper]
```

base 3D 是 body-frame 平面速度；TCP 6D 是 robot-root/base-frame 增量；夹爪
范围为 `[-1,1]`。`fixed_base` 的 base 三维必须为零；`whole_body_policy`
使用项目内固化的 Go2-X5 locomotion policy，不允许直接运动学改写根节点。

特权状态 teacher 负责生成最小可验证轨迹，并为后续 VLA 数据采集提供监督，
不代表最终部署策略。near 单目标复用动态抓取与投放逻辑；连续任务在外层顺序
协调器中复用单目标 teacher；remote teacher 的阶段至少覆盖抓取后收臂、原地
对准、持物导航、到达稳定、预放置、下降、开爪、撤离和稳定性验证。

## 7. 时钟、相机与权限

V2 继承 V1 的 `400/50/25 Hz` physics/control/camera-model 时钟和同步语义。

| 相机 | 语义 | 权限 |
| --- | --- | --- |
| `head_rgb` | 安装在狗头，沿机身 `+X` 水平直面前方 | `policy_observation` |
| `wrist_rgb` | 安装在夹爪中线上方，向下微俯视抓取/放置区 | `policy_observation` |
| `overview_rgb` | 固定第三视角，观察机器人、传送带、目标与移动路径 | `observer_only` |

near 沿用 V1 的远景 observer 相机。remote 为覆盖更长路线，把 observer 相机
拉远到 `(-2.80, -2.60, 3.20) m`，分辨率仍为 `480×320`、25 Hz。
`overview_rgb` 绝不能成为策略输入；目标真值、接触、未来标签、当前目标索引
也只能用于 teacher、监督或 evaluator。相机采集必须保持 Fabric 开启，并通过
时变门禁，PNG 文件存在本身不是视觉链路成功证据。

## 8. 记录、事件与成功判据

V2 episode 继续原子发布 V1 的：

```text
manifest.json
steps.jsonl
objects.jsonl
action_chunks.jsonl
events.jsonl
summary.json
camera_frames.jsonl + cameras/   # 开启无损相机帧时
```

V1 strict validator 先检查 canonical 协议、时间、引用、图像、未来标签和通用
成功证据；V2 validator 再检查 suite metadata 和 V2 特有语义。成功 episode
额外要求：

- continuous：`target_selected → object_spawned → object_placed` 事件按目标
  序列完整出现，spawn 不早于 `not_before_s`，且后一目标不得早于前一目标
  完成 service gate；step 中的 `selected_object_id` 也必须按同一顺序切换；
- remote：robot mode 是 `whole_body_policy`，并满足至少 `0.65 m` 的连续
  持物底盘位移。

失败 episode 仍应完整发布并保留物理失败原因；V2 特有成功证据只在
`summary.success=true` 时强制，但 canonical 结构与 suite metadata 对成功和
失败都必须合法。运行异常、流损坏或版本不一致不是任务失败，而是数据损坏。

## 9. ConveyorVLA AL0 与 DynamicVLA 投影

V2 exporter 包装 V1 的 lossless 离线投影，不改变 canonical 文件，也不把
模型专用动作写回原始流：

- DynamicVLA 保留 V1 的 base-frame state/action、视觉历史、未来 TCP 标签和
  20-step action chunk；
- AL0 的 legacy `m0` 视图保留 V1 的 world-frame state/arm delta、右臂 14D 适配、base 3D 和
  16-step action chunk；
- 两种投影都增加 `scene_id`、`task_family`、`target_sequence_ids`、
  `destination_zone_by_target`、`current_target_id` 和
  `current_subtask_index`；
- `current_target_id` 与 `current_subtask_index` 明确标记为
  `supervision_only_fields`，不能当成部署时可直接获得的策略观测。

导出前必须先通过 canonical V1 与 V2 严格校验，并保存 head/wrist 训练相机。
observer-only overview 是否存在不决定训练资格。

### 9.1 算法可见性边界

为了让 AL0 的分层控制和 DynamicVLA 风格的视觉 action chunk 在同一数据上
公平对比，输入、监督和评价信息必须分层：

| 信息 | teacher/记录 | 在线 VLA |
| --- | --- | --- |
| 中英文任务指令 | 是 | 是 |
| head/wrist 历史帧 | 是 | 是 |
| 允许的机器人本体状态 | 是 | 是 |
| canonical 10D action/chunk | teacher 输出 | 策略预测目标 |
| overview 图像 | 是 | 否，observer only |
| 物体真值、接触、未来轨迹 | 是 | 否，仅监督/evaluator |
| current target/subtask index | 是 | 否，仅监督 |

模型比较应始终回到 canonical episode 的统一评价：目标选择是否正确、是否在
出口前完成抓取、是否稳定放入正确区域、机器人是否跌倒/碰撞，以及 continuous
吞吐或 remote loaded displacement。模型自己的 world/base frame 投影和 chunk
长度不能改变成功定义。

## 10. 当前验证状态与冻结条件

纯 Python 的 V2 配置、任务矩阵、顺序协调、验证和导出逻辑可以在不启动
Isaac 的情况下回归；俯视 SVG 也完全从本地 JSON/manifest 生成。synthetic
data-plane smoke 已跑通 Recorder、双目标 V2 strict validation 与两种 iterator，
且 canonical 6 个源文件哈希未变化。

冻结源码候选 `0a2fd7c…` 已完成真实 Isaac CPU PhysX 的 near fixed-base
continuous 正例：seed 0、`0.06 m/s`、45 s 上限，两个目标在 `13.98 s` 与
`26.16 s` 稳定放置；最终 2/2 成功，记录 1327 个 control sample、2654 个
object row，并得到 `1/1 valid, 0 errors` 的 V2 strict validation。

同一工作站还完成 remote whole-body 蓝/黄双向单目标正例。黄色 seed 0 的最大
连续持物根部位移为 `0.778166 m`，蓝色 seed 2 为 `0.735903 m`，两者均通过
V2 strict validation。远端托盘高度 `0.46 m` 和 `x=-0.16 m` 携物走廊由双向
物理可达性结果覆盖。

RTX 4060 上的 near fixed single 与 remote whole-body 三相机正例分别记录
294/600 个同步 tick、882/1800 张 PNG，均通过 temporal camera gate、V2 strict
validation 和 DynamicVLA/AL0 双导出。全部最终正例携带同一 source-tree SHA-256
`0a2fd7c20f2ef62e1ab8c13ef6d871f779b5871088fb46093f994753a291514b`。早期 Isaac
默认 ground 的远程 USD 依赖已由 V2 本地程序化地面消除；near whole-body、near
continuous 相机、语言条件与其余物体/seed/速度矩阵仍未通过。

因此在实际得到并保存以下证据前，不得声明 V2 已可大规模采集：

1. 两个场景都能在目标 Isaac 环境离线启动；
2. 传送带动态接触、两个目标连续服务和远端持物导航分别有成功 episode；
3. 三相机随物理状态变化且 head/wrist 含有效任务证据；
4. canonical、V2 strict validation、相机门禁和双导出全部通过；
5. asset lock、源码指纹、run summary 和无 `.inprogress` 残留均可审计；
6. 少量不重叠 seed 覆盖 train/val/unseen、蓝/黄目的地和冻结速度档后，再决定
   正式采集规模。

## 11. 后续资产与任务扩展原则

现阶段不需要下载其他资源即可跑通协议、场景与采集框架。完成首轮物理门禁后，
高价值扩展按以下顺序进行：

1. 为现有 8 类程序化零件制作保持相同碰撞与质量的视觉材质变体，用于 texture/
   illumination domain randomization；
2. 增加尺寸、质量、抓取 affordance 可控的本地 CAD 零件，并先完成许可证、
   单位、碰撞简化和哈希清单；
3. 在已有无障碍 remote 闭环稳定后，再增加带明确感知输入的窄通道、静态障碍
   和移动投放点；
4. 完成全身模式返程/再武装后，才开放 whole-body continuous 与 remote
   continuous；
5. 最后增加带速变化、短时扰动、遮挡、光照和相机标定扰动。

任何外部资源都必须先拷入本仓库、记录来源与许可证、离线可加载并纳入 V2
asset lock；不能让采集运行依赖下载链接，也不能为了视觉丰富改变冻结的物理
评价边界。
