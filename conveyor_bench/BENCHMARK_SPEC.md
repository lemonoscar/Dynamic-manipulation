# ConveyorBench V0 协议

## 1. 范围

协议版本为 `conveyor-bench-v0`。V0 的目标是验证 Go2-X5 固定机身条件下，从物理仿真、动态抓取控制到可审计数据落盘的完整链路。

V0 包含：

- 单个 Go2-X5，腿部保持默认站立姿态。
- 单个红色长方体目标。
- 横向布置在机器狗正前方、使用 PhysX surface velocity 驱动的直线传送带。
- 头部、腕部两个机器人 RGB 相机，以及一个 observer-only 第三视角相机。
- 使用特权物体位姿和传送带命令速度先验的匀速预测 oracle。
- C0/C1 统一数据契约、成功判定和失败分类。

V0 不包含：

- 底盘移动或 whole-body 控制。
- 多物体、语言条件分拣和连续流任务。
- 学习策略训练或大规模并行环境。
- 将特权 oracle 状态声明为视觉策略输入。

相机安装是协议的一部分：

- `head_rgb` 挂载于 `base` 的狗头前缘，偏移为
  `(0.355, 0.0, 0.06) m`，沿机器人 `+X` 水平前视。
- `wrist_rgb` 挂载于 `arm_link6` 的夹爪中线上方，偏移为
  `(0.02, 0.0, 0.125) m`，沿夹爪前方下俯 25°。
- `overview_rgb` 固定在环境坐标
  `(-1.20, -0.70, 1.80) m`，从更远的斜上方观察机器人、
  传送带和抓取区；它只用于人工观察和数据质检。
- 三个外参均使用 `world` orientation convention，并冻结到每条 episode
  的 `manifest.json`。

狗头相机是物理第一人称视角，不承担全局 overview 功能。当前 V0 中传送带
顶面为 `0.50 m`，接近机器狗头部和背部的工作高度；水平前视保留真实的
机载前向视角，但近侧皮带边缘会遮挡顶面目标的大部分区域。腕部相机承担
末端抓取观察，固定第三视角承担全局观察。视觉策略的默认观测只包含
`head_rgb` 和 `wrist_rgb`。

场景坐标约定为：机器狗朝世界 `+X`，机器狗左侧为世界 `+Y`、右侧为世界
`-Y`。传送带长轴平行世界 `Y`，目标从 `+Y` 端生成并沿 `-Y` 运动，因此从
机器狗和 overview 视角看均是从左向右经过抓取区。当前布局标识为
`transverse_y_negative_low_v2`。

## 2. 任务定义

### 2.1 C0：静态抓取

协议名称：`c0_static_pick`。

- 传送带任务速度为 `0.0 m/s`。
- `|belt_speed_mps|` 不得大于 `0.01 m/s`。
- 目标位于固定机身机械臂的拦截区域。
- 用于验证机器人资产、夹爪接触、IK、抬升、判定和记录。

C0 是系统校准任务，不作为动态抓取结果。

### 2.2 C1：动态抓取

协议名称：`c1_dynamic_pick`。

- 传送带沿世界坐标 `-Y` 方向运动，即机器狗视角从左向右。
- 默认任务速度为 `0.08 m/s`。
- 合法动态任务速度的绝对值不得小于 `0.02 m/s`；当前运行入口要求正速度。
- `belt_speed_mps` 始终记录沿运输方向的正标量；世界速度向量为
  `(0, -belt_speed_mps, 0)`。
- 目标从传送带上游生成，并在越过冻结的出口位置前完成抓取验证。
- oracle 根据传送带命令速度预测短时未来位置，逐控制周期更新拦截目标；首次
  接触后的刚体反弹速度不得反馈为输送速度。

C1 是 V0 的正式 benchmark 任务。

## 3. 多速率时钟

| 数据域 | 频率 | 周期 | 说明 |
|---|---:|---:|---|
| PhysX 物理步 | 200 Hz | 0.005 s | 每控制周期执行 4 个物理步 |
| 控制与结构化状态 | 50 Hz | 0.020 s | oracle、动作、状态和接触采样 |
| RGB 相机 | 25 Hz | 0.040 s | 每 2 个控制周期记录一帧 |

同一 episode 内：

- `sim_step` 严格递增。
- `sim_time_s` 严格递增。
- 相机帧通过 `camera_frames.jsonl` 关联到 `sim_step` 和 `sim_time_s`。
- 物理判定使用仿真时间，不使用墙钟时间。
- 推理链路时间戳使用进程启动后的单调墙钟秒数。

每个控制样本可记录：

1. `observation_capture_s`
2. `inference_start_s`
3. `inference_end_s`
4. `action_enqueue_s`
5. `action_execute_start_s`
6. `action_execute_end_s`

这些时间戳用于后续分析观测年龄、推理延迟和动作执行延迟，不参与 C0/C1 物理成功判定。

## 4. Task 与 Episode manifest

任务必须在 episode 开始前解析并冻结。`TaskManifest` 至少包含：

- `task_id`、`task_type` 和自然语言指令。
- `target_object_id` 与该 episode 的全部 `object_ids`。
- 任务 seed。
- 名义传送带速度、皮带顶面高度、运输方向单位向量
  `transport_direction_xyz` 和出口平面点 `exit_plane_point_xyz`。
- 最大任务时长。
- 已采样的 `lane_axis_xyz`、`lane_offset_m`、生成位置和布局标识等 metadata。

越过出口统一定义为
`dot(object_xyz - exit_plane_point_xyz, transport_direction_xyz) >= 0`。
早期纵向开发数据中的 `exit_x_m` 只作为 `+X` 方向的兼容字段读取；新 episode
不再依赖轴绑定字段。

`EpisodeManifest` 至少包含：

- `episode_id`、`run_id` 和协议版本。
- 完整 `TaskManifest`。
- UTC 创建时间与 `env_id`。
- 机器人资产 SHA-256。
- episode、布局等子 seed。
- Isaac Sim、Isaac Lab、计算设备和多速率配置。

一个 task manifest 必须能在不依赖运行时随机采样的情况下还原任务参数。

## 5. 50 Hz 样本契约

每条 `steps.jsonl` 记录至少包括：

- `sim_step`、`sim_time_s`、`env_id`。
- 目标 `object_xyz` 与 `object_linear_velocity`。
- TCP 位置与姿态。
- 机器人关节位置与速度。
- 传送带命令速度和测量速度。
- 夹爪闭合命令状态、左指接触和右指接触。
- `target_in_gripper`、`target_crossed_exit`。
- `robot_fallen`、`forbidden_collision` 和 `wrong_object_grasped`。
- oracle 阶段。
- 目标 TCP、机械臂关节目标和夹爪目标等动作。
- 六类观测到执行时间戳。
- 可选相机帧索引和诊断 metadata。

GT 目标位姿、速度和接触可用于 oracle、标签与 evaluator；未来视觉策略不得把这些特权字段作为策略观测。

命令速度、测量速度以及 `object_transport_speed_mps` 都是世界速度向量在
`transport_direction_xyz` 上的投影，因此本布局中的 `-Y` 世界速度仍记录为
正的前向速度。

`gripper_closed` 表示闭合命令正在执行，而不是空夹时的最小关节位置。物体
位于夹指之间时，关节无法到达空夹极限；物理夹持必须继续由双指目标接触和
`target_in_gripper` 的几何条件共同证明。

## 6. 成功判据

某一控制样本只有同时满足以下条件，才是 secure sample：

1. `gripper_closed == true`。
2. `left_contact == true`。
3. `right_contact == true`。
4. `target_in_gripper == true`。
5. `object_z - belt_surface_z >= 0.05 m`。

成功要求 secure sample 连续保持至少 `1.0 s`。保持期间任一条件中断，连续计时立即清零。

C1 额外要求：上述 1 秒验证必须在任何 `target_crossed_exit == true` 的样本之前完成。C0 不使用出口作为失败条件。

## 7. 失败原因与优先级

逐样本判定按以下顺序执行：

1. `robot_fallen`
2. `forbidden_collision`
3. `wrong_object`
4. C1 的 `target_missed`
5. 连续保持达到阈值后的成功

episode 未成功且没有上述终止事件时：

- 超过任务时长：`timeout`
- 曾进入 secure 状态但未保持完成：`dropped`
- 从未形成有效 secure 状态：`grasp_not_secured`

协议层还定义：

- `no_samples`
- `invalid_task_configuration`
- `aborted`
- `recorder_error`
- `runtime_error`

失败 episode 必须保留。训练数据导出可以选择成功轨迹，但原始 benchmark 数据不得静默删除失败尝试。

## 8. 指标

每个 episode 的 `summary.json` 至少包含：

- `success`
- `failure_reason`
- `sample_count`
- `duration_s`
- `completion_time_s`
- `verification_time_s`
- `max_lift_m`
- `tcp_path_length_m`
- `belt_speed_mae_mps`
- `target_crossed_exit`
- 本协议的抬升高度与保持时间阈值

一次运行的顶层 summary 还应包含请求 episode 数、成功 episode 数以及每条 episode 的路径和指标。

当前 V0 只生成单次运行报告，不宣称统计显著性。进入正式 benchmark 比较后，应使用冻结 seed manifest，按 seed 汇报成功率，并保留逐 episode 结果。

## 9. 事件流

`events.jsonl` 使用稀疏事件记录任务语义。已定义事件包括：

- `episode_start`
- `object_spawned`
- `phase_changed`
- `camera_frame`
- `gripper_closed`（闭合命令开始）
- `target_lifted`
- `grasp_verified`
- `target_crossed_exit`
- `failure`
- `episode_end`

每个事件包含 `kind`、`time_s` 和 `payload`。`episode_end` 必须包含最终 `success` 和 `failure_reason`。

## 10. 输出与事务边界

输出根目录契约：

```text
<output-root>/
├── run-<UTC>-summary.json
└── episodes/
    └── <episode-id>/
        ├── manifest.json
        ├── steps.jsonl
        ├── events.jsonl
        ├── summary.json
        ├── head_rgb.mp4
        ├── wrist_rgb.mp4
        ├── overview_rgb.mp4
        └── camera_frames.jsonl
```

无视频模式不生成四个相机文件。

写盘规则：

1. episode 开始时创建隐藏的 `.inprogress` staging 目录。
2. manifest 先原子写入 staging。
3. steps 与 events 写入临时 JSONL。
4. 视频和帧索引直接写入同一个 staging。
5. 所有流关闭后，JSONL 和 summary 在 staging 中原子发布。
6. 整个 staging 目录一次重命名为最终 episode 目录。
7. 失败与异常中止走同一发布路径。

最终 episode 目录一旦可见，就必须至少包含 manifest、steps、events 和 summary。现有最终目录不得被覆盖。

## 11. Seed 与重现性

- 命令参数 `--seed N` 对应第一条 episode。
- 第 `i` 条 episode 使用 `N + i`。
- V0 用 episode seed 决定目标沿 `lane_axis_xyz=(1,0,0)` 的前后车道偏移；
  左右输送方向不随 seed 改变。
- manifest 保存 episode 和布局 seed。
- 相同协议、资产、设备配置和 seed 应产生相同任务 manifest。
- 训练、验证和测试使用互不重叠的 seed 区间与输出目录。

PhysX 接触轨迹仍可能受到设备和仿真版本影响，因此报告中必须保留版本、设备和资产哈希。

## 12. 采集门禁

开始正式采集前必须通过：

1. [tests/](tests/) 中全部纯 Python 测试。
2. 单条 C0 成功。
3. 单条 C1 成功。
4. 带相机 C1 生成三个可读 MP4 和非空帧索引。
5. 结构化流满足严格时间单调性。
6. 成功和失败 episode 都能生成完整 summary。
7. 运行结束后没有对应 run 遗留的 `.inprogress` 目录。

命令和检查顺序见 [README.md](README.md)。
