# Benchmark 规范

本文描述当前唯一的 ConveyorBench 任务。机器可读参数在
[`configs/benchmark.json`](../configs/benchmark.json)，数据字段以
`src/conveyor_bench/schema/` 为准。

## 1. 任务目标

一个完整 episode 包含四个连续阶段：

```text
Navigation-to-conveyor
  → 机器狗移动到可见、可抓、安全的操作位姿
  → 通过位置、航向和速度驻车门禁
Dynamic grasp
  → 俯视观察并持续跟随传送带目标
  → 缓慢下降、闭合并稳定抬升
Loaded navigation-to-bin
  → 机械臂收回到紧凑负载位姿
  → 机器狗转向、移动到分类箱前并再次驻车
Overhead place
  → 从上方移动、开爪投放并确认瓶子进入目标框
```

完整成功要求四个阶段都成功。`fixed_base` 仅用于隔离机械臂和动态抓取问题，
属于消融/教师模式，不能代替移动操作结果。

## 2. 场景

静态背景来自 Liangzhu NuRec/3DGS，动态前景由 Isaac Sim 生成：

| 组成 | 渲染 | 物理 |
| --- | --- | --- |
| Liangzhu 房间 | NuRec/RTX | 扫描碰撞经过验证后禁用，任务区使用解析地面 |
| Go2-X5 | Isaac USD/URDF | Articulation |
| 传送带 | Isaac 程序化几何 | kinematic rigid body + surface velocity |
| 零件 | sidecar USD 视觉 | SI 单位解析碰撞 fixture |
| 投放盒 | Isaac 程序化几何 | 静态碰撞 |

NuRec 保持原始世界坐标，不移动 Gaussian 根节点。机器人和工位放置在 PCT 可乐抓取
区域的已标定锚点。运行时先验证 NuRec volume、field、碰撞层和任务地面，再开始 reset。

## 3. 机器人和相机

机器人资产严格参考 `arm-vla-grasp-sim/pct_scene` 的 Go2-X5、FinRay 夹爪、TCP
和 D436 标定。策略只读取 head 和 wrist；overview 只供人类观察和审计。

| 相机 | 分辨率 | 频率 | 安装与用途 |
| --- | ---: | ---: | --- |
| `head_rgb` | 640×480 | 25 Hz | 狗头前向，策略输入 |
| `wrist_rgb` | 640×480 | 25 Hz | `arm_link6` 手眼外参，俯视夹爪，策略输入 |
| `overview_rgb` | 480×320 | 25 Hz | 拉远第三视角，仅审计 |

相机内外参是数据合同。修改 mount、分辨率、帧率、畸变或角色后，必须同步更新
`benchmark.json`、测试和数据 schema，旧数据不得静默混入。

## 4. 传送带和教师

当前传送带尺寸为 `0.252 × 1.56 × 0.06 m`，表面高度 `0.34 m`，沿世界 `-Y`
运输；从机器人视角观察为从左向右。表面使用深绿色线性 RGB
`(0.015, 0.10, 0.035)`。

当前动态速度为 `0.01 m/s`。教师不得停在预测点等待目标撞入夹爪，而必须：

1. 在 episode 初始化时把目标放到上游并立即随传送带运动；
2. 机器狗接近传送带时让目标进入策略相机视野；
3. 从目标上方建立俯视观察并在水平面持续跟随；
4. 保持跟随的同时缓慢下降并平滑闭合夹爪；
5. 接触后继续跟随，直到稳定夹持；
6. 垂直抬升，底盘锁定，机械臂收回标准携带位；
7. 收臂完成后才允许直退、左转、导航和投放；
8. 可乐进入目标框且仍被夹持后才松爪。

当前教师 profile 是 `overhead_target_follow_pick_place_v4`。历史 `v3` episode 保持
原身份，但不能混入要求“初始化即出现、收臂后再移动”的新训练配额。

## 5. 物品

当前正式注册物品只有 `cola`：

- 视觉来自 SSH sidecar 的真实 USD；
- 碰撞为半径 `0.0325 m`、高度 `0.12 m` 的圆柱；
- 质量 `0.12 kg`；
- 抓取采用 top-down、Y 轴闭合的 FinRay affordance。

sidecar 中虽然已有 apple、orange、bottle、box2 等文件，但“文件存在”不等于“可采
训练数据”。每个新物品必须分别冻结尺度、稳定姿态、质量、摩擦、碰撞、夹爪开度和
抓取 affordance，并完成 stationary 与 dynamic 成功门禁后才能加入对象池。

## 6. 时钟

| 时钟 | 频率 | 用途 |
| --- | ---: | --- |
| physics | 400 Hz | PhysX 与关节控制 |
| control | 50 Hz | canonical 状态与动作 |
| camera/model | 25 Hz | 三路 RGB、future label 和模型动作 |
| policy query | 5 Hz | ConveyorVLA 在线查询 |

所有状态、图像、动作和事件必须携带明确 tick；禁止按文件名或读取顺序猜测时间。

## 7. 成功和失败

训练可用 episode 必须同时满足：

- 仿真运行完整，无 runtime error 或 `.inprogress` 残留；
- 任务 `success=true`；
- canonical strict validator 通过；
- quality audit 通过；
- head/wrist/overview 相机门禁通过；
- canonical 文件在 export 前后哈希不变；
- 教师 profile、物品 fixture 和场景 provenance 与当前合同一致；
- 没有 assisted diagnostic 控制；
- 目标曾被正确夹持，开爪释放后中心首次进入指定目标框；进入时仍在滚动不影响成功，
  不要求静止或驻留；
- `mobile_approach → track → close → lift → carry_retract → carry_backoff → carry_turn →
  carry_navigate → place_descend → open → verify_place` 顺序完整；
- 物体在初始化时出现并连续运动；收臂期间底盘锁定且物体保持夹持；
- 接近传送带至少 `0.20 m`，收臂后负向直退至少 `0.30 m`，负载导航至少 `0.10 m`；
- 到达蓝框并开始放置后，底盘保持为零，直至物体入框且夹爪松开。

任务失败可保留用于诊断或未来失败建模，但不能计入专家成功配额。结构损坏、丢帧、
哈希不一致、运行异常和资产不一致的数据必须隔离，不得作为负样本。

## 8. 规划中的正式矩阵

目标矩阵是 `4 个训练零件 × 2 档动态速度 × 每 cell 48 条成功轨迹 = 384 条`。
这是目标，不是当前实现已支持的配置。启动前必须满足：

1. 四个训练零件全部完成 fixture 和逐物品门禁；
2. 第二档速度完成教师、相机和闭环节拍验证；
3. 每个 cell 的 seed 池互不重叠；
4. GPU 2/3 双 worker 小规模 pilot 全部成功；
5. raw → LeRobot v3 往返和四视频首帧解码通过；
6. locomotion 若进入训练范围，必须先通过 Navigation 门禁。

在这些条件完成前，只允许小批量单可乐罐采集，不能把重复同一条件的数据描述为
384 条正式 benchmark。
