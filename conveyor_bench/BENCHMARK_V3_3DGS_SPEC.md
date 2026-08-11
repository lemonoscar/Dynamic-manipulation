# ConveyorBench V3：arm-vla 风格 3DGS 场景方案

本文件定义一个独立的 3DGS 视觉场景版本。机器可读配置为
[configs/v3_3dgs.json](configs/v3_3dgs.json)。当前状态是 `design`，尚未取得真实
场景的 Gaussian PLY 与标定文件，因此 `collection_ready=false`；不能把本分支
误当成已经通过采集门禁的版本。

## 1. 结论

V3 采用与 `arm-vla-grasp-sim` 相同的核心分层方式，并补齐其尚未完成的数据采集
环节：

```text
Isaac/PhysX：机器人、传送带、物体、盒子、碰撞、控制和标签
             │ 每个 25 Hz 相机 tick 的 RGB/depth/alpha + 世界位姿
             ▼
3DGS/gsplat：静态真实工位背景的 RGB/depth/alpha
             │ 同一内参、同一外参、同一 frame_index
             ▼
深度合成：   静态 3DGS + 动态 Isaac 前景
             │
             ▼
V1 raw PNG → 既有校验器 → 既有 raw-to-LeRobot MP4 → ConveyorVLA AL0
```

纯 3DGS 不能提供可靠碰撞，也不能直接表示运动的机器狗、机械臂、零件和传送带。
因此不能把整个场景换成一份 Gaussian PLY。参考工程当前已具备 Gaussian PLY
检查、`gsplat` 离线背景渲染和 overview 相机轨迹导出，但机器人/物体前景合成仍
是待完成项。V3 保留其正确的双层结构，同时把三相机轨迹、前景深度和合成纳入
正式验收，才能生成训练可用数据。

## 2. 版本边界

- V1 canonical 协议不变：400 Hz physics、50 Hz control、25 Hz camera/model、
  10D action、状态、事件、对象流和成功判据全部保持原样。
- ConveyorVLA AL0 的 head/wrist 输入、时序窗口和 LeRobot MP4 格式不变。
- V3 首版只覆盖当前近端横向传送带的单目标 stationary/dynamic grasp。V2 remote
  delivery 需要更大的实景重建范围，不能在同一次标定尚未验证时顺带开放。
- 3DGS 是视觉 profile，不是新的 teacher、控制器或任务定义。
- 正式运行只允许读取 `Dynamic/conveyor_bench/` 内的文件，不联网、不下载、
  不使用指向其他仓库的软链接。

## 3. 照搬与改造范围

调研基线固定为 `arm-vla-grasp-sim` 的本地提交
`b0f4f39ddf7ce2a94ad5c174e48da0ec31f6534a`。V3 按行为复现以下模式，最终实现和
资产均落在本仓库中：

| 参考模式 | V3 对应实现 |
| --- | --- |
| Gaussian PLY header/property 检查 | 本地 PLY validator，拒绝普通 mesh PLY |
| 分块读取、stride 与 `max_gaussians` | preview 限点，正式采集加载全量 |
| `gsplat` CUDA 离线 rasterization | 三相机批量 RGB+D 渲染 |
| USD 相机世界位姿导出 | head/wrist/overview 每帧统一 pose JSONL |
| visual 与 collision 分离 | 3DGS 静态视觉 + 既有 V1 collision/PhysX |
| 坐标变换后再渲染 | 显式 `Sim(3)` 标定，把 Gaussian 变换到 sim world |

不直接复制参考工程的导航、多楼层、PCT 或 locomotion 代码；它们与传送带抓取无关。
也不沿用只用 look-at 的静态相机接口，因为 head 与 wrist 都随机器人运动，必须记录
完整世界变换。

## 4. 场景分层

### 4.1 3DGS 静态层

静态层包含真实房间壳体、地面、固定背景、固定照明外观、传送带机架和安全围栏。
3DGS 只负责可见外观，永远不参与 PhysX 碰撞和任务判定。

### 4.2 Isaac 动态层

以下内容必须从 3DGS 训练图像中移除或做 mask，并由 Isaac 渲染：

- Go2-X5、X5 机械臂和夹爪；
- 运动皮带表面；
- 全部任务零件、接料盒和任务标记；
- 人员和其他会移动的物体。

这样可避免“实景里留着一台旧机器人，仿真里又出现一台机器人”的重影。传送带
固定机架可以进入 3DGS，但深绿色运动皮带仍由 Isaac 生成，保证物理表面、颜色和
运动状态一致。

## 5. 3DGS 资产制作与标定

资产目录固定为：

```text
assets/workcells/conveyor_station_v3_3dgs/
├── scene_static_gaussians.ply
├── calibration_sim_from_gs.json
├── photometric_calibration.json
├── capture_masks/
└── ASSET_MANIFEST.json
```

推荐使用空工位实拍：固定曝光、焦距和白平衡，覆盖 overview 距离、狗头高度和
腕部近距离三类视域。训练集必须包含传送带上下游、抓取区和放置区的近景；只围绕
房间远距离拍摄，会导致 wrist camera 在抓取时出现孔洞。若无法清空机器人和零件，
必须为每张训练图像提供同名 mask，不能用一张固定 mask 代替所有视角。

Gaussian PLY 使用 binary little-endian float32，至少包含 `xyz`、DC SH 颜色、
opacity、三个 scale 和四元数 rotation。正式资产必须写入顶点数、训练参数、原始
图像集合 ID、许可证和 SHA-256；目录内不得使用软链接。

标定文件保存一个带尺度的相似变换 `S_sim_from_gs`。加载时先把 Gaussian 的中心、
旋转和尺度变换到 Isaac 世界坐标，之后三台相机可直接使用仿真世界位姿渲染。至少
使用 12 个覆盖地面、传送带四角、机架高点和放置区的控制点，目标为：

- 中位重投影误差不高于 1.5 px；
- P95 重投影误差不高于 3 px；
- 实际尺度误差不高于 1%。

## 6. 三相机同步

V3 保留现有已校准安装：head 为 `base` 前向相机，wrist 位于 `arm_link6` 的夹爪
上方并向下微俯视，overview 使用当前拉远的第三视角。分辨率、角色和频率不变：

| camera | 分辨率 | 频率 | 权限 |
| --- | ---: | ---: | --- |
| `head_rgb` | 640×480 | 25 Hz | policy observation |
| `wrist_rgb` | 640×480 | 25 Hz | policy observation |
| `overview_rgb` | 480×320 | 25 Hz | observer only |

每个相机 tick 在 Fabric transform 更新之后读取 USD world transform，并写入
`camera_poses.jsonl`。每条记录必须包含相同的 `frame_index/sim_step/timestamp`、
3×3 内参、4×4 世界位姿和裁剪面。记录的是 USD camera frame（局部 `-Z` 向前、
`+Y` 向上），渲染器负责一次性转换为 gsplat 的相机约定。禁止根据机器人状态重新
近似 wrist pose，也禁止把 overview 输入训练。

## 7. 两遍渲染与合成

第一遍运行现有 Isaac 物理主循环。静态外观隐藏但碰撞保持启用，输出动态前景的
RGB、metric depth 和 instance ID，并由 instance ID 生成 alpha，同时写 canonical
状态/action/label 与三相机位姿。

第二遍离线批量运行 `gsplat`，从同一条 pose 记录输出静态背景 RGB、metric depth
和 alpha。合成器按最近有效深度选像素，并使用 3 mm 深度容差处理软边缘。合成完成
后才把无损 PNG 原子发布到 canonical `cameras/<camera_id>/`；中间图默认只保留失败
episode 和少量 debug 样本。

两遍渲染是首版的有意选择：它不要求 3DGS 达到实时 25 FPS，只要求最终帧严格对应
25 Hz 仿真时间。这样不会因 gsplat 推理抖动改变控制频率或 teacher 行为，也便于在
GPU 2/3 上把物理采集与背景批渲染分开调度。

## 8. 验收与放量顺序

必须依次通过以下阶段，任一阶段失败都不能开始正式采集：

1. **资产门禁**：PLY 属性、manifest、SHA-256、许可证和无软链接检查。
2. **静态预览**：三台相机各至少 20 个 held-out pose；PSNR ≥ 25 dB、SSIM ≥ 0.85。
3. **几何合成**：标定道具跨静态/动态边界的 P95 edge error ≤ 3 px，无遮挡黑洞。
4. **静止抓取 smoke**：一条完整成功 episode，无缺帧、重帧、黑帧和冻结帧。
5. **低速动态 smoke**：至少三个 seed 跑通现有慢速俯视抓取流程。
6. **数据门禁**：V1 strict validator、temporal camera gate 和 LeRobot round-trip
   全部通过。
7. **小批量**：20 条 pilot 数据人工查看 head、wrist、overview 和失败样本后，才
   允许设置 `collection_ready=true` 并开启大规模采集。

任务失败可以作为 benchmark 失败样本保留；丢帧、错位、重影、校准错误或合成失败
属于数据损坏，不能计作任务失败。

## 9. 分支实施顺序

本分支首先冻结本规范和机器可读配置。审核后按最短闭环实施：

1. 在本仓库实现小型 PLY validator、分块 loader 和 `gsplat` RGB+D renderer；
2. 扩展 V1 recorder，一次写出三台相机的真实世界位姿；
3. 增加 Isaac 动态前景 pass 与深度合成器；
4. 放入真实 3DGS 资产和 `S_sim_from_gs` 标定；
5. 运行 stationary → low-speed dynamic → 20 episode pilot 三层门禁；
6. 通过后再接回现有 raw-to-LeRobot 转换和 AL0 训练。

首版不实现原生 Isaac viewport 3DGS 插件、不重写 canonical recorder、不改变训练
模型，也不扩展 remote delivery。等离线两遍路径稳定后，再评估实时渲染是否值得。
