# Manipulation_Navi_v1 系统接线与启动级验证

- 日期：2026-08-27 CST
- 分支：`Manipulation_Navi_v1`
- 范围：无 fresh data、无 checkpoint、无训练的 runtime/recorder 接线
- 结论：纯 Python 系统接线与批准 Isaac 接口启动门禁通过；真实 stage/reset 生命周期通过，
  但 H20 Vulkan/RTX device-creation 警告仍阻塞相机和 GPU 物理健康结论

## 1. 远端旧任务处置

操作前按实时 host、用户、canonical root、Git、Conda、GPU、tmux 和 PID 逐项核验。确认旧
Waypoint v2 训练的 launch contract 标记为 agent-owned 后，只向该 tmux session 发送一次
`Ctrl-C`，没有按名称模糊杀进程，也没有触碰 GPU1 上归属其他用户的 StarVLA。

停止前最后一个有限、完整 optimizer event 为 step 1722；最后 durable checkpoint 仍为
step 1500。训练进程和两个 rank 均退出后，GPU2/3 为 `0 MiB / 0%`，旧日志、run state 和
checkpoint 原样保留。旧 `run_state.json` 仍写着 `running`，这是 SIGINT 没有完成状态落盘的
历史事实，本轮没有回写或伪装修复。

## 2. 新增接线

### 2.1 NAV：完整 reference、批准 PCT/DWA、固定重观察

`joint_trajectory_system.py` 新增独立 successor executor：

1. 校验并把模型的 `[10,3] @ 0.20 s` 全部从 query-body 变换到 world；
2. 保留全部 10 点到 trace，拒绝 K、prefix selector 和本地 stall detector；
3. 批准版 PCT API 只有起点/终点接口，因此只把第 10 点作为 PCT local goal；前 9 点保留作
   reference 形状与连续性审计，不虚构为 PCT 原生 via-point；
4. PCT path 交给真实 DWA 接口，每个指令再经过既有纵向 locomotion 包络：
   `|vx|<=0.30`、`vy=0`、`|wz|<=0.35`，非零 `|vx|<0.16` 归零；
5. 最多执行精确 `2.0 s = 100 × 0.02 s`，到达 local goal 可提前重观察；planner/controller
   异常只下发一次零速 arm/gripper hold 后 fail-closed。

这里有一个必须继续保留的诚实边界：现有批准 PCT 不是多 via-point API。若后续需要让前 9
点直接塑造规划器路径，必须升级 PCT 合同并单独验证，不能在当前 adapter 中暗中循环调用或
宣称已经支持。

### 2.2 Mani：direct joint、连续夹爪、零底盘

Isaac 批准 runtime 已有 `RobotAction.arm_joint_positions`，也支持 metadata 中的连续
`gripper_joint_positions`。新 adapter 因此直接生成：

- 6 维 arm position target；
- `[0,1]` open fraction 线性映射到实际夹爪 joint range，`gripper_command="hold"`，不再
  二值化成 open/close；
- 每个 0.04 s target 执行两个 50 Hz control tick，10 点共 20 tick；
- PICK/PLACE requested 和 applied base command 都必须精确为三维零；
- 不调用 IK、cuRobo、`plan_pose`、可行性 selector，也不启用会 teleport base 的外部
  manipulation base lock。

pending 和 NAV 继续使用 inference session 给出的上一安全 arm/gripper endpoint。route 仍只
来自 Qwen Pass 1 的双观测 commit，system executor 没有改 route 的接口。

### 2.3 evaluator truth 隔离

目标 box 的 world-frame valid area 从 episode 的 `place.placement_region` 解析。released 由
固定抓取约束的明确释放报告，或 measured gripper open fraction + object/TCP separation
联合判断；与目标内部条件连续满足 1.0 s 才 success。

truth adapter 与 system executor 是两个独立对象。object pose、TCP truth、valid area 和
success dwell 均不会进入 model request、route commit、PCT/DWA 或 direct-joint action。

### 2.4 fresh raw recorder

`joint_trajectory_recording.py` 新增不可覆盖的 staging→atomic rename recorder：

- `joint_commands_50hz.jsonl`：要求连续 tick、严格 20 ms、measured q/dq/gripper、requested
  和 controller-applied arm/gripper/base、base pose/twist、route 与 applied provenance；
- 每 tick 要求 Isaac arm/gripper apply count 相对前一 state 真实增加，不能把旧 report 或
  measured q 冒充 applied target；
- `joint_queries_5hz.jsonl`：要求连续 0.20 s query、精确 head/wrist `[t-0.20,t]`、合法 split、
  route、物理 progress provenance 和已存在的 episode-relative JPEG；
- overview 可保存并用于审计，但不进入 head/wrist 模型资产；
- 只有 `finalize()` 后才发布 episode；它不生成正式 dataset manifest、normalizer 或训练 row。

共享 applied-command validator 同时收紧为：`sim_step`、`model_tick` 和
`base_command_requested` 必填；PICK/PLACE requested/applied base 任一非零都会拒绝。

## 3. 已执行验证

### 3.1 本地合同与回归

- 新 system/recording 测试覆盖：10 点 world trace 与第 10 点 PCT endpoint、100 tick NAV、
  locomotion envelope、10×2 Mani 时钟、连续夹爪、零 base、无 IK/cuRobo、DWA fail-closed、
  fresh Isaac apply count、success dwell、atomic recorder 和时钟/资产拒绝；
- joint-trajectory 定向集合：`26 passed`；
- 加冻结 Waypoint 回归的联合集合：`74 passed, 2 skipped`；两个 skip 分别是当前环境缺少
  `accelerate` 导致一个旧训练模块跳过，以及无 CUDA 时的旧 device-alignment 测试；
- 新 Python 文件通过 `py_compile`，diff 通过 whitespace 检查；
- `scripts/check_joint_trajectory_system.py` 对批准 reference commit/cleanliness、真实
  `RobotAction`/`IsaacLabNavigationRuntime` import、连续夹爪、NAV→PCT/DWA fixture 和目标区域
  解析返回 `startup_wiring_ready`。

```bash
python scripts/check_joint_trajectory_system.py \
  --reference-root /path/to/clean/arm-vla-388b681
```

上述 fixture 只验证合同和调用链，不冒充模型开环或真实机器人闭环。

### 3.2 4×H20 真实 stage/reset smoke

启动前再次核验 GPU UUID、显存、tmux、精确 reference commit 和 clean status，只使用物理
GPU2。前两次启动分别在 argparse 和 scene/task identity 门禁前失败，均未创建 Isaac context；
失败 run 原样保留。第三次使用 clean-reference 配套 task/source 后：

- run：`manipulation-navi-v1-system-stage-smoke-r3-388b681-20260827T141443CST`；
- exit code：`0`；
- startup status：`completed / simulation_smoke`；
- lifecycle：`config_ready → isaac_app_starting → isaac_app_started → episode_spec_preparing →
  pipeline_creating → pipeline_created → episode_finished → completed`；
- episode：`success=true`，状态链 `build_stage → reset_episode → cleanup_episode → done`；
- scene binding：`standalone_asset_path_rewrite` 成功；
- 结束后 GPU2 再次为 `0 MiB / 0%`，tmux 自动退出，GPU1 外部任务未动。

证据摘要 SHA-256：startup status
`7b4fd1cfc6fc0908d360d8ace1cf9eb0eec713b7f2dcb03aca4d2432a45b5cd0`，episode summary
`1cf2e7ad57e42add33b34cff4de052504090f7a2102757efaac753c32377b95c`。

Isaac 日志同时出现 `No device could be created`、Vulkan/RTX foundation 和部分 robot visual
reference warning。因此本门禁只晋级“stage/episode 生命周期可启动”，不能晋级以下项目：

- head/wrist/overview 相机可用；
- GPU PhysX 或 RTX 渲染健康；
- PCT/DWA 实际运动；
- direct-joint tracking；
- fresh recorder 在真实 episode 上完整产出；
- 模型开环、闭环或任务成功。

## 4. 当前剩余门禁

1. 在具备健康 Vulkan/RTX 的 Isaac 节点运行一次真实 hold→NAV→Mani control smoke，逐 tick
   检查 apply report、零 base、连续夹爪和 recorder 文件；
2. fresh 数据到齐后只读审计 raw episode，再 materialize 全新 immutable release；
3. 用 12 条完整成功 episode 做 disposable overfit，检查四 route、boundary、NAV/Mani 10 点
   和两次夹爪主转换；
4. 固定 noise 开环、真实 PCT/DWA + direct-joint 多 seed 闭环；
5. 门禁通过后才允许从 selective warm-start 启动新训练。

因此现在的准确状态是：系统代码已经能进入下一步真实控制 smoke；训练仍被 fresh data 明确
阻塞，而且本轮没有启动任何训练。
