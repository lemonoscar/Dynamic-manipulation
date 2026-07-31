# 从 V0 到毕业设计 Benchmark

本项目仍遵循“先跑通可信闭环，再扩大数据规模”的路线。V0 固定机身单目标
流水线完整保留，V1 已成为冻结的可采集基线，当前增量转入 V2 suite。V1 不是
训练模型，也不是大规模轨迹集，其接口冻结见 `BENCHMARK_V1_SPEC.md` 和
`configs/v1.json`。

## 当前增量：ConveyorBench V2 suite

V2 不修改 V1 canonical episode 协议，而是在其上增加面向新 VLA 策略的场景
与任务上下文。规范、采集手册和机器快照分别为 `BENCHMARK_V2_SPEC.md`、
`COLLECTION_V2_GUIDE.md` 与 `configs/v2.json`。

当前已经落地的 V2 结构包括：

- `transverse_near_sort_v2` 与 `mobile_remote_delivery_v2` 两个场景；
- 7 个显式允许的 scene/task/mode 组合，未列组合启动前 fail closed；
- 近端 fixed-base 双目标 `continuous_multi_target`，采用 service-gated 顺序
  生成和共享 episode 时间预算；
- 远端蓝/黄投放台、无障碍持物导航走廊和 `>=0.65 m` loaded displacement
  成功门槛；
- 保持 head/wrist 为策略观测、overview 为 observer-only 的三相机契约，remote
  第三视角拉远以覆盖完整路线；
- 8 类 V1 本地程序化零件、机器人、移动策略与远端新增资产的独立 V2 asset
  lock；
- 在 V1 strict canonical validation 之上的 V2 目标顺序/持物距离验证；
- 在 V1 M0/DynamicVLA 投影之上的 scene、目标序列与当前子任务监督字段；
- 不启动 Isaac 即可重建的 V2 场景 SVG 与纯 Python 回归测试。

冻结源码候选 `0a2fd7c…` 已完成真实 Isaac CPU PhysX 的 near fixed-base 双目标
continuous：seed 0、`0.06 m/s`、45 s 上限，两个目标分别在 `13.98 s` 与
`26.16 s` 稳定放置，最终 2/2 成功；1327 个 control sample 和 2654 个 object
row 通过 V2 strict validator。

remote whole-body 已完成蓝/黄双方向单目标：黄色 seed 0 为 1201 个 sample、
`24.02 s`、`0.778166 m` loaded displacement；蓝色 seed 2 为 1542 个 sample、
`30.84 s`、`0.735903 m`。两者均通过 V2 strict validator。远端托盘标定为
`(-0.16, ±1.20, 0.46)`，投放和真实四足携物走廊均已物理可达。

本机 RTX 4060 还完成 near fixed single 与 remote whole-body 的三相机正例：
分别记录 294/600 个同步 tick、882/1800 张 PNG，均通过 strict validator、
temporal camera gate 和 DynamicVLA/M0 双导出。早期默认 ground 的远程 USD
依赖已由 V2 本地程序化地面替代。near whole-body、near continuous 相机、语言
条件及其余物体/seed/速度矩阵仍未通过，不能把当前少量正例外推为 V2 全矩阵
成功。

V2 下一步只做以下递进工作：

1. 跑通 near whole-body、语言条件及尚未覆盖的任务组合；
2. 若 continuous 进入视觉训练主矩阵，为 near continuous 增补一条三相机正例；
3. 用几十条以内、不重叠 seed 覆盖 train/val/unseen、蓝/黄目的地和三个冻结
   带速档，统计失败模式；
4. 只有回归稳定后才冻结正式采集清单和数据规模。

## 已落地：V0 单目标流水线

- Go2-X5 固定机身、X5 机械臂和双指夹爪。
- 机器狗正前方的横向物理传送带、左到右单目标输送、C0 静态校准和 C1
  匀速动态抓取。
- 特权状态 oracle 与项目内确定性 IK，作为数据生成 teacher。
- 头部/腕部策略 RGB、observer-only 第三视角、完整关节状态、TCP、目标
  状态、接触力和动作。
- 目标恒速未来位置、离开时间和抓取窗口标签。
- 六段观测到执行墙钟时间戳，为异步链路分析预留接口。
- 自动成功判定、失败原因、原子 episode 发布和 V0 数据集校验。

V0 契约保持不变，继续作为定位机械臂、传送带接触和历史数据兼容问题的最小
基线。

## 当前主线：V1 可采集框架

V1 已经在项目内提供以下组成部分：

- `+Y → -Y` 的横向低位传送带；机器人视觉中物体从左到右运动。
- 运输碰撞面与视觉建模解耦的真实工位：皮带/滚筒、机架/支腿、护罩/安全栏、
  光电传感器、急停、出口标记和漏件接料盘。
- 8 个本地程序化物品：6 seen（4 train、2 val）与 2 unseen。
- 运行时使用互斥的 train 4、val 2、unseen 2 三个 curriculum split，并支持
  单目标和带干扰物的中英双语目标选择。
- 蓝色、黄色两个计分分拣盒，以及不计分的下游接料盒。
- 使用本地策略权重的 `whole_body_policy` 主路径，以及 `fixed_base` 消融。
- physics/control/camera-model 固定为 `400/50/25 Hz`。
- body base 3D、base-frame TCP delta 6D 和夹爪 1D 组成的 canonical 10D
  action。
- head/wrist 两路 policy observation；overview 只允许观察与质检。
- `manifest/steps/objects/action_chunks/events/summary` 原子 episode 记录。
- 任务判定、V1 结构 validator、逐 episode quality audit。
- 相机物理时变和 head/wrist 目标证据的 fail-closed camera gate。
- 不改写 canonical 数据的 DynamicVLA 与 M0 离线导出。
- 机器人 USD/URDF/mesh、移动策略、物体注册表、分拣盘和工位 manifest 的
  asset lock，以及每 episode 的源码树指纹。
- 全部项目资产、策略权重和代码位于 `Dynamic/conveyor_bench/`；采集运行不
  联网，也不依赖 `Dynamic/` 外的项目文件。

宿主机预装的 Isaac Sim、Isaac Lab 及 Python 包属于运行环境前置条件，不是
需要在线获取的项目资产。

## 当前里程碑：核心采集闭环已通过，转入小批量前回归

当前本地 Isaac 烟测已经获得以下物理结果：

- fixed 单目标抓取—运输—释放—稳定放置成功，约 `10.48 s`；
- whole-body 单目标移动—动态抓取—携带—放置成功，约 `21.60 s`；
- whole-body 三物体、中英双语目标选择成功，约 `21.58 s`。

三条 canonical 输出均已通过 strict validator 和 quality audit；带三相机的
最终 whole-body 单目标 release 还生成了 540 个同步 tick、1620 张 PNG，
并实际通过 temporal camera gate。相机最大结构变化率为 head `0.704164`、
wrist `0.688824`、overview `0.039858`，head/wrist 目标证据为 `0.760409`。
同一 canonical episode 已完成各 540 条的 M0/DynamicVLA 双导出，源文件哈希
未改变；其源码树 SHA-256 为
`a5c2802447abd4e4c50365549b7b0cc83db313f01800cb26d734fc8fc695f39c`。
核心 whole-body 单目标的物理—视觉—记录—审计—导出链路因此已经形成一条
与当前代码一致的完整正证据。

Fabric 修复前的冻结相机 episode 保留为负例，证明 strict validator 和
quality clean 不能替代相机时变门禁。fixed 与三物体语言烟测未保存相机，只能
声明各自的物理/data 路径成功。

当前仍不应直接大规模采集。进入小批量前只做以下矩阵回归：

1. 完成最终纯 Python、V0、scene probe、移动策略 probe 和无
   `.inprogress` 残留检查；
2. 若 fixed 与三物体语言配置进入视觉训练矩阵，分别生成相机 episode，并通过
   strict validator、quality audit、camera gate 和双导出；
3. 用少量不重叠 seed 覆盖 train/val/unseen、两个分拣盘和冻结速度档；
4. 复核同 seed 任务/资产可复现、asset lock 与源码树指纹一致；
5. 把 report、run summary、audit、camera gate 与 export manifest 固化为
   同代码版本的交付记录。

可执行命令、正负证据路径和状态边界见 `COLLECTION_GUIDE.md`。

## 门禁后：小批量回归集

只有 fixed 与 whole-body 单条链路都通过后，才进入小批量：

1. 每个 seen/val/unseen 物体至少覆盖两个分拣盒；
2. 使用不重叠 seed 覆盖少量冻结速度档；
3. 统计选择正确率、抓取成功率、正确盒放置率、漏件率、跌倒率和吞吐量；
4. 检查 PNG 完整性、相机黑帧/模糊、磁盘吞吐与 `.inprogress` 残留；
5. 固化失败样例，不把 smoke 或调试 episode 混入训练导出。

先以几十条以内的回归规模验证分布和失败模式，再决定正式采集规模。

## 后续任务扩展

V1 已经支持同一 episode 中的目标加干扰物和中英双语目标选择。门禁稳定后再
依次增加：

- 多个计分目标的连续分拣、对象回收和复位；
- 低速、中速、高速、加减速与短时扰动；
- 语言条件目标选择和更严格的 seen/unseen 组合泛化；
- 旋转/往复工作台、移动餐车等共享协议场景；
- 固定机身和全身协同的分层对照实验。

新增任务必须继续复用相同坐标、时间、相机权限、canonical action、事件与
评价接口；不能以新场景为由绕过 V1 数据门禁。

## 算法接入：VLA、预测与异步

框架支持的替换路径是：

```text
特权 oracle 数据 teacher
  → DynamicVLA / M0 同步视觉基线
  → VLA + 短时未来状态预测
  → VLA + 预测 + 异步 action chunk
```

视觉策略只能使用 head/wrist、语言和允许的机器人本体状态；overview、目标
真值、接触真值和未来状态只用于人工观察、teacher、监督标签或 evaluator。
当前交付只完成采集和导出接口，不包含上述模型的训练结果。

## 数据规模原则

- 烟测：每个物理门禁 1 条。
- 回归：每种冻结配置少量、不重叠 seed。
- 小批量：先验证失败率、标签、磁盘吞吐和图像质量。
- 正式扩容：任务 manifest、资产哈希、协议版本、validator 和导出契约全部
  冻结并通过后再进行。

任何完整发布的失败 episode 都保留在 canonical 原始数据中；训练导出可以
按任务结果筛选，但不能改写 benchmark 原始记录。
