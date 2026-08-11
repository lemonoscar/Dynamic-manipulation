# ConveyorBench V3：Liangzhu NuRec 场景与进阶采集方案

机器可读配置见 [configs/v3_3dgs.json](configs/v3_3dgs.json)。V3 当前处于
`integration`，`collection_ready=false`。这表示代码已经具备资产校验和场景组合入口，
但在远端画面、碰撞、示教状态机和真实物品刚体四项门禁通过前，不能开始正式放量。

## 1. 当前选择

现有 Liangzhu 资产不是普通 Gaussian PLY，而是 Omniverse NuRec USDZ：

- `liangzhu/usdz/liangzhu.usdz` 内含原生 NuRec 场和约 1.5 GB 的 `.nurec`；
- `liangzhu/liangzhu.usda` 已定义 Gaussian 视觉的 270° 坐标旋转；
- `liangzhu/usd/liangzhu_collision.usda` 是独立静态碰撞网格；
- `objects/` 已包含 cola、apple、orange、bottle、box2 等带纹理 USD。

因此首选路径是 Isaac/RTX 原生 registered compositing，而不是把 NuRec 解包为不存在的
Gaussian PLY，再自行维护 `gsplat` 两遍渲染器：

```text
SSH sidecar asset bundle + SHA-256 gate
                    │
                    ▼
Liangzhu NuRec visual ─┐
Liangzhu collision USD ├─ Isaac/RTX 单遍组合 ─ head/wrist/overview RGB
Go2-X5 + conveyor      │                     └─ V1 action/state/event
task objects           ┘
                    │
                    ▼
V1 raw PNG strict gate → raw-to-LeRobot MP4 → ConveyorVLA AL0
```

只有原生 NuRec 出现动态物体遮挡或语义标签无法解决的确定性问题时，才启用离线
RGB/depth 两遍合成作为回退；现在不提前建设第二套渲染栈。

## 2. 代码与资产边界

代码、配置、测试和文档正常进入 Dynamic 仓库。大体积场景与物品资产不进入 Git，
由专用传输任务通过 SSH 放入 4xH20 的工作根目录。运行时只接受一个不可变 sidecar：

```text
<asset-root>/
├── TRANSFER_MANIFEST.sha256
├── liangzhu/
│   ├── liangzhu.usda
│   ├── runtime_asset_manifest.json
│   ├── usdz/liangzhu.usdz
│   └── usd/liangzhu_collision.usda
└── objects/
    ├── cola/
    ├── apple/
    ├── orange/
    ├── bottle/
    ├── box2/
    └── ...
```

启动 shell 用 `CONVEYOR_BENCH_V3_ASSET_ROOT` 指向这个目录。校验器在 Isaac 启动前
拒绝以下情况：路径不在允许根目录、文件缺失、SHA-256 不一致、软链接、NuRec USDZ
成员缺失、视觉/碰撞 prim 契约不一致。运行时不联网，也不从其他仓库读取文件。

资产预检入口：

```bash
python conveyor_bench/scripts/validate_v3_asset_bundle.py \
  --asset-root "$CONVEYOR_BENCH_V3_ASSET_ROOT" \
  --allowed-root /diff/wallx_workspace/dzb
```

## 3. 坐标和任务布局

Liangzhu 原场景中 PCT 机器人根 XY 是
`(-1.4849319648, 5.1261365028)`。V3 把它平移到 ConveyorBench 的 `(0, 0)`，
并用已审计的 root-to-ground 值把源地面抬到仿真 `z=0`：

```text
T_sim_from_liangzhu.translation =
  (1.4849319648, -5.1261365028, 0.1381941050)
```

这样现有传送带中心 `(0.70, 0.0)` 对应源场景约
`(-0.78493, 5.12614)`；长轴沿世界 Y，零件沿 `-Y` 运动。正式固定布局前生成
`x=-0.25/0/+0.25 m` 三个横向候选预览，并对每个候选执行：

1. 从 Liangzhu collision 向下 raycast 求真实地面，不写死场景高度；
2. 检查机器人足端、机身、传送带支架在 reset 时无穿插；
3. 检查 head/wrist 中抓取区无遮挡，overview 能看到全任务；
4. 选定一个候选后冻结变换和哈希，数据集中不得逐回合漂移。

V3 禁用 V1 的程序化地面与两面背景墙，避免重叠；深绿色传送带、机器人、零件和接料
区域仍由 Isaac 生成并参与物理。

## 4. 机器人与相机保持不变

V3 不建立第二份机器人。它继续使用已对齐 PCT 的项目内 Go2-X5 URDF、FinRay 夹爪、
TCP 和 D436 相机标定：

| 相机 | 分辨率/频率 | 用途 | 安装 |
| --- | --- | --- | --- |
| head | 640×480 @ 25 Hz | policy | `base` 前向 |
| wrist | 640×480 @ 25 Hz | policy | `arm_link6`，夹爪上方俯视 |
| overview | 480×320 @ 25 Hz | observer | 拉远第三视角 |

head/wrist 的 OpenCV 内参和手眼外参沿用 V1。overview 永远不能进入模型输入。场景
切换若改变任何相机 mount、intrinsics、帧率或数据字段，严格测试必须直接失败。

## 5. 物品策略

首轮不做分类，只做“一个目标、一个放置点”：

- 训练 pilot：cola、apple、orange；
- unseen gate：bottle；
- 放置目标：box2；
- blanket 暂缓，因为它是柔性物，不能混入首轮刚体抓取。

不能只把真实 USD 当贴图覆盖到旧碰撞上就直接采集。每种物品必须先完成刚体 fixture：
真实尺度、质量、稳定姿态、凸碰撞、摩擦、夹爪开度和 top-down grasp affordance。教师、
成功判据和记录器继续使用 canonical instance ID，不依赖物品文件名。

## 6. 采集前的已知控制阻塞

最近的 45 秒 PCT 对齐回放已经通过三相机帧门禁并进入 `carry`，但一直没有转入
`preplace`，最终超时。这不是 3DGS 问题，也不能靠增加数据量掩盖。V3 正式采集前必须：

1. 为 carry→preplace 写确定性状态转移回归测试；
2. 静止传送带连续 5 条成功；
3. 低速动态传送带连续 5 条成功；
4. 抓取/放置保持俯视，动作速度与 25 Hz VLA 推理能力匹配。

## 7. 分阶段数据计划

| 阶段 | 内容 | 目标成功条数 | 是否进入训练集 |
| --- | --- | ---: | --- |
| A | NuRec、碰撞、三相机和三种带位预览 | 0 | 否 |
| B | 旧 canonical 物体的 stationary 控制诊断 | 5 | 否 |
| C | cola 真实 USD stationary fixture | 10 | pilot |
| D | cola/apple/orange 低速动态 | 60 | pilot |
| E | bottle unseen 验证 | 20 | 只验证 |
| F | whole-body mobile pilot | 80 | pilot |
| G | 固定底盘 3×200 + 移动底盘 3×200 | 1200 | 正式 |

阶段 C 以前不以数量为目标。阶段 D 的 60 条用于估计成功率和节拍后再决定是否执行 G，
避免在场景或教师仍有错误时制造大量“格式正常但任务错误”的 PNG。

建议正式平衡维度是对象、速度和模式，而不是类别标签：

- object：cola/apple/orange 各 1/3；
- belt speed：0、0.005、0.01 m/s 分层；
- robot mode：fixed-base 先稳定，再加入 whole-body；
- failure：任务失败单独保存，数据损坏一律隔离，不能冒充负样本。

## 8. 远端门禁和启动入口

场景预览沿用现有 probe，仅切换 profile：

```bash
python conveyor_bench/scripts/probe_v1_scene.py \
  --scene-profile v3_nurec \
  --v3-asset-root "$CONVEYOR_BENCH_V3_ASSET_ROOT" \
  --belt-speed 0.005 \
  --settle-seconds 2 \
  --output-dir conveyor_bench/outputs/v3_nurec_probe \
  --enable_cameras --headless
```

单回合采集入口：

```bash
python conveyor_bench/scripts/run_benchmark_v3.py \
  --asset-root "$CONVEYOR_BENCH_V3_ASSET_ROOT" \
  --episodes 1 --seed 0 --belt-speed 0 \
  --save-camera-frames --require-all-success \
  --output-dir conveyor_bench/outputs/v3_stationary_gate \
  --enable_cameras --headless
```

这些命令只定义入口，不代表已经通过。远端必须限制在获授权的 GPU 2/3；资产预检、
场景 probe、三相机人工查看、控制成功、V1 strict validator、temporal camera gate 和
LeRobot round-trip 全部通过后，才能把 `collection_ready` 改为 `true` 并开启长采集。
