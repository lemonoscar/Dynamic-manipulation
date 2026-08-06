# ConveyorBench V2 数据采集与验收手册

本文给出从本地资产检查、场景预览、最小物理 smoke、严格校验到 ConveyorVLA
AL0/DynamicVLA 离线导出的操作顺序。V2 当前交付目标是“可开启采集的框架”，不是
大规模轨迹集。先逐条关闭门禁，再扩大 seed 和物体覆盖；不要用协议测试或 SVG
预览代替 Isaac 物理与 RTX 相机证据。

规范与机器可读配置见 [BENCHMARK_V2_SPEC.md](BENCHMARK_V2_SPEC.md) 和
[configs/v2.json](configs/v2.json)。所有命令均从 `Dynamic/conveyor_bench/`
执行。

## 1. 当前状态先读

当前开发工作站使用 NVIDIA RTX 4060 Laptop GPU（8 GB）完成 RTX 渲染；V2
PhysX 仍固定为 `--device cpu`。一次早期启动发现 Isaac 默认 `GroundPlaneCfg`
会组合远程 USD；两个 V2 scene 已改用项目内程序化 cuboid 地面，源码和 V2
manifest 不再需要在线 ground 资产。

冻结源码候选的 source-tree SHA-256 为
`0a2fd7c20f2ef62e1ab8c13ef6d871f779b5871088fb46093f994753a291514b`
（62 个执行/配置文件）。同一指纹下的 near fixed-base continuous 正例使用
seed 0、`0.06 m/s` 和 45 s 上限，记录 1327 个 control sample、2654 个 object
row；两个目标在 `13.98 s` 与 `26.16 s` 稳定放置，最终 2/2 成功。episode ID 为
`run-20260731T143748059454Z-1a8ae5fc-ep0000-seed0-fixed_base`。

remote whole-body 在同一冻结指纹下完成了蓝/黄双方向物理正例：seed 0 的黄色
投放在 `24.02 s` 结束，1201 个 sample/object row，连续持物位移
`0.778166 m`；seed 2 的蓝色投放在 `30.84 s` 结束，1542 个 sample/object row，
连续持物位移 `0.735903 m`。两者均通过 V2 strict validator，episode ID 分别为
`run-20260731T143907372855Z-655bb0c7-ep0000-seed0-whole_body_policy` 与
`run-20260731T144007414026Z-942292b2-ep0000-seed2-whole_body_policy`。

RTX 三相机也各保留一条成功正例。near fixed single 记录 294 个同步 tick、
882 张 PNG，strict validator 与 temporal camera gate 通过，并导出 294 条
DynamicVLA 和 294 条 AL0 记录；remote whole-body 记录 600 个同步 tick、1800 张
PNG，同样通过两级校验，并各导出 600 条记录。episode ID 分别为
`run-20260731T144222799582Z-20b7af64-ep0000-seed0-fixed_base` 与
`run-20260731T144429166398Z-50776f40-ep0000-seed0-whole_body_policy`。overview
始终为 observer-only，不计入策略目标证据。

另有一条 synthetic data-plane smoke 通过
`EpisodeRecorder → continuous 两目标成功 episode → V2 strict validator →
DynamicVLA/AL0 iterator`，且 canonical 6 个源文件哈希未变化。它只补充证明数据
平面接线，不能替代上述物理证据或实际 RTX 门禁。

## 2. 环境、测试与资产预检

准备已安装 Isaac Sim、Isaac Lab、PyTorch、NumPy 和 OpenCV 的 Python 3.11
环境：

```bash
cd Dynamic/conveyor_bench
conda activate env_isaaclab
python -m pip install -e .
python scripts/check_environment.py
```

先执行不启动 Isaac 的全部测试：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider
```

验证 V2 独立资产锁；输出应是被锁文件数量，且命令不得访问网络：

```bash
PYTHONPATH=src python -c \
  "from pathlib import Path; from conveyor_bench.v1.assets import verify_asset_lock; print(len(verify_asset_lock(Path('assets/asset_lock_v2.json'))))"
```

同时检查仓库内没有意外的在线资产引用：

```bash
rg -n "https?://|omniverse://|omniverse-content|s3[.-]" \
  src/conveyor_bench/isaac/scene_v2.py \
  src/conveyor_bench/isaac/scene_remote_delivery.py \
  assets/workcells/remote_delivery_v2 \
  assets/receptacles/remote_delivery_v2.json
```

该命令应无输出。许可证和来源字段本身允许出现在其他资产文档；这里检查的是
V2 运行时和新增 scene manifest。

## 3. 不启动 Isaac 的场景预览

以下脚本只读取项目内 JSON/manifest，生成确定性的俯视 SVG：

```bash
python scripts/render_v2_layout.py
```

默认输出为
[docs/images/conveyorbench_v2_layout.svg](docs/images/conveyorbench_v2_layout.svg)。
图中同时示意两个 profile：near 场景使用近端分拣盘；remote 场景关闭近端盘，
显示远端蓝/黄投放台、无碰撞导航走廊、持物路线和拉远的第三视角。SVG 能检查
布局和方向，不能证明刚体接触、teacher、locomotion 或相机同步正确。

需要输出到临时位置时：

```bash
python scripts/render_v2_layout.py --output /tmp/conveyorbench_v2_layout.svg
```

## 4. 冻结采集矩阵

只允许以下 7 个组合：

| scene | family | mode |
| --- | --- | --- |
| `transverse_near_sort_v2` | `single_target` | `fixed_base` |
| `transverse_near_sort_v2` | `single_target` | `whole_body_policy` |
| `transverse_near_sort_v2` | `language_conditioned` | `fixed_base` |
| `transverse_near_sort_v2` | `language_conditioned` | `whole_body_policy` |
| `transverse_near_sort_v2` | `continuous_multi_target` | `fixed_base` |
| `mobile_remote_delivery_v2` | `single_target` | `whole_body_policy` |
| `mobile_remote_delivery_v2` | `language_conditioned` | `whole_body_policy` |

V2 固定带速档为 `0.06/0.08/0.10 m/s`。near 默认时限 `45 s`，remote 默认
时限 `60 s`。continuous 首版固定为两个顺序计分目标，使用 service-gated
spawn；remote 成功必须包含至少 `0.65 m` 的连续持物根部平面位移。

运行前使用正式入口的 `--dry-run-task` 解析 task context，验证
scene/family/mode、split、目标顺序和目的地映射；该操作不启动 Isaac：

```bash
python scripts/run_benchmark_v2.py \
  --scene transverse_near_sort_v2 \
  --task-family continuous_multi_target \
  --robot-mode fixed_base \
  --split train \
  --seed 0 \
  --dry-run-task
```

输出 JSON 中 `simulator_started` 必须为 `false`，canonical protocol 必须为
`conveyor-bench-v1`。被禁组合（例如 remote + fixed-base）在这里就应返回 `2`，
而不是启动 simulator 后才失败。

语言任务把 `--task-family` 改为 `language_conditioned`，并显式选择
`--instruction-language en` 或 `--instruction-language en_zh`。物体数量、目标
和干扰物由所选 split 与 seed 的冻结 task builder 决定，V2 CLI 不提供绕过
manifest 的临时 `--active-objects`、`--target-asset` 或 `--destination` 覆盖项。

## 5. 物理 smoke 顺序

在目标版本 Isaac Sim/Isaac Lab 的采集机上按以下顺序进行。无相机物理 smoke
使用 CPU PhysX；第 5.5 节相机 smoke 还需要可用 RTX GPU。每一步只采 1 条，
使用 `--require-all-success`，失败时保留完整 episode 和日志，不要直接扩大
样本数。

### 5.1 near 单目标固定机身

先验证本地地面、横向传送带、机械臂、夹爪和 V1 canonical 记录。该门禁是
continuous 的前置条件，但不能代替连续目标切换测试。

```bash
python scripts/run_benchmark_v2.py \
  --scene transverse_near_sort_v2 \
  --task-family single_target \
  --robot-mode fixed_base \
  --episodes 1 \
  --seed 0 \
  --belt-speed 0.06 \
  --max-duration 45 \
  --split train \
  --output-dir outputs/gate/v2_near_fixed_single \
  --require-all-success \
  --headless \
  --device cpu
```

### 5.2 near 双目标连续服务

验证两个目标是否按 manifest 中的 `target_sequence_ids` 顺序完成：

1. 第一个目标在 episode start 后生成；
2. 第一个目标稳定放置后发出 `object_placed`，但 episode 不终止；
3. 机械臂回到可抓取姿态后，第二个目标才解除 service gate 并生成；
4. `steps.jsonl.selected_object_id` 切换到第二个目标；
5. 第二个目标稳定放置后才发布成功 summary。

```bash
python scripts/run_benchmark_v2.py \
  --scene transverse_near_sort_v2 \
  --task-family continuous_multi_target \
  --robot-mode fixed_base \
  --episodes 1 \
  --seed 0 \
  --belt-speed 0.06 \
  --max-duration 45 \
  --split train \
  --output-dir outputs/gate/v2_near_fixed_continuous \
  --require-all-success \
  --headless \
  --device cpu
```

### 5.3 near whole-body 单目标

当前 V1 locomotion probe 使用的旧场景仍可能组合 Isaac 默认在线 ground，因而
不把它作为 V2 离线门禁。直接在使用本地地面的 V2 near scene 中执行单目标
whole-body smoke，检查浮动根稳定、速度响应、抓取与放置。不能用固定机身成功
替代移动策略门禁。

```bash
python scripts/run_benchmark_v2.py \
  --scene transverse_near_sort_v2 \
  --task-family single_target \
  --robot-mode whole_body_policy \
  --episodes 1 \
  --seed 2 \
  --belt-speed 0.06 \
  --max-duration 45 \
  --split train \
  --output-dir outputs/gate/v2_near_whole_body \
  --require-all-success \
  --headless \
  --device cpu
```

### 5.4 remote whole-body 单目标

remote smoke 必须观察完整的抓取后收臂、原地对准、持物导航、到达稳定、
预放置、下降、开爪、撤离和稳定性验证。人工观察机器人确实移动还不够；最终
由 V2 validator 从 `objects.jsonl` 与 `steps.jsonl` 复核连续持物段的根部
位移是否至少 `0.65 m`。

```bash
python scripts/run_benchmark_v2.py \
  --scene mobile_remote_delivery_v2 \
  --task-family single_target \
  --robot-mode whole_body_policy \
  --episodes 1 \
  --seed 3 \
  --belt-speed 0.06 \
  --max-duration 60 \
  --split train \
  --output-dir outputs/gate/v2_remote_whole_body \
  --require-all-success \
  --headless \
  --device cpu
```

### 5.5 三相机 smoke

对 near continuous 和 remote 各至少保存一个三相机成功 episode：

- head：狗头水平前视，策略可见；
- wrist：夹爪正上方微俯视，策略可见；
- overview：拉远第三视角，只供观察和质检。

保持 Fabric 开启，同时保存无损 PNG 和 `camera_frames.jsonl`。必须运行时变相机
门禁，不能只检查帧数量或视频可播放。

near continuous 相机正例：

```bash
python scripts/run_benchmark_v2.py \
  --scene transverse_near_sort_v2 \
  --task-family continuous_multi_target \
  --robot-mode fixed_base \
  --episodes 1 \
  --seed 11 \
  --belt-speed 0.06 \
  --max-duration 45 \
  --split train \
  --output-dir outputs/gate/v2_near_continuous_camera \
  --enable_cameras \
  --save-camera-frames \
  --require-all-success \
  --headless \
  --device cpu
```

remote 相机正例：

```bash
python scripts/run_benchmark_v2.py \
  --scene mobile_remote_delivery_v2 \
  --task-family single_target \
  --robot-mode whole_body_policy \
  --episodes 1 \
  --seed 13 \
  --belt-speed 0.06 \
  --max-duration 60 \
  --split train \
  --output-dir outputs/gate/v2_remote_camera \
  --enable_cameras \
  --save-camera-frames \
  --require-all-success \
  --headless \
  --device cpu
```

`--enable_cameras` 来自 Isaac AppLauncher，参数名保留下划线；
`--save-camera-frames` 是 V2 collector 的短横线参数，两者必须同时出现才能生成
训练用无损帧。正常完成全部 episode 返回 `0`；使用
`--require-all-success` 时，任一物理任务失败返回 `3`；参数/组合错误返回 `2`，
未处理的运行错误返回非零。

## 6. 输出目录与审计顺序

每次 smoke 或采集使用新的输出目录和不重叠 seed。预期目录仍遵循 V1
canonical 原子发布契约：

```text
outputs/gate/v2_<profile>/
├── run-<UTC>-summary.json
└── episodes/
    └── <EPISODE_ID>/
        ├── manifest.json
        ├── steps.jsonl
        ├── objects.jsonl
        ├── action_chunks.jsonl
        ├── events.jsonl
        ├── summary.json
        ├── camera_frames.jsonl       # 开启帧保存时
        └── cameras/
            ├── head_rgb/*.png
            ├── wrist_rgb/*.png
            └── overview_rgb/*.png
```

录制中间态位于隐藏的 `.inprogress` 目录；正常结束或有明确物理失败的 episode
都应原子发布。运行异常、磁盘失败或半写流不能被当成有效失败样本。

每条数据按以下顺序验收：

1. V1 strict validator：canonical 结构、时钟、引用、事件、成功证据、PNG 与
   未来标签；
2. V2 validator：suite 元数据、7 项矩阵、service-gate contract 与实际事件
   时序、continuous 的 selected/spawned/placed 顺序和 remote loaded
   displacement；
3. V1 quality audit：把物理任务失败与数据损坏分开；
4. temporal camera gate：三路图像时变、head/wrist 目标证据和 overview 权限；
5. V2 AL0/DynamicVLA exporter：仅生成派生视图，并复核 canonical 哈希未变化。

例如验收 remote 相机 smoke；把 `EPISODE_ID` 替换为 run summary 中发布的实际
目录名：

```bash
python scripts/validate_v2_dataset.py outputs/gate/v2_remote_camera

python scripts/audit_v1_episode.py \
  outputs/gate/v2_remote_camera/episodes/EPISODE_ID

python scripts/check_v1_camera_gate.py \
  outputs/gate/v2_remote_camera/episodes/EPISODE_ID

python scripts/export_v2.py \
  outputs/gate/v2_remote_camera/episodes/EPISODE_ID \
  --profile both
```

`validate_v2_dataset.py` 接受单个 episode、`episodes/` 目录或 collection root；
全部有效返回 `0`，发现验证错误返回 `1`，输入路径/发现错误返回 `2`。
`export_v2.py` 接受相同三类目录，默认 `--profile both`；已有派生文件时会拒绝
覆盖，只有明确重建 `exports/` 时才使用 `--force`。`--force` 不得改写
canonical 六个流或相机索引。

当前 validator 会同时检查 service-gate 静态契约和
`target_selected → object_spawned → object_placed` 实际事件时序，包括
`not_before_s` 与前序目标完成门控。人工回放仍用于发现物理异常，但不能替代
validator，也不再承担第二目标是否提前生成的唯一判定。

失败 episode 只有在前两层确认结构有效后，才可作为 benchmark 失败样本保留；
默认训练集筛选策略应另行记录，不能修改原始 summary。

## 7. V2 导出检查

V2 的两种投影都应在 canonical V1 记录上增加以下任务上下文：

```text
scene_id
task_family
target_sequence_ids
destination_zone_by_target
current_target_id
current_subtask_index
supervision_only_fields
```

`current_target_id` 和 `current_subtask_index` 是 teacher/evaluator 监督，不是
在线策略可见观测。抽查导出首尾记录时，至少确认：

- DynamicVLA 与 AL0 的记录数和 model tick 对齐；
- continuous 的 current target 顺序只从目标 0 前进到目标 1；
- canonical 10D action 和有效位 mask 仍存在；
- AL0/DynamicVLA 的坐标系与 chunk 长度继续遵循 V1；
- `export_manifest.json` 记录源与输出哈希，canonical 文件导出前后不变。

## 8. 从 smoke 到正式采集的清单

以下 `x` 是 2026-07-31 本次开发验收记录；迁移环境、改变代码或 asset lock 后
必须清空状态并全部重跑。

### A. 纯逻辑与离线资产

- [x] 全部纯 Python 测试通过；
- [x] `asset_lock_v2.json` 校验通过；
- [x] 两个 V2 scene 源码与 manifest 无网络资产引用；
- [x] SVG 可重生成，传送带方向、近端/远端盒和相机语义正确；
- [x] 7 个允许组合可构建，禁用组合全部 fail closed；
- [x] 同 seed 的目标序列、物体、目的地和指令可复现。

### B. 单 episode 物理门禁

- [x] near fixed 单目标成功；
- [x] near fixed 双目标 continuous 按序成功（seed 0、`0.06 m/s`、无相机）；
- [ ] near whole-body 单目标成功，且根部响应/稳定性符合移动策略契约；
- [x] remote whole-body 单目标成功且 loaded displacement `>=0.65 m`
  （黄侧 `0.778166 m`、蓝侧 `0.735903 m`，均为无相机方向回归）；
- [x] near fixed single 与 remote whole-body 三相机正例通过 temporal camera gate；
- [x] 上述两条视觉正例依次通过 V1/V2 strict validation、quality audit 和双导出；
- [x] 本轮所有输出根无 `.inprogress` 残留。

### C. 小规模回归

- [ ] 使用几十条以内、互不重叠的 seed；
- [ ] 覆盖 train/val/unseen、蓝/黄目的地与 `0.06/0.08/0.10 m/s`；
- [ ] 7 个允许组合至少各有结构有效样本，关键主任务各有成功正例；
- [ ] 汇总选择正确率、抓取成功率、正确放置率、漏件率、跌倒率、连续吞吐和
  remote loaded displacement；
- [ ] 人工抽查 head/wrist 可学性与 overview 回放，但不把 overview 导入训练；
- [ ] 固化配置、asset lock、源码指纹、run summary、validator/camera/export
  报告后再决定放量。

## 9. 已知限制与排障边界

- 当前仅支持 CPU PhysX 的已审计传送带 surface-velocity 接触路径；RTX 4060
  已验证三相机渲染，但不要把 `--device cuda` 当作 V2 并行采集选项。
- continuous 首版仅支持 near + fixed-base + 2 targets；whole-body 的返程与
  再武装未通过前保持拒绝。
- remote 仅支持 single/language + whole-body；不含避障和 remote continuous。
- 本机已完成 near fixed single 与 remote whole-body 的 RTX 三相机 E2E；near
  continuous 相机、near whole-body、语言条件、其余物体/split/速度矩阵仍未
  验证。迁移采集机或改变源码/asset lock 后必须从第 2 节重新开始，不能把当前
  少量正例外推到完整矩阵。
- 发现远程 URL、缺失资产或 hash 漂移时立即停止；先把资产本地化、补来源/
  许可证并更新 V2 lock，不能在运行时临时下载。
- 本地 whole-body `policy.pt` 尚无已确认的权重专属再分发许可证；可用于本地
  研究与验收，但公开分发二进制前必须取得授权或替换为许可明确的权重。边界见
  `assets/policies/go2_x5_pct_dog_only/PROVENANCE.md`。
- 任何 task/mode 绕过允许矩阵、overview 进入策略输入、持物距离不足却标记
  remote success、第一目标完成即提前结束 continuous，均属于实现错误而不是
  “较弱结果”。

## 10. 扩展资产建议

首轮门禁不需要下载额外资源。物理闭环稳定后，如果视觉域过于单一，优先增加
本地材质/光照变体，再增加经过许可证、单位、碰撞和质量审核的 CAD 零件。
每项资源必须复制进 `assets/`、记录 provenance、离线加载并纳入 V2 asset
lock。新资产先进入小规模 held-out 回归，不直接混入正式训练 split。
