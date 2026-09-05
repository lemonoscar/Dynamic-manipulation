# SPCGVLA 实现交接

状态：implementation handoff
目标分支建议：`feat/spcgvla-semantic-3d-memory`
设计合同：`spcgvla-design-draft-v0.2`
最后核对日期：2026-08-31

## 1. 交接目标

在一个新的短期 feature 分支上，把 [README.md](README.md)、[train.md](train.md) 和 [play.md](play.md) 中的 SPCGVLA v0.2 架构实现为可测试的代码与配置。最终系统应满足：

1. 精确语言中的任务物体经 head RGB 的 Grounding DINO、SAM2 mask 和 LiDAR 投影形成语义对象点云；
2. 每对象最多 128 个 `xyz+RGB` 点，不足零填充且绝不复制有效点；
3. PointNeXt-S 加 learned query-token resampler 固定产生每对象 4 个 token；
4. 3D token 只经 Qwen 后段的 4 个 Semantic-3D Cross-Attention Adapters 融合；
5. 保留两次完整 Qwen 前向，Pass 1 只做 `NAVIGATION / MANIPULATION` 二分类并生成子任务文本；
6. Pass 2 使用同一冻结 observation、同一 3D snapshot、同一 FPS 采样和同一 token bank；
7. NAV/Mani expert 不接收独立语义点云旁路；
8. 机器人接近可乐、当前 LiDAR 点消失时，保留并传播最后可靠的 3D object track，而不是立即删除；
9. 原始 LiDAR occupancy/costmap 继续走独立的 planner/collision hard-safety path；
10. 仿真 GT object ID、world mesh、head depth 和真实任务 phase 不进入模型输入。

这次交接的“实现完成”是指代码合同、数据结构、单元测试、离线/仿真 smoke path 和配置均可运行；不要求完成大规模训练，也不允许把一次成功视频写成模型性能结论。

## 2. 当前 Git 基线

交接创建时的基线为：

```text
repository: ${CONVEYORVLA_ROOT}
branch: Manipulation_Navi_v1
HEAD: 134cf61fc13b0eca5c9c985812dece556c2ac1ca
upstream: origin/Manipulation_Navi_v1
```

工作树在交接前已经是 dirty，且这些修改属于既有工作，禁止 reset、clean、checkout 丢弃或用 stash 隐藏后遗忘：

```text
 M .gitignore
 M README.md
 M docs/checkpoint_step2000_lookahead_evaluation_20260821.md
 M docs/operations.md
?? docs/SPCGVLA/
?? scripts/probe_liangzhu_lidar.py
?? scripts/probe_sam2_coke.py
?? scripts/segment_sam2_lidar_coke.py
?? scripts/view_lidar_pointcloud.py
?? src/conveyor_bench/isaac/liangzhu_lidar_probe.py
?? src/conveyor_bench/perception/
?? tests/test_lidar_probe.py
?? tests/test_lidar_web_viewer.py
```

其中 `.gitignore`、根 `README.md`、`docs/operations.md` 和 checkpoint 评测文档的修改主要属于另一项本地 `artifacts/` 路径治理工作，不应自动混入 SPCGVLA 行为提交。LiDAR probe、SAM2 probe、`src/conveyor_bench/perception/`、相关 tests 和 `docs/SPCGVLA/` 是本方案的重要前置证据，但仍需逐文件审查后再精确 stage。

## 3. 安全创建分支

先确认基线没有变化：

```bash
cd ${CONVEYORVLA_ROOT}
git branch --show-current
git rev-parse HEAD
git status --short
```

如果仍为上述分支和 HEAD，且目标分支不存在，可在当前工作树直接创建分支，使现有未提交文件继续留在工作树：

```bash
git switch -c feat/spcgvla-semantic-3d-memory
```

如果分支已存在，或 HEAD/dirty state 已变化，先审计差异，不要使用 `-f`，不要 reset。创建分支后再次保存 `git status --short` 到 handoff 回报中。

提交时使用精确路径 `git add <paths...>`。不要使用 `git add .`，不要把大型 checkpoint、数据、视频、点云、Isaac 缓存或 `artifacts/` 加入 Git。

## 4. 已有代码证据与限制

### 4.1 可复用内容

- `src/conveyor_bench/perception/lidar_probe.py`
  - 已有 `LidarScan`、原始点云记录、TF/clock evidence 和三画面渲染；
  - 原始点云和 simulator object-ID audit 已分离。
- `src/conveyor_bench/isaac/liangzhu_lidar_probe.py`
  - 已有 Liangzhu 场景下的模拟 LiDAR scan 生成路径。
- `scripts/probe_liangzhu_lidar.py`
  - 已有 head RGB、LiDAR 与同步 capture 的 probe 入口。
- `scripts/probe_sam2_coke.py`
  - 已证明手工 point/box prompt 下 SAM2 可以分割 head RGB 中的可乐。
- `scripts/segment_sam2_lidar_coke.py`
  - 已有单帧 SAM2 mask 到 LiDAR 点的投影 POC。
- `src/conveyor_bench/perception/lidar_web_viewer.py`
  - 已支持通过浏览器远程交互查看 raw/audit/SAM 点云。
- `joint_trajectory_*`
  - 已有两次 Qwen 前向、last-16 layerwise states、NAV/Mani FM experts、13D Mani state 和运行时安全框架。

### 4.2 不能直接沿用的 POC 行为

当前 `scripts/segment_sam2_lidar_coke.py` 是研究 probe，不是正式模型输入管线：

- 使用手工 point/box，不是语言驱动的 Grounding DINO；
- 使用 simulator head depth 做 depth consistency；SPCGVLA v0 不允许依赖该 privileged depth，正式实现应改为仅基于投影 LiDAR 的 z-buffer/遮挡规则；
- simulator object ID 只能在推理后计算 precision/recall，绝不能参与筛点；
- camera mount 常量和标定仍是 probe 级约定，正式路径必须使用带 hash 的 calibration contract；
- 脚本包含模型加载、投影、评测和文件写出，需抽成可测试 package 模块；
- 当前 PointNeXt-S、Grounding DINO、Qwen 3D Adapter 和 semantic object track 尚未实现。

### 4.3 现有 VLA 的兼容性风险

`src/conveyor_bench/conveyorvla/joint_trajectory.py` 当前仍定义四个 route：

```text
NAV_TO_SOURCE / PICK / NAV_TO_TARGET / PLACE
```

`joint_trajectory_model.py`、data、training、runtime、recording 和 tests 都依赖这一 v1 合同。不要把 v1 enum 和 schema 原地改成二分类，否则会破坏已有数据、checkpoint 和测试。SPCGVLA 应建立独立 v2 contract/module，并尽量复用动作专家实现。

建议的新 identity：

```text
model_contract_id: spcgvla-mobile-manipulation-v0.2
dataset_schema_version: spcgvla-mobile-manipulation-v0.2
point_encoder_id: pointnext-s-spcgvla-v0
semantic_token_schema: four-tokens-per-object-v0
```

## 5. 目标模块边界

建议先建立以下独立边界；最终名称可按仓库风格小幅调整，但不要把所有逻辑放入脚本或 `utils.py`：

```text
src/conveyor_bench/perception/
├── language_entities.py           # 精确语言实体合同；detector-independent
├── grounded_segmentation.py       # Grounding DINO + SAM2 adapter
├── lidar_camera_projection.py     # 无 simulator truth 的投影/z-buffer
├── semantic_object_cloud.py       # 128 点、mask、voxel/FPS、metadata
├── semantic_object_tracker.py     # current observation + memory track
└── semantic_3d_snapshot.py        # immutable snapshot and provenance

src/conveyor_bench/conveyorvla/
├── spcgvla_contract.py            # v0.2 IDs、binary domain、shapes
├── spcgvla_pointnext.py           # PointNeXt-S adapter/interface
├── spcgvla_tokenizer.py           # 4-token resampler + entity/meta fusion
├── spcgvla_adapter.py             # Qwen Semantic-3D adapters
├── spcgvla_model.py               # two-pass VQA/VLA + expert routing
├── spcgvla_data.py                # schema/materializer/loader
├── spcgvla_training.py            # losses/optimizer groups/checkpoint state
└── spcgvla_runtime.py             # request freeze/degradation/logging

configs/
└── spcgvla_v0.json

scripts/
├── materialize_spcgvla.py
├── train_spcgvla.py
└── run_spcgvla.py
```

Stable VLA code may depend on typed perception contracts, but core geometry/tracking code不得 import Isaac Sim。Isaac adapter 负责产生普通 RGB/LiDAR/pose 数据；正式 student path 不得看到 simulator object ID。

## 6. 必须实现的对象点云合同

每个语言对象至少包含：

```text
entity_id: str
entity_text: str
points_xyz_rgb: float32[128, 6]
point_valid_mask: bool[128]
centroid_query_body_m: float32[3]
size_query_body_m: float32[3]
point_count: int
segmentation_confidence: float
observation_age_s: float
object_valid: bool
object_observed_now: bool
object_track_valid: bool
last_observed_timestamp_s: float | None
track_source: current | memory_occluded | invalid
```

`point_count` 表示当前编码张量中 padding 前的有效点数；raw/current/memory/voxel/FPS 各阶段的点数另外写入派生 evidence 和日志。

固定规则：

- 距离裁剪：`0.05–10 m`；
- `>128`：固定米制 voxel 去重，再 deterministic FPS 到 128；
- `<128`：零填充并使用 mask，禁止重复点填充；
- `0` 且无可用 memory：全零点集、全 false mask、`object_valid=false`；
- 有 memory：输入最后可靠点云经当前 pose 变换后的结果，`object_valid=true`、`object_observed_now=false`；
- PointNeXt 的 xyz 使用 object-centered metric coordinates，不做 unit-sphere scaling；
- 米制 centroid/size 通过 metadata 分支融合回 token；
- entity 顺序由语言中的稳定顺序决定，不依赖 detector 返回顺序。

## 7. 近距离 LiDAR 盲区与 object track

这是本次实现的关键新增要求。

### 7.1 状态语义

至少区分：

```text
CURRENT_OBSERVATION
MEMORY_OCCLUDED
NOT_OBSERVED
STALE
INVALID_ASSOCIATION
INVALID_POSE
INVALIDATED_BY_MANIPULATION
```

`MEMORY_OCCLUDED` 不是“当前 LiDAR 看到了物体”，也不是普通 invalid。它表示最后可靠 3D track 仍可以通过 ego-motion 传播。

### 7.2 Track 的稳定坐标

最后可靠对象点云、centroid 和 size 应保存于连续的稳定 frame，例如局部 `odom`。每次 query 使用 query-time pose 把它们重新表达为重力对齐的 `query_body_t`：

```text
p_query = T_query_body_from_odom @ p_odom
```

不能只把旧的 query-body 坐标原样重复输入，否则机器人向前移动后对象相对位置不会变化。

### 7.3 更新与保留

当前观测满足最低点数、mask/projection confidence 和关联门限时：

- 更新稳定 frame 中的 track；
- 设置 `object_observed_now=true`；
- 更新 `last_observed_timestamp_s`；
- 保存当前点数、投影证据和关联结果。

当前 LiDAR 点下降或为零时，只要同时满足：

- 历史 track 尚未超过 provisional `memory_max_age_s`；
- ego pose 连续且可用；
- 没有实体关联冲突；
- 没有抓取、释放或已知对象运动事件；

则转为 `MEMORY_OCCLUDED`，使用传播后的历史点云。confidence 应随 age 和 pose uncertainty 单调衰减。具体阈值必须在 config 中标为 provisional，不能硬编码到算法。

### 7.4 失效与重建

只有以下情况清除或重建 track：

- age 超过上限；
- odom/pose reset、时间回退或 frame 不连续；
- 重观测 centroid/size 与旧 track 超出关联门限；
- 新检测与 entity mapping 冲突；
- 抓取开始/完成、释放完成或明确物体运动；
- calibration/config identity 改变。

抓取后必须使桌面上的旧可乐 track 失效，不能继续向 Qwen 提供“可乐仍在原处”的 ghost cloud。首版不额外增加 stall 状态机；执行层只使用既有安全 hold/fail-closed 合同。

### 7.5 近距模态职责

- NAV 阶段：当前/记忆 3D 提供距离、方向和结构；
- 接近后：track 变为带 age/confidence 的空间记忆；
- Mani 阶段：head/wrist RGB 是精细抓取主要依据；
- 3D memory 只通过 Qwen Adapter提供尺度、方位和上下文；
- 不得因此新增 semantic-3D-to-Mani-expert 的直接旁路。

## 8. PointNeXt-S 与 4-token tokenizer

PointNeXt-S 必须通过明确接口接收：

```text
points_xyz_rgb: [B, N_obj, 128, 6]
point_valid_mask: [B, N_obj, 128]
object_valid_mask: [B, N_obj]
```

learned query-token resampler 对每对象固定输出：

```text
1 object-summary token
3 local-geometry tokens
```

再融合：

```text
entity language embedding
centroid / size
point count
segmentation confidence
observation age
observed-now / track-valid / source flags
```

最终：

```text
semantic_3d_tokens: [B, N_obj * 4, D_qwen]
semantic_3d_token_mask: [B, N_obj * 4]
semantic_3d_entity_ids
```

不要用通用 PointNet 静默替换 PointNeXt-S。如果外部实现或权重不可用，先实现 typed adapter、配置检查和可注入的 test fake，并把真实 PointNeXt 集成标成显式未完成；下载依赖或 checkpoint 前需按运行环境的授权流程执行。外部权重必须记录来源、版本和 SHA-256。

## 9. Qwen Semantic-3D Adapter

3D token 不拼入 Hugging Face processor 生成的原生 input sequence。对实际 Qwen decoder 的后段层加入：

```text
A_l = CrossAttention(
    Q=LN(H_l),
    K=LN(Z_3D),
    V=LN(Z_3D),
    mask=M_3D,
)
H_l_out = H_l + tanh(g_l) * W_l(A_l)
```

要求：

- 在最后 16 层覆盖范围内均匀选择 4 层；
- 第一个 adapter 位于或早于首个导出的 last-16 state；
- 绝对层号由加载后的 Qwen config 计算并写入 resolved config；
- gate/output projection 使用 zero-impact initialization；
- 全 mask 和 gate-off 时数值上近似原 RGB-language Qwen；
- 训练、保存、恢复时 adapter 参数和 layer mapping 不丢失；
- 不用不稳定的全局变量向 forward hook 偷渡 token；通过显式 request/context 生命周期传递，并测试异常后的清理。

先用 fake/minimal Qwen layer stack 完成 CPU unit test，再做真实 Qwen checkpoint smoke。不要为了测试通过而修改 v1 Qwen 行为。

## 10. 两次前向与二分类合同

SPCGVLA v0.2 只包含：

```text
NAVIGATION
MANIPULATION
```

Pass 1 输出 binary domain 与自由形式 `subtask_text`。Pass 2 是第二次完整 Qwen forward，输入同一语言、head/wrist RGB、semantic snapshot 和 Pass 1 实际生成的 prefix。不得把 v1 四 route 作为 student 输入。

Pass 2 的 last-16 hidden states 进入被选中的动作专家：

- NAV `[10,3] @ 0.20s`，query-body `(x,y,yaw)`；
- Mani `[10,7] @ 0.04s`，六关节加 gripper；
- 13D `q6+dq6+gripper` 仅进入 Mani；
- 动作专家接口中不得出现 semantic point/tokens 参数。

旧四阶段可作为离线审计、采样和 label 派生来源，但必须在 materializer 中折叠成 binary domain，并列入 forbidden model keys。

## 11. 数据与推理快照

建立 immutable semantic snapshot，并绑定：

```text
snapshot_id
raw observation IDs
query timestamp and causal cutoff
head frame ID
object clouds and valid masks
FPS indices
PointNeXt/tokenizer output or deterministic reconstruction inputs
detector/SAM2/PointNeXt identifiers and hashes
calibration/config IDs
track states and last-observed timestamps
```

必须验证：

- `causal_cutoff <= query_time`；
- Pass 1/Pass 2 的 `request_id`、`snapshot_id`、FPS indices 和 token tensors 相同；
- 新 snapshot 在两次前向之间到达时只能供下一 query 使用；
- planner 可以独立使用更新更快的 raw LiDAR map；
- invalid semantic token 在 Adapter mask 中被完全排除；
- 语义路径失效时仍可退化为 RGB-language Qwen。

## 12. 配置建议

`configs/spcgvla_v0.json` 至少应包含并验证：

```text
model_contract_id
dataset_schema_version
point_encoder_id
semantic_token_schema

lidar.min_range_m = 0.05
lidar.max_range_m = 10.0
lidar.accumulation_seconds
lidar.max_accumulated_frames
lidar.deskew_required

segmentation.detector_id
segmentation.detector_checkpoint_sha256
segmentation.sam2_id
segmentation.sam2_checkpoint_sha256
segmentation.thresholds

projection.calibration_id
projection.occlusion_mode = lidar_z_buffer
projection.voxel_size_m

object_points.max_points = 128
object_points.padding = zero_with_valid_mask
object_points.sampling = deterministic_fps

tracking.enabled = true
tracking.memory_max_age_s
tracking.confidence_decay
tracking.reassociation_distance_m
tracking.pose_discontinuity_thresholds
tracking.invalidate_on_grasp = true
tracking.invalidate_on_release = true

tokenizer.tokens_per_object = 4
tokenizer.summary_tokens = 1
tokenizer.local_geometry_tokens = 3

qwen.semantic_3d_adapter.count = 4
qwen.semantic_3d_adapter.region = last_16_layers
qwen.semantic_3d_adapter.zero_init = true

runtime.pass_snapshot_policy = identical
runtime.allow_rgb_fallback = true
```

所有尚未实测的 tracking/segmentation 阈值标记为 `provisional: true`，并在 resolved run manifest 中保存最终值。

## 13. 建议实施波次与提交边界

### Wave 0：基线与合同

- 创建 feature branch；
- 保存 dirty-state 证据；
- 运行当前轻量相关 tests；
- 新增 v0.2 contract/config schema；
- 不改 v1 行为。

建议提交：`feat: add SPCGVLA v0.2 contracts and config`

### Wave 1：可复用语义点云与 tracker

- 把 probe 中的投影逻辑移入 package；
- 删除正式路径对 head depth/object ID 的依赖；
- 实现 128 点物化和 deterministic FPS；
- 实现 current/memory object track 与 ego-motion propagation；
- 保留 probe script 作为薄 CLI。

建议提交：`feat: add causal semantic object cloud tracking`

### Wave 2：PointNeXt-S tokenizer

- 实现可选依赖边界；
- 实现 masked PointNeXt-S adapter；
- 实现每对象 4-token resampler 和 metadata/entity fusion；
- 完成 checkpoint provenance。

建议提交：`feat: add PointNeXt semantic tokenization`

### Wave 3：Qwen Adapter 与双 Pass

- 实现 4 个 late-layer Adapter；
- 实现 zero-init identity tests；
- 建立二分类 Pass 1 和完整 Pass 2；
- 复用 NAV/Mani experts，但不改其输入合同。

建议提交：`feat: condition two-pass Qwen on semantic 3D tokens`

### Wave 4：data/training/runtime wiring

- 新 dataset schema/materializer/loader；
- optimizer groups、loss 和 checkpoint state；
- immutable runtime request/snapshot；
- RGB fallback、raw-map safety 和日志；
- scripts 与操作文档。

建议提交：`feat: wire SPCGVLA training and runtime`

不要把依赖下载、自动格式化、代码迁移和模型行为变化混在同一个提交里。

## 14. 必须新增的测试

### 感知与点集

- LiDAR projection 不读取 simulator object ID 或 head depth；
- 未来点被 causal cutoff 拒绝；
- `>128` 点 deterministic FPS，重复运行索引一致；
- `<128` 点零填充且 mask 正确，没有重复填充；
- 全空点集产生 explicit invalid；
- padding 点不影响 PointNeXt pooling/tokenizer 输出。

### 近距 object track

- 当前观测有效时创建/更新稳定-frame track；
- 近距当前点变零但 age 合法时转为 `MEMORY_OCCLUDED`；
- 机器人平移/旋转后，记忆对象在新 query-body frame 的坐标正确变化；
- confidence 随 age/pose uncertainty 单调下降；
- 超时、pose discontinuity 和 association mismatch 使 track invalid；
- grasp/release event 立即使旧位置 track invalid；
- `object_observed_now=false` 时不会伪造当前点数或时间戳。

### Tokenizer 与 Adapter

- 每个有效对象严格输出 4 token；
- object padding 和 token mask 正确；
- entity 顺序稳定；
- Adapter 全 mask/gate-off 等价于 baseline tolerance；
- Adapter 参数进入 state dict 并可 round-trip；
- Pass 1/Pass 2 使用同一 token tensor 和 snapshot ID。

### VLA 与安全边界

- 只有 binary domain 可作为模型 route；
- 旧 phase/route/object truth 被 forbidden-key validator 拒绝；
- 13D state 只进入 Mani；
- NAV/Mani expert 签名不包含 semantic tokens；
- planner path 读取 raw map，不依赖语义 snapshot；
- 语义 3D invalid 时 RGB-language fallback 可运行。

## 15. 验证命令与分层门禁

先运行当前轻量 baseline：

```bash
pytest -q tests/test_lidar_probe.py tests/test_lidar_web_viewer.py
pytest -q tests/test_joint_trajectory_contract_data.py
pytest -q tests/test_joint_trajectory_model.py
pytest -q tests/test_joint_trajectory_runtime_training.py
```

新增实现后，建议新增并运行：

```bash
pytest -q tests/test_semantic_object_cloud.py
pytest -q tests/test_semantic_object_tracker.py
pytest -q tests/test_spcgvla_pointnext.py
pytest -q tests/test_spcgvla_adapter.py
pytest -q tests/test_spcgvla_data.py
pytest -q tests/test_spcgvla_model.py
pytest -q tests/test_spcgvla_runtime.py
```

随后运行：

```bash
pytest -q
git diff --check
```

GPU/Isaac/真实 Qwen 检查分开报告，不得因环境缺失伪造通过：

1. recorded evidence 离线生成可乐/箱子 semantic cloud；
2. Web viewer 检查 current 与 memory track；
3. fake Qwen CPU smoke；
4. 真实 Qwen + PointNeXt 单 batch forward/backward；
5. Liangzhu 仿真 perception-only；
6. NAV-only low-speed；
7. Mani-only；
8. 完整移动抓取。

任何会驱动真实硬件的命令都必须单独获得用户明确授权。

## 16. 完成标准

新 agent 在交付时必须报告：

- 实际创建的分支名、base SHA 和最终 SHA；
- pre-existing dirty changes 如何保留；
- 修改/新增文件清单；
- v1 compatibility 决策；
- config/schema/model identity；
- 当前观测和 memory track 的状态转移；
- PointNeXt 来源、版本、加载报告和 hash；
- Qwen Adapter 的实际绝对层号；
- 所有测试命令与结果；
- 未运行的 GPU/Isaac/硬件检查及原因；
- 仍然存在的风险和下一步最小实验。

不得声称以下内容，除非有对应证据：

- 自动 Grounding DINO 已优于手工 prompt；
- 近距 3D memory 已改善抓取成功率；
- PointNeXt 预训练不是必要的或一定必要；
- Adapter 已被模型有效使用；
- 仿真结果可以直接代表真机。

## 17. 明确非目标

- 不删除或重写现有 joint-trajectory v1；
- 不加入传送带和动态可乐预测；
- 不实现模糊语言和开放词汇对话；
- 不让 wrist RGB 参与 v0 的 2D-to-3D 语义赋值；
- 不使用 D436/head depth 补充 student semantic cloud；
- 不添加 expert-specific 3D selector/bypass；
- 不训练大模型或提交 checkpoint/dataset/video；
- 不在没有授权时运行真机控制；
- 不顺手整理无关代码或提交现有 artifact-path 修改。

## 18. 给新 Agent 的最短入口

开始前完整阅读：

1. 本文件；
2. [README.md](README.md)；
3. [train.md](train.md)；
4. [play.md](play.md)；
5. `joint_trajectory.py/model.py/data.py/training.py/runtime.py`；
6. 当前未跟踪的 LiDAR/SAM2 probe 与 tests。

先提交一份简短 implementation plan，再创建分支并按 Wave 0 开始。能在本地证实的事项直接推进；遇到需要下载依赖、启动 Isaac/GPU 长任务、改变安全规则或驱动硬件时，按权限和风险边界请求确认。
