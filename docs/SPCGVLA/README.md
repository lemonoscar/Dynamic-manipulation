# 语义点云引导下的视觉语言动作模型的移动抓取

英文名：**Semantic Point Cloud Guided VLA for Mobile Manipulation**
简称：**SPCGVLA / SPC-MobileVLA**
文档版本：`spcgvla-design-draft-v0.2`
状态：架构草案，尚未实现或完成训练验证

## 1. 目标与首版边界

SPCGVLA 在现有 ConveyorVLA 的语言、双路 RGB、两次 Qwen 前向和双动作专家框架上，引入由语言约束的语义 3D 点云。首版目标是在 Liangzhu 3DGS 场景中完成静态可乐罐到目标箱子的移动抓取，并回答三个问题：

1. 语言指定的物体能否由 head RGB 中的检测与 SAM2 分割稳定定位；
2. LiDAR 点能否通过标定投影获得可靠的目标物体 3D 点云；
3. 语义 3D 信息经 Qwen 内部适配器融合后，能否改善导航、阶段判断和完整任务成功率。

首版明确不包含：

- 传送带和动态抓取；
- 模糊语言、指代消解和开放词汇搜索；
- wrist RGB 参与检测、SAM2 或点云语义赋值；
- D436 深度对 LiDAR 的补点；
- 将完整场景点云直接拼入 Qwen 原生 token 序列；
- 语义点云到 NAV/Mani 动作专家的独立旁路；
- 仿真 object ID、世界 mesh、真实任务阶段等特权信息作为模型输入。

## 2. 已冻结的设计决策

| 项目 | v0.2 决策 |
| --- | --- |
| VLA 主体 | 保留 ConveyorVLA 的两次完整 Qwen 前向与双动作专家 |
| 第一遍输出 | 二分类 `NAVIGATION / MANIPULATION`，并生成自由形式当前子任务文本 |
| 第二遍输入 | 与第一遍完全相同的冻结观测，再附加第一遍模型生成的回答或前缀 |
| RGB | head 与 wrist 都输入 Qwen；只有 head 用于语义 3D 构建 |
| 语义来源 | 精确语言实体 → text-grounded detector → SAM2 → LiDAR 投影筛点 |
| 检测器 | 首选 Grounding DINO；具体版本和 checkpoint 待实验锁定 |
| 3D 编码器 | 轻量 `PointNeXt-S`，不要求巨量 3D 预训练 |
| 每物体点数 | 最多 128 点；不足零填充并使用 valid mask；禁止重复填充 |
| 每物体 token | 固定 4 个：1 个 object-summary token + 3 个 local-geometry tokens |
| Qwen 融合 | Qwen 后段的 Semantic-3D Cross-Attention Adapters |
| Adapter 初版布局 | 在 Qwen 最后 16 层覆盖范围内均匀布置 4 个；绝对层号按实际配置确定 |
| 动作专家融合 | 不设额外 3D 旁路；两个专家只通过 Qwen layerwise hidden states 接收 3D 信息 |
| NAV 输出 | `[10, 3] @ 0.20 s`，query-body frame 下的 `(x, y, yaw)` |
| Mani 输出 | `[10, 7] @ 0.04 s`，六关节加连续夹爪；绝对量或增量量待锁定 |
| 机械臂状态 | 13D `q6 + dq6 + gripper1` 仅进入 Mani expert |
| 硬安全 | 原始 LiDAR 地图直接供规划器和碰撞检查器，模型无权绕过 |

不存在 `NAV_TO_SOURCE / PICK / NAV_TO_TARGET / PLACE` 四路模型路由。任务阶段可以保留为离线标注和评测字段，但不得作为模型输入。

## 3. 总体架构

```text
Task language ── entity extraction ──────────────────────────────────────┐
                                                                         │
Head RGB ── Grounding DINO ── SAM2 masks ──┐                            │
                                            ├─ LiDAR-to-image projection │
Raw LiDAR ── deskew / TF / causal buffer ───┘                            │
                     │                                                   │
                     ▼                                                   │
       Language-grounded semantic object clouds                         │
       N objects × [128, xyz+RGB] + mask + metric metadata              │
                     │                                                   │
                     ▼                                                   │
        PointNeXt-S + learned query-token resampler                      │
                     │                                                   │
                     ▼                                                   │
    Semantic 3D Token Bank: N objects × 4 tokens × D_qwen ◄─────────────┘
                     │
                     ▼
Language + head/wrist RGB ── Qwen + Semantic-3D Adapters
                     │
          ┌──────────┴──────────┐
          │ Pass 1              │
          │ VQA / domain        │
          │ NAVIGATION or       │
          │ MANIPULATION        │
          └──────────┬──────────┘
                     │ model-produced answer/prefix
                     ▼
          Pass 2: complete Qwen forward
          same RGB, language and 3D snapshot
                     │
          last-16 layerwise hidden states
                     │
             ┌───────┴────────┐
             ▼                ▼
        NAV Expert        Mani Expert ◄── 13D robot state
          [10,3]             [10,7]
             │                │
             └────── control ─┘

Raw LiDAR ── rolling occupancy / costmap ── PCT/DWA/collision checker
                         hard safety path
```

语义 3D 具有两个互不替代的用途：

- 学习通路：任务物体点云经 PointNeXt-S 和 Qwen Adapter 影响 VQA 与动作表征；
- 硬通路：未经语义筛选的原始几何地图供规划器和碰撞检查器使用。

## 4. 语言引导的语义点云

### 4.1 语言实体

首版要求指令明确列出任务物体，例如“找到可乐罐并放入蓝色箱子”。实体抽取生成有序的 `entity_id` 与 `entity_text`。只对语言中出现的任务物体构建语义对象点云，不把墙、地面和所有背景物体都对象化。

首版不额外预测 `focus_entity`。Qwen 根据完整语言、RGB 和全部有效对象 token 自行判断当前关注对象。

### 4.2 2D 到 3D

语义点云不是 SAM2 原生输出的 3D 分割，而是以下投影链：

1. Grounding DINO 使用 `entity_text` 在 head RGB 上产生候选框；
2. SAM2 在候选框约束下产生像素 mask；
3. 对 LiDAR 点做时间对齐、运动补偿和坐标变换；
4. 使用 `T_head_lidar` 和相机内参把 LiDAR 点投影至 head RGB；
5. 保留投影落在有效 mask 内且通过深度、遮挡和距离检查的点；
6. 在 query 时刻的重力对齐 `query_body_t` 坐标系内形成对象点云。

LiDAR 有效范围首版固定为 `0.05–10 m`。允许约 `1 s` 或最近 `3–5` 帧的短时因果积累，但所有点的时间戳必须不晚于 query time。机器人接近后，可乐罐可能因扫描几何、机身或机械臂遮挡而暂时没有当前 LiDAR 点；这种可解释的近距丢失不得立即删除对象。系统应保留最后一次可靠的 3D object track，依靠 odometry/pose 将其重新表达在当前 `query_body_t`，同时增加 age 并降低 confidence。只有超过最大 age、pose 不连续、重关联失败、确认物体运动，或发生抓取/释放状态变化时，才清除或重建 track，避免 ghost cloud。

### 4.3 每物体数据契约

模型可见的最小对象契约为：

```text
entity_id
entity_text
points_xyz_rgb: [128, 6]
point_valid_mask: [128]
centroid_query_body_m: [3]
size_query_body_m: [3]
point_count
segmentation_confidence
observation_age_s
object_valid
object_observed_now
object_track_valid
last_observed_timestamp_s
track_source
```

点数处理必须确定性且训练、推理一致：

- 多于 128 点：固定米制 voxel 去重，再使用 deterministic FPS 采样到 128 点；
- 少于 128 点：使用零行填充，并由 `point_valid_mask` 排除；不得复制已有点；
- 没有点：点张量全零，但 `object_valid=false`；未观测、空观测和过期状态应在元数据中可区分；
- PointNeXt-S 的局部坐标使用 `point_xyz - centroid`，不做单位球归一化；米制质心和尺寸通过元数据分支重新融合。

`object_valid` 表示本次送入编码器的对象表示是否有效；它可以来自当前观测，也可以来自尚未过期的历史 track。`object_observed_now=false` 不等于 `object_track_valid=false`。历史 track 被使用时，应记录 `track_source=memory_occluded`、当前点数、记忆点数、最后观测时间、age 和衰减后的 confidence，不能伪装成当前 LiDAR 仍然看到了物体。

原始证据与派生证据可以额外保存 box、mask、投影点索引以及 grounding/mask/projection 的分项置信度，但不应无条件扩大模型输入。

## 5. PointNeXt-S 与语义 3D Tokenizer

PointNeXt-S 处理带 mask 的 `[128, xyz+RGB]` 对象点集，负责提取局部几何与颜色特征。它可以从随机初始化开始，也可以加载结构兼容的 PointNeXt 权重；首版不依赖 ULIP、Concerto 或大规模 3D-language 预训练。

每个对象的 token 化流程为：

```text
masked xyz+RGB points
        │
        ▼
   PointNeXt-S
        │
        ▼
Learned Query Token Resampler
        │
        ├─ 1 object-summary token
        └─ 3 local-geometry tokens
        │
        ▼
fuse(entity language embedding,
     centroid, size, point count,
     confidence, age, valid flag)
        │
        ▼
project to D_qwen
```

若指令解析出 `N` 个实体，输出：

```text
semantic_3d_tokens: [N * 4, D_qwen]
semantic_3d_token_mask: [N * 4]
semantic_3d_entity_ids: [N * 4]
```

对象顺序由语言实体顺序和稳定的 `entity_id` 决定，禁止依赖检测器的偶然返回顺序。

## 6. Qwen 内部 Semantic-3D Adapter

3D token 不直接拼接进 Qwen 原生输入序列，因此不会改写文本、图片 token 的位置和处理器契约。对选定的 Qwen 后段层加入门控 cross-attention：

```text
A_l  = CrossAttention(
           Q = LN(H_l),
           K = LN(Z_3D),
           V = LN(Z_3D),
           mask = M_3D)

H'_l = H_l + tanh(g_l) * W_l(A_l)
```

初版在 Qwen 最后 16 层覆盖范围内均匀布置 4 个 Adapter。第一个 Adapter 必须位于或早于第一个被导出的 last-16 hidden state，使随后导出的层状态都可能携带语义 3D 信息。绝对层号、attention heads 和内部维度需读取实际 Qwen 配置后锁定。

`g_l` 与输出投影采用零影响初始化，使新模型在初始化时近似原 RGB-language Qwen。所有无效 3D token 由 mask 排除；整组 3D 无效时，Qwen 退化为 RGB-language 路径。

## 7. 两次 Qwen 前向

### 7.1 Pass 1：VQA 与二分类 domain

输入：

- 任务语言；
- head RGB temporal observation；
- wrist RGB temporal observation；
- 当前冻结的 semantic 3D token bank。

输出：

- `NAVIGATION` 或 `MANIPULATION`；
- 自由形式、尽可能具体的当前子任务文本。

这里的二分类决定第二遍后启用哪个动作专家，但不把完整任务硬切成四个 route。

### 7.2 Pass 2：VLA 表征

Pass 2 是一次完整 Qwen 前向，而不是复用 Pass 1 中间缓存。它使用：

- 与 Pass 1 完全相同的语言、head/wrist RGB；
- 同一个 immutable semantic 3D snapshot；
- 同一批 PointNeXt 采样结果与 token；
- Pass 1 由模型实际生成的回答或前缀。

Pass 2 导出最后 16 层的 layerwise hidden states，供被选中的动作专家逐层 cross-attention。训练初期可以使用 teacher prefix，之后必须逐步暴露 model-produced prefix，消除训练与推理不一致。

## 8. 双动作专家与控制

```text
Pass 2 last-16 hidden states
          │
     binary domain
       ┌──┴───┐
       ▼      ▼
 NAV Expert  Mani Expert ◄── q6, dq6, gripper
  [10,3]       [10,7]
```

首版不加入 `NAV object selector`、`Mani object selector`、`g_NAV/g_Mani` 或语义点 token residual bypass。理由是语义 3D 已在 Qwen 内部进入 VQA 和第二遍 layerwise states；再添加专家旁路会引入重复融合和难以归因的训练路径。

一次 query 只运行或只采纳被二分类选中的专家。NAV waypoint 必须经过 PCT/DWA 和最新占据地图检查；Mani trajectory 必须经过关节限位和碰撞检查。学习策略没有最终安全否决权。

## 9. 数据、因果性与可追溯性

本地必须保存可重新生成派生结果的原始证据：

- head/wrist RGB 与时间戳；
- 原始 LiDAR 点云、per-point time（若设备提供）、IMU/pose；
- 相机内参、LiDAR-camera/body 外参与 calibration hash；
- 机器人状态、控制命令、任务语言和结果。

派生证据包括：

- entity phrases；
- detector boxes/scores；
- SAM2 masks/scores；
- LiDAR 投影索引和时序对象点云；
- voxel/FPS 采样索引；
- PointNeXt/tokenizer 配置与 checkpoint ID。

每个派生 snapshot 必须绑定原始 observation ID、因果截止时间、检测器/SAM2/PointNeXt 标识及 SHA-256、标定 ID 和配置 ID。训练与推理必须调用同一套投影、采样和 token 化实现。

## 10. 论文 Overview 图的固定语义

论文总览图建议采用从左到右的四块布局：

1. **Language-Grounded Semantic 3D Perception**：语言实体、head RGB、Grounding DINO、SAM2、LiDAR 投影和对象彩色点云；
2. **PointNeXt-S Semantic 3D Tokenizer**：每物体 128 点、valid mask、4-token resampler、语言与米制元数据融合；
3. **Two-Pass Qwen VLM with Semantic-3D Adapters**：第一遍 VQA/domain，第二遍带模型前缀的完整前向，四个后段 Adapter；
4. **Routed Action Experts and Safety**：last-16 states 分别进入 NAV/Mani，13D 状态仅到 Mani，原始 LiDAR costmap 走独立硬安全路径。

图中必须明确画出两条边界：

- 3D token 进入 Qwen Adapter，而不是拼进原生 Qwen 输入 token 序列；
- 首版没有语义点云直达动作专家的箭头。

可直接交给绘图模型或设计人员的详细 prompt：

```text
Create a publication-quality horizontal overview figure for a robotics and
embodied-AI paper. Title: “Semantic Point Cloud Guided VLA for Mobile
Manipulation (SPCGVLA)”. Use a clean white background, vector-like scientific
graphics, consistent typography, thin dark-gray arrows, rounded modules, and a
restrained color palette. The figure must be readable at a two-column paper
width and must not look like a marketing infographic.

Organize the figure into four left-to-right stages.

Stage 1 — Language-Grounded Semantic 3D Perception, colored teal and green.
At the top, show a precise task instruction such as “Find the Coke can and
place it into the blue box.” Split the sentence into two ordered entity phrases:
“Coke can” and “blue box”. Below it, show a head-camera RGB image of an indoor
Liangzhu 3DGS room containing a quadruped mobile manipulator, one Coke can, and
two boxes. Draw Grounding DINO bounding boxes and SAM2 masks only on the head
RGB image. Explicitly show that wrist RGB does not enter Grounding DINO or SAM2.
Beside the image, show a raw LiDAR scan, labeled “0.05–10 m, causal, deskewed”.
Draw a calibrated LiDAR-to-head-camera projection module that intersects the
LiDAR points with each SAM2 mask. Its outputs are two separately colored 3D
object point clouds, each bound to its language entity. Add a small note:
“SAM2 segments 2D masks; semantic 3D points are obtained by projection.” Show a
short causal accumulation buffer labeled “about 1 s / latest 3–5 frames” and a
gravity-aligned query-body coordinate frame.

Stage 2 — PointNeXt-S Semantic 3D Tokenizer, colored orange. For each language-
grounded object, show a tensor labeled “up to 128 xyz+RGB points”. Visualize
deterministic voxel deduplication and farthest-point sampling. For fewer than
128 points, show zero padding plus a valid mask, never duplicated points. Show
centroid subtraction for local shape encoding, while retaining metric centroid
and metric size in query-body coordinates. Feed the masked point set into a
compact PointNeXt-S encoder. Then show a learned query-token resampler producing
exactly four tokens per object: one larger object-summary token and three
local-geometry tokens. Fuse every object token set with the corresponding
language entity embedding and minimal metadata: centroid, size, point count,
segmentation confidence, observation age, and valid flag. Project all tokens to
the Qwen hidden dimension and label the result “Language-Grounded Semantic 3D
Token Bank: N × 4 × D_qwen”, with a token-validity mask.

Stage 3 — Two-Pass Qwen VLM with Semantic-3D Adapters, colored blue and violet.
Show the native Qwen inputs as task language plus head RGB video and wrist RGB
video. Draw the Semantic 3D Token Bank entering four gated cross-attention
adapter blocks distributed across the late Qwen layers, specifically within the
last-16-layer region. Do not draw the 3D tokens concatenated into the native
language/image token sequence. Annotate one adapter with:
H'_l = H_l + tanh(g_l) W_l CrossAttn(LN(H_l), LN(Z_3D)),
and note “zero-impact initialization”. First show Pass 1 as a complete Qwen
forward producing only a binary domain decision, NAVIGATION or MANIPULATION,
plus a detailed free-form current-subtask answer. Do not show four task routes
and do not show a focus-entity classifier. Then show the model-produced answer
feeding Pass 2. Pass 2 is another complete Qwen forward using exactly the same
frozen language, head/wrist RGB, semantic 3D snapshot, sampled points, and token
bank. Label its output “last 16 layerwise hidden states”.

Stage 4 — Routed Action Experts and Hard Safety, colored red for navigation and
purple for manipulation. Route the Pass-2 last-16 hidden states to one of two
experts according to the binary Pass-1 decision. The NAV Expert outputs ten
query-body waypoints, “[10 × (x, y, yaw)] at 0.20 s”. The Mani Expert outputs
ten arm-and-gripper actions, “[10 × (6 joints + gripper)] at 0.04 s”. Draw a
separate 13D robot-state arrow, “q6 + dq6 + gripper”, entering only the Mani
Expert. Do not draw semantic point tokens, object selectors, residual 3D
bypasses, or separate 3D gates directly entering either action expert; the only
learned 3D route to both experts is through the Qwen hidden states.

Along the bottom, draw a clearly separated gray hard-safety path: raw LiDAR to
rolling occupancy/costmap to PCT/DWA and collision checker. Connect proposed
NAV and manipulation trajectories to this safety layer before execution. Label
it “latest raw geometry, final veto authority; VLA cannot bypass”. Use solid
arrows for runtime data, dashed arrows only for metadata/provenance, and add a
small legend. Clearly distinguish the immutable semantic snapshot shared by the
two Qwen passes from the potentially newer raw LiDAR map used by the planner.

Avoid decorative 3D renders, dense text, photorealistic humans, conveyor belts,
four-stage route labels, D436 depth input, wrist-camera segmentation, simulator
ground-truth IDs, and any direct semantic-3D-to-action-expert connection.
```

## 11. 代码落点建议

```text
src/conveyor_bench/perception/
├── language_entity_extractor.py
├── grounded_sam2.py
├── lidar_camera_projection.py
├── semantic_object_cloud.py
└── semantic_3d_snapshot.py

src/conveyor_bench/conveyorvla/
├── spcgvla_pointnext.py
├── spcgvla_tokenizer.py
├── spcgvla_adapter.py
├── spcgvla_model.py
├── spcgvla_data.py
└── spcgvla_runtime.py
```

新的实验必须使用独立 identity，例如：

```text
model_contract_id: spcgvla-mobile-manipulation-v0.2
dataset_schema_version: spcgvla-mobile-manipulation-v0.2
point_encoder_id: pointnext-s-spcgvla-v0
semantic_token_schema: four-tokens-per-object-v0
```

这些名称目前是设计占位符，不表示代码已经存在。

## 12. 晋级标准与待定项

进入正式联合训练前至少应证明：

- 可乐罐和目标箱子的 2D mask 可重复；
- mask 内 LiDAR 点在斜视 3D 可视化中与物体位置一致；
- 短时积累不会留下明显 ghost；
- 每对象 128 点和 4 token 的实现满足确定性与 mask 契约；
- Adapter 关闭时能近似复现 RGB-language 基线；
- raw-only planner 路径保持独立可用。

仍需实验锁定：

- Grounding DINO、SAM2 和 PointNeXt 的具体版本与权重；
- voxel size、遮挡检查、短时积累帧数；
- Qwen Adapter 的绝对层号、head 数和中间维度；
- Mani action 使用绝对关节量还是增量量；
- 二分类切换的迟滞、确认和失效保持规则；
- 各损失权重和联合解冻节奏。

训练细节见 [train.md](train.md)，在线推理与安全流程见 [play.md](play.md)。
