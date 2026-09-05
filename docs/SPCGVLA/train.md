# SPCGVLA 训练方案

文档版本：`spcgvla-training-draft-v0.2`
状态：训练设计草案，尚无可宣称的训练结果

## 1. 训练目标

训练目标不是让一个新 3D 模型替代 ConveyorVLA，而是在保留两次 Qwen 前向与 NAV/Mani 双专家的前提下，让语言指定的任务物体点云通过 Qwen 内部 Semantic-3D Adapter 同时影响：

- Pass 1 的 `NAVIGATION / MANIPULATION` 判断和当前子任务描述；
- Pass 2 的最后 16 层 hidden states；
- 由这些 states 条件化的 NAV 或 Mani 动作生成。

首版训练对象为 Liangzhu 3DGS 静态可乐罐到箱子的移动抓取，不含传送带、动态物体追踪、模糊语言和 D436 深度补点。

## 2. 模型可见输入与禁止信息

每个 query 的模型可见输入为：

```text
language
head RGB temporal observation
wrist RGB temporal observation
semantic object point sets
    points_xyz_rgb: [N, 128, 6]
    point_valid_mask: [N, 128]
    centroid/size/count/confidence/age/object_valid
Mani-only robot state: [13]
```

其中 wrist RGB 只进入 Qwen，不用于 Grounding DINO、SAM2 或点云语义赋值。13D 状态只进入 Mani expert，不进入 Qwen、Pass 1 或 NAV expert。

以下字段可用于 teacher、loss、审计或评测，但禁止进入 student observation：

- simulator object ID、GT segmentation、world mesh；
- GT object pose、base pose 或未来轨迹；
- `NAV_TO_SOURCE / PICK / NAV_TO_TARGET / PLACE` 等真实任务阶段；
- query 时刻之后的 RGB、LiDAR 或 pose；
- evaluator truth、成功标签和人工修正答案。

## 3. 原始数据与派生数据

### 3.1 原始证据

每个 episode 至少保存：

```text
episode/
├── rgb/
│   ├── head/
│   └── wrist/
├── lidar/
│   ├── clouds/
│   ├── imu.jsonl
│   └── pose.jsonl
├── calibration/
├── robot_state.jsonl
├── control.jsonl
├── queries.jsonl
└── instruction.json
```

点云必须尽可能保留原始字段和时间戳。模型输入可由派生缓存加速，但原始 RGB、LiDAR、pose 和标定应足以重新生成所有 mask、对象点云和 token 输入。

### 3.2 派生证据

派生目录建议记录：

```text
derived/
├── language_entities.json
├── grounding_dino/
├── sam2_masks/
├── lidar_projection/
├── semantic_object_clouds/
├── fps_indices/
└── manifests/
```

每个派生样本必须记录：

```text
raw_observation_id
query_timestamp_s
causal_cutoff_timestamp_s
entity_id / entity_text
detector_id + checkpoint_sha256
sam2_id + checkpoint_sha256
pointnext_config_id + checkpoint_sha256 or random_init_id
calibration_id + calibration_sha256
projection_config_id
voxel_fps_config_id
semantic_snapshot_id
```

禁止只保存渲染后的语义点云图片而丢弃原始证据。

## 4. 数据物化契约

### 4.1 语言实体与 2D 分割

首版指令必须精确且显式包含对象。实体顺序由语言解析结果固定。每个实体在 head RGB 上依次执行：

1. Grounding DINO 文本条件检测；
2. 候选框筛选与置信度记录；
3. SAM2 mask 生成；
4. mask、box 与实体 ID 绑定。

不使用 wrist RGB 参与这条链。检测器和 SAM2 的具体版本必须写入 manifest，不能在同一数据集版本中静默变化。

### 4.2 LiDAR 投影与短时积累

LiDAR 点先完成 deskew、时间同步、TF 变换和距离裁剪，再投影到 head RGB。首版有效范围为 `0.05–10 m`。语义点云允许使用约 `1 s` 或最近 `3–5` 帧，但必须满足：

- 所有点的 timestamp `<= query_timestamp`；
- 全部点表达在 query 时刻重力对齐的 `query_body_t`；
- 短暂或近距可解释丢失时保留并传播历史 track；只有 track 超时、pose 不连续、重关联失败、抓取/释放或确认物体运动时才清理或重建；
- train 与 runtime 使用同一遮挡判断、边界膨胀和投影代码。

### 4.3 固定点集

每对象物化为：

```text
points_xyz_rgb: [128, 6]
point_valid_mask: [128]
centroid_query_body_m: [3]
size_query_body_m: [3]
point_count: scalar
segmentation_confidence: scalar
observation_age_s: scalar
object_valid: bool
object_observed_now: bool
object_track_valid: bool
last_observed_timestamp_s: scalar
track_source: enum
```

规则：

- 多于 128 点：固定米制 voxel 去重，然后 deterministic FPS；
- 少于 128 点：零填充，不复制有效点；
- 没有有效点：全零点集和全 false mask，同时 `object_valid=false`；
- PointNeXt 的 xyz 输入减去对象质心，但不做单位球尺度归一化；
- RGB 的范围、色彩空间和归一化常数必须写入配置；
- 质心、尺寸、点数和 age 的归一化统计只由训练集计算。

训练样本必须区分当前 LiDAR 观测与历史 track。近距遮挡时可以使用由最后可靠观测经 ego-motion 传播得到的记忆点云，但必须令 `object_observed_now=false`、`object_track_valid=true`，记录 `track_source=memory_occluded`、最后观测时间和衰减后的 confidence。不得把记忆点重新标注成当前传感器返回。

数据加载器应验证 `point_valid_mask.sum() == min(point_count_after_voxel, 128)`，并保证同一 snapshot 在重复物化时产生相同 FPS 索引。

## 5. 3D 编码与 token 对齐

### 5.1 PointNeXt-S 初始化

首版使用轻量 PointNeXt-S。可比较两种初始化：

- 随机初始化；
- 结构和输入通道兼容的 PointNeXt 权重 warm-start。

不把大规模 3D-language 预训练作为启动条件。如果外部权重的输入通道、邻域尺度或分类头不兼容，只加载可验证匹配的 backbone 参数，并在 manifest 中记录 missing/unexpected keys。

### 5.2 固定四 token

PointNeXt-S 输出由 learned query-token resampler 压缩为每对象 4 个 token：

- 1 个 object-summary token；
- 3 个 local-geometry tokens。

随后融合对应的 entity language embedding 与最小米制元数据，再投影到 `D_qwen`。padding 对象和无效点不能影响 pooling、归一化或 attention。

### 5.3 轻量 3D 预对齐

PointNeXt-S 不需要先做巨量预训练，但在联合训练前建议做短周期预对齐：

1. **对象一致性**：同一对象点云的两个扰动视图，其 summary embedding 应接近；
2. **RGB–3D 对齐**：head RGB 中 SAM2 区域的 Qwen pooled visual feature，与对应 3D object-summary token 对齐；RGB 目标分支 stop-gradient 或保持冻结。

允许的点云扰动包括点 dropout、坐标/距离噪声、局部遮挡、轻微外参抖动和颜色扰动。扰动后必须重新维护 valid mask。首版不要求点云重建 loss。

## 6. Qwen Semantic-3D Adapter 训练

3D token 不拼接到原生 Qwen 序列，而是在最后 16 层覆盖范围内通过 4 个 cross-attention Adapter 注入：

```text
A_l  = CrossAttention(LN(H_l), LN(Z_3D), LN(Z_3D), mask=M_3D)
H'_l = H_l + tanh(g_l) * W_l(A_l)
```

训练初始化要求：

- adapter output projection 使用零影响初始化；
- gate `g_l` 初始化为使残差接近零的值；
- 无效 token 必须被 attention mask 完全排除；
- 全部 3D 无效时，输出应数值上接近 RGB-language Qwen；
- 第一个 Adapter 位于或早于第一个导出的 last-16 state。

绝对层号、head 数和内部维度必须在读取实际 Qwen checkpoint 配置后固化到实验 manifest。

## 7. 损失函数

建议总损失为：

```text
L_total =
    lambda_answer      * L_answer
  + lambda_domain      * L_domain
  + lambda_nav         * L_nav_fm
  + lambda_mani        * L_mani_fm
  + lambda_boundary    * L_boundary
  + lambda_progress    * L_progress
  + lambda_consistency * L_3d_consistency
  + lambda_rgb3d       * L_rgb3d
  + lambda_gate        * L_gate_reg
```

含义：

- `L_answer`：Pass 1 当前子任务文本的语言建模损失；
- `L_domain`：`NAVIGATION / MANIPULATION` 二分类损失；
- `L_nav_fm`：NAV expert 的 flow-matching 动作损失；
- `L_mani_fm`：Mani expert 的 flow-matching 动作损失；
- `L_boundary`：二分类切换边界的辅助损失；
- `L_progress`：任务进度辅助损失，仅由合法标注监督；
- `L_3d_consistency`：同对象双扰动的 3D 表征一致性；
- `L_rgb3d`：head RGB mask 区域与 3D summary token 的对齐；
- `L_gate_reg`：防止 Adapter gate 无约束爆炸的可选正则。

不得加入要求 NAV/Mani 直接读取 semantic point token 的旁路损失。损失权重必须通过小规模实验确定，不能在文档中伪装成已验证超参数。

## 8. 分阶段训练

### Gate P：感知链路验收

在训练模型前，先对自动派生数据做人工与统计验收：

- 可乐罐和箱子的 2D box/mask 正确率；
- mask 内投影点的 3D precision/recall；
- 质心和尺寸误差；
- 每对象点数分布和空对象比例；
- 斜视 3D 可视化中的错配、穿透和 ghost。

未通过时先修正标定、同步或投影，不应依靠 PointNeXt 学习掩盖系统性错位。

### Stage A：小样本可丢弃过拟合

使用少量 episode 验证：

- 128 点与 mask 的前后向正确；
- 4-token resampler 和 Adapter 梯度存在；
- Pass 1/Pass 2 使用同一 3D snapshot；
- 二分类只激活对应动作损失；
- checkpoint 可以完整保存和恢复新模块。

该阶段只证明管线可训练，不作为性能证据。

### Stage B：冻结 Qwen 主干

优先训练：

- PointNeXt-S；
- learned query-token resampler；
- entity/meta fusion 与 3D projector；
- 4 个 Qwen Semantic-3D Adapters 和 gates；
- 二分类相关新增参数；
- NAV/Mani action experts 中需要适配新表征的参数。

Qwen 原始语言与视觉主干保持冻结，以验证新增 3D 路径是否能独立学到有效残差。

### Stage C：有限联合微调

按验证集结果逐步开放：

- Qwen 顶部层 LoRA 或少量后段层；
- vision projector 或顶部视觉参数；
- domain/VQA 相关参数；
- 两个动作专家。

参数组使用独立学习率和梯度统计。通常 PointNeXt、resampler 和 Adapter 学习率高于 Qwen LoRA；具体数值以 pilot sweep 为准。

### Stage D：模型前缀与闭环暴露

训练早期 Pass 2 可使用 teacher prefix。随后逐步混入 Pass 1 实际生成的 model prefix，并分别报告：

- teacher-prefix open-loop；
- model-prefix open-loop；
- model-prefix closed-loop。

只有最后一项可支撑部署结论。

## 9. 鲁棒性增强

训练应覆盖：

- 点 dropout、距离噪声和局部空洞；
- LiDAR 帧丢失、timestamp delay、pose noise、extrinsic jitter；
- Grounding DINO 低置信度或 SAM2 mask 腐蚀/膨胀；
- 对象部分遮挡、短时未观测和过期；
- 接近可乐后当前 LiDAR 点降为零、但历史 track 仍有效的 near-field blind-zone 样本；
- track 超时、pose 跳变、抓取和释放导致的显式 track invalidation；
- 整个 3D modality dropout。

建议 `10–20%` 样本使用整组 3D dropout 作为起始 sweep 区间，而非固定最终值。无效 3D 必须通过 mask 和 valid flag 表达；全零点集不能被解释为“自由空间”。

## 10. 采样与批处理

batch 采样应至少平衡：

- `NAVIGATION / MANIPULATION`；
- 可乐罐/箱子对象；
- 近距离/远距离；
- 高/低点数；
- 有效/部分有效/无效语义 3D；
- 阶段切换附近与稳定段。

对象数可以随语言而变化，但 batch 内通过 object-slot padding 和 object/token mask 对齐。固定的是每对象 128 点和 4 token，不要求不同样本真实对象数相同。

## 11. 必做消融

| 实验 | 目的 |
| --- | --- |
| A. RGB-language ConveyorVLA baseline | 确立不含 3D 的科学对照；不是启动训练的硬性前置 checkpoint |
| B. PointNeXt geometry tokens，不融合 entity embedding | 判断提升来自几何还是语言语义绑定 |
| C. 完整 PointNeXt-S + 4-token + Qwen Adapter | 主方案 |
| D. C + 3D modality dropout | 判断失效鲁棒性 |
| E. C 推理时关闭 Adapter gate | 检验模型是否真正使用 3D |
| F. C 对象点打乱或实体映射置换 | 检验是否学习了正确语言—对象对应关系 |
| G. PointNeXt 随机初始化 vs 兼容 warm-start | 判断是否需要外部 3D 权重 |

首版不做“Qwen Adapter + NAV/Mani 3D 旁路”的组合消融，因为旁路不属于 v0.2 架构。如未来 Adapter 信息传递不足，应建立新合同版本后再研究。

## 12. 指标与晋级门

### 感知层

- detector/mask precision、recall 和失败率；
- semantic point precision、recall；
- centroid/size error；
- empty-object ratio、point-count distribution；
- temporal jitter 和 ghost duration；
- snapshot age 与物化延迟。

### 模型层

- domain accuracy、切换延迟和抖动；
- VQA answer accuracy/一致性；
- NAV ADE/FDE、waypoint clearance；
- Mani trajectory error；
- Adapter gate magnitude、attention entropy；
- 3D dropout degradation；
- Adapter-off 与 wrong-entity control 的性能下降。

### 任务层

- source/target arrival success；
- grasp success；
- place success；
- full-task success；
- collision、planner reject 和 completion time；
- observation-to-action age。

从一阶段晋级到下一阶段必须保存对应配置、checkpoint、数据 manifest 和评测报告。不得以单个成功视频替代数据集指标。

## 13. Checkpoint 与恢复

checkpoint 必须显式包含：

```text
PointNeXt-S weights and config
query-token resampler
entity/meta fusion
3D projector
four Semantic-3D Adapters and gates
Qwen LoRA or unfrozen parameters
domain/VQA parameters
NAV and Mani experts
optimizer/scheduler/scaler state
global step and RNG state
model/data/token schema IDs
detector/SAM2/calibration provenance
```

恢复测试应验证：

- 同一输入产生相同的 FPS 索引、token mask 和近似相同输出；
- Adapter gate 状态未遗漏；
- teacher/model prefix 模式与训练阶段一致；
- 不兼容 schema 或 calibration hash 被明确拒绝。

## 14. 当前待定项

- 具体 Grounding DINO、SAM2 和 PointNeXt checkpoint；
- voxel size、FPS 实现和短时积累长度；
- Qwen Adapter 的绝对层号、attention head 和内部维度；
- 各损失权重、学习率与解冻节奏；
- Mani action 的绝对/增量定义；
- domain 边界标注和 model-prefix curriculum。

总体模型结构见 [README.md](README.md)，运行时流程见 [play.md](play.md)。
