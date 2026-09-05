# SPCGVLA 推理与闭环运行流程

文档版本：`spcgvla-play-draft-v0.2`
状态：运行时设计草案，不代表当前代码已经具备这些接口

## 1. 运行目标与边界

运行时在 Liangzhu 3DGS 静态可乐罐到箱子任务中完成：

1. 从精确任务语言抽取对象实体；
2. 只用 head RGB 做 Grounding DINO + SAM2；
3. 将 `0.05–10 m` 的 LiDAR 点投影到 mask，形成语言绑定对象点云；
4. 用 PointNeXt-S 将每对象最多 128 点编码成固定 4 个 token；
5. 通过 Qwen 后段 Semantic-3D Adapters 完成两次 Qwen 前向；
6. Pass 1 二分类选择 NAV 或 Mani，Pass 2 生成对应动作；
7. 使用独立的原始 LiDAR 地图和碰撞检查器执行硬安全约束。

首版不运行传送带动态抓取，不用 D436 补点，不把 wrist RGB 用于语义分割，也不把 3D token 直接旁路到动作专家。

## 2. 运行时线程与信息流

```text
Head RGB thread ───────────────┐
Wrist RGB thread ──────────────┤
LiDAR / pose thread ───────────┤
Robot-state thread ────────────┤
                               ▼
                 time-indexed sensor buffers
                               │
Language entities ─────────────┤
                               ▼
          semantic perception worker (asynchronous)
 Grounding DINO → SAM2 → projection → causal accumulation
                               │
                               ▼
             immutable Semantic3DSnapshot
       points/masks/metadata + sampled indices + tokens
                               │
                        5 Hz policy query
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
        Pass 1 Qwen VQA/domain          latest raw LiDAR map
               │                               │
       model answer/prefix                      ▼
               │                         PCT/DWA/safety
               ▼
        Pass 2 full Qwen
               │
       last-16 hidden states
               │
          NAV or Mani expert
               │
         proposed trajectory
               └──────────► safety/execution
```

语义感知首版不要求严格实时，可以低于 policy 频率异步更新。policy query 只能读取一个已经完成且不可变的最近因果 snapshot，不能在两次 Qwen 前向之间等待新分割。

## 3. 建议运行时数据结构

```python
@dataclass(frozen=True)
class SemanticObjectCloud:
    entity_id: str
    entity_text: str
    points_xyz_rgb: np.ndarray          # [128, 6]
    point_valid_mask: np.ndarray        # [128]
    centroid_query_body_m: np.ndarray   # [3]
    size_query_body_m: np.ndarray       # [3]
    point_count: int
    segmentation_confidence: float
    observation_age_s: float
    object_valid: bool
    object_observed_now: bool
    object_track_valid: bool
    last_observed_timestamp_s: float | None
    track_source: str                 # current | memory_occluded | invalid


@dataclass(frozen=True)
class Semantic3DSnapshot:
    snapshot_id: str
    query_timestamp_s: float
    causal_cutoff_timestamp_s: float
    query_frame: str
    objects: tuple[SemanticObjectCloud, ...]
    fps_indices: tuple[np.ndarray, ...]
    semantic_3d_tokens: torch.Tensor    # [N * 4, D_qwen]
    semantic_3d_token_mask: torch.Tensor
    semantic_3d_entity_ids: tuple[str, ...]
    detector_id: str
    sam2_id: str
    point_encoder_id: str
    calibration_id: str
    config_id: str


@dataclass(frozen=True)
class SPCObservationRequest:
    request_id: str
    query_timestamp_s: float
    language: str
    head_frames: tuple[FrameRef, ...]
    wrist_frames: tuple[FrameRef, ...]
    semantic_snapshot: Semantic3DSnapshot | None
    robot_state_13d: np.ndarray
```

具体实现可以不在 snapshot 中缓存 GPU tensor，但必须缓存或锁定能确定性重建相同 token 的对象点集、valid mask、FPS 索引、配置和 encoder checkpoint。Pass 1 与 Pass 2 不允许各自重新随机采样点。

## 4. 语义感知 worker

### 4.1 初始化

episode 开始时：

1. 解析精确任务语言，得到稳定有序的 `entity_id/entity_text`；
2. 加载 Grounding DINO、SAM2、PointNeXt-S 与 token resampler；
3. 校验相机内参、`T_head_lidar`、`T_base_lidar` 和 calibration hash；
4. 初始化每对象短时因果点云 buffer；
5. 记录所有模型和配置 ID。

语言实体在同一任务中保持稳定，不按检测器候选框顺序重新编号。

### 4.2 head RGB 检测与分割

对选定的 head frame：

```text
entity_text
    │
    ▼
Grounding DINO box candidates
    │ threshold / NMS / entity binding
    ▼
SAM2 masks
```

wrist RGB 不进入该流程。若某实体没有通过阈值的 box 或 mask，应产生显式 invalid/empty 对象状态，而不是把其他候选错误绑定给它。

### 4.3 LiDAR 投影

对每个点按实际采样时间完成 deskew，并变换到投影所需坐标。投影应检查：

- range 在 `0.05–10 m`；
- 点位于相机前方且落在图像范围内；
- 像素属于目标 mask；
- 深度与局部 z-buffer 或遮挡规则一致；
- timestamp 不晚于 snapshot 的因果截止时间。

通过检查的点携带 head RGB 对应颜色，随后变换至重力对齐 `query_body_t`。若没有 per-point time，必须记录退化模式，不能伪称完成了逐点 deskew。

### 4.4 短时积累

允许约 `1 s` 或最近 `3–5` 帧的因果积累。当前观测与对象记忆必须分开维护。机器人接近可乐后，当前 LiDAR 点可能因近距扫描几何或自体遮挡消失；只要最后 track 尚未过期、ego pose 连续且没有物体运动证据，就保留最后可靠对象点云，并通过 odometry/pose 将它从稳定 frame 变换到当前 `query_body_t`。此时设置：

```text
object_observed_now = false
object_track_valid = true
track_source = memory_occluded
observation_age_s = query_time - last_observed_timestamp_s
```

历史 track 的 confidence 应随 age 和 pose uncertainty 衰减。下列情况才应清空或重建对应对象 buffer：

- track age 超过配置阈值；
- 重观测后的质心或实体关联发生不合理跳变；
- 机器人开始/完成抓取、完成释放或对象明显运动；
- pose/calibration 状态无效；
- 时间戳回退或 frame 变换不连续。

buffer 的更新频率可以低于 LiDAR 原生频率，首版优先保证几何正确和可审计。

### 4.5 固定点集与 token 化

对每对象：

1. 固定米制 voxel 去重；
2. 若超过 128 点，deterministic FPS 到 128；
3. 若不足 128 点，零填充并设置 valid mask；
4. 计算 query-body 米制质心和尺寸；
5. xyz 减质心后输入 PointNeXt-S，不做单位球缩放；
6. learned query-token resampler 生成 1 个 summary + 3 个 local tokens；
7. 融合 entity embedding 和最小 metric metadata；
8. 投影为 `D_qwen`，生成 token mask 与 entity IDs。

同一个 snapshot 的重复 token 化必须确定性一致。

## 5. 冻结 ObservationRequest

每个 policy query 在开始时一次性冻结：

- query time；
- head/wrist temporal frames；
- 任务语言；
- 最近的合法 semantic snapshot；
- snapshot 对应的对象点集、FPS indices 和 3D tokens；
- 13D 机器人状态；
- calibration/config/model IDs。

快照验收条件至少包括：

```text
semantic_snapshot.causal_cutoff_timestamp_s <= query_timestamp_s
semantic_snapshot.query_frame == current query_body frame contract
calibration_id == runtime expected calibration_id
point_encoder_id == runtime expected point_encoder_id
config_id == runtime expected config_id
all tensor shapes and masks valid
```

Pass 1 和 Pass 2 必须引用相同 `request_id` 与 `snapshot_id`。如果 Pass 1 期间新 snapshot 到达，只能留给下一次 query。

## 6. Pass 1：VQA 与 domain

Pass 1 输入为语言、head/wrist RGB 和 semantic 3D token bank。Qwen 原生语言/视觉序列保持不变，3D 只通过后段的四个 Semantic-3D Adapter 注入。

输出合同：

```text
domain: NAVIGATION | MANIPULATION
subtask_text: free-form text
domain_confidence: optional calibrated score
```

不输出四路 route，也不要求输出 `focus_entity`。例如：

```text
domain: NAVIGATION
subtask_text: Approach the Coke can while keeping the target box in view.
```

二分类切换可以在执行层使用确认或迟滞，但具体规则仍待闭环实验锁定。分类输出本身不得覆盖安全检查。

## 7. Pass 2：完整 Qwen 前向与动作生成

Pass 2 重新运行完整 Qwen，并使用：

- 同一语言；
- 同一 head/wrist RGB；
- 同一 semantic 3D tokens 和 mask；
- Pass 1 实际生成的 `domain + subtask_text` 前缀。

Pass 2 导出最后 16 层 hidden states。3D 信息已经由 Qwen Adapter 写入这些 states，因而首版不再执行 expert-specific token selector 或 3D residual bypass。

```text
if domain == NAVIGATION:
    trajectory = nav_expert(qwen_last_16_states)

if domain == MANIPULATION:
    trajectory = mani_expert(
        qwen_last_16_states,
        robot_state_13d,
    )
```

一次 query 只运行或只采纳一个动作专家。

## 8. NAV 执行路径

NAV expert 输出：

```text
shape: [10, 3]
step: 0.20 s
frame: frozen query_body_t
meaning: x, y, yaw
```

执行前：

1. 检查有限值、尺度和连续性；
2. 使用 query-time map 做初步 clearance 检查；
3. 交给 PCT/DWA；
4. PCT/DWA 使用独立线程维护的最新 raw-LiDAR occupancy/costmap 再检查；
5. 记录原轨迹、修改后轨迹、reject reason 和实际控制。

语义对象点云不能替代完整占据地图。规划器拥有最终否决权。

## 9. Mani 执行路径

Mani expert 输出：

```text
shape: [10, 7]
step: 0.04 s
meaning: six arm joints + continuous gripper
```

13D `q6 + dq6 + gripper1` 只在该分支使用。局部抓取仍以 head/wrist RGB 为主要依据；语义 3D 通过 Qwen states 提供物体尺度、位置和场景上下文，不直接旁路到 Mani expert。

执行前检查：

- joint limits、速度/加速度和有限值；
- self-collision 与环境碰撞；
- 轨迹首点相对当前关节目标的连续性；
- gripper 范围与控制语义；
- action 是绝对量或增量量的合同一致性。

## 10. 语义 3D 失效与降级

失效状态必须显式表示，不能把全零点云解释为可通行空间或有效对象。典型状态：

- `not_observed`：当前相机/雷达未覆盖；
- `empty_after_projection`：有 mask 但无通过投影检查的点；
- `low_confidence`：检测、mask 或投影置信度不足；
- `stale`：snapshot age 超阈值；
- `calibration_invalid`：标定或 TF 不匹配。

另有一个不是失效的状态：`memory_occluded`。它表示当前 LiDAR 没有对象点，但历史 3D track 仍可经 ego-motion 传播使用。它必须携带真实 age 和衰减后的 confidence，且近距离 Mani 主要依靠 head/wrist RGB。`memory_occluded` 超时后才转为 `stale/invalid`。

处理原则：

1. 对应对象的 3D token 在 Adapter 中 mask 掉；
2. 全部语义 3D 无效时退化为 RGB-language Qwen；
3. 不存在需要关闭的 expert 3D gate，因为首版没有专家旁路；
4. planner 仍可读取最新 raw LiDAR map；
5. domain-specific hold/fail-closed 的最终规则由安全实验锁定。

在没有验证之前，推荐 NAV 对关键地图或 pose 失效采用 base zero/hold；Mani 对关键感知失效保持最后安全 joint target，而不是继续开放环执行新动作。

## 11. 一次控制循环

```python
def policy_step(query_time: float):
    request = freeze_observation_request(query_time)
    validate_request(request)

    # Both passes consume the exact same immutable request/snapshot.
    domain, subtask_text = run_qwen_pass1(request)

    qwen_last_16 = run_qwen_pass2(
        request=request,
        model_prefix=(domain, subtask_text),
    )

    if domain == "NAVIGATION":
        proposal = nav_expert(qwen_last_16)
        command = planner_filter_with_latest_raw_map(proposal)
    else:
        proposal = mani_expert(
            qwen_last_16,
            request.robot_state_13d,
        )
        command = manipulation_safety_filter(proposal)

    log_query(request, domain, subtask_text, proposal, command)
    execute(command)
```

伪代码强调的是数据与安全合同，不是当前已存在的函数名。

## 12. 日志与可视化

每次 query 建议记录：

```text
request_id / snapshot_id / timestamps
head/wrist frame IDs
entity IDs and phrases
Grounding DINO boxes/scores
SAM2 mask IDs/scores
per-object raw/voxel/FPS point counts
point valid masks and FPS indices
centroid/size/confidence/age/object_valid
PointNeXt/tokenizer/config/checkpoint IDs
semantic 3D token mask
four Adapter gate values and attention summaries
Pass 1 domain/subtask text
Pass 2 prefix and hidden-state provenance
selected expert
proposed trajectory
safety/planner result and reject reason
executed command
```

调试可视化至少包含：

- head RGB 上的 entity box 与 SAM2 mask；
- 斜视 raw LiDAR point cloud；
- 按实体着色的语义点云；
- 每对象有效点数、质心、尺寸和 age；
- 4-token validity 与 Adapter gate 统计；
- NAV proposal 与 planner 修改后的轨迹。

不再显示“selected NAV semantic tokens”或“selected Mani semantic tokens”，因为 v0.2 没有 expert-specific selector。

## 13. 启动前检查

### 几何与标定

- head RGB 与 LiDAR 时间戳单调且时基一致；
- `T_head_lidar`、相机内参和 frame convention 已验证；
- 斜视 3D 可视化中可乐罐和箱子点与 RGB mask 对齐；
- `0.05–10 m` 裁剪生效；
- 短时积累无明显双影和 ghost。

### 模型

- PointNeXt-S 输入 mask 对 padding 完全无响应；
- 每对象严格输出 4 token；
- Adapter 层号与 checkpoint 一致；
- Adapter 全 mask 或 gate-off 时能退化到 RGB-language 路径；
- Pass 1/Pass 2 的 snapshot、FPS indices 和 tokens 完全相同；
- 只有 Mani 收到 13D 状态。

### 控制与安全

- NAV/Mani 输出形状、单位、坐标系和时间步正确；
- raw LiDAR planner path 与 learned semantic path 相互独立；
- planner/collision checker 可以拒绝模型轨迹；
- stale/invalid/NaN/timeout 的 hold 行为经过 dry run；
- 日志足以从 request 重放两次前向。

## 14. 推荐试运行顺序

1. 离线重放并人工查看 2D mask、raw cloud 和 semantic cloud；
2. 固定机器人，只运行 PointNeXt/tokenizer 和两次 Qwen，不发控制；
3. 开启 Adapter，比较 gate-on/gate-off 输出；
4. NAV-only 低速闭环，保留最新 raw map 的规划器硬约束；
5. Mani-only 静态抓放和碰撞检查；
6. 完整移动抓取；
7. 语义 3D dropout、stale 和错误实体映射故障注入。

## 15. 当前待定项

- semantic perception worker 的目标频率和最大 snapshot age；
- Grounding DINO/SAM2 阈值与多候选消歧；
- voxel size、遮挡检查和积累清空阈值；
- Qwen Adapter 的绝对层号与运行开销；
- domain 切换迟滞和确认规则；
- Mani action 的绝对/增量定义；
- 各类失效下最终经安全验证的 hold/fail-closed 行为。

总体架构见 [README.md](README.md)，训练方案见 [train.md](train.md)。
