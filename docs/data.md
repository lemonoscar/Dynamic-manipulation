# 数据格式与质量门禁

## 1. 两层数据

采集先生成审计友好的 canonical raw，再转换为训练友好的 LeRobot v3：

```text
Isaac episode
  → canonical JSON/JSONL + 三路 PNG
  → strict validation / audit / camera gate / export
  → temporal JSONL
  → LeRobot v3 parquet + H.264 MP4
```

raw 是证据源，LeRobot 是派生训练集。转换器不得改写 raw canonical 文件。

## 2. Canonical episode

一个已发布 episode 至少包含：

```text
episode/
├── manifest.json
├── summary.json
├── steps.jsonl
├── objects.jsonl
├── action_chunks.jsonl
├── events.jsonl
├── camera_frames.jsonl
├── frames/
│   ├── head_rgb/*.png
│   ├── wrist_rgb/*.png
│   └── overview_rgb/*.png
└── exports/
```

关键含义：

- `manifest.json`：任务、seed、资产、相机、场景和 teacher provenance；
- `summary.json`：成功、失败原因、阶段和计数；
- `steps.jsonl`：50 Hz 机器人状态、canonical action 和事件状态；
- `objects.jsonl`：目标/干扰物真值，仅用于教师和审计；
- `camera_frames.jsonl`：25 Hz 捕获 tick、路径、尺寸、角色和质量统计；
- `action_chunks.jsonl`：模型动作块身份和时间；
- `events.jsonl`：抓取、释放、失败、阶段切换等离散事件。

目录先以 `.inprogress` 写入，完整关闭并 `fsync` 后原子发布。中断目录不是有效 episode。

当前采集清单将 `benchmark_config.evaluation.require_settled_placement` 写为 `false`：
目标必须先被夹持并释放，释放后中心首次进入指定框即记为完成。线/角速度仍逐帧记录供
质量分析，但不参与任务成功门槛。旧清单缺少该字段时按 `true` 解释，保持历史严格
静止驻留语义。

## 3. Temporal export

ConveyorVLA AL0 的 temporal 记录使用：

- head `[t-5, t]`；
- wrist `[t-5, t]`；
- 当前 `state28`；
- 未来 `20 × 10` 动作；
- 25 Hz action rate，覆盖 `0.8 s`；
- 5 Hz query，query stride 为 5 个 model tick；两帧跨度固定为 0.20 秒。

当前 profile 为 `conveyorvla_al0_temporal_v3`，任务范围固定为
`navigate_grasp_deliver`。每个来源 episode 必须按顺序包含接近传送带、动态跟随抓取、
负向直退、转向目标框、负载导航、驻车投放和入框确认；三段实际平面位移分别不得小于
`0.20 m`、`0.30 m` 和 `0.10 m`。`carry/preplace/place_descend/open` 阶段底盘动作必须
严格为零，底盘实际平面漂移不得超过 `0.05 m`。旧的 `grasp_only` 或不含直退证据的
记录不能进入该训练集。放置阶段还必须记录低层 `root_pose_hold` 为 active；它对应真实
机器人站立控制器，上层 VLA 的底盘标签仍为零。

四张输入图来自两台物理相机的两个时刻。overview 不进入 temporal export。

未来 TCP 行是相对 observation 时刻的独立目标，不是必须从第 0 行依次积分的增量。
因此在线推理延迟导致前缀过期时，剩余目标仍有明确定义。

## 4. LeRobot v3

转换环境固定：

- Python 3.10；
- `lerobot==0.4.4`；
- H.264；
- PyAV 解码；
- `use_videos=true`。

视频特征：

| key | 来源 |
| --- | --- |
| `observation.images.head_tminus2` | head，历史帧 |
| `observation.images.head` | head，当前帧 |
| `observation.images.wrist_tminus2` | wrist，历史帧 |
| `observation.images.wrist` | wrist，当前帧 |

另有 `observation.state`（28）、扁平化 `action`（20×10）和语言 `task`。

## 5. 转换

转换输入只能是已经通过全部门禁的成功根列表：

```bash
python scripts/convert_dataset.py \
  --episode-list outputs/collection/successful_episode_roots.txt \
  --output-root outputs/lerobot
```

小规模 smoke 使用 `--max-episodes 1`。输出目录必须是新的独立目录，转换器不会把
数据上传到 Hugging Face，也不会依赖网络。

PCT Liangzhu raw 使用 `scripts/convert_pct_dataset.py`。它保留 5 Hz 双相机原始时序，
并从稀疏物理结点重建 25 Hz、20×10 的独立未来目标。四个 LeRobot feature 的
`tminus2` 仅是现有 loader 的兼容字段名；PCT 派生集的权威 manifest 必须写明历史为
`[-5, 0]`、跨度 0.20 秒，以及状态插值和命令零阶保持方法。该静态 box1→box2 任务
属于 PCT 适配 pilot，不能伪装成传送带动态抓取配额。

Liangzhu seen 层不修改旧 base，而是生成包含 navigation/planning 与专家切换
verifier transition observation 的 expanded base，并使用新 sidecar schema
`conveyor-vla-al0-liangzhu-seen-dense-transition-view-5`。它保留全部 phase boundary row，
用 20 位 `action_valid_mask` 屏蔽跨 Navigation/Manipulation 专家的未来后缀，并提高两个
导航终点前 2～4 秒和切换前后 1 秒的采样权重。sidecar 不包含
`subtask_history`，split 继续按完整 `source_episode_id` 复用既有 seed。

## 6. 完整性检查

每次转换至少验证：

1. episode ID、帧计数和 task 索引一致；
2. state 为 28 维、action 为 200 维扁平向量；
3. 四个视频 feature 均存在；
4. 每个视频至少首帧可由 PyAV 解码；
5. 解码图像尺寸、dtype 和通道顺序正确；
6. dataset frame count 与 manifest 一致；
7. state 统计量有限，标准差下限有效；
8. raw canonical 哈希在转换前后不变。

建议除四路首帧外，再对第一/中间/最后 episode 抽取中间帧，检查时间对应和视觉内容。

## 7. 任务与分层规则

当前不增加独立的物品类别分类头；策略学习的是完整导航、抓取、配送和按任务指定目标
框投放。数据按以下维度分层统计：

- `target_asset_id`；
- `belt_speed_mps`；
- `robot_mode`；
- `seed`；
- `task_success`；
- `training_eligible`；
- 接近、负向直退和负载导航的实际位移与 phase 顺序；
- 场景、相机和 teacher profile 哈希。

专家成功、任务失败和结构损坏三类必须分开：

| 类型 | 保留 | 进入专家训练 |
| --- | --- | --- |
| 成功且全部门禁通过 | 是 | 是 |
| 任务失败但数据完整 | 是，诊断区 | 否 |
| runtime/结构/哈希/相机失败 | 隔离 | 否 |

## 8. 数据兼容

`conveyor-bench-v1` 是现有 raw 的稳定协议名，因此源代码迁移不会重写它。读取器必须
按 schema 显式校验。旧 teacher、旧场景或 assisted 数据即使字段可读，也可能因当前
训练合同不兼容而被 exporter 拒绝。

`temporal_v2/grasp_only` 是历史派生格式，不就地改写。若其 canonical raw 具备完整
联合轨迹证据，可从 raw 重新导出 temporal v3；缺少负载导航证据的旧成功数据必须保留
为消融或诊断数据。

若未来必须改变 canonical 字段：

1. 写清变化和兼容边界；
2. 增加新 schema version；
3. 提供只读 migration 或明确拒绝；
4. 禁止在原目录就地改写历史 raw；
5. 重新执行 LeRobot round-trip。
