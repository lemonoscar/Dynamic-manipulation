# 数据格式与质量门禁

本文先描述现行 Waypoint v1 派生数据，再说明仍保留的 ConveyorBench canonical/LeRobot
历史数据。批准语义以 [Waypoint Policy v1 第 9 节](conveyorvla_waypoint_policy_contract_v1.md)
为准。

## 1. 冻结数据身份

Waypoint v1 只从只读的 Liangzhu 0815 `n200 + n400` 成功轨迹生成新派生目录，不修改
raw，也不复用旧 dense-transition sidecar、state28 normalizer 或 checkpoint：

| 项目 | 冻结值 |
|---|---|
| schema | `conveyorvla-waypoint-dense-transition-v1` |
| split unit | 完整 `source_episode_id` |
| split seed | `conveyor-vla-al0-liangzhu-seen-split-v2` |
| camera calibration | `liangzhu-0815-go2-x5-head-wrist-v1` |
| episode | 522 |
| row | 119,700 |
| train / val / test | 108,603 / 5,771 / 5,326 |
| manifest SHA-256 | `0db6169d726b2165a90ec6e833403666179eb68135248af5681de92a400ec957` |
| normalizer SHA-256 | `75a60ba125a83383f1d00ef4151933a77c796faee5d5c559364310cb64acfca0` |

这些哈希标识 2026-08-20 正式长训使用的数据快照。可变别名或目录名不能替代 manifest
hash。

## 2. 派生目录

```text
waypoint-dataset/
├── manifest.json
├── normalization.json
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

派生集保存监督、provenance 和源图像引用，不复制或重编码原始图像。构建先写同级唯一
staging 目录，完成后原子发布；目标目录已存在时直接拒绝。

manifest 绑定：

- 每条 source episode 的 `samples.jsonl`、`task.json`、`summary.json` 和源 manifest
  SHA-256；
- split 文件 SHA-256、route/boundary 计数和 episode 计数；
- visual history、动作 shape/stride/frame、相机标定和 prompt SHA-256；
- normalizer 相对路径与 SHA-256；
- `robot_state_field_count=0`、`robot_state_tensor_count=0` 和 forbidden model keys。

## 3. 每行 schema

每个训练 row 至少包含：

```text
schema_version
source_dataset_id
source_episode_id
source_row_id
split
timestamp
global_instruction
head_images[2]
wrist_images[2]
history_timestamps_s[2]
route
route_token
subtask_text
assistant_solution
action_domain
nav_waypoints_body[20,3] | null
arm_targets_base[20,7] | null
action_valid_mask[20]
waypoint_time_offsets_s[20]
label_frame_id
calibration_id
boundary/provenance metadata
```

监督记录可以保存 `previous_route`、`next_route` 和 source pose 字段的来源，用于审计
和边界标签；loader 输出的模型 batch 只有：

```text
video, lang, solution, route, route_token, action_domain,
action, action_valid_mask, sample_id, split
```

`state`、`state28`、`observation.state`、joint/TCP/base pose、phase、operation、
history、object state 和 target truth 均为 forbidden key。标签来源可以读取物理 state，
但 manifest 明确记录 `source_data_used_as_model_input=false`。

## 4. 视觉、动作与 mask

视觉历史固定为 head/wrist 各两帧，顺序 oldest→newest，时间为
`[t-0.20s, t]`。模型 query 为 5 Hz；兼容字段名不允许改变这一定义。

Navigation 标签为 `[20,3]`：

- 每行 `[dx_body, dy_body, dyaw]`；
- stride 为 0.60 s，总覆盖 12.0 s；
- 所有行相对 query 时刻同一个 `B_t`，不是相邻 waypoint delta；
- body→world round-trip 后与 source base pose 对齐。

Manipulation 标签为 `[20,7]`：

- `[x,y,z,roll,pitch,yaw,gripper_open_fraction]`；
- stride 为 0.20 s，总覆盖 4.0 s；
- 是 query base 下 absolute TCP target，不是 TCP delta；
- pose round-trip 后与 source TCP pose 对齐。

跨 route、缺失未来 source row 和 episode 尾部只把剩余
`action_valid_mask=false`，不删除当前视觉/route row。mask 必须是真前缀。DONE 只有
route 监督，不含动作域。

## 5. Split、平衡与归一化

split 只按完整 source episode 计算，禁止 frame-level 泄漏。正式 audit 的 route 计数为：

| split | NAV_TO_SOURCE | PICK | NAV_TO_TARGET | PLACE | DONE |
|---|---:|---:|---:|---:|---:|
| train | 14,785 | 17,496 | 48,565 | 18,460 | 9,297 |
| val | 779 | 924 | 2,613 | 977 | 478 |
| test | 781 | 848 | 2,360 | 877 | 460 |

训练 sampler 同时平衡 route 和 boundary window。normalizer 只由 train split 的有效动作
构建，按 horizon/维度保存连续量分位数；ARM gripper 单独映射到 `[-1,1]`。正式数据的
train 最大单侧饱和率为 0.9982%，双侧合计为 1.9964%。val/test 只应用冻结统计，不参与
拟合。

## 6. 数据门禁与现有证据

`scripts/audit_waypoint_dataset.py` 检查：

1. schema、split、route、boundary、文件与 source hash；
2. head/wrist 两帧存在且时间严格递增；
3. action shape、domain、mask、stride、frame 和有限值；
4. NAV/ARM 几何 round-trip；
5. normalizer hash、train-only provenance 和各 split clip rate；
6. 模型字段与 tensor 的 state 泄漏为零。

正式数据 audit 为 `ok=true`；NAV round-trip 最大误差
`1.33e-15 m/rad`，ARM 最大误差 `4.22e-8 m/rad`。

`scripts/extract_waypoint_videos.py` 可生成 train/val/test × 五 route 的 head+wrist
并排片段，并叠加首个有效 GT waypoint/TCP target。0815 source 没有外部第三视角，因此
这些数据 review 片段不能标为 three-view 证据。step 1000 的真实 Isaac 测试后来另行
生成了 overview/front/wrist 三路视频；两类证据不得混用，且该自主 episode 本身未成功。

## 7. 构建原则

- raw source 永远只读；输出必须是新目录；
- 先 `--audit-only` 确认 eligible episode，再 materialize；
- 数据、视频和 audit JSON 不提交 Git；
- 正式 run 必须绑定 manifest 和 normalizer 的 SHA-256；
- schema、horizon、stride、frame、route token 或 normalizer 不一致时 loader/checkpoint
  必须显式拒绝，不能自动猜测或转换。

具体命令见 [operations.md](operations.md)。

## 8. Legacy canonical 与 LeRobot v3

ConveyorBench 采集链仍采用：

```text
Isaac episode
  → canonical JSON/JSONL + head/wrist/overview PNG
  → validator / quality audit / camera gate
  → temporal export
  → LeRobot v3 parquet + H.264 MP4
```

`conveyor-bench-v1` raw 仍是审计证据源；`temporal_v3`、旧 dense-transition view
和 LeRobot `observation.state(28) + action(20×10)` 只服务旧 direct-action 实验。
它们不与 Waypoint v1 schema 兼容。旧数据若要用于新策略，必须从可追溯 raw 重新生成
全新的 waypoint 派生集，不能就地删除 state 字段或替换 normalizer。

canonical raw 继续保留 manifest、summary、steps、objects、action chunks、events、
camera frame index 与三路 PNG；失败但结构完整的 episode 可留作诊断，只有成功且通过
全部门禁的 episode 才能进入任何专家派生集。
