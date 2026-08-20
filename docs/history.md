# 版本迁移与兼容策略

## 1. 单一 live tree

仓库不按 V0/V1/V2/V3 复制 runtime、scene、trainer 或 README。源码只维护一套现行
实现；历史由 commit、branch、不可变数据/checkpoint manifest 和诊断文档保存。

采集协议中的 `conveyor-bench-v1`、LeRobot `v3.0` 和场景/teacher 名称是持久数据
身份，不表示源码中存在并行版本目录。

## 2. Waypoint v1 迁移

公开开发分支为 `feature/conveyorvla-waypoint-v1`，批准起点为
`7d7617d8c4225ff6105497c2e3dcce252fb6cd92`。2026-08-20 正式训练 source 为
`724ead21be2c27d9b40c200375ee4ab49ccedc84`；2026-08-21 runtime/eval 基线推进到
`121512903667e16578525ec22dcfb2d0deca92e5`，没有改写 checkpoint 的 source 身份。

主要历史节点：

| commit | 迁移内容 |
|---|---|
| `fe2b4ea` | 新建无 state waypoint 数据合同 |
| `d03f386` | 受约束 Qwen route 与双 Layerwise FM head |
| `720f5ca` | 独立 Waypoint 训练入口 |
| `a852b5b` | runtime/v1 与 receding-horizon executor |
| `8fcccd9`–`c547e34` | 数据 clip、checkpoint binding、分布式 route/batch 修复 |
| `405bb34` | Waypoint 自主 route + oracle-prefix 开环门禁 |
| `747f9f6`–`f390036` | PCT/DWA、cuRobo adapter 与数值门禁 |
| `55e433a`–`c25e7cf` | 四卡 accumulation 和 eval launch 合同 |
| `9562c05` | 绑定的单卡 inference export/service |
| `3cc16ec` | query-base 到 cuRobo planner-base 变换 |
| `724ead2` | 模型自主管理 route 的 arm-vla Isaac rollout loop |
| `aa06479` | 后续 Waypoint checkpoint 默认每 500 effective step 保存 |
| `23afff4` | 分离 cuRobo code root 与 arm-vla runtime asset root |
| `13f6e87` | 正确处理 Qwen tied weights 的 inference export |
| `1215129` | 复用并 capability-gate 外部 Waypoint cuRobo 服务 |

这些 commit 记录实现演化；最终结果仍须读取 checkpoint、数据和运行 manifest，不能只靠
branch 名或最新 commit 猜测。

## 3. 不兼容边界

Waypoint v1 是有意的 breaking contract：

| surface | 旧 direct-action | Waypoint v1 | 兼容性 |
|---|---|---|---|
| 模型输入 | 双帧视觉 + `state28` | 双帧视觉，无 state/phase/history | 不兼容 |
| route | canonical 自由文本/dispatcher | 受约束 ACTION/DONE + 单 route token | 不兼容 |
| Pass 2 | 旧预测文本合同 | 模型自己的完整 prefix，完整 Qwen forward | 不兼容 |
| NAV | `20×[vx,wz]` | `20×[dx,dy,dyaw]` body waypoint | 不兼容 |
| ARM | TCP delta + gripper | query-base absolute TCP target + gripper | 不兼容 |
| 执行 | direct composer | PCT/DWA 或 cuRobo/IK，首目标后 requery | 不兼容 |
| dataset | temporal/dense view + state/action | `conveyorvla-waypoint-dense-transition-v1` | 不兼容 |
| runtime | legacy serve/evaluate | `conveyorvla-waypoint-runtime/v1` | 不兼容 |

因此禁止：

- 用旧 state28 checkpoint 初始化或 optimizer-resume 新模型；
- 用旧 action scale/normalizer 反归一化 waypoint；
- 把 `[dx,dy,dyaw]` 当 `[vx,vy,wz]`；
- 把 absolute TCP target 当 TCP delta；
- 把旧 `scripts/train_hierarchical.py`、`serve.py` 或 `evaluate.py` 指向新 checkpoint；
- 通过补零 state 或 silent key mapping 伪造兼容。

读取器和 checkpoint gate 必须以 schema/model/protocol ID、shape、token ID、frame、
stride 和 SHA-256 显式拒绝错误组合。

## 4. 保留的采集兼容面

下列身份仍用于读取已有证据，不因模型迁移而重写：

- canonical raw：`conveyor-bench-v1`；
- LeRobot：`v3.0`；
- 历史 teacher/scene/profile ID；
- `temporal_v2/grasp_only`、`temporal_v3` 和 dense-transition view 的只读历史目录。

旧 raw 若有完整 provenance，可以通过新 builder 生成独立 Waypoint 派生集；旧派生文件
本身不得就地迁移。缺少状态、相机、标定或 source hash 的数据应明确拒绝。

## 5. Git 与制品策略

- branch 承载开发，不作为实验档案；
- 正式结果至少绑定 source commit、dirty state、resolved config、dataset/normalizer
  hash、环境、GPU、checkpoint 和评测协议；
- 普通 push 必须是非 force；共享历史不为美观重写；
- 数据、checkpoint、日志、视频、cache、sidecar 和 `handoff_private/` 永不加入 Git；
- 训练正在运行时，远端 worktree 必须固定在 source commit，文档或后续代码 push 不得
  在运行中 fast-forward 它；当前正式训练已暂停，但 checkpoint 的 source 身份不变；
- 有可复现价值的训练状态后续应由 annotated tag/release + manifest 保存，而不是永久
  保留实验 branch。

## 6. 查找旧实现

```bash
git log --all -- .
git show COMMIT:PATH
git worktree add /tmp/conveyorvla-old COMMIT
```

不要把旧目录复制回 live tree。若需要恢复旧能力，应从历史提交提取最小组件，接入现行
边界并增加对应合同测试。
