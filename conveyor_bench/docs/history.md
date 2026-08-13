# 版本迁移与兼容策略

## 为什么不再保留 V0/V1/V2/V3 目录

过去每次实验迭代都新增一套 runtime、scene、collector、validator、config 和文档，
造成同一功能多处修复、运行入口不明确和配置双重真相。当前仓库改为单一现行实现：

| 历史层 | 当前归属 |
| --- | --- |
| V1 episode 协议、记录、校验 | `src/conveyor_bench/schema/` |
| V2 顺序目标协调器 | `src/conveyor_bench/task_coordinator.py` |
| V3 NuRec 与物品 sidecar | `src/conveyor_bench/sidecar/` |
| V1 采集主循环 | `isaac/runtime_core.py` |
| V3 当前场景入口 | `isaac/runtime.py`、`isaac/scene.py` |

V0 和已经被覆盖的 V2 场景/CLI 不在 live tree 中维护，可从 Git 历史恢复。

## 兼容边界

文件路径可以整理，已发布数据身份不能随意重命名：

- canonical raw 继续是 `conveyor-bench-v1`；
- 历史 teacher profile `overhead_target_follow_pick_place_v3` 继续可读；当前要求初始化即
  连续运输且收臂后再移动的数据使用 `overhead_target_follow_pick_place_v4`；
- scene ID 中已有的 `_v3` 继续保留；
- 当前联合训练使用 `temporal_v3`；历史 `temporal_v2/grasp_only` 派生文件保留但不再
  被联合训练入口接受；
- LeRobot 继续使用数据格式 `v3.0`。

这些名称用于判断已有数据是否可读，不表示源码中存在并行版本。

## 后续迭代

普通行为修正直接修改当前文件并提交 Git。只有以下情况升级 schema：

- 字段增加/删除或含义变化；
- 时钟、坐标系或动作语义变化；
- 相机角色或模型输入变化；
- 成功判据变化会影响训练资格。

升级时提供迁移说明和测试，不创建 `runtime_v4.py`、`scene_v4.py` 或另一套 README。

## 查找旧实现

需要复现实验时使用 Git：

```bash
git log --all -- conveyor_bench
git show COMMIT:conveyor_bench/PATH
git worktree add /tmp/conveyorbench-old COMMIT
```

不要把旧目录复制回现行分支。若旧能力确实需要恢复，应提取最小可复用部分并接入当前
入口，同时添加当前合同的验证。
