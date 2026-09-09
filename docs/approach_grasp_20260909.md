# 接近并抓起：独立任务分支

分支：`feat/approach-grasp-task-20260909`，基于
`fix/mani-visual-conditioning-20260909@4e0fabf8c09fdf9fab8a130dfe0a363fa3369fad`。
2026-09-09 用户明确将此新分支目标简化；原完整搬运分支与数据保留。

## 已实现

任务只有 `NAV_TO_SOURCE → PICK`。使用现有真实双视角 RGB、关节状态、
causal 5 Hz 动作与实际导航执行路径，无 depth。默认指令为
`Approach the cola can, grasp it, and lift it stably.`。
只有 NAV 阶段请求任务切换；PICK 不再预测切到 NAV_TO_TARGET 或 PLACE，
即使服务错误返回终端 ADVANCE，也保留合法低层动作并记录被忽略的切换。
NAV 完成仍是模型声明，不能当作物理完成证据。

沿用用户要求的唯一正常终止媒介：60 秒仿真任务时间，加载和暂停推理不计入。
达到抓起标准时记录首次达标时间，不提前结束、不要求随后搬运或放置。
在余下时间继续 PICK；因此本任务统计“曾稳定抓起”，不承诺第 60 秒仍持物。
当前目标是先验证接近到抓取的衔接。不存在 GT 触发切换或强制闭合。

独立只读评分复用 relative-grasp-evidence-v2 的窗口：抬起 ≥ 4 cm、
TCP 与物体距离 ≤ 8 cm、连续 ≥ 1 s、相对位置漂移 ≤ 1 cm、
相对旋转漂移 ≤ 0.15 rad、采样间隔 ≤ 0.05 s。
新任务额外要求同一窗口内 joint7 的实际开度和生效命令均 ≤ 0.5 open fraction
（标定满量程 0.04 m），并要求原 physical pick 事件。
必须来源评分有效、60 秒正常闭合、无物体固定约束或中途重置。
继承已声明的 MANI 底盘/支持关节锁；不声称无任何物理辅助。
这是稳定抬起的几何代理，严格双指接触证据缺失时保持 unknown。

结果写入 summary 的 `task_contract=approach-grasp-v1`、`approach_grasp` 和
`success`。`full_task_success=null`，不得拿简化任务成功冒充完整搬运成功。
评分与首次成功时间只进入审计日志，不进入模型条件或执行决策。

## 入口与未运行项

沿用 `scripts/run_full_episode_diagnostic.py` 及既有独立模型服务。
本任务参数为：

```text
--mode approach_grasp --timer-only
--task-context configs/approach_grasp/task_context.json
--condition-label approach-grasp-v1-sim51-new-condition
```

其余 source-episode、query-tick、expected-sha256、reference-root、单 episode
运行参数仍绑定实际源实例、step1700 权重和现有 Sim5.1 配置。
必须使用新输出目录；不复用旧搜索 launcher 的四任务 context。
该入口仍校验 step1700，未来新训练 checkpoint 需要独立修改服务身份合同。

本次只实现、CPU 验证并发布新分支，不启动训练或占用 GPU。
没有此新任务的物理成功或新训练结果。旧权重是诊断起点；缩短任务提示、
删除后缀不等于模型已训练过该分布。后续训练应另建只含合法 NAV/PICK
动作与 NAV→PICK 标签的派生视图，沿用 family 划分和 train-only normalizer，
不能把旧 PICK→NAV_TO_TARGET 标签直接混入，也不能制造 FINISH 训练标签。

CPU 回归入口：

```text
PYTHONPATH=src:scripts python -m pytest -q -p no:cacheprovider \
  scripts/test_approach_grasp.py scripts/test_timer_only_diagnostic.py \
  scripts/test_full_episode_diagnostic.py tests/test_formal_physics.py
```

覆盖两任务上下文、PICK 无后续任务、实际物理步进函数的连续抓起窗口、
实测张开/未抬起/辅助与无效证据拒绝、旧四任务和 3000 tick 时钟回归。
这些是 CPU 测试，不计作 Isaac 物理实验。
