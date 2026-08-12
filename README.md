# Dynamic Manipulation

Dynamic Manipulation 面向 Go2-X5 移动操作机器人，研究在横向传送带上完成导航、
动态跟踪抓取和放置的视觉语言动作策略。当前策略称为 **ConveyorVLA AL0**，
仿真与数据平台位于 [`conveyor_bench/`](conveyor_bench/)。

仓库只维护一套现行实现。过去的 V0/V1/V2/V3 是迭代历史，不再以并列源码存在：

- 旧实现由 Git commit/tag 保存；
- 已有 episode 继续使用不可变的 `conveyor-bench-v1` 数据协议；
- LeRobot 的 `v3.0` 表示数据格式，不表示第三套 benchmark；
- 当前运行时统一使用 Liangzhu NuRec 背景、Isaac 动态前景和 PCT 对齐的 Go2-X5。

## 当前状态

- 固定底盘、单可乐罐、`0.01 m/s` 动态抓取—投放教师正例已通过；
- 相机、场景、sidecar 资产和 raw → LeRobot v3 转换链路已经接入；
- 完整移动专家已接入两段导航状态与联合数据门禁，真实 Isaac 联合烟测仍需单独记录；
- 当前正式代码只启用 `cola`，四零件 × 两速度 × 48 条的 384 条矩阵仍是后续目标。

## 快速入口

```bash
cd conveyor_bench
conda activate env_isaaclab
python -m pip install -e .
python scripts/check_environment.py
```

完整说明：

- [项目与命令](conveyor_bench/README.md)
- [任务和场景规范](conveyor_bench/docs/benchmark.md)
- [代码与模型架构](conveyor_bench/docs/architecture.md)
- [数据格式](conveyor_bench/docs/data.md)
- [采集、训练和测评操作](conveyor_bench/docs/operations.md)
- [状态与已知问题](conveyor_bench/docs/status.md)
- [版本迁移说明](conveyor_bench/docs/history.md)

大体积 NuRec 与物品资产通过 SSH sidecar 交付，不进入 Git，也不允许运行时联网下载。
