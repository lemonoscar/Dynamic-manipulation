# Joint-Trajectory 手动停止前保存：待实现设计

状态：待实现。当前训练入口尚未启用本设计。

## 目标

允许操作者请求训练在下一个合法 optimizer-step 边界完成以下动作：

1. 停止获取新的 micro-batch；
2. 由全部分布式 rank 同步保存 ZeRO-3 checkpoint；
3. 校验 checkpoint manifest 与分片完整性；
4. 将运行状态写为 `stopped_after_checkpoint`；
5. barrier 后由所有 rank 正常退出。

“合法边界”指当前梯度累积周期已经完成、optimizer 和 scheduler 已共同推进的边界。不得在任意 micro-batch 中间保存。

## 推荐控制面

首选共享输出目录中的原子请求文件，而不是直接依赖 Ctrl+C：

```text
<output-dir>/control/stop-and-save.request.json
```

建议另提供命令行工具：

```bash
python scripts/request_joint_trajectory_stop.py --output-dir <output-dir>
```

工具应验证 `run_state.json` 的状态为 `running`，再通过临时文件加 `os.replace` 原子发布请求。重复请求必须幂等，不得覆盖已存在的未处理请求。

## 训练进程协议

每个有效 optimizer step 完成并记录 metrics 后：

1. 主 rank 检查请求文件；
2. 通过 distributed broadcast/reduce 将请求同步到所有 rank；
3. 所有 rank 共同调用现有 `_save_checkpoint(...)`；
4. 主 rank验证 checkpoint manifest、期望分片和 step 一致；
5. 主 rank 将请求文件原子改名为 `stop-and-save.handled-step-XXXXXX.json`；
6. 写入 `stop_and_save` event 和最终 `run_state.json`；
7. 所有 rank barrier，随后以成功状态退出。

若当前 step 已因周期保存产生同一步 checkpoint，不得重复保存，只需完成验证和退出协议。

## 信号处理边界

不应直接把 SIGTERM 处理为保存请求。`torchrun`/elastic launcher 会使用 SIGTERM 清理失败 worker；屏蔽它可能导致剩余 rank 在 collective 中永久等待。

Ctrl+C/SIGINT 可以作为尽力而为的辅助入口，但只有在确认 launcher 的信号传播行为并完成多 rank 故障测试后才能启用。共享请求文件是保证路径。

## 失败语义

- 保存成功：`stopped_after_checkpoint`，退出码 0。
- 请求已收到但保存失败：`stop_checkpoint_failed`，保留请求与错误 event，退出码非零。
- rank 丢失或 collective 失败：不得声称 checkpoint 完整。
- 只有通过 manifest 和分片校验的目录才能被后续评估或恢复使用。

## 验证清单

- 2-rank ZeRO-3、8 次梯度累积时，在累积中间发出请求；确认只在下一 optimizer step 保存。
- 请求恰好与周期 checkpoint 同 step；确认不重复写。
- 连续发出两次请求；确认幂等。
- 保存期间模拟一个 rank 失败；确认状态不是成功。
- 检查 checkpoint 可由全新进程加载，model/optimizer/scheduler/RNG/global step 全部一致。
- 验证 formal 与 disposable overfit 两种 run kind。

## 当前实验说明

2026-09-01 启动的 `conveyorvla-abot-m0-overfit32-20260901-v5` 在进程启动时没有加载本设计，因此仍按原计划在 step 250 和最终 step 300 保存。修改工作区文件不会热更新已运行的 Python 进程。
