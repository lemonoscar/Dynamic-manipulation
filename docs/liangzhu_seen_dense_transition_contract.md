# Liangzhu seen dense-transition 合同

本轮不续训旧 `seen-subtask-view-2`，也不继承 `step_007000` 的 optimizer/scheduler。
旧 base LeRobot v3 保持只读；新的 expanded base 采用
`conveyor-vla-al0-lerobot-v3-dense-transition-manifest-3`，sidecar schema 为
`conveyor-vla-al0-liangzhu-seen-dense-transition-view-5`，split seed 继续使用
`conveyor-vla-al0-liangzhu-seen-split-v2`。

expanded base 显式保留 navigation planning、`plan_pick`、`plan_nav_to_place`、
`plan_place`，以及三个专家切换处的 reachable/success verifier 物理观测，并映射到
下一可执行子任务，使三个专家切换处的 query 仍保持 0.20 秒连续。

每个 row 保留 `source_episode_id`、`base_index`、phase/next phase、距边界时间、边界原因、
20 位 `action_valid_mask` 和导航 reference mode。所有 expanded-base query row 都保留；导航远端帧
降权，终点前 4 秒、尤其前 2 秒，以及切换前后 1 秒加权。跨专家的 action suffix 置为
false，VLM subtask loss 仍覆盖整个边界窗口。

视觉合同唯一为 head/wrist `[-5,0]` model tick、0.20 秒、5 Hz query。历史 feature 名
`tminus2` 只是兼容名称，不表示 0.08 秒。

主 prompt 不包含真实 `subtask_history`。训练只允许单个 previous label 作为早期 teacher
forcing，并同时执行 dropout/corruption；teacher forcing 在配置的 step 区间内线性降到
0。动作 pass 在非 teacher-forced 样本上先贪心生成当前 subtask，解析成功且路由正确时，
把该模型文本与原始观测送入第二次完整 Qwen forward。Qwen3-VL 与两个 DiT 均全量更新。

Navigation DiT 只预测 `[vx,wz]`。在线 composer 输出显式 joint-space reference：

- `NAV_TO_SOURCE`: `stow_open`，夹爪 open；
- `NAV_TO_TARGET`: `carry_closed`，夹爪 closed。

关节目标按每控制步限速；导航路径不再构造零 TCP delta。Manipulation DiT 的 7 维输出
补成 10 维时底盘三维严格为零。

训练前依次运行：

```bash
python scripts/audit_dense_transition_view.py --hierarchy-root DATASET --output AUDIT.json
python scripts/probe_dense_loader.py --hierarchy-root DATASET
python scripts/extract_dense_transition_videos.py --hierarchy-root DATASET --output-root CLIPS
```

正式运行必须记录 commit、manifest SHA-256、resolved config、环境、GPU UUID、tmux、日志、
checkpoint 和连续训练事件；`scripts/audit_training_events.py` 检查有限 loss/LR/梯度以及
VLM、Navigation DiT、Manipulation DiT 三条非零梯度路径。
