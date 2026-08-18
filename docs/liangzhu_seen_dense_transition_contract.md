# Liangzhu seen dense-transition 合同

本轮只使用 ModelScope `liangzhu_0815_n200` 与 `liangzhu_0815_n400`。旧 raw、base、
sidecar 和 checkpoint 均保持只读，但不再进入训练引用；不继承旧 optimizer/scheduler。
新 expanded base schema 是
`conveyor-vla-al0-liangzhu-0815-lerobot-v3-dense-transition-manifest-5`，sidecar schema 是
`conveyor-vla-al0-liangzhu-0815-seen-dense-transition-view-7`。split 仍以完整
`source_episode_id` 为单位并复用 `conveyor-vla-al0-liangzhu-seen-split-v2`。

expanded base 显式保留 navigation planning、`plan_pick`、`plan_nav_to_place`、
`plan_place`，以及三个专家切换处的 reachable/success verifier 物理观测。planning row
映射到下一可执行子任务；verifier row 保持在其正在验证的前一专家，随后一帧再切换，
使三个专家切换处的 query 保持 0.20 秒连续且动作域与真实物理命令一致。

每个 row 保留 `source_episode_id`、`base_index`、phase/next phase、距边界时间、边界原因、
20 位 `action_valid_mask` 和导航 reference mode。所有 expanded-base query row 都保留；导航远端帧
降权，终点前 4 秒、尤其前 2 秒，以及切换前后 1 秒加权。跨专家的 action suffix 置为
false，VLM subtask loss 仍覆盖整个边界窗口。

视觉合同唯一为 head/wrist `[-5,0]` model tick、0.20 秒、5 Hz query。历史 feature 名
`tminus2` 只是兼容名称，不表示 0.08 秒。

训练与推理使用同一个无 semantic-memory prompt：只有完整任务、head/wrist 双帧视觉和
“现在应做什么”的格式要求；annotation previous phase 和上一 query 预测都不进入 Pass 1
或 Pass 2。视觉历史仍保留。动作路由 teacher forcing 是独立机制，只决定训练时使用真值
route 还是模型当前预测 route，并在配置区间内线性降到 0。

实际 Qwen chat 没有额外 system message。Pass 1 的唯一 user message 按以下顺序构造：

```text
Head camera, oldest to newest: + head[t-0.20s, t]
Wrist camera, oldest to newest: + wrist[t-0.20s, t]
Task: Walk to the box holding the Coke can. Keep the base still and pick up
the can. Lift it and retract the arm. Turn around and walk to the other empty
box. Keep the base still and place the can on top of it.
The head and wrist videos are ordered from oldest to newest.
What should the robot do now? Output exactly one canonical subtask as
<|pred_action|><|subtask|><subtask><|end_subtask|>
```

允许的 answer 只有四条完整 canonical 文本：

```text
<|pred_action|><|subtask|>Walk to the box holding the Coke can.<|end_subtask|>
<|pred_action|><|subtask|>Pick up the Coke can, lift it, and retract the arm.<|end_subtask|>
<|pred_action|><|subtask|>Turn around and walk to the empty box.<|end_subtask|>
<|pred_action|><|subtask|>Lower the Coke can onto the empty box and release it.<|end_subtask|>
```

Pass 2 重新提交同一个 user message，并追加 Pass 1 的 assistant answer；不追加真实 phase、
completed list 或上一步专家标签。28 维 proprioception 不被文本化进 prompt，而是直接进入
被路由的 DiT state encoder。这样训练和在线的语言输入完全相同，teacher forcing 只影响
“动作 loss 是否采用真值 route”，不改变 prompt 内容。

Pass 1 的 batch 输出在 token 级逐行裁到首个 `<|end_subtask|>`（含结束 token）后再解码，
从而丢弃 batch 对齐 padding；没有结束 token 或非四种 canonical 文本仍由严格 parser
fail closed。Pass 2 使用模型当前预测文本与原始观测做完整 Qwen forward。Qwen3-VL 与
两个 DiT 均全量更新。

Navigation DiT 只预测 `[vx,wz]`。在线 composer 输出显式 joint-space reference：

- `NAV_TO_SOURCE`: `stow_open`，夹爪 open；
- `NAV_TO_TARGET`: `carry_closed`，夹爪 closed。

关节目标按每控制步限速；导航路径不再构造零 TCP delta。Manipulation DiT 的 7 维输出
补成 10 维时底盘三维严格为零。新数据 manifest 按专家统计 train split 有效动作的绝对
P95/P99/P99.5/P99.9，并以 `1.05 × P99.9` 生成建议物理 scale；夹爪单独检查 `[0,1]`。
正式配置必须记录采用值与相对旧 scale 的裁剪率。
本轮完整 train split 冻结的 action10 scale 为
`[0.473, 1.0, 0.525, 0.3, 0.3, 0.265, 0.5, 1.424, 0.562, 1.0]`；只向上扩展旧尺度，
不因样本分布变小而收窄已有动作合同。

完整 view 首先作为只读统计探针生成；冻结 train-derived scale 与导航中位姿后，使用
`scripts/finalize_dense_transition_view.py` 复制同一 annotations 到新的最终目录，并在
manifest 中绑定统计探针 SHA、最终 temporal config SHA、动作 scale 和 composer reference。
该步骤不重新选样，也不修改或覆盖统计探针。

manifest 另记录导航观测中的夹爪开度中位数，但它不是 composer 命令目标：携物时接触到
可乐会使实测指间开度非零。门禁检查的是显式 `stow_open/carry_closed` 命令 trace，不要求
接触状态下的实测开度等于无物体命令参考。

训练前依次运行：

```bash
python scripts/audit_dense_transition_view.py --hierarchy-root DATASET --output AUDIT.json
python scripts/probe_dense_loader.py --hierarchy-root DATASET
python scripts/extract_dense_transition_videos.py --hierarchy-root DATASET --output-root CLIPS
```

视频产物既包含 12 个 split×phase 边界片段，也包含 train/val/test 各一个四阶段完整
episode 和三者拼接版。源数据只有 front/head 与 wrist；不得把它们标成外部第三视角。

正式运行必须记录 commit、manifest SHA-256、resolved config、环境、GPU UUID、tmux、日志、
checkpoint 和连续训练事件；`scripts/audit_training_events.py` 检查有限 loss/LR/梯度以及
VLM、Navigation DiT、Manipulation DiT 三条非零梯度路径。routing 计数按所有 rank 求和，
并同时记录观测样本、实际送入 DiT 的样本及其比例，不能再把 rank 0 局部计数称为 global。

balanced empty-history 路由门禁要求严格 parser invalid rate 为 0、预测不全部坍缩为
`NAV_TO_SOURCE`，并且四个真值阶段都至少有一个正确预测。全 INVALID、自由文本、缺少
结束 token，或完全学不到任一 canonical phase 都是失败，不能通过 dispatcher fallback
伪装成有效专家样本。
