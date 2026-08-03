# Dynamic Manipulation handoff — 2026-08-04

## 仓库状态

- 本地仓库：`/home/lemon/research/Issac/Dynamic`
- 分支：`main`
- 远端：`https://github.com/lemonoscar/Dynamic-manipulation.git`
- 本轮基线提交：`fddcb1affb445b6557243074885a6a99c35eecf7`
- 本文档对应的正式提交和远端状态以 `git log -1`、`git status -sb` 为准；全程
  直接推送 `main`，没有 PR。

## 已完成的主体工作

新增了正式的 V1 `stationary_sort` 诊断：传送带命令与逐帧实测速度均为
`0.0 m/s`，完整执行抓取、携带、投放和 `0.5 s` 稳定判据，但不计入动态
benchmark 分数。固定合同为 `single_target`、一个 `part_red_block`、目标
`sort_bin_blue`，只允许以下场景：

| split | seeds |
| --- | --- |
| train | 1101, 1102, 1103 |
| val | 2101 |
| test | 3101 |

运行时已同步物体 spawn X/Y、静态 intercept、staging、oracle 和任务描述；配置
位于 `conveyor_bench/configs/v1.json`。协议、strict validator、M0 exporter、M0
dataset 和训练 CLI 均已加入静态任务支持。训练接口目标用法：

```bash
python conveyor_bench/scripts/train_m0_mobile.py \
  --episode-root TRAIN_EPISODE_ROOT \
  --state-statistics TRAIN_STATE_STATISTICS \
  --initial-action-checkpoint ACCEPTED_M1_ACTION_MODEL \
  --model-root LOCAL_MODEL_ROOT \
  --output-dir NEW_EXPERIMENT_ROOT \
  --belt-speed 0 \
  --task-type stationary_sort
```

另新增默认关闭的 `--mobile-approach-assist`，只用冻结 service command 完成
`mobile_approach`，物体生成前抑制 M0 请求，用于隔离判断静态抓取 primitive。
它是诊断功能，不能把 assisted 回合计作 policy-only 成功或训练数据。

## 已生成并验收的静态 oracle 数据

这些输出位于被 `.gitignore` 排除的 `conveyor_bench/outputs/`，canonical 数据不要
手工修改：

- train：`outputs/gate/v1_stationary_train_oracle_final_1101_1103`
  - 3/3 success
  - 2906 control steps，4356 张三相机 PNG
  - 1428 条 M0-Mobile train records
- val：`outputs/gate/v1_stationary_val_oracle_2101`
  - 1/1 success，957 steps，1434 PNG，470 records
- test：`outputs/gate/v1_stationary_test_oracle_3101`
  - 1/1 success，966 steps，1449 PNG，475 records

五条 episode 均已通过 strict validator、quality audit 和 temporal camera gate；
实测最大带速为零，无跌倒和禁区碰撞。详细 episode ID、导出 SHA 和在线证据在
`conveyor_bench/docs/m0_stationary_followup_20260803.json`。

## M0 结论

接受的基线仍是 M1：

- action SHA：`2f6c10a55f857cab198daa1886be0d8b2df5fbb7f93d3e8c648df3f3bd795024`
- state statistics SHA：`29bc9a04a9c0eb03947e21e3fb752959c3d6841097df52cbb001acb5558d0e66`
- training report SHA：`ac80d39d764412a4782ac94988e4f73c332869d83e6389682f507ac36eef8a02`
- 1183 records，1200 steps

静态 policy-only seed 1101 在 `mobile_approach` 超时，没有进入机械臂 full action。
只辅助 approach 的干净隔离回合证明 M1 确实能完成闭夹、112 个连续双侧持有
step、约 `2.24 s` 持有和约 `0.18264 m` 抬升；但它在 `7.38 s` 主动开爪，未完成
carry/place。canonical failure reason 是 `runtime_error`；对应 `summary.json` 的
`metrics.abort_metadata` 已持久化 pregrasp `IKConvergenceError` 及误差，因此它是
可复核的诊断原因，但不替代顶层标准 failure reason。

M2 `m2-carry-retract-step800-20260803-2100` 已拒绝：离线 grasp transition 从
M1 的 `1/3` 退化为 `0/3`，在线仍在 approach 超时。其 action SHA
`b1dbc623020cd432f5d247043fa75103384ecd94e9f0bed64901c0d96824b936` 不得用于
服务、采集或后续初始化。

## 远端状态

- SSH：`4xH20`
- 工作根：`/diff/wallx_workspace/dzb`
- 仓库：`/diff/wallx_workspace/dzb/dynamic-m0-mobile-smoke-20260803-bundle`
- 环境：`/diff/wallx_workspace/dzb/abot-m0-repro-h20-20260731/envs/abot_m0_v2`
- 模型：`/diff/wallx_workspace/dzb/dynamic-m0-mobile-models-20260803`
- 运行：`/diff/wallx_workspace/dzb/dynamic-m0-mobile-runs-20260803`
- GPU0/1：StarVLA 占位进程，各约 90 GB，保持不动。
- GPU2：tmux `m1-restored-service-gpu2`，M1 服务端口 `18765`，健康。
- GPU3：结束时空闲。
- 本地 SSH 端口转发已退出；不影响远端服务。检查服务：

```bash
ssh 4xH20 curl -s http://127.0.0.1:18765/health
ssh 4xH20 nvidia-smi
ssh 4xH20 tmux list-sessions
```

## 已收口的 fail-closed 门禁

- runtime、strict validator 与 exporter 共用 seed registry；真实
  `object_spawned` 坐标必须与 registry 推导的场景位置一致，不能只伪造 manifest。
- 伪造 split、scenario ID、episode/layout seed、object/root offset、root yaw 或
  使用未注册 seed 均会被拒绝。
- 标准 M0-Mobile exporter 拒绝已声明或逐控制步记录的 approach、staging、teacher
  diagnostic assist，并在合法记录中显式写入 `source_assisted=false`。
- dataset 对 `source_assisted`、`source_task_type`、belt speed、object curriculum
  split 和 train statistics 全部采用缺字段即拒绝；`--all-belt-speeds` 必须显式给出
  task type。
- 当前 exporter 已重导动态 release、静态 train/val/test。训练用 SHA 为：
  - static 1101：`c83b96db8fc74cae13214987465f6eb8ef1229cb1bea13f1989ade02fb4420cc`
  - static 1102：`9bc76aeb2cace9e85b29085c226344b3f2e398d2d4ebf5ba9d9551223a8bc58f`
  - static 1103：`3018bd1d7bc34fe320ccd45d3d5454ebc0f523ad3b8576629cfb625d92360414`
  - dynamic release：`0d3f9d54f5a12d2ccaa05bfad2c0e227041193f20502d74a2e8cb1476d6193d6`

完整 `conveyor_bench/tests` 为 393/393 green；关键定向测试另行复核 107/107，五条
静态 canonical episode 均再次通过 strict validator。`git diff --check` 通过。
全套测试中的 localhost server/client 用例需要允许只在本机打开临时 HTTP 端口。

## 训练已通过；采集仍需通过硬件门禁

最终小混合为 2492 records：静态 1428（57.3%），动态 release 532 加一次完整等权
replay 532（42.7%）。state statistics 的 `count=2492`、`split=train`。GPU3 单卡
100-step 真实 pipeline smoke 已从 accepted M1 成功完成，最终 loss 与 gradient norm
有限，模型、统计和 report 哈希均已复核；GPU0/1 StarVLA 与 GPU2 M1 service 未受
影响。首次 tmux wrapper 把退出文件误写为字面量 `0n`，因此另跑 2-step exit-proof，
其退出文件字节为 `30 0a`。完整机器证据见
`conveyor_bench/docs/m3_training_pipeline_smoke_20260804.json`。

远端新的隔离 Isaac 环境位于
`/diff/wallx_workspace/dzb/dynamic-isaaclab-5.1-20260804`。轻量 Isaac Sim 5.1
compatibility 包已安装，但首次 Kit 启动要求用户明确接受 NVIDIA Omniverse EULA；
同时远端 H20/570 驱动组合不在官方测试覆盖内，必须先通过 compatibility checker、
三相机非黑帧、静态 1101 和动态 seed 0 门禁，才能启动批量采集。

首批动态生产矩阵为 4 个 train 目标 × 2 个分拣盘 × 2 种语言 × 每格 8 回合，共
128 回合。先采每格 1 条的 16-cell pilot，全部通过 strict、quality、camera gate 后
再启动剩余 112 条；失败回合保留作 benchmark 证据，但训练集只接收未辅助成功回合。
生产入口是 `conveyor_bench/scripts/collect_v1_train_matrix.py`；它冻结 seed 矩阵、
采用单一写锁，从 canonical 数据恢复，拒绝重复/orphan/inprogress，并原子维护
`matrix_report.json` 与训练候选清单。bulk 启动前会重新跑完 16 条 pilot 的四层门禁。
