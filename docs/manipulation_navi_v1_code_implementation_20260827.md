# Manipulation_Navi_v1 代码实施报告

- 日期：2026-08-27 CST
- 分支：`Manipulation_Navi_v1`
- 模型合同候选：`conveyorvla-joint-trajectory-policy-v1`
- 数据 schema：`conveyorvla-joint-trajectory-v1`
- 数据 profile：`conveyorvla-liangzhu-fresh-joint-trajectory-v1`
- 结论：离线代码、Isaac/controller 接线和启动级门禁已完成；没有 fresh data，也没有启动
  任何新训练

## 1. 本轮完成了什么

本轮没有继续修补 Waypoint v1/v2，而是为 breaking successor 建立独立文件和身份。旧数据、
normalizer、config、checkpoint、模型与 runtime 路径均保持不变。

| 层 | 已实现内容 | 主要文件 |
|---|---|---|
| 合同 | 四 route、无 DONE、10 点 NAV/Mani、13D Mani state、M=1/10-step inference | `joint_trajectory.py` |
| raw/derived data | 50 Hz applied-command provenance、5 Hz query、两域派生、边界 terminal-hold | `joint_trajectory_data.py` |
| normalizer/materializer | train-only 分域 normalizer、round-trip、不可覆盖发布、manifest hash | `joint_trajectory_data.py`、`materialize_joint_trajectory.py` |
| audit | schema、state leakage、shape、mask、progress provenance 与 normalizer 审计 | `audit_joint_trajectory.py` |
| model | 共享 Qwen/router，两个独立 16-block action expert，Mani-only state token | `joint_trajectory_model.py` |
| warm-start | step1250 选择性 key map，逐 key loaded/reinitialized/rejected 报告 | `joint_trajectory_model.py` |
| loss | answer/route 解耦、hard/soft route CE、boundary rank、physical progress、分组 Mani FM | `joint_trajectory_model.py` |
| sampler | global64：28 NAV interior、28 Mani interior、8 boundary，episode 去重 | `joint_trajectory_training.py` |
| schedule | 0.25 epoch Stage A + 1.75 epoch Stage B、独立参数组 LR | `joint_trajectory_training.py` |
| trainer | 12-episode disposable overfit、immutable identity、warm-start≠resume、save250、两 data-equivalent epochs | `train_joint_trajectory.py` |
| runtime | Pass 1 双确认、pending hold、Pass 2、NAV reference、direct-joint Mani | `joint_trajectory_runtime.py` |
| evaluator | released + target interior 连续 1.0 s，truth 不回流控制 | `joint_trajectory_runtime.py` |
| system adapter | 10 点 NAV→PCT/DWA、10×2 direct-joint、连续夹爪、零 base | `joint_trajectory_system.py` |
| Isaac truth | placement region、released + 1.0 s dwell，独立于 request/control | `joint_trajectory_system.py` |
| raw recorder | 50 Hz applied report、5 Hz query、三视角资产、atomic episode publish | `joint_trajectory_recording.py` |

配置冻结在 `configs/manipulation_navi_v1.json`。三个脚本都只服务新合同，不复用旧训练 output
或 checkpoint identity。

## 2. 关键语义已经怎样落到代码

### 2.1 数据不会把未来实测关节误当教师动作

Mani 标签必须来自 controller saturation 之后实际下发的 applied joint/gripper command。raw
记录缺少 requested/applied command、q/dq/gripper measured、base command/pose/twist、route 或
provenance 时直接拒绝；不存在回退到未来 measured q 的路径。

NAV 每 0.20 s 采样，10 个 target 全部相对同一个 query-body frame。Mani 每 0.04 s 采样，
10 个 `delta_q` 全部相对同一个 query `q_measured` anchor。遇到真实 route boundary、episode
tail 或 evaluator-success tail 时保留真实前缀并 terminal-hold 到第 10 点；训练 mask 始终为
10 个有效 target，不再存在推理无法复现的 suffix mask。

### 2.2 route 分类与文本 CE 已分离

`L_answer` 不训练 route token；transition window 中也不让模糊的 route-specific subtask 文本
反向制造第二个 route 监督源。`L_route` 是唯一 route 目标：interior 用硬 CE，boundary window
只在 old/new 两个 route 间使用连续软目标。boundary ranking 只配同 episode、同 transition
event 的 before/after；progress 无物理标签就 mask，禁止用 elapsed time 或 row index 补值。

### 2.3 runtime 不再把第一次 route 概率抖动当切换

Pass 1 每次都只读任务和视觉。初始 route 及后续切换都需要连续两次新观测确认；切换的两次
还必须满足同一个 new route 且 `P(new)>P(committed)`。pending 时不运行 Pass 2，base 为零并
保持当前关节/夹爪。确认后才用本次模型自产 assistant prefix 运行第二次完整 Qwen forward。

NAV 时变换并审计完整 10 点 reference；批准 PCT API 只接受 endpoint，因此第 10 点是 local
goal，前 9 点保留到 trace 而不虚构为 PCT via-point。Mani 时 base
严格为零，按 query anchor 重建并顺序执行 10 个 joint/gripper target；policy 层不调用 IK、
cuRobo、`plan_pose`、K* 或 feasibility selector，只保留底层 position/rate saturation 与 trace。

Isaac adapter 把每个 0.04 s Mani target 执行成两个 50 Hz tick，夹爪以连续 joint position
metadata 下发。raw recorder 只有在 arm/gripper apply count 对本 tick 真实增加时才接受
controller-applied target，并要求 PICK/PLACE requested/applied base 同时精确为零。

## 3. 已执行的验证

新合同测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider \
  tests/test_joint_trajectory_contract_data.py \
  tests/test_joint_trajectory_model.py \
  tests/test_joint_trajectory_runtime_training.py -q
```

加入 system/recorder 后结果：`26 passed`。

旧基线定向回归：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider \
  tests/test_waypoint_model.py \
  tests/test_waypoint_v2_model.py \
  tests/test_waypoint_runtime.py \
  tests/test_train_waypoint.py -q
```

当前环境结果：`48 passed, 2 skipped`；联合集合为 `74 passed, 2 skipped`。两个 skip 是
缺少 `accelerate` 的旧训练模块和无 CUDA 的旧 device-alignment test。此外，新 Python 文件
通过 `py_compile`，公开 diff 通过
`git diff --check`。

数据无关启动脚本已对批准 reference 的 commit/clean 状态、真实 RobotAction/Isaac runtime
import、连续夹爪、NAV→PCT/DWA fixture 和 placement region 解析返回
`startup_wiring_ready`。4×H20 的真实 headless stage/reset smoke 也以 exit 0 完成，但 H20
Vulkan/RTX device-creation warning 仍存在，所以只证明 stage/episode 生命周期，不外推相机、
GPU PhysX 或真实 control loop。完整证据见
[系统接线与启动级验证](manipulation_navi_v1_system_wiring_20260827.md)。

全仓 pytest 在当前 ROS Python 环境的 collection 阶段受到 `/opt/ros/humble/.../scripts`
包遮蔽仓库 `scripts` namespace 的既存环境冲突，未进入八个旧测试模块的断言；本轮没有为
规避该环境问题而改变旧包结构。

## 4. 现在明确没有什么

- 没有 fresh episode、derived rows、正式 split、normalizer 或 dataset manifest；
- 没有拿 12 episode 跑 disposable overfit；
- 没有 materialize 或加载新 checkpoint；
- 没有启动、停止、续接或监控任何新 GPU 训练；
- 没有固定 noise 开环动作图、gripper 时序评测或 route crossover 实测；
- 没有用 fresh episode 实测 recorder/FK 可视化或 materializer；
- 没有在健康 Vulkan/RTX 节点跑真实 hold→NAV→Mani control smoke；
- 没有真实闭环视频或任务 success 证据。

因此当前可准确称为“代码基本功能实现完成”，不能称为“模型完成”或“训练完成”。

## 5. fresh data 到达后的最短执行顺序

1. 用 audit 脚本拒绝字段、provenance、时钟、route、gripper 时序或 physical progress 不合格
   的 episode。
2. 在 Git 外新路径 materialize immutable release，并复核 split、normalizer/hash 和统计图。
3. 取独立 12 episode 做 overfit；验证 route、完整 10 点 joint/gripper、terminal-hold 与
   gripper open→close / close→open。
4. 丢弃 overfit optimizer/权重，从同一 step1250 selective warm-start 启动正式 run。
5. 正式 global batch 保持 64，训练约两个数据等效 epoch，每 250 optimizer step 保存；
   启动后至少观察 10 个健康 optimizer step。
6. 再做固定 noise 开环、route boundary、NAV→PICK→close→lift milestone、多 seed 真闭环和
   released-in-target 1.0 s evaluator success。

只有第 1–6 项用真实数据和运行证据通过，才冻结正式 resolved config/manifest/checkpoint
identity 并晋升合同。
