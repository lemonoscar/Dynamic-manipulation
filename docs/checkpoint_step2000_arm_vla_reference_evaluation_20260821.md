# step 002000 原始 arm-vla 规则闭环复测

复核时间：2026-08-21 10:12 CST。被测模型为
`step_002000@a8d57a22c515`，runtime 为
`feature/conveyorvla-waypoint-v1@ace7d6e9f2026b55be2f9cc55cf4a355b4dde339`。

## 1. 结论

移除本地额外导航门控、改用固定参考
`arm-vla-grasp-sim@388b6818f4c605a707d13c519fbb58b1d07acd92` 的原始导航规则后，
step 2000 不再因为 20 点 horizon 尾部、PCT snap、重复 DWA 速度门禁或本地 3 s/1 cm
stall 规则提前退出。模型完成 22 次 `NAV_TO_SOURCE` query，并自主切换到 `PICK`。

这仍不是闭环成功：22 个首 waypoint 全部落在 arm-vla 的 `0.18 m` 到达容差内，机器人
没有实际走向可乐；模型随后在距可乐约 `1.56 m` 时切到 `PICK`。首个 absolute TCP
target 的平移变化合法，但 yaw 变化为 `47.74 deg`，超过原始 arm-vla 与批准合同共同规定的
`35 deg` 单轴变化限值，因此在 cuRobo 规划前被拒绝。

本轮证明的是“额外门控已不再主导退出，并观测到模型真实能力边界”，不是模型通过
NAV、ARM 或完整 pick-place 门禁。

## 2. 规则边界

新增显式 `arm-vla-reference` profile，只用于与批准 reference 做行为对照：

- 只检查模型第一个 raw NAV waypoint，不审计其余 19 个尾部点；
- 保留 arm-vla 首点 `0.80 m / 45 deg` 上限和 `1e-3` 退化阈值；
- 使用 reference executor 的 `0.18 m` 位置容差、`pi` yaw 容差和 250 control-step chunk；
- 使用 reference `NavigationStallDetector`；stall/chunk timeout 零速并重新 query，不结束 episode；
- PCT 仍要求结构和有限值正确，但不再叠加本地 `0.10 m` snap 拒绝；
- DWA 命令仍来自 arm-vla DWA 自身限幅，不再叠加一份本地重复速度拒绝；
- 保留协议有限值、arm-vla workspace/rate、cuRobo/IK reachable/collision、关节和执行限值。

默认 `contract` profile 没有被改写。`arm-vla-reference` 也没有取消 arm-vla 自己的 ARM
规则：reference 的 `RemoteVLAArmTargetShadowValidator` 与批准合同第 8.2 节都规定相邻
absolute TCP target 平移不超过 `0.15 m`、单轴旋转不超过 `35 deg`。

## 3. step 2000 闭环对照

四次运行均使用 Liangzhu、seed 861、完整任务、head/wrist 双时刻四图、真实 PCT/DWA、
真实 locomotion policy 和外部 Waypoint cuRobo 服务。模型 request 的 state field 数为 0，
没有 GT phase、外部 FSM 或 route gate。

| profile | query / control step | 结果 |
|---|---:|---|
| `contract` | 1 / 58 | 第 19 段平移超过 0.8 m，完整 horizon fail-closed |
| `executable-prefix-diagnostic` | 26 / 473 | 旧本地 `navigation_stall` 结束 episode |
| `unbounded-translation-diagnostic` | 9 / 301 | 取消 NAV 平移上限后仍被同一本地 stall 结束 |
| `arm-vla-reference` | 23 / 140 | 22×NAV 后自主 PICK；首 ARM target 超过原始 35° rate gate |

`arm-vla-reference` trace 中：

- `arm_vla_reference_rules=true`；
- `full_horizon_contract_passed=null`，说明尾部没有被拿来否决首点；
- 22 次导航结束均为 `first_waypoint_reached`、`failed=false`；
- `navigation_failed=0`，没有 stall/timeout 被升级成 episode failure；
- route 计数为 `NAV_TO_SOURCE=22`、`PICK=1`。

## 4. 初始化与 NAV 能力

对象先完成 59 个物理 settle step，位移 `0.00453 m`，随后才发生第一条模型 query；query
anchor 为 control step 58。没有“机器人/物体未初始化就推理”的顺序错误。

22 个 NAV 首点平移统计：

| 指标 | 值 |
|---|---:|
| min / mean / max | `0.0060 / 0.0819 / 0.1356 m` |
| 落在 0.18 m 到达容差内 | `22 / 22` |
| episode 内底盘 XY 位移 | `0.0440 m` |
| 初始 / 最终底盘到可乐距离 | `1.5458 / 1.5558 m` |

因此本轮无导航 stall；问题是模型输出的首点都被原始执行器视为已经到达，且 route 在没有
接近物体时提前切换为 PICK。overview 视频也确认机器人没有走到源箱附近。

## 5. ARM target 诊断

模型自主 PICK 的第一个 target 为：

```text
[0.41004, 0.00701, 0.22645, 0.57318, 0.00047, -0.83337, 0.87372]
```

相对于当时 query-base 下的 TCP：

- 平移变化 `0.04126 m`，低于原始 `0.15 m`；
- roll/pitch/yaw 变化分别为 `33.44 / 0.91 / 47.74 deg`；
- yaw 超过原始 `35 deg`，失败原因为
  `arm_plan_failed:ValueError:arm target 0 exceeds rotation step limit`。

失败发生在原始 rate gate，cuRobo 没有收到这个 target。若将来需要检验“完全忽略 ARM
rate gate 后 cuRobo 如何处理”，必须另建明确标记的 unsafe diagnostic；不能把它称为
arm-vla 原始规则或 production 结果。

## 6. 视频与证据

| stream | 分辨率 | 帧数 | 时长 | SHA-256 |
|---|---:|---:|---:|---|
| overview | 1280×720 | 69 | 2.76 s | `c4e26b6fab4e1ffbf50f46b342b7a7b517e2790f0c8131965338138a4e1555da` |
| front | 640×480 | 70 | 2.80 s | `fafb89e4e4826d65c8cd5d69f13b773630f5b1292cf663bc0ddad25c97a79dae` |
| wrist | 640×480 | 70 | 2.80 s | `22c4b1e9d145b0b612c8a097db4db6f7a1843b081509361933fa92039bb144c6` |

视频、summary、完整 trace、启动脚本和日志已由远端下载到 Git 忽略目录
`artifacts/evaluation/waypoint_step002000_arm_vla_reference_20260821T015504Z/`。远端/本地
SHA-256 一致，三路视频通过 `ffprobe` 和末帧目检。评测结束后仅停止本任务创建的 tmux；
端口 18081/8766 已释放，四张 H20 均回到 0 MiB。

## 7. 当前判断

step 2000 的 route 生成已有从 NAV 到 PICK 的自主切换能力，但动作/时序还不可用：NAV
首点尺度小于 reference 到达容差，且 PICK 发生在未接近物体时；首个 ARM orientation 又
违反 reference rate gate。下一 checkpoint 若要重测，应同时关注 NAV 首点尺度、真实目标
距离下降、route 切换时机和 ARM RPY step，而不是继续叠加新的 runtime 门控。
