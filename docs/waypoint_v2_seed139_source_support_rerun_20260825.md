# Waypoint v2 seed139 源箱支撑面修正与闭环复测

日期：2026-08-25 CST

## 1. 现象与根因

seed139 的近景视频中，可乐底部低于箱体可见顶面，看起来像嵌入箱体。USD 网格逐面审计证明
这不是 PhysX 穿透：可乐所在 XY 的真实向上碰撞面为 `z=0.1395388350 m`，而 task 中
`pick.support_geometry.support_surface_z=0.1442482562 m` 取自箱体/边缘的全局最高点。
两者相差 `4.709421 mm`。

修正前落稳后的可乐底部为 `z=0.1397730905 m`，仍比局部碰撞面高 `0.234255 mm`，因此
物理碰撞本身成立；错误是局部碰撞面与任务语义/可见顶面不一致。单纯抬高初始 pose 不能
修复，因为自由落稳后仍会回到局部低面。

## 2. 最小修正

`d2bff854941fb6081fe015abae2eb07f3139ee85` 为 rollout 增加显式、默认关闭的源物体支撑
代理：

- `--source-support-proxy-radius-m` 默认 `0`，只有非零时启用；
- `--source-support-proxy-height-m` 默认 `0.005 m`；
- 每个 episode 生成新的 USD wrapper，不覆盖原场景资产或 task；
- 在物体 XY 下方创建不可见静态圆柱，顶面严格等于 task 的语义支撑面；
- 半径必须不大于 task 声明的物体 footprint，避免把碰撞体扩展到夹爪侧面；
- 不改变 object initial pose，不进入模型输入，不参与 route 或动作选择；
- trace 和 summary 记录几何参数、wrapper SHA-256 和 `model_input=false`；
- 默认关闭时原基线行为不变。

本次 seed139 使用半径 `0.026 m`、高度 `0.005 m`。远端 Isaac 环境的定向测试 `13/13`
通过；其中覆盖默认关闭、pose 不变、代理顶面和 footprint 越界拒绝。

## 3. 无推理落稳门禁

完整闭环前先运行 `--seed-preflight-only`，不启动模型、不执行落稳后的动作：

| 指标 | 结果 |
|---|---:|
| requested center z | `0.1978932364 m` |
| settled center z | `0.1978932023 m` |
| requested-position error | `1.03e-7 m` |
| settled can bottom z | `0.1442482221 m` |
| semantic support top z | `0.1442482562 m` |
| model query / post-settle action | `0 / 0` |

修正后罐底与语义顶面仅差 `3.41e-8 m`，首帧与 10.74 s 近景目检均不再下沉。

## 4. 同轨迹闭环复测

为了只隔离支撑面和已批准 runtime 修正，复测固定 step1250、seed139、真实 PCT/DWA 与
真实 cuRobo/IK；query 0--99 精确回放历史模型响应，query 100 起使用当前四图调用实时
step1250 模型。模型 request 仍只有 instruction 与 head/wrist 双时刻四图，state field 为
0。

| 指标 | 结果 |
|---|---:|
| 总 query / control step | `170 / 2,534` |
| episode 时长 | `50.68 s` |
| route 切换 | q92，`10.74 s`，`NAV_TO_SOURCE -> PICK` |
| PICK 底盘零速 | `1,997 / 1,997` control step |
| live q100--169 cuRobo 成功规划 | `52` |
| live 合法 pose `plan_pose=None` 保持重询 | `17` |
| 最近 TCP--物体距离 | `0.193309 m`（q146） |
| live target0 最小夹爪值 | `0.398771`（q162） |
| PICK 期间物体 z 变化 | `0 m` |
| 最终失败 | q169，真实 SO(3) step `36.628° > 35°` |

q101 等合法 TCP 的 `plan_pose=None` 均只触发保持与下一次完整重推理，没有再次提前终止。
q162 首次在 live 部分跨过 close 阈值并实际发出 close 控制，但三视角显示夹爪没有包围可乐，
物体未抓住或抬升。q169 的平移 step 为合法的 `0.095160 m`，但真实旋转 step 超过未放宽的
原始 arm-vla `35°` 限制，因此 fail-closed 正确。

## 5. 结论与边界

源箱局部支撑高度缺陷已经修复，并有无推理落稳与同 seed 三视角闭环两层证据。修正没有
改变模型输入、route owner、NAV/PICK 控制链或 arm-vla 安全阈值。

本次仍不是完整任务通过：模型已经证明 NAV 到达、自主切 PICK、合法无规划后的持续重询、
继续靠近和一次闭合尝试；尚未证明稳定对准、抓取、抬升或 `PICK -> NAV_TO_TARGET`。当前
主要剩余问题仍是 MANI target 的抓取时序、空间对准与相邻姿态连续性，不应把支撑面修正
误报为模型能力提升。

完整视频、trace、summary 和前后对比图保存在 Git 忽略的
`artifacts/evaluation/waypoint_v2_step1250_seed139_support_proxy_20260825T180427CST/`，不得
加入公开 Git。
