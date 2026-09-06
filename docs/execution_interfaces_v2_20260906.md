# 执行接口 v2：评价、命令来源与条件抓取诊断

本轮在 `fix/grasp-evaluation-execution-v2` 上继续 `e0292c0` 的 validation 工作。
没有重新训练，也没有修改发布数据、最终 checkpoint、0.10 m snap 门槛或 saturation≤0.5% 门槛。
原始结果保留在 Git 外的 `execution-interface-v2-20260906-r1/`；旧轨迹没有覆盖。

原冻结迁移测试也已结束：50 个任务 × 3 个 seed × 2 种条件，共 300 次尝试。
无抓取辅助/源辅助分别有 60/62 次产生模型请求，178 次没有模型请求。
两组旧合约整任务成功均为 0，按任务口径的原报告 95% 区间均为 [0%, 7.13%]；
这是包含启动失败、旧执行接口和旧判据的结果，不能当作纯模型抓取失败率。
原 `closed_test_full_v1/report.json` SHA256：
`cc7aaa47b3bd87e46ecafc58f81c9a119ab3bc84cd6a29045ffb23477d681767`。

## 1. 已确认的问题与解释边界

### 最大裁剪样本包含规划阶段的缓存目标

`liangzhunew500-000109:frame-000019`，PICK 路由、`plan_pick` 阶段，查询时间 3.8 s：

| arm_joint2 | 数值 |
|---|---:|
| 查询实测 q | 0.0000295283 rad |
| 第一未来标签的绝对目标 | 2.2235598564 rad |
| 3 rad/s × 0.2 s 限制后的目标 | 0.6000295283 rad |
| 最大改变量 | 1.6235303282 rad |

对应源帧的显式 `arm_joint_positions=None`、`gripper_command=None`，动作来源为
`pick_base_settle`。采样向量中的目标却与 reset 前缓存关节状态相同。
参考记录器 `_update_full_action` 在显式目标为空且已有缓存时继续使用旧目标；不能把该向量
直接解释为这个规划时刻新下发的控制命令。这是命令来源不一致的证据，不是提高限速的理由。

50 个 validation 示范、3484 个 MANI 查询的来源审计：

| 查询阶段 | 查询数 | 速率裁剪事件 |
|---|---:|---:|
| plan_pick | 310 | 1187 |
| exec_pick | 1675 | 575 |
| plan_place | 3 | 0 |
| exec_place | 1496 | 0 |

370 个查询的真实未来前缀包含没有显式机械臂命令的采样点；终端 padding 不计入这个计数。
缺少显式命令不自动证明每个缓存目标无效，需要按 reset、保持和执行阶段进一步区分。
脚本 `audit_sampled_command_provenance.py` 保留源帧动作、实测 q、原目标、限制后目标及轴/时域位置，
并核对被归因的源目标与发布标签一致，不回填标签。

### 限制逐轴核对

六轴部署限制均为 3 rad/s；源 URDF 的速度标注也是 3 rad/s。
参考 IsaacLab implicit actuator 配置为 10 rad/s，当前仿真的六轴
`root_physx_view.get_dof_max_velocities()` 实读也均为 10 rad/s。
位置范围与部署范围一致至浮点精度。夹爪映射仍采用既有合同。
URDF 标注、PhysX actuator 设置、部署目标变化率是三个不同层次，不能互相替代。

238 次位置、1762 次速率、0 次夹爪裁剪：冻结门槛采用
`sample_mean = 2000/(3484×10×7) = 0.820075%`，**不通过**。
示范等权均值另报 0.819193%，示范聚类 bootstrap 95% CI 为 [0.730077%, 0.905617%]。
该 CI 对应示范等权均值，不是门槛估计量的区间。两个统计入口现在共享 `saturation_gate()`；
回归测试覆盖两种均值恰好位于门槛两侧的情况。

## 2. 绝对目标与部署转换的配对物理回放

固定当前 Sim5.1 迁移环境、同刻标签、无抓取固定约束，保留操作底座/支撑锁定。
部署组实际执行 `query-relative 编码 → 发布 normalizer 往返 → 恢复绝对目标 → 位置/速率限制`。
每 10 点重新读取 live q 作为编码锚点；每点保持 10 个 50 Hz tick。
尾部为满足 decoder 形状而补齐的点不额外执行。

| 源示范 | exec_pick 直接回放最终抬升 | 部署转换后最终抬升 | v2 持续几何代理 |
|---|---:|---:|---|
| 000006 | 22.8072 cm | 22.8477 cm | 两组均出现 |
| 000024 | 21.5784 cm | 21.5797 cm | 两组均出现 |
| 000030 | 22.9816 cm | 22.9442 cm | 两组均出现 |
| 000109 | 22.6027 cm | 22.6084 cm | 两组均出现 |

这是 4 个示范的 8 条轨迹，000109 为极值定向选样，其余三个延续前轮选样；不是八个独立任务，
也不是随机抽样的抓取成功率估计。接触记录缺失，这些结果只支持持续几何抬升。
旧的世界速度保持指标继续保留；000006 两组仍未通过该旧指标。

**规划前缀对照改变了结论边界。** 对 000109 增加 `plan_pick` 缓存目标后，
直接/部署两组的峰值抬升分别只有 0.5222/0.4091 cm，最终 TCP 距物体约 30.76/29.99 cm，
均没有持续几何夹持。这个额外实验执行的是“缓存采样向量构造的命令”，不能称为源教师完整控制。
从 `exec_pick` 开始的成功，不能证明规划前缀或完整迁移执行链通过；裁剪也不是这个失败的必要条件。

## 3. 评价与辅助已拆开

`physical_events.py` 的评估器没有仿真句柄，只读取测量和记录事件。
`grasp_assistance.py` 单独拥有施加/释放约束的权限；不读取 `pick` 或几何代理的布尔值。
`FormalPhysics` 仅组合两者，旧事件与阈值仍保留，v2 是并列证据。

v2 在末端坐标系计算物体相对位置与旋转，要求持续 1 s、相邻采样间隔≤0.05 s：
闭合命令≤0.5、抬升≥0.04 m、TCP 距离≤0.08 m、相对位置漂移≤0.01 m、相对旋转漂移≤0.15 rad。
这只能构成 `geometry_hold_proxy`。接触验证还要求整窗双指接触、接触法向相对、无外部支撑；
缺少任一接触证据时为 unknown。世界速度≤0.30 m/s 独立记录为安全指标。

探针记录 PhysX **finger-body collider** 接触，不宣称是独立分区 fingerpad 传感器。
辅助仍冻结为旧 admission rule，没有悄悄改成 v2 代理；新辅助规则必须另行冻结并重跑辅助组。
改变评估器输出不能启用或禁止旧辅助的回归测试已加入。

合成测试覆盖刚性共同平移/旋转、空抓、桌面支撑、短暂撞起、托举、双指同向支撑、滑落、张开及采样间断。
托举可通过几何代理而不通过接触判据，作为明确反例保留。合成测试不是物理误报率校准。
`rescore_grasp_trace.py` 对前轮 12 条原始轨迹只读重评分，12 条出现持续几何代理，接触成功率未知；
没有给旧辅助组补记约束或改变旧得分。

真实物理正负对照 `contact-calibration-05` 已完成（同一 000006，两次独立启动）：
源命令回放的 340 个控制采样中，188 个连续采样有带载双指接触且法向相对；
单个接触点记录的最大法向力约 11.90 N。强制保持张开组 340 个采样均无双指带载接触，最终抬升 0。
这些是 GPU 接触张量的实测记录，尚不足以校准空抓、托举、滑落等完整物理负例集的误报率。

全物体接触覆盖目前**未通过**。GPU 回调没有事件；改用对象接触张量、显式碰撞体路径、
在源初态恢复前重建 PhysX actor 后能读到桌面接触，但仍漏掉指侧已经读到的双指接触。
桌面正样本不能证明对 articulation 的覆盖。新增双向一致性检查，只有刚体和 articulation 接触覆盖
都获得证据时，零外部接触才允许解释为“无外部支撑”。
`contact-calibration-08/none` 原摘要中的严格接触 True 因覆盖不完整而**作废**；
只读重评分 `contact-coverage-rescore.json` 为几何代理 True、严格接触 unknown。
原轨迹与原摘要保留，辅助未重跑或补记。该组的负例启动因临时磁盘写满失败，不进入判据成绩。

## 4. 连续 PCT 末段与退化路径

没有放宽 0.10 m 门槛，也没有覆盖原始粗端点 B。
`continuous_endpoint.py` 提供离线候选：保留 B、原 snap 与 A，只有完整末段的扫掠验证通过才追加真实 A。
验证器用机器人导航/转向完整包络的外接圆覆盖整条线段，对所有可能相交栅格逐一检查，
包括线段之间、中心线旁、原地旋转包络、地图外和未知地面；支持旋转地图坐标。
单纯的 bool 障碍图不能证明地面可通行，调用方必须提供全栅格几何/支撑证据和包络来源哈希。
不支持未经证明的垂直连接。

**当前只是几何候选，`deployment_approved=false`。** 现有发布地图还没有与完整机器人包络、
地面覆盖和控制可行性绑定的证书，不能拿单元测试或 .6 m 规划成本半径代替真实场景验收。
尚未接入正式执行、尚未改善实际 C，也没有产生修正导航后的 M3/M4 交接快照。

DWA 重复 XY 路径现在在执行器中区分：

- 位置和朝向都到达：正常 `local_goal_reached`。
- 位置到达、朝向未到：`validated_in_place_turn_required`，请求新决策，不记成功。
- 位置未到：`reconnect_from_measured_pose_required`，请求从实际 C 重新连接，不记成功。

后两种暂不自动旋转或移动，尚未证明恢复能力；不会再把空投影段传给 DWA。

## 5. 固定模型诊断与首帧混杂检查

仅用最终 `step_002414`，权重 SHA256：
`d86360e96d97f45467281ca77a006eba85c085c737e4156170efbf8a58a351b9`。
独立 localhost 诊断服务固定 PICK 和 canonical subtask，跳过自主 route；模型只接收 RGB、指令及实测 q/dq/gripper。
每个 query 固定 diffusion seed=17；比较每次执行 10 点与 2 点、12 s 仿真预算，无抓取固定约束。
此诊断不代表自主四阶段成绩，也不衡量实时推理能力。

最终独立进程矩阵 `model-pick-05` 完成 3 个源示范 × 2 个周期，共 6 条轨迹：

| 条件 | 持续几何代理 | 描述性 Wilson 95% CI | 严格接触成功率 |
|---|---:|---:|---|
| 每次 10 点 / 2 s | 2/3，66.67% | [20.77%, 93.85%] | unknown |
| 每次 2 点 / 0.4 s | 0/3，0% | [0%, 56.15%] | unknown |

区间以每个条件的三个不同源示范为分母，不把六条配对轨迹算成六个独立任务。
这是沿用前轮选样的 convenience pilot，区间不是总体能力认证，不能据此作显著性或普遍优劣结论。

| 示范 | 10 点最终抬升 / TCP 距离 | 2 点最终抬升 / TCP 距离 | 已执行闭合目标（10 点 / 2 点） |
|---|---|---|---|
| 000006 | 20.21 cm / 2.10 cm | −1.73 cm / 73.81 cm | 47/60 / 0/60 |
| 000024 | −2.40 cm / 16.52 cm | −2.41 cm / 20.36 cm | 22/60 / 0/60 |
| 000030 | 21.87 cm / 2.35 cm | −2.17 cm / 4.75 cm | 31/60 / 26/60 |

000006/000030 的长块有持续抬升；000024 两组的峰值抬升都不足 1 cm，不能解释成旧开度判据漏计。
000006 短块曾短暂抬升约 9 cm，但未保持，末态物体速度约 2.37 m/s，不能算稳定抓取。
短块 000006/000024 各有 29/30 次查询只在未执行尾部预测闭合，实际模型目标中没有执行闭合；
000030 短块有 15/30 次此现象，也执行过闭合，仍没有抬升。这是“截断重规划导致闭合被推迟”
的具体诊断证据，不足以说明它是所有失败的唯一原因。当前结果不支持优先投入更短周期或异步推理。

最终矩阵首次 query 的底座、全关节 q/dq、物体姿态/速度和 TCP 完全一致。
头部/腕部图像平均每通道差异为 0.050–0.135（像素范围 0–255），第一块预测最大元素差为
0.0148/0.0518/0.0592；仍保留渲染扰动，不宣称像素与预测逐位相同。
六条轨迹均未创建抓取固定约束，旧 pick 均为 False，严格接触覆盖均未知。
每次查询完整十点预测的 saturation 为 6.57%–18.33%，**全部不通过 0.5% 门槛**；
该分母包含短块未执行的预测尾部，不可当成实际执行点的裁剪率。

逐条摘要哈希、首帧一致性、截断闭合统计和门槛见机器可读
[实验摘要](experiments/execution_interfaces_v2_20260906.json)。

试运行 `model-pick-04` 的 000006：10 点组出现持续几何夹持，最终抬升 20.8318 cm；2 点组没有。
但首次 query 的物体/TCP/关节状态完全相同，腕部图像视角明显不同，第一块预测最大元素差 1.4366。
因此这不是合格的反馈周期因果对照，**不能据此说 2 点执行更差**。原始图像与失败记录保留。
因此最终入口限制每个条件一个新进程，在源状态恢复后显式同步 Fabric/RTX、不推进物理时间，
再建立共同的 0.4 s 相机历史；上述最终矩阵与这批混杂试运行分开统计。

启动失败（脚本调用接口、localhost 沙箱权限、临时磁盘写满）与物理失败分开记录，不进入模型分母。
原始命令、摘要、视频、模型服务身份和接触覆盖均保存在独立工件目录。

## 6. 复现入口与下一步验收

沿用前轮文档的 Isaac 资产/运行参数，新增以下诊断参数；不要把它们混入冻结正式测试：

```text
audit_sampled_command_provenance.py --prepared PREPARED --validation-records RELEASE/val.jsonl --output-dir NEW_DIR
audit_source_action_contract.py --validation-records RELEASE/val.jsonl --output-dir NEW_DIR
replay_sampled_joint_targets.py --paired-contracts --offset 0 --normalization RELEASE/normalization.json ...
replay_sampled_joint_targets.py --source-phase pick_with_planning ...
replay_sampled_joint_targets.py --record-contacts --negative-control open_gripper ...
replay_sampled_joint_targets.py --record-contacts --contact-backend finger-tensors ...
rescore_grasp_trace.py --trace-root OLD_REPLAY --output NEW_REPORT.json
serve_conditioned_pick.py --checkpoint FINAL_CHECKPOINT ... --port 18086
run_conditioned_pick.py --execute-points 10 --num-episodes 1 ...
run_conditioned_pick.py --execute-points 2 --num-episodes 1 ...
summarize_conditioned_pick.py --root MODEL_PILOT --expected-source-episodes 3 --output NEW_REPORT.json
```

真实接触正负校准完成前，严格接触成功率保持未知；只有传感覆盖的正负物理案例才能估计误报率。
连续末段仍需真实几何证书、控制可行性及源目标实测 C 的对照；M3/M4 必须来自该导航真正执行后的完整状态，
不能用 A/B/G 代替。源 Sim6 重执行、运输和放置仍是单独的迁移/整任务验收，本轮没有宣称通过。

回归验证：597 passed、1 skipped（外部 locomotion 权重未安装）；两项 localhost 测试在允许本机套接字的环境通过。
原维护工作树及 pinned arm-vla checkout 均干净；原维护源码 SHA256 仍为
`1fd7bfb2f104bebf3260b0461549e6ae37ce6e2716476b8e692c19c15ed5ad84`。
