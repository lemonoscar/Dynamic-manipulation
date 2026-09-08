# 训练种子 16100000：第 1700 步拟合与放宽限制的完整任务尝试

本轮使用未修改的 `best.pt` 第 1700 优化步，不是第 1700 epoch。没有训练、没有 depth。完整已录 episode 的模型视图拟合已实际跑完；自由物理实验与该离线拟合分开报告。

## 冻结身份与资源

- 权重 SHA256：`474d03ff22da939f763574bf434c75226a2a2533ae984ced80e25e34a1df7fa0`。
- 原模型源码：`e03cb0bffd1a3bdeabd39ad8e94e1bbfd9a3e532`；原 normalizer 和 causal_command_5hz profile 保持不变。
- 新评测分支：`feat/train-seed-fit-diagnostic-20260908`，初版完整任务诊断 `92dfddc5d12b345daa5e8a88964a79153841825d`，真实物体读数与测量容差修正 `691a14444d22d18ef2c3c742ebbfab2062732b23`。
- 物理仍是 pinned `388b6818f4c605a707d13c519fbb58b1d07acd92` 的 IsaacSim 5.1 新条件，不是 Sim6 solver 重放。
- H20 `/diff/wallx_workspace/dzb/integration_runs/train_seed_fit_20260908_v1`；推理 GPU2，仿真 GPU3，推理仅 loopback 端口 18170。
- 上限：2 次物理尝试，每次 1200 秒；总测试墙钟 2700 秒，自 22:04:29 开始，截止 2026-09-08 22:49:29 CST，早于原 23:12:46 截止。所有失败保留，不额外补齐。

## 训练实例选择与实际拟合

预先按排序选第一个满足四路线和图片解码要求的 train family：`liangzhu_seed_16100000`，episode UUID `97550c83-0fe7-46b6-9ac9-be34c00a0a70`。无排除候选，也没有看评测结果后换种子。该 episode 含 96 个动作 query 和 6 个交接 query，1.6–40.4 秒，392 张唯一 RGB 全部核验。

所有合法 query 实际通过完整 Qwen＋动作专家采样，扩散 seed17，参数 FP32、前向 BF16 autocast，墙钟 118.719 秒、退出 0。严格重载与发布/normalizer/图片哈希检查通过。这里的“完整”指遍历该 episode 的所有合格模型视图，不代表对未被派生的原生 50Hz tick 逐个推理。

按冻结调度、原训练源码和优化完成日志重建，该 family 的 96 个动作与 6 个交接在 step140–153 完成第一轮；第二轮排在 step1869–1882。第 1700 步因此已训练过此完整 family 一轮。没有直接逐样本执行日志，不能把确定性重建称为逐样本直接见证。

|阶段|全部有效前缀误差|前 0.4 秒误差|夹爪分类准确率，query 宏平均|
|---|---:|---:|---:|
|NAV_TO_SOURCE|XY ADE 0.01975m|0.01607m|—|
|PICK|关节 MAE 0.04344rad|0.01850rad|96.84%|
|NAV_TO_TARGET|XY ADE 0.02168m|0.01346m|—|
|PLACE|关节 MAE 0.02549rad|0.01162rad|46.67%|

PICK 首两点的夹爪分类是 100%；PLACE 首点为 50%，首两点宏平均为 41.67%。PLACE 全有效点的微平均为 38.06%。宏/微权重不同，不能混用。

PLACE 错误均为应保持 `open_fraction=0.4492` 而预测偏开，许多点只略高于 0.5 阈值，不能称为全部完全张开。错误点主 joint7 目标偏差中位约 4.22mm，首点错误偏差中位约 2.64mm；离线实验不能把这个数直接换算为掉落概率。原 0.5% 限幅统计门仍失败，本 seed 为 8.0695%。

交接 6 条 JSON 全合法但只答对 3 条：应从源导航进 PICK 时没有进、PICK 未完成前提前进携物导航、应从携物导航进 PLACE 时没有进。模型在已录训练观测条件下也未充分拟合阶段交接。

独立复算、各预测窗口/边界/夹爪指标和曲线见 `FIT_ANALYSIS.md`、`fit_metrics_independent.json`、`training_seed16100000_fit.png`。每个图中标记是独立 query 预测，不是自由物理轨迹。

## 放宽限制后的物理实验条件

实测关节速度门从 3 提高到 30 rad/s。命令关节机械范围、夹爪 [0,1]、0.2 秒预测点的 3 rad/s 目标限速仍保留，非有限值停止。原 0.5% 限幅门继续评分，未改判部署通过。

从训练 episode 第一条合格导航 query（源 tick80/1.6秒）恢复状态一次；因此是已知源状态下的完整任务尝试，不是从源 reset 的第一物理 tick 精确重演。最大 90 秒仿真、225 次 query，0.4 秒重规划，50Hz 控制。推理暂停物理，不证明实时能力。

四任务顺序 NAV_TO_SOURCE→PICK→NAV_TO_TARGET→PLACE 保持。每次阶段推进来自模型自己的视觉 CONTINUE/ADVANCE，不读 teacher phase，不使用 evaluator 触发推进。历史显式是 MODEL_ADVANCE_CLAIM，不伪造 Evidence 或 TASK_COMPLETED；换任务递增 epoch 并丢弃旧任务动作。最后 PLACE 不生成尚未训练的 FINISH，继续动作至有界结束。普通部署所需独立完成反馈仍缺失，这属于显式未认证的研究诊断。

NAV 模型输出仍是 query-body 未来实测 reference，末点送已有 PCT/DWA，不改名为已验证目标命令。保存 A/B_raw/C、0.10m snap 检查；实际补执行纯转向，底盘仍受 vx≤0.30m/s、|wz|≤0.35rad/s 限制。导航完整几何/制动认证 unknown。

物理 profile 为 no_grasp_assist，仍声明 manipulation base/support locks；无固定抓取辅助，无中途物体重置。不能标为纯物理成功。

## 第一条尝试及评分修正

第一条 `physical_01_full_off` 已实际执行 2 次模型推理、42 个物理 tick，其中 22 个 NAV 控制 tick。0.84 秒时 arm_joint3=-0.000016596rad，超过原实测边界容差 1e-5rad；停止原因是数值边界容差，不是 30rad/s 速度门，也不能据此判模型导航失败。

对物体状态的独立审计还发现：旧 SingleRigidPrim 在 physics handle 无效时可回退到 USD；源包装器却无条件报告 physics tensor。reset 也会使物体 sleep。第一条及上一轮相关轨迹没有足够证据区分这两种情况，物体相关成功/失败评分应为 unknown；原 trace、summary 原样保留，不能采信其中的对象失败布尔值来证明策略失败。机械关节/底盘状态和离线真实 RGB 拟合不受该对象读数问题影响。

第二条冻结为：测量位置容差 0.02rad，命令范围不变；写前检查真实 physics handle，无效只重绑一次；要求动态刚体启用、显式唤醒并确认非 sleep；直接读取 PhysX tensor 校验初始化，每个物理步比对真实 tensor 与公开状态读数。它是独立新物理条件，不冒充与第一条完全相同的配对实验。

## 第二条实际结果

`physical_02_full_off_live` 已在 22:25:34 关闭，墙钟 175.923 秒。完成 3 次真实模型推理、60 个物理 tick，其中 40 个 NAV 控制 tick；模型三次都选择 CONTINUE NAV_TO_SOURCE。没有进入 PICK。停止于第 3 次 PCT 规划，其端点量化偏移超过保留的 0.10m 门；不能把这当成完整策略任务失败率。

真实物体检查全部通过：初始化物理句柄有效、刚体启用、非运动学；睡眠状态从 True 显式变为 False。60 次 raw PhysX tensor 与公共状态比对误差均为 0，60 个速度样本各不相同，位置相对首次读数最大变化约 0.000178m。此条排除了 USD 静态回退，仍未执行到抓取，严格接触仍 unknown。

已向用户询问是否追加 1 次、最多 15 分钟，将本研究 PCT 终止门提高到 0.50m，并保留原 0.10m raw snap 失败评分；总 22:49 截止和 GPU2/3 不变。此追加在收到回复前未启动，不把本地准备好的补丁算作已运行。

## PCT 独立重建与剩余边界

审核使用 command.json 绑定的地图、同一 PCT 源码和实际 query root pose/z 在 CPU 重算。前两次 B_raw 与 snap 与原 trace 精确一致。第 3 次未把失败 trace 写入原日志；其重建端点 B_raw=[-0.9152450561523438, 6.4248662948608395, 1.393521708708934]，相对模型端点的 snap 为 0.11618134279328975m，超过 0.10m 门 0.016181342793289746m。PCT 本身返回 ok/same_floor_direct，可通行的正常 0.2m 栅格量化，无需 BFS 移到更远可行点。第三次 B/snap 是独立重建，不是原日志直接记录。

两次尝试均在 NAV_TO_SOURCE 停止，因此本轮自由物理完整 episode 尚未完成；不能给出完整任务成功率、RTC 改善、实时能力或严格接触通过结论。离线 96+6 条实际模型推理已完成，足以定位本训练样本的动作与交接拟合缺口。

本地 deployment_edit 中含待批准的 PCT 0.50m 诊断补丁，仅完成 CPU 检查，未上传、未运行。实际第二条源码保存在 executed_source/run_full_episode_diagnostic_691a144.py；不能用待运行补丁冒充 691a144 的执行身份。

## 本轮收尾

2026-09-08 22:40:02 CST 已按已核实 PID 878648 停止本任务 loopback 推理服务；端口 18170 关闭，GPU2/3 无本任务进程。GPU0/1 原有四个进程保持。两条物理尝试均已关闭；没有重启训练，没有新采集。

追加第三次未获回复，未启动；当前完整物理任务仍未完成。若后续继续，使用新的不可变尝试目录，明确剩余墙钟，加载同一 step1700 与原 normalizer，并在提交待执行补丁后记录新源码 SHA。当前待执行补丁保留原 0.10m snap 评分，仅将研究停止门设为 0.50m；不能据此放行部署。

已运行的 691a144 推送两次遇到 GitHub TLS 中断/低速超时。将尝试从本地已有仓库连接发布同一 Git 对象；推送结果另以 publication.json 记录，未确认前不声称成功。

发布确认：通过本地裸仓库中转同一 Git bundle 后，正常非强制推送成功，分支 `feat/train-seed-fit-diagnostic-20260908`，最终提交 `3a6570007e1e8950267eed11266e0c0f33ffc01f`。H20 工作树干净；冻结训练树仍为 e03cb0b、干净。已运行第二条的代码提交仍是 691a144，后续提交只添加报告。发布回执为 publication.json。

## 用户授权后的第三次追加（2026-09-08 23:02结束）

用户随后明确要求继续放宽第二次限制并再试运行。已追加一次，使用新目录 integration_runs/train_seed_fit_20260908_retry3，源码772d5a1833f449f8e3e6e4a7993b527130c936f4已推送；PCT研究停止门0.50m、原0.10m评分保留。8次真实模型query、161个物理tick，运行到3.22秒，底盘净移动0.5764m，越过原端点停止点。模型始终CONTINUE NAV_TO_SOURCE，未进入抓取；新停止原因为dwa_zero_control_before_reach。

最后一次query距B为0.118729m，先进入原地转向；执行1tick后距B为0.120681m，越过0.12m边界而切回DWA，随后收到零命令。实际DWA容差已绑定为0.12m，不能归因默认0.35m与上层冲突。失败分支没有保存原始DWA命令，零速度来自DWA本身还是guard仍待进一步证据。

23:03:33已释放GPU2/3，端口关闭，未追加第4次、未启动训练。第三次完整记录见retry3/RETRY3_REPORT.md和retry3/validation.json。此前“待回复/未运行补丁”段落是当时历史状态，已被本节的实际授权和执行更新。
