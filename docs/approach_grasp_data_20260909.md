# 接近并抓起：数据核查与新服务器交接

2026-09-09 用户指定所有后续工作转到 10.130.130.37；4×H20 不再作为工作服务器。
这是执行目标切换，未删除或控制旧服务器数据与任务。

## 当前工作位置

- SSH 配置实际身份：dzb_xhq@10.130.130.37:23698，hostname=node04。
- 新仓库：`/hdd1/dzb_xhq/VLA/ConveyorVLA-approach-grasp-20260909`。
- 分支：`feat/approach-grasp-task-20260909`；运行实现继承 `3e446cb`。
- 从本地主控传入源码，无需再访问旧服务器。保留本地浅历史边界
  `92dfddc5d12b345daa5e8a88964a79153841825d`，不声称迁移了完整 Git 历史。
- 当前 checkpoint 和训练元数据：
  `/hdd1/dzb_xhq/migration-4xH20-20260908/dzb/integration_runs/rgb_full_episode_20260908_v1/`。
- 原 RGB：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/`。
- 核查工件：`/hdd1/dzb_xhq/master_integration_20260908/approach_grasp_data_audit_20260909/`。

## 可以复用的当前 116 条 RGB 数据

按已存在的动作视图与原教师事件计数，保留原 family 划分：

| 划分 | 原 episode 数 | NAV/PICK 动作 query | NAV→PICK 标签 | 成功抓起前段候选 |
|---|---:|---:|---:|---:|
| train | 90 | 2755 | 176 | 71 |
| validation | 13 | 366 | 26 | 10 |
| test | 13 | 415 | 26 | 11 |

候选 92 条均有接近成功、抓起成功事件、两阶段动作和一对 CONTINUE/ADVANCE 标签。
其中 15 条后来完整搬运失败（train 13、test 2），但成功抓起前段仍可用于新任务。
候选按 N/B/R 分别为 71/12/9。只用成功前段时，预计保留 train 2176 个动作
query 与 142 个切换标签；validation 289/20，test 355/22。
这些是数据记录数量，不是已验证新模型成功次数，也不是已完成的专用训练发布。

92 条的固定约束创建事件都晚于 pick_success 的墙钟事件顺序；两者可能同一仿真 tick。
所选 NAV/PICK query 未出现在约束创建后，但其中 35 个原先有效的未来应用区间
跨过该边界（train 30、validation 5），派生前段时需要进一步 mask。
不能仅按 query 所属 PICK 阶段筛选而忽略其未来动作窗口。
原教师成功门不等同于新分支连续 1 秒稳定抬起门；新门仍需逐帧重新评分。
所有 116 条的严格接触标定状态仍是 not_calibrated。

目标机实际验证：5 个派生文件哈希全部匹配、116 条 UUID/family 对齐、232 个原始
manifest/任务事件文件哈希匹配；14,304 张所需 RGB 均存在；跨三个 split 和两阶段
抽样解码 24 张图，均正常。未重新全量解码、重验所有 raw 控制或运行训练。
原始数据只读，目标机路径映射保存在 `node04_verification.json`。

## 350 条和 500 条旧数据也在目标机

- `/hdd1/dzb_xhq/modelscope_uploads/liangzhuNeW_350_50fps`：manifest 确有 350 条，
  35 个实际 tar 文件，总计 170,625,382,400 字节；原快照记载 734,668 个 50 Hz 样本，
  双 RGB、11 维目标、frames 状态、阶段事件、summary。原快照记载这 350 条均通过
  旧成功和逐样本检查。本轮抽读首条元数据，确认导航/抓取事件、动作及状态记录存在。
  尚未逐条重新校验所有归档内容或转换成新动作合同。
- `/hdd1/dzb_xhq/modelscope_uploads/liangzhuNeW_500`：manifest 确有 500 条，
  20 个实际 tar 文件，共 25,026,600,960 字节；这是独立的旧批次。
  其旧模型派生数据采用双夹爪目标平均及 legacy 时间语义，不能直接混入当前 reader。

350 条原始记录比旧 5 Hz 派生包更值得优先适配，但没有每通道 raw-control-v2
事务身份，需用 frames 的实际控制报告检查能恢复哪些 issued/held 事实；缺失部分
保持未知。不得照搬旧双指平均，也不得将离线阶段或 evaluator 真值放入模型输入。
此前只清点 H20 派生目录，遗漏了 350 条原始包；本报告修正该遗漏。

## 本轮结论和未启动事项

已有可用的数据来源，不需要先重新采集。最短路径是先由当前已对齐的 116 条中
构造 92 条成功抓起前段视图，再适配 350 条 50 Hz 原始数据以扩充。
派生需统一简化任务指令、两任务后缀及 parent memory 哈希，保留 NAV→PICK 标签，
排除 PICK→NAV_TO_TARGET；不造 FINISH 标签，失败前段可单独加入合法动作池。
新派生继续保留 family 划分；normalizer 的继承或重新拟合必须有独立身份。

尚未启动训练：迁移 release 仍含旧绝对路径；复制的环境有待验证的旧 shebang/
editable 路径；物理诊断入口仍含 H20 GPU/Sim5.1 绑定。目标机是 A40/3090 混合卡，
不能复用 H20 GPU2/3 授权和编号，也不能将源码同步解释为部署环境已验收。
本次未使用 GPU、未启动推理或训练、未更改旧数据与已运行任务。
