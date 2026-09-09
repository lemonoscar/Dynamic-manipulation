# MANI 双视角分工与权重诊断

2026-09-09。用户要求把front/wrist视觉权重纳入考量。本轮完成真实相机干预及输入token审核；没有改模型权重、像素亮度、视角顺序或生产融合比例，也没有训练。

## 冻结条件与输入核查

沿用step1700、同normalizer、同8个训练query（16100021/16100026各PICK与PLACE两点）及固定噪声171/172；状态、标签、有效动作mask与解码anchor不变。七条件：原输入、front两帧置黑、wrist两帧置黑、只换front、只换wrist、全部置黑、交换两相机角色；共112次动作采样。单相机替换来自同阶段另一family，保留各自历史→当前的顺序；不是用另一相机的某帧补时间。

真实源图像四张均640×480。processor实际video_grid_thw为[[1,30,40],[1,30,40]]，spatial_merge_size=2，两段时序clip各300个视觉token，共600个。8个query均一致；300是每相机两帧clip的融合token数，不是每帧300。已同时核对input_ids里的video_token_id数量，与grid推导一致。

输入先明确标记Head/front，再标记Wrist，均oldest→newest。两视角经过Qwen联合处理，再给动作专家混合token bank；没有独立固定front:wrist标量。token数量相等不等于实际贡献各50%，融合后的hidden不能不加区分地当成原始独立相机特征做简单倍乘。

## 实测误差

各阶段4query×2noise均值。关节标签MAE单位rad；夹爪标签MAE单位open_fraction。

| 阶段/条件 | 关节MAE | 夹爪MAE |
|---|---:|---:|
| PICK 原输入 | 0.04531 | 0.03240 |
| PICK front置黑 | 0.07215 | 0.02676 |
| PICK wrist置黑 | 0.07213 | 0.08828 |
| PICK 只换front | 0.07048 | 0.03278 |
| PICK 只换wrist | 0.05318 | 0.03705 |
| PLACE 原输入 | 0.01735 | 0.05152 |
| PLACE front置黑 | 0.04174 | 0.14528 |
| PLACE wrist置黑 | 0.02155 | 0.09800 |
| PLACE 只换front | 0.01790 | 0.05031 |
| PLACE 只换wrist | 0.01726 | 0.05057 |

在这组query中，PICK遮住任一相机都增加关节误差；wrist遮挡对夹爪更明显。PLACE遮front的误差上升大于遮wrist。单相机普通换图与黑图的幅度、含义不同：PLACE只换图几乎不改变误差，而黑图是更强的分布外干预。不能把黑图结果换算为最优权重或因果贡献百分比，也不能把PICK遮front后夹爪误差小幅下降解释为front无用。

交换两个相机角色后，PLACE关节误差0.03748、夹爪误差0.15929，均高于原输入；已有权重对视角角色敏感，不能随便交换角色。完整逐query预测和其余条件在queries.jsonl/report.json。

## 本轮决定与下一轮验收

保留完整双视角和已有学习式融合，不硬编码全程50:50、30:70或单纯提高wrist。当前证据支持分阶段、分关节/夹爪评估的必要性，没有证明存在token分配偏置，也没有找到一个应立即用于部署的固定权重。

下一轮训练沿用独立的front/wrist干预诊断作为验收项，记录PICK对准/闭合、PLACE接近/释放的正确图像误差和单相机退化；本轮只有各阶段两点，不冒充已覆盖全部细阶段。若后续确需显式相机门控，应作为需要训练的新模型版本，在清楚的相机token/特征边界进行融合，保留两侧信息，不能直接改旧checkpoint推理增益。优先验证训练能否降低正确图像下的动作误差及实际稳定抓取，而不是最大化换图敏感性。

这是两个训练family的小样本诊断，不是总体成功率、训练后增益或新的物理实验。高层未持物却ADVANCE的问题仍独立存在，不由改变相机权重自动解决。

## 实现、资源与复核

复用scripts/diagnose_mani_conditioning.py，新增--study cameras和显式最大墙钟，旧state诊断默认为原路径；新增完整时序对干预测试。GPU诊断源码d671765，旧权重SHA256为474d03ff22da939f763574bf434c75226a2a2533ae984ced80e25e34a1df7fa0。16次原输入预测与上一轮state诊断逐项完全相同。

H20 GPU2 UUID GPU-0b4f9aa0-7c5b-0aa1-3652-51b7ec9fc8b7，单进程，10分钟墙钟上限，0优化步；实际成功耗时95.220秒，峰值allocated显存28,710,004,224字节。10项CPU相机/RGB定向测试通过。GPU与测试均退出0，自己的tmux会话全部退出，其他用户进程未操作。

远端工件根：/diff/wallx_workspace/dzb/integration_runs/mani_camera_diagnostic_20260909_v1。结果已下载至本地mani-camera-conditioning-20260909-evidence，三个文件与远端SHA256逐一匹配，见VERIFICATION.json。本次只修改诊断及文档，没有新增相机加权生产代码或训练权重。
