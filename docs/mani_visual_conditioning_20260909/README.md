# MANI 视觉与状态：诊断及最小训练调整

用户提出 MANI 的 qpos 可能压过视觉，要求修复。结论：当前真实小样本不支持“视觉被qpos覆盖”是已证实主因。代码已加入可选、版本化的训练期状态token屏蔽，但未训练新权重，不能宣称抓取改善。

## 真实输入诊断

固定step1700权重SHA256 `474d03ff22da939f763574bf434c75226a2a2533ae984ced80e25e34a1df7fa0`、其原normalizer、RGB四帧、causal_command_5hz、任务上下文和两个扩散噪声seed171/172。取训练family16100021/16100026各PICK、PLACE的1/4与3/4位置，共8个真实query（4 PICK、4 PLACE），每例六种条件，总96次动作采样。源query时钟不是此前60秒闭环视频时钟。

对照分别为原输入、同阶段另一family RGB、全黑RGB、另一family qpos、另一family dq和完整13维状态。仅变条件，不变标签/有效mask，不修改解码所用真实q anchor。指标针对delta joint proposal，不把必然随q anchor变化的绝对目标算作状态敏感性。换图与换状态的扰动幅度并不等价，所以不能把下表比值叫作视觉贡献比例。

PICK的平均结果（4 query × 2固定噪声，关节为rad）：

| 条件 | 相对原预测的关节MAE变化 | 对真实动作标签的关节MAE | 对夹爪标签MAE(open fraction) |
|---|---:|---:|---:|
| 原输入 | 0 | 0.04531 | 0.03240 |
| 换RGB | 0.07129 | 0.08021 | 0.03001 |
| 黑RGB | 0.17026 | 0.14885 | 0.41263 |
| 换qpos | 0.01907 | 0.05423 | 0.02930 |
| 换dq | 0.01033 | 0.04355 | 0.03259 |
| 换完整状态 | 0.02986 | 0.05748 | 0.03011 |

视觉对当前预测有明显作用。黑图会显著破坏抓取关节与夹爪预测；普通换图对夹爪误差并非总变坏，这也是不能简单说“所有动作都主要由视觉决定”的原因。PLACE与逐query全量结果在report.json和queries.jsonl。

这是两个训练family的有意输入干预，不是留出集成功率、物理抓取、注意力因果贡献或所有闭环偏离状态的证明。它不能排除某些时刻的状态捷径。现有闭环seed16100021的PICK共581tick中，q任一分量超出训练分位数区间约2.58%，归一化最大绝对值1.079；dq约3.96%，最大4.082。此项由已下载真实trace计算，提示少量状态尾部偏离，不能据此擅自clip旧normalizer。

## 实际结构与调整

本轮实际使用StagedRGBBackend → StagedExperts → M0DiTActionHead，不能与另一套joint_trajectory head混淆。13维状态(q6+dq6+gripper)经MLP生成一个token；视觉通过交替结构中的8个cross-attention层参与。Qwen训练通路没有detach，视觉参数可训练。仍有可疑的捷径来源：较短的状态通路、纯文本task bank与含文本live bank的冗余、视觉学习率低于动作专家。但这些结构本身不构成已经定位的根因。

最小新增配置：`StagedConfig.mani_state_dropout`，旧配置缺失时为0。仅MANI loss且train模式，按每个query一次Bernoulli遮罩，丢掉整个编码后的状态token（包括MLP bias），保留槽位和序列长度，幸存token不放大。避免仅将输入q乘小数而改变物理数值含义。原始记录、真实q anchor、标签、夹爪有效命令、normalizer、NAV、eval、普通采样与VJP RTC均不改变。

完整episode训练入口可在下一轮独立配置的training部分显式加入：

```json
{"mani_state_dropout": 0.3}
```

0.3是待验证的起点，不是已选优超参数；1只适合极端屏蔽诊断。新candidate及training binding将记录该值。旧checkpoint加载、零概率训练及旧staged resume将补齐默认0；真正改变概率不能假装原配置resume。未给完整训练预算配置、未启动训练，历史8小时预算不续用。

这项正则只抑制依赖状态的捷径，不能保证模型转而学习正确视觉关系，文本也可能成为捷径。后续训练需固定这批干预query，并另加不同family验证，比较正确图像的动作误差及闭环持物；不能只追求“换图后变化更大”。高层过早ADVANCE与低层视觉对准/夹爪执行仍应分别检查。

## 版本、资源、验收

诊断源码身份4fba360d853d7fe9e5f6a3e5244bfc0fcd58f876，使用旧p=0候选和原权重。后续两个提交只增加旧resume兼容与测试，GPU诊断结束后才同步远端工作树，没有热改正在运行的代码。

GPU2 UUID `GPU-0b4f9aa0-7c5b-0aa1-3652-51b7ec9fc8b7`，单进程，0优化步，统一20分钟墙钟上限；GPU3未使用。首次诊断因脚本把numpy状态构造成float64而在首个动作forward失败；修为FP32后，在原预算剩余975秒内使用新结果目录完成8query，成功进程耗时90.813秒，峰值allocated显存28,705,637,376字节。失败日志仍保留，不记作模型能力失败。

远端：`/diff/wallx_workspace/dzb/ConveyorVLA-mani-visual-20260909`，分支`fix/mani-visual-conditioning-20260909`；工件根`/diff/wallx_workspace/dzb/integration_runs/mani_visual_diagnostic_20260909_v1`。

63项CPU定向测试通过（CUDA隐藏）：状态p0输出/梯度/RNG精确兼容、p1状态不敏感且视觉梯度保留、p0.5每query一次混合遮罩、编码后整token零化、不放大保留token、NAV/eval/sample/RTC不改变、旧checkpoint严格权重加载与旧resume身份兼容，加原DiT、staged、RGB和完整episode相关回归。独立只读审核未发现阻塞项。不是63项物理实验。

GPU诊断及CPU测试均退出0，所有本轮tmux会话已自然退出；只剩原用户任务会话，未操作他人进程。protocol/report/queries三个文件已传回本地并逐一与源SHA256核对，见TRANSFER_HASHES.json。
