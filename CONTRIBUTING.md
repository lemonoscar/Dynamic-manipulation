# 开发与提交约定

当前维护线是 `Manipulation_Navi_v1` 的 Joint-Trajectory 5 Hz 合同。
先读 [README](README.md) 和 [开发指南](docs/joint_trajectory_development_guide.md)，
确认正在修改的模型/数据合同；历史 Waypoint 与 SPCGVLA 草案各自有独立范围。

## 改动与验证

使用范围明确的分支和提交，说明触发条件、行为变化、验证和已知限制。
动作单位/坐标、时间采样、数据 split、normalizer、checkpoint、抓取辅助、成功判定或门槛变化，
须同步配置、合同、对应测试与迁移说明。新结果保存在新的实验目录，不能改写冻结结果。
README 陈述能力时引用固定实验摘要，区分运行完成、性能门槛通过与完整物理任务成功。

运行 [分层检查](docs/joint_trajectory_development_guide.md#5-测试与实验晋级)。
对影响模型/控制的修改追加适当 strict-load 或仿真验证；CPU mock 通过不等于实机测试。
不为绕过依赖或数值错误删除检查。可选权重未安装的测试可以明确跳过；错误权重不得跳过。
使用 AI 辅助开发也遵循相同证据要求，提交者负责确认输出正确及来源适用。

## Git 与外部工件

允许提交源码、配置、测试、小型实验摘要、来源及许可证记录，以及已有的受控机器人资源。
权重、checkpoint、优化器/RNG 状态、数据集、逐帧记录、视频、点云、日志、环境、下载包和凭据放在
Git 之外或被忽略的 `artifacts/` 下。正式数据与训练输出还必须位于工作树之外。
不要使用 `git add -f` 绕过 `.gitignore`，不要把工件改后缀规避排除规则。

`.gitignore` 不会自动移除已经跟踪的文件。对误跟踪的工件，使用 `git rm --cached -- <文件>`
只移除索引，并核验本地文件仍在；本次已对旧 locomotion `policy.pt` 这样处理。
权重哈希仍由其 contract 保存。该操作不清除既有 Git 历史中的旧对象，历史清理是另一个需协调的任务。

推荐明确暂存路径后检查：

```bash
git diff --check
git diff --cached --check
git diff --cached --stat
git diff --cached --name-status
git ls-files -ci --exclude-standard
```

最后一条应无输出：若发现已跟踪且命中忽略规则的文件，逐项处理。
还需检查暂存文件内容和大小，确认无 token、私有端点、账号凭据、完整样本或模型文件。
小型 JSON 指标摘要可以版本化，但须保留来源报告哈希、统计口径、时间戳和完成状态，去掉机器绝对路径。

推送前确认当前分支、目标远端及非快进冲突；不要强推覆盖其他人的更新。
此次发布保留既有提交历史，不自动合并其他模型合同分支，不移动已发布 tag。
不要在活跃冻结评估的工作树内修改源代码；使用独立工作树做下一轮实验。
问题反馈提供复现命令、配置/源码身份与脱敏错误摘要，数据及模型的分享遵循各自来源许可。
