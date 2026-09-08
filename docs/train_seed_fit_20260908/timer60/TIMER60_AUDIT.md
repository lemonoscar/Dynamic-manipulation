# 60 秒诊断独立审计

计时区间：0.0 → 60.0 s；记录 3000 个物理 tick。连续且完整 60 秒：True；仅由计时正常退出：True。

模型请求/响应：150/150；任务分布：{'NAV_TO_SOURCE': 53, 'PICK': 27, 'NAV_TO_TARGET': 70}；转换：{'CONTINUE': 148, 'ADVANCE': 2}。最后任务：NAV_TO_TARGET。这是完整 episode 起点的有界尝试；实际进入的阶段单独列出。

导航零控制 638 次：{'dwa_zero_control_before_reach': 1, 'local_goal_reached': 484, 'reconnect_from_measured_pose_required': 153}。原始 DWA 命令与 debug 逐条保存在 JSON；缺失值不推断。

底盘观测净位移/路径长度：0.7479925175034334 / 3.730069836475625 m；物体观测净位移：0.0001283854716221802 m；评分证据有效性：True。

完整任务评分：False；严格成功：None。计时完成不代表搬运成功；未进入阶段不视为已执行失败，未知接触证据保持未知。详细输入哈希、异常、控制与任务证据见 TIMER60_AUDIT.json。

PICK 实际物理区间：21.22–31.8 s，共 521 ticks；最近 TCP–物体距离：0.2183147135858655 m。全程物体最大位移：0.0002997275732136524 m，正向峰值抬升：0.0 m。PICK 主夹爪目标范围：{'min': 0.0, 'max': 0.04, 'valid_count': 521}；实测 joint7 范围：{'min': 6.3069487623579334e-06, 'max': 0.040023017674684525, 'valid_count': 521}。
