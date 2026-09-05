# CFPI v1 — 冻结候选集的闭环决策纠错，首轮协议

冻结时间：2026-09-05。内部名称 CFPI 不代表已成立的论文创新。
研究判断：ACCEPT selector-only 的 V3 进化；AMEND 为自然重规划下的
决策价值学习与能力保留；REJECT 默认认可 IPCT、basin 或声称 GRPO 在扩散上数学失效。

## 研究问题和边界

局部整条轨迹评分与“现在执行该候选的下一步，此后自然重规划”的返回不同。
本轮仅检验真实模拟器闭环分支反馈是否可以从 selector 可见输入学到，
以及全候选分类更新是否优于相同反馈下的 GRPO 和普通 Q 回归。
不声称全局因果最优 Q、真实世界安全或连续部署单调提升。

训练与部署均冻结原感知、DiffusionDrive 生成器及 base selector，保留原始 20 候选。
复用 V3 scene-conditioned selector，不增加 basin、历史模块、搜索或候选。
训练使用 privileged 分支标签；推理只使用既有几何、候选特征、BEV、ego/status/agent context。
生成器不训练，部署不读取奖励、未来、Q 标签、成功/失败分层或日志 ID。

采集参考策略固定 rare-tuned scalar V3 seed 0：
`bd35f0293c878c6cddb985ac0b8aaae91d4f7f8e0bf2c3bf70ba49ba329c02a3`。
噪声 namespace 为 `selector_cfpi_v1_noise0`，绑定 sample token，不绑定 worker 编号。
相同历史/候选下仅干预一次，之后恢复参考策略。保持现有 LogPlayController、
0.5 秒重规划和 reactive IDM，不 latch 整条轨迹，不更换控制器。
返回取原始最终闭环 PDM score，范围 [0,1]，不另造奖励。
固定 namespace 的重复只检查固定条件重复性，不估计未来随机性的期望。

## 数据与冻结

新 run ID、新 CFPI source/target/treatment/cache schema，不接受旧 V4 cache 训练。
上游只读复用 V4 origin-log membership 与 train shards；只采集 original rare_union
和 matched_common，各 512 场景，共 1024，跨两类合计每 origin log 最多 4 场景。
2026-09-05 source 启动修复：两类配额采用联合容量分配。先复现原 hash-greedy
选择；若单独顺序选择未满，使用确定性残量增广重新分配少量共享日志名额。
日志内仍按原 scene hash 取样，不改变 family、不降低配额、不读取 outcome/Q。
只有联合容量确实不可行才报源池不足。此修复发生在任何 baseline 采集之前，
须使用新 run ID；旧失败记录与源码合同保留。
`exclusions.json` 冻结排除历史 CCV 的 15 个 origin logs。历史 V3 训练/调参曝光不能
自动视为不存在；本轮 CV 只对新增反馈训练 log-disjoint，不称全流程完全未见。
任何 development/certification 必须另做全部历史来源曝光审计后再冻结。
不读取新 V4 outcome、旧 CCV Q 训练、不采集 development/test。

先采集 train baseline A/B。同一 namespace 的 A/B 轨迹/logits、部署动作与返回必须一致。
局部 PDM 的轻微重复漂移仅报告，冻结 A 标签用于 local-GRPO 对照，不据此筛 target。
不稳定闭环结果直接停止，不仅排除不稳定行后继续。

target 仅从 decisions 4..11 中选择。每场景 1 个、每日志最多 2 个，共 64：
family × 原策略 failed/solved × near/ordinary，共 8 层，每层 8 个。
failed near 从首次违规前最后 3 个合法决策 hash 选择；
failed ordinary 从所有合法违规前决策 hash 选择；
solved near 逐 family 匹配 failed near 的 step histogram；
solved ordinary 从全部 decisions hash 选择。near/ordinary 是抽样规则，
不是互斥的物理时间区间。先 near 后 ordinary，禁止重复场景。
任一分层不足则停止并报告，不降低配额或改源。
以上规则读取 baseline 的事件状态/时刻，但不读取候选 Q 或用局部 reward 排序。

targets 确定后按 origin log 分配四折，hash 排序、负载/分层计数平衡。
每层预先 hash 选 1 个（共 8）用于 sentinel 和 repeat8；不得看到 Q 后选择。
sentinel 只干预为原策略候选；pilot64 与 repeat8 都运行全部 20 分支。
任何 target/protocol/source/code/checkpoint/noise 漂移必须新 run ID，禁止静默混合。

## 最小方法和公平对照

分类权重 `w_i = Q_i - min_j Q_j + delta * 1[i=b]`；
loss 为 `-mean_rows(sum_i(w_i * log_softmax(z)_i))/Z`。
Z 固定为当前训练折上 sum(w) 的平均值（下限 1e-8），不逐行归一化。
delta=0 和 0.01；正值仅是小幅原选择偏好，不是 retention 保证。
它属于已知的成本敏感分类/策略改进思想，不以公式本身冒充创新。

对照固定为：

- 冻结 V3。
- local reward + exact group GRPO，T=1。
- 同一闭环 Q + exact group GRPO，T=1 和 5。
- 同一闭环 Q + MSE 回归。
- 同一闭环 Q + 分类更新，delta=0/0.01。

GRPO 为精确 `-sum(pi * normalized_advantage)`，无 sampled-PPO ratio/clipping，
这一轮匹配目标实验 KL=0；不宣称复现 rare-tuned V3 的全部历史训练合同。
所有新方法同样训练完整 V3 selector，从同一 V3 表示初始化。
Q 回归仅将最后输出层重置为训练折 Q 均值，输出 absolute Q；其余模型保留 V3
residual 初始化与 base logits 相加。该初始化差异显式记录，不称完全同初始化。
不把 Q 回归值与 base logits 相加，不把 logit 概率当作校准不确定性。

四折、seed 0/1/2；LR={3e-5,1e-4}，AdamW weight_decay=1e-4，batch=16，
有足够数据时每 batch 无放回随机抽样，500 updates，检查点 50/150/500。
同配置优化预算一致；复用 Hopper 的 CPU global gradient norm clipping，max_norm=10。
共 144 个独立训练 job，最多 8 GPU 并行，无需 DDP。
报告每个方法/LR/step 的完整折外预测与各 seed，不把 CV winner 的区间当正式认证。
筛选先通过 gate 的配置，再按 mean gain、solved gain、较早 step、较低 LR、名称决胜。
未通过配置中排名仅用于诊断，不授权推广。

## 质量、信息与方法 gate

工程：全候选/时刻覆盖、source/checkpoint/code 哈希、一次干预、前缀与动作对齐、
incumbent sentinel、全部 target 的 incumbent 返回、repeat8 完整反馈。
数组 max_abs <=1e-5，动作 <=1e-4，返回 <=1e-3；
NC/DAC、success 与首次违规时刻要求分类一致。NaN、缺失候选不得补零。

信息：64 中至少 16 个满足 max(Q)-Q_incumbent >0.02，才授权训练 CV。
不满足只表示该 pilot 未达到采集推进门槛，不证明任务无选择空间。
方法筛选：三 seed 平均的完整 OOF mean gain >=0.005、四折至少三折正，
solved 子集 mean gain >=-0.005。日志聚类 bootstrap 10000 次，seed=20260905，
同时报告场景平均与等日志权重平均。多个种子不当作独立场景，
20 个候选不当作 20 个独立统计样本。所有阈值仅为研究筛选，不是显著性声明。

若只有 Q 回归有效：接受标签方向，不接受新损失贡献。
若 oracle 足够但学不出来：区分样本、优化、可见输入，不直接下不可学习结论。
无论哪种结果都停在 pilot 报告，expansion_authorized=false。

## 执行预算与后续决策

默认 8 H100（sm_90）、torch 2.0.1+cu118/CUDA 11.8，独立 MMCV/gsplat 扩展逐卡验算。
默认 144 GPUh 上限，估计 80–140 GPUh；中断时保存各 collection 与训练 checkpoint。
运行器记录实际 GPU 预留时间（并非功耗/利用率积分），断电后未知区间保守计费。
小卡数支持新 run 的硬件配置，不改变 token 噪声；已有 run 不静默改变拓扑。
只管理本次启动的进程组，不全局 pkill/ray stop，不覆盖旧合并产物。

首轮完成后才另行批准：扩到 train256 → 冻结胜出配置 → 独立 development 连续部署
→ 等预算比较 static vs refreshed policy evaluation → 正式测试及跨噪声泛化。
最终还要比较 gate-conditioned 等最强 V3；pilot 的 scalar V3 不是最终最高基线。
内部目标为主闭环指标较最强可比基线 +2pp、配对区间支持正值、其他主要指标不退化 >1pp；
这些目标不等于顶会录用保证。不根据未见 test 改方法。

## 论文定位

CRAFT 已研究代理反事实与真实执行纠正，RAD-2 已研究 RL discriminator 和时间一致性；
我们不能声称首个 selector-only RL、首个 Q scorer 或首个分类策略改进。
必须验证完整实际分支反馈在自然重规划下的可学性、错误替换控制、连续部署，
及相对于普通 Q 学习/等模拟预算方法的收益。
参考：https://arxiv.org/html/2605.04470v1 ，https://arxiv.org/html/2604.15308v1 ，
https://www.jmlr.org/papers/v17/10-364.html 。
