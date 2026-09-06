# CFPI pilot 独立复核与下一阶段研究决定

日期：2026-09-06。来源：cfpi_pilot_20260905_r2 与 cfpi_pilot_20260905_r2_cv1。
本文件是完成 pilot 后的研究复核与执行建议；没有启动新实验，没有改写旧 run 的
contract、gate、选择规则或 expansion_authorized。下一阶段须使用新的运行合同。
复核脚本：[analysis/review_pilot_20260906.py](analysis/review_pilot_20260906.py)。

## 研究判断

ACCEPT：继续投入一次有预算上限的连续闭环验证。完整分支反馈包含可利用的选择空间，
可见输入下训练的 selector 在训练侧折外评估中获得正向收益。

AMEND：主线聚焦自然重规划下的执行价值、策略更新后的价值变化与原有能力保留。
下一轮先验证持续部署，再决定扩大分支采集。原“直接扩 train256”的次序暂缓。

REJECT（当前证据下的主张）：分类更新已优于 GRPO、GRPO 在扩散候选选择中数学失效、
温度调整本身构成论文创新。CE 仍保留完整负结果，Q 回归仍是必须保留的强对照。
这些判断不意味着 CE 永远无效，也不意味着整个研究方向已达到顶会要求。

## 数据和结果是否可信

本次通过 JSON target manifest、20 个已审计 arm 的原始 CSV 和 144 个训练报告，
独立重建结果，未反序列化 pickle 或模型。核对 20 份指标文件/审计 SHA，
432 个 checkpoint SHA；36 组方法/LR/step 的平均收益均按原 float32 汇总规则复现。
所有 CV 训练在新反馈层面按 origin log 隔离，48 场景训练、16 场景折外评估。
repeat8 是固定条件重复性检查，不能视为跨噪声泛化或未来随机性置信度。

pilot：64 场景、59 个 origin logs；32 failed、32 solved；每场景只选一个 target，
在该时刻替换一次候选，然后恢复 scalar V3。成功定义沿用 NC==1 且 DAC==1。

- incumbent mean score = 0.4327469。
- 全分支 oracle mean score = 0.8618842，平均 gap = 0.4291373。
- 46/64 场景存在 >0.02 的改进空间。
- 32 个失败场景中，27 个存在可使其成功的单次干预分支。
- oracle 是 privileged 诊断，不是可部署策略。

以下 gain 都是选中分支的最终 PDM 相对 incumbent 分支的差值，
不是 selector 全程接管后的分数，更不是总体驾驶性能提高的百分比。

| 配置 | 平均 gain | solved gain | failed gain | 各 seed 挽回失败数 /32 | 各 seed 破坏成功数 /32 | 原 screening |
|---|---:|---:|---:|---|---|---|
| Q-GRPO T1, LR=1e-4, 500 | +0.125434 | -0.004197 | +0.255064 | 8 / 10 / 8 | 0 / 1 / 0 | 通过；2/3 seed 通过 |
| Q-GRPO T5, LR=3e-5, 500 | +0.123715 | +0.009879 | +0.237552 | 8 / 8 / 8 | 0 / 0 / 0 | 通过；3/3 seed 通过 |
| Q-MSE, LR=1e-4, 50 | +0.163437 | -0.023487 | +0.350361 | 17 / 12 / 11 | 0 / 1 / 1 | 未通过 |
| Q-CE d0, LR=3e-5, 500 | +0.160256 | -0.008077 | +0.328589 | 12 / 12 / 10 | 1 / 1 / 1 | 未通过 |
| local-GRPO T1, LR=1e-4, 500 | +0.076184 | -0.013574 | +0.165942 | 6 / 5 / 5 | 1 / 0 / 0 | 未通过 |
| local-GRPO T1, LR=1e-4, 50 | +0.089138 | -0.007834 | +0.186109 | 5 / 7 / 6 | 0 / 0 / 0 | 未通过 |

原预注册排序的 winner 仍是 T1，不追溯改成 T5。
T5 作为下一阶段的能力保留候选：rare/common、near/ordinary 四个子集平均 gain 均正，
每个 seed 的四个 fold 均正。它仍有每 seed 4–6 个 score 下降场景，
“零 solved-to-failed”仅适用于这 32 个已成功场景的单次干预，不能保证未来安全。
T1/T5 的 driving-direction component 平均分别下降约 0.00781/0.00260，也应继续报告。

完整的同 LR、同 step T1-Q 与 local 对比中，6 组有 4 组差值为正、2 组为负。
不能凭一个最优配置证明闭环 Q 在所有设置中优于 local。

新的探索性配对 log bootstrap（10,000 次，先对 seed 平均，等日志权重）：

| 比较 | 场景平均差 | 等日志平均差 | 等日志 95% 区间 |
|---|---:|---:|---|
| T1-Q vs 同 LR/500-step local | +0.049249 | +0.037269 | [-0.032921, +0.109597] |
| T5-Q vs T1-Q | -0.001718 | -0.006471 | [-0.041009, +0.021906] |
| Q-MSE vs T1-Q | +0.038003 | +0.044300 | [-0.039266, +0.128273] |

均不足以确认两方法存在差异。原报告 winner 的 [0.04739, 0.19523] 对应
“相对冻结 V3”的等日志加权均值 0.11509；不能把它当作场景均值 0.12543 的区间，
也不能当作“优于 local-GRPO”的区间。所有这些区间均在方法筛选之后计算，非正式认证。

## 两个会改变推进方式的限制

1. 原 source1024 的 V3 baseline 只有 48 个失败，失败率 4.6875%，成功率 95.3125%；
   pilot 人为平衡到失败率 50%，并包含违规前 near targets。当前总体 gain 会强烈
   受到困难场景富集影响。不能仅按 4.7%/95.3% 重加权就宣称总体效果，时间和 family
   抽样同样不同。必须有按 ID 抽样的自然比例全程驾驶评估。

2. 历史成员表显示，59 个 pilot logs 中有 54 个在已知 V3 rare-tuning train，
   2 个在已知 V3 development，0 个在其 certification；其他历史曝光并未穷尽。
   当前 CV 只对新增 CFPI 反馈训练 log-disjoint。现有 development_consumed=false
   是本轮 split 的记录，不意味着从未接触历史 development。
   论文的独立验证必须另审计历史 adaptation/tuning/diagnostic 成员表。

直接把当前平衡抽样扩为 train256 会要求 128 个 failed 场景，但 source1024
总共只有 48 个；即便 train128 平衡配额也要求 64 个。不得简单改常量运行、
复制同一失败场景充数或静默降低配额。将来若扩大，需要先冻结新的训练来源/
抽样合同，再观察其候选分支 Q。

## 下一阶段 A：用现有折外模型检验持续接管

目的：测量当前“一次改选，旧策略后续执行”的收益，在“此后每一步均使用新 selector”
时能否保留。这直接决定现有标签是否足以支持策略部署。

使用原 64 场景及四折归属，每个场景始终路由到未使用该 log 的 CFPI 训练模型，
seed 0/1/2 全部保留；保持原控制器、0.5 秒重规划、噪声 namespace、冻结生成器。
生成器参数和同状态下候选不变，不要求不同后续状态下生成的候选相同。

每个场景先用旧 V3 复现原 prefix，到预定 target 起持续使用新 selector。
target 上的动作应与原 OOF prediction 对齐，prefix 必须通过原有 parity 检查。
“固定时刻开始接管”仅是机制实验设置，不能伪装成部署时使用未来事件的 gate。
单次干预结果直接读取已审计 full-branch CSV，避免重跑 20 个分支。

冻结四个方法，各三个 seed：

- T1-Q：LR=1e-4，500 steps，保留原预注册 winner。
- T5-Q：LR=3e-5，500 steps，检验当前能力保留候选。
- local-T1：LR=1e-4，500 steps，与 T1-Q 构成标签比较。
- Q-MSE：LR=1e-4，50 steps，保留普通回归强对照。

核心约 4×3×64=768 条 rollout，另少量 incumbent sentinel。
暂不新增 loss、结构或重新搜索超参。各方法完整报告 score、SR、NC、DAC、EP、
TTC、direction、挽回和破坏场景，以及 one-shot→持续接管的差值。

最终批准的推进条件：工程检查通过，持续接管平均 PDM 收益 >=0.005，且至少 2/3 seed 为正；
均值 SR/NC/DAC 相对旧 V3 的退化分别不超过 0.01。它们是新阶段筛选门槛，
须在采集持续接管 outcome 前冻结，不能用来改写旧 pilot 判定。
若没有方法满足，暂停扩大采集；定位 target 后的首次新增错误，并检验 continuation
policy 变化与新状态覆盖，而不是从同一 pilot 再搜索几十个 loss。
分层样本上的通过仍不是总体有效性结论。

## 下一阶段 B：自然比例、从第一步开始的驾驶筛选

A 通过后，以同一固定配置在全部 pilot64 上拟合各方法的 3 个 seed。
从现有 train source 中排除全部 pilot origin logs，以固定 hash、仅按 family 配额
选 128 个场景（rare/common 各 64，严格每 log 1 个），不按成功/失败、违规时刻、
奖励、Q 或新方法结果筛选；先检查 log/family 容量并冻结名单。
这一步称 training-pool deployment screening，不能称独立 development/test。

对这些场景从第一个决策起全程部署，使用 A 中四种已冻结方法，
同时评估冻结 scalar V3 和预先固定的 gate-conditioned 强 V3 checkpoint。
gate 的指定 checkpoint、score semantics 与哈希须在开始前锁定，筛选阶段可以单个
预指定 seed；论文主结果再补齐可比三种子。不能用新方法的结果决定挑哪个旧 checkpoint。

约 12 个学习策略条件×128 + 两个固定基线×128 = 1,792 条 rollout；
若 A 的方法因工程原因失败，应先修复，不能静默删掉强对照。
未通过 retention 的 Q-MSE仍可作为诊断对照，但不因此被授权推广。

完整报告配对 gain、方法差异、三 seed、rare/common 与保留指标。
小样本区间跨零只能标为证据不足，不自动等于方法无效；明确负收益或保留失败时
暂停 train256，避免仅靠更大样本掩盖目标/执行不一致。
独立 development 的数据成员审计和冻结可以在准备本阶段时并行开展，
但本计划不自动消费新的 development/test outcome。

## 之后的方法与论文主线

要验证的问题可以表述为：

> 对冻结扩散规划器，如何把同一候选集的实际执行反馈转成持续重规划的策略改进，
> 并控制对原有可靠行为的错误替换？

当前 Q 是固定噪声、参考策略续接和原模拟器下的终局返回，记作 G(history, i; π_ref)
更谨慎；它不是对未来随机性积分的真实世界 Q，也不是换策略后不变的候选属性。

若 A/B 显示反馈续接策略变化是主要损失来源，下一轮优先检验有限次 policy evaluation
refresh：冻结 π_k → 在其访问状态获取一次分支反馈 → 更新 selector → 重新评估。
必须与“同等额外模拟预算、仍为旧策略增加静态标签”的对照比较，避免把更多数据的
收益误写成策略刷新贡献。标准 policy iteration、KL 和温度本身不构成新方法。
若 A/B 已稳定提高效果，先扩大到经审计的独立验证，再决定是否需要新增机制；
不为制造创新而增加没有证据需要的模块。

分类/IPCT、basin 暂不作为主线。普通 Q 回归更激进且挽回数量多，不能因原 gate
没过就从论文移除；以后任何能力保留改动，都应给 Q 回归可比的保留对照，
再判断专用方法是否有价值。

论文主结果最终应超过最强可比 V3（包括 gate-conditioned）并保留主要能力；
+2pp 主闭环指标、其他主要指标退化不超过 1pp 是内部研究目标，非录用保证。
现有训练侧、单个行为初始化、单个噪声 namespace 的结果还不能承担这个结论。

相关工作已再次核对原文：
[CRAFT](https://arxiv.org/html/2605.04470v1) 已有 dense proxy + grounded correction
与 teacher regularization；
[RAD-2](https://arxiv.org/html/2604.15308v1) 已有 RL discriminator、latched temporal
rollout 和 generator 更新，也报告 discriminator-only 实验。
自然重规划、生成器冻结和全分支可审计反馈是我们可验证的设定与证据区别，
仅有设定区别不足以声称新颖性；必须给出重要的经验发现和相应的方法/效率收益。

## 预算和执行准备

三个正式相关 runs 账本累计 72.095045 GPUh；相对原 144 GPUh 留有约 71.90 GPUh。
这是 runner 记录的 GPU 预留时间，不能替代实例实际计费记录。
本次 CV 实际 404.18 秒（6 分 44 秒），0.78037 GPUh；
先前 3–7 小时的续跑估计明显过高，训练小 selector 远快于模拟器。

实测旧 pilot 每个 64-scene arm 平均 586.95 秒，20 arms 合计 26.09 GPUh。
下一阶段拟分别设置 A ≤24 GPUh、B ≤40 GPUh，总上限 64 GPUh，
保留约 7.9 GPUh 余量；8 卡实例粗估数小时至一个工作日内，启动开销/
折模型切换会影响墙钟，需按首批吞吐更新估计。该预算不能支持自动 train256。

新实现需要：
1. 导出/路由四折 selector 并验证非 selector 张量不变，Q-MSE 维持 direct_q。
2. 增加旧策略 prefix 后切换并持续执行的管理逻辑，以及全程部署入口。
3. 锁定 scene→fold→model 清单、128 场景来源、基线身份、预算与停止条件。
4. 在采集前验证同模型路由、prefix/target 动作对齐，以及后续不再读取标签；
   新旧采集协议分开记录，新 run 只引用经过哈希校验的旧产物。
5. 先完成软件与小规模 sentinel 检查，再在 H100 节点启动上述有上限的阶段。

实现状态更新：连续部署新入口、审计和固定配置 refit 已实现，详见
[DEPLOYMENT_RUNBOOK_20260906.md](DEPLOYMENT_RUNBOOK_20260906.md)。
尚未启动正式 A/B 实验；不能把原 first_phase 或 cv_continuation 当作下一阶段命令。
