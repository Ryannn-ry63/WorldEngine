# Rare-first V3 selector research

独立分支：`research/frozen-selector-rare-v1`，基于 CFPI `26ab020`。
旧 CFPI 工作树、模型和负结果保持不变。这里只实现批准的固定实验，
不把 adapter / KL / replay 当作已经成立的顶会创新。

## 科学问题与本轮边界

问题是：固定生成器已有可行候选时，怎样让 selector 的闭环难例修正
在自然重规划中持续生效，同时不破坏已解决场景？
已有 pilot 的分支机会支持研究动机，但不能证明这些机会可预测、可迁移；
旧 B 的退化是需要保留的负证据。不能声称 GRPO 因为迁移到 diffusion 就数学失效。

本轮只用已有 pilot64 的20分支反馈；不新增全20候选分支采集。
先把信息来源、参数更新范围和 retention 控制拆开，建立能继续研究的强基线。
即使通过本轮效果门槛，也不等于已满足顶会的新颖性、泛化或安全论证要求。

## 固定设计

| 条件 | 修正标签 | 可训练部分 |
|---|---|---|
| local_anchor | local PDM | 冻结完整V3后的 256→128→1 残差，末层零初始化 |
| q_anchor | 既有分支闭环回报 | 同一残差，33,025参数 |
| q_full | 同一分支闭环回报 | 从V3初始化的完整 scene selector |

所有条件：种子0/1/2，T=1，AdamW lr=1e-4、wd=1e-4、500更新、梯度裁剪10。
每步16个 correction 状态＋16个 replay 状态；
loss = exact-group loss + 0.01 KL(πV3 || πcurrent)，只有step500可部署。
Q-Full与Q-Anchor同时改变容量和更新范围，不能据此把差异唯一归因于某个正则公式。
Q-MSE旧模型作为强效果对照；若之后声称本目标优于回归，还需matched-retention回归对照。

correction 固定原pilot64。replay从baseline1024中选V3成功且EP≥0.2的64场景，
SHA顺序、每原始日志最多2场景，取执行决策4..11共512状态。
排除pilot、全部B128、rare评测、已知legacy development/certification、CCV日志。
采样为均匀场景再均匀决策，不接入奖励、分支回报、日志或未来信息作为模型输入。

评测固定 rare development58／22日志、confirmation230／67日志，
common为既有B的matched_common64。rare必须重新跑两个V3基线，不能跨噪声协议复用旧rare结果。
common旧结果只在场景、模型、噪声、控制器和动作审计匹配后复用。
旧CSV的`overall_average`不是场景：289行实际代表288场景，不沿用旧错误分母。

这些rare划分是legacy-exposed，confirmation只是固定候选的后续验证，
**不是未见、盲测或未使用过的测试集**。不按新模型增益、oracle可救性重新筛评测场景。
主结果报告完整队列，同时列baseline失败、成功、低EP分层。

## 阶段和晋级

1. audit：只读来源核验＋新run内的派生数据/哈希清单，不启动训练或仿真。
2. preflight：8H100、扩展、冻结基线张量与真实缓存打分预检。
3. bridge：固定OOF q_grpo_t1 seed0/2、local_grpo_t1 seed2，在pilot64从决策4开始。
   与原A目标时刻接管配对比较；只检查初始公共状态，不要求策略分叉后仍复现旧轨迹。
   同时离线比较OOF/refit在同一缓存的分数、动作、所选标签。
   这是基于已知负结果选种子的posthoc诊断，不是三种子泛化效果。
4. screen：新3条件×3种子×(rare58＋common64)；另跑scalar/gate V3和旧Q-GRPO/Q-MSE三种子的rare58。
   共26个完整采集条件。缺条件或动作审计不全时禁止晋级。
5. confirm：仅一个已冻结方法的全部3种子＋两个V3基线，在230场景上评测；不重新拟合、不换赢家。
6. report：CPU报告在预算停止后仍可使用；工程PASS和效果决策分开。

新方法须同时：rare三种子均值PDM比scalar和gate各≥0.03，两种比较各至少2/3种子正增益；
rare成功率和EP不低于scalar；common PDM/SR/NC/DAC相对scalar均≥−0.01。
通过者按rare PDM、成功率、EP、较少参数、名字依次排序。
无赢家即停止，不读confirmation新结果。

先对种子求场景均值，再做原始日志cluster bootstrap：重采样日志簇，
统计量为场景加权均值（簇内总和／总场景数），另列等日志权重敏感性。
confirmation效果门槛未过记FAIL；门槛过但任一基线比较CI跨0记INSUFFICIENT_EVIDENCE。
这些是探索性、legacy-exposed证据，不是无人干预的独立认证。

## 预算与安全

新run累计上限64 GPUh：bridge含预检6、screen32、confirm22、共用失败/时间缓冲4。
缓冲只计算一次；所有正式子进程含CPU准备/报告都按整机8卡预留计费。
单独CPU audit/report不预留卡、不记GPUh；建议在拿GPU实例之前完成audit。
每个采集条件启动前按历史和本run耗时估算，超预算则停止，不缩场景、不缩种子、不重置账本。
被中断的条件归档到其自身attempt_archive，可恢复查看，下一次从该条件场景0重新跑；
已完整审计条件只核验并复用。只清理本runner创建的进程组，不执行全局ray stop/pkill。

首次真实软件审计的保守rollout预测：bridge4.60、screen38.12、confirm21.61 GPUh，
合计64.33，尚不含训练/预检。因此**不能保证完整方案装入64 GPUh**。
预测有余量、实际运行可能更快；程序仍执行固定边界，预算不足时必须停止。
扩预算、减少条件或改变设计需要另行决定，不能用新run ID绕过同一次研究预算。

## 运行

在本工作树根目录执行，每条命令独立完整，不需要Bash数组。
默认来源已指向已完成的CFPI continuous run，默认8卡/64 GPUh。
首次audit建议在CPU节点完成，预检及后续阶段要在8×H100实例运行：

```bash
bash run_diffusiondrive_selector_rare_8h100.sh audit rare_retention_20260906_v1
bash run_diffusiondrive_selector_rare_8h100.sh preflight rare_retention_20260906_v1
bash run_diffusiondrive_selector_rare_8h100.sh bridge rare_retention_20260906_v1
```

检查bridge完成与预算后，使用同一个run ID继续：

```bash
bash run_diffusiondrive_selector_rare_8h100.sh screen rare_retention_20260906_v1
bash run_diffusiondrive_selector_rare_8h100.sh report rare_retention_20260906_v1
bash run_diffusiondrive_selector_rare_8h100.sh confirm rare_retention_20260906_v1
```

confirm无完整赢家会返回NOT_AUTHORIZED，不会启动预检/评测。
阶段不会自动串联。失败时先看run/logs和decision_ledger.json，再用原命令原ID续跑。
代码、数据、超参的哈希契约不同会拒绝续跑；不应在正式run中途修改实现。

产物在`experiments/diffusiondrive/selector_rare_v1/runs/<ID>/`：
rare_inputs.json、replay512.pkl、baseline_tensor_audit.json、
train/*/step_500.pt、collections/*/collection_audit.json、
bridge/screen/confirm_report.json、winner.json、decision_ledger.json。

## 软件验证

```bash
/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python -m unittest discover -s research/rare/tests -v
/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python -m unittest discover -s research/cfpi/tests -p test_continuous_deployment.py -v
```

开发用`rare_software_audit_20260906_01`已完成旧来源的全量哈希核验和真实缓存CPU预检；
开发中又收紧了校验，该ID不是正式run，**不要拿它续跑GPU实验**。
它的审计得到replay可用279场景／99日志，最终选64场景／53日志。
真实64＋512缓存：零适配器V3分数完全相同；在线/离线最大差0；
teacher最大复算差6.8665e-5，低于1e-4，argmax无不一致。
这里只验证软件与数据，不构成新模型效果结论。
