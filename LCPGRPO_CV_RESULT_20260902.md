# LC-PGRPO 五折 CV 结果与下一步锁定

更新时间：2026-09-02  
状态：smoke 与正式 40-run CV 均完整结束；无方法 winner；未消费 development 或 certification。

## 1. 证据完整性

- Smoke 完成 1-step A3 plumbing，作用仅是验证训练、checkpoint 与评测链路。
- 正式实验完成 8 configurations × 5 folds = 40 次训练，以及
  4 checkpoints × 40 = 160 次 held-out fresh-noise 评测。
- 五折按 821 个 log 划分；训练 noise 为 0/1/2，held-out fresh noise
  为 9/10/11。
- 最终机器决定：
  STOP_LCPGRPO_OBJECTIVE_AND_RUN_TEMPORAL_INFORMATION_AUDIT。
- 权威结果：
  experiments/diffusiondrive/selector_lcpgrpo_v1/runs/lcpgrpo_v1_20260902a/cv/cv_gate.json。

## 2. 关键数值

| 配置 | Equal gain | Common | Rare | Log-bootstrap lower | V3-solved | Suppressed recovery | KL | Positive folds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 Direct, epoch 2 | +0.005134 | +0.004472 | +0.005796 | +0.002198 | -0.007436 | +0.009892 | 0.692653 | 5/5 |
| A0 Direct, epoch 4 | +0.005512 | +0.005596 | +0.005428 | +0.002029 | -0.009320 | +0.011889 | 0.949699 | 5/5 |
| A1 Lineage-Direct, epoch 4 | +0.006138 | +0.005943 | +0.006333 | +0.001636 | -0.008657 | +0.012106 | 0.843752 | 5/5 |
| A2 Per-draw proximal, delta 0.10 epoch 2 | +0.000333 | +0.000151 | +0.000514 | -0.000370 | -0.002557 | +0.000807 | 0.033695 | 3/5 |
| A3 Lineage-proximal, delta 0.10 epoch 2 | +0.000427 | +0.000551 | +0.000304 | -0.000419 | -0.002261 | +0.001202 | 0.033504 | 5/5 |

Direct 的结果不是无效：它在 common/rare、三个 fresh noise 和五个 folds
上都产生稳定总增益。但所有 Direct/Lineage-Direct checkpoint 都违反预注册
的 V3-solved >= -0.001 保留约束，因此不能把总分最高者事后宣布为新方法。

以 A0 epoch 2 为例：

- common/rare 选择改变率约为 15.1% / 11.7%；
- 在 unsolved 子集上 gain 为 +0.012689 / +0.017850；
- 在 V3-solved 子集上却为 -0.005330 / -0.009543；
- suppressed-recoverable gain 为 +0.009872 / +0.009912。

这说明同一个更新同时产生了真实纠错与真实遗忘。

## 3. 两个方法假设的判定

### 3.1 Lineage 均值没有增量价值

A1 相对同 epoch A0 的 paired-log 比较：

| Epoch | Mean | 95% CI |
|---:|---:|---:|
| 1 | -0.000249 | [-0.001428, +0.001106] |
| 2 | -0.000002 | [-0.001279, +0.001203] |
| 4 | -0.000471 | [-0.002000, +0.000955] |
| 8 | -0.000123 | [-0.001707, +0.001556] |

A3 相对匹配 A2 的 12 组比较也全部跨 0，paired mean 范围仅为
[-0.000655, +0.000443]。因此不能声称跨 draw lineage supervision
提高了 log-level transfer。

机制解释是：Direct 在每轮训练中本来就联合看到三个 draw；三者同 index
reward correlation 已高达 common/rare 0.921842/0.877522。leave-one-draw
均值主要重述 Direct 已见过的监督，而不是提供新的可辨识信息。

### 3.2 静态 proximal 只得到“少动”，没有得到“选择性地动”

Proximal 的 KL 降到约 0.02–0.05，说明二分求解和目标分布实现有效；
训练中约 46% 的集合被直接保留。但最佳 A2/A3 总增益仅约
+0.0003/+0.0004，置信下界为负，suppressed recovery 也只有约
+0.0008/+0.0012。

因此失败不是数值错误，而是机制限制：全局固定 KL budget 无法知道某个
fresh-noise 样本上 V3 是否已经正确，只能沿着“更新越强，纠错和遗忘都越大；
更新越弱，两者都越小”的 Pareto 曲线移动。

## 4. 当前真正的问题

本轮把问题从“GRPO 会追逐一次 diffusion noise 的 winner”修正为更精确的：

> 在冻结 generator 的随机 proposal set 上，scalar-GRPO 已能学到有用的
> 候选纠错；瓶颈是 selector 的 opportunity-retention 决策——仅凭当前帧
> 可观测量，它不能可靠判断何时应推翻强 V3 incumbent、何时应保留。

这和单纯增加 loss、调 rare/common 比例或增加一个通用 evaluator 不同。
此前 IVPS、hard-pair、DCSR 与 interaction probes 已分别表明：当前帧存在
候选质量信息，但 pairwise/structured verifier 不能稳定把它转换成安全
override。故不应再做新的 current-frame loss 网格。

## 5. 下一步：边界对齐的 proposal-history 信息审计

下一步不是直接训练“第二个 selector”，也不是迁移 AutoVLA。先用现成 train
cache 做零新增标签的信息审计：

1. 只用 V3 logits 锁定 top-5 与 incumbent，禁止 reward 参与 shortlist；
2. 主任务预测 challenger 是否以 scalar PDM 胜过 incumbent；
3. 比较 current-only、真实 immediate-predecessor proposal history、同
   stratum 跨 log shuffled history；
4. 五折 log-disjoint；前序 reward/outcome 永不作为输入；
5. 一般随机 candidate-pair AUC 仅作辅助，不决定授权。

只有主边界任务中真实 history 同时显著胜过 current-only 与 shuffled
history，且 fold/stratum 稳定，才值得设计 history-conditioned、
state-dependent conservative GRPO。届时方法应让 temporal proposal
consistency 调节“是否/多大幅度偏离 V3”，而 scalar PDM 仍是唯一训练 reward。

若只有一般 pairwise AUC 提升、边界任务不提升，则 history 不能解决本轮
瓶颈，停止该路线。若两个任务都不提升，则现有观测信息不足，应保留 V3，
寻找真正的新信息源，而不是继续调 loss。

固定运行入口：

    cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-rapg-v1
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      ./run_diffusiondrive_selector_proposal_history_audit_v1.sh history_boundary_20260902a

runner 会主动隔离 GPU 0；该审计只使用一张 H100/H200。
