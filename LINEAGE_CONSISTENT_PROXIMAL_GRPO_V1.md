# Lineage-Consistent Proximal GRPO V1：锁定研究协议

更新时间：2026-09-02  
状态：四臂、五折 log-disjoint CV 已完成；LC-PGRPO 假设未通过归因与有效性 gate。完整结果见 LCPGRPO_CV_RESULT_20260902.md。

## 1. 论文主线与本轮问题

论文主线保持不变：**生成强，选择弱**。冻结 DiffusionDrive generator，并固定每个场景原有的 20 条候选，只通过 reward-driven selector post-training 改善最终 top-1 决策。

V3 已证明简单的 exact full-action group-relative objective 有效，但 Direct 进一步暴露出一个可复现的瓶颈：

- 同一批 train-scene token 换新 diffusion noise（seed 9/10/11）时，Direct 相对 V3 的 equal-stratum hard PDM gain 为 `+0.011472`；
- 换到 log-disjoint development（seed 3/4/5）时，equal-stratum gain 为 `-0.006298`；其中 common 为 `+0.003361`，rare 为 `-0.015958`；
- 历史 Direct 与本轮重现 Direct 的 54 个 selector state tensor 完全逐元素一致，因此不是复现漂移。

这说明需要解决的不是“GRPO 没把训练 reward 推得足够高”，而是：

> 逐 draw 的 scalar GRPO 会把概率推向一次 diffusion realization 中 reward 较高的候选，但一次 noisy realization 的局部极值并不一定代表可迁移的 plan quality；同时，不受控的大幅概率搬移可能破坏 V3 已经正确解决的场景。

本轮不引入 AutoVLA 的快慢思考范式，不增加人工工况标签，也不增加 reward component supervision。AutoVLA 只曾作为“额外评价机制可能有意义”的启发，不定义当前方法。

## 2. 为什么这是 diffusion-specific selector 问题

每个场景有 `D=3` 个独立 diffusion noise draw，每个 draw 有 `K=20` 条候选。候选索引并非三个 draw 间互不相关的离散 action：它由固定 plan anchor 派生，因而同一个 `k` 表示跨 draw 的 trajectory lineage。

全量机制审计得到：

| 机制统计 | Common | Rare |
|---|---:|---:|
| 同 index reward 的跨 draw pooled correlation | 0.921842 | 0.877522 |
| 用另外两个 draw 预测的 lineage 落入 held-out reward top-4 | 0.953340 | 0.925719 |
| tie-aware held-out top-1 | 0.878764 | 0.788785 |
| 预测 lineage 的 held-out mean PDM | 0.952005 | 0.870359 |
| held-out oracle mean PDM | 0.980643 | 0.935192 |

几何上，同 index 是另一 draw 最近候选的 pooled 比例为 `0.964289`，同 lineage 的轨迹 ADE 中位数为 `0.358966 m`。

### Top-k 定义校正

PDM 存在大量并列最优。以下两个方向不能混写：

1. 主统计：由另外两个 draw 预测出的 top-1 lineage，是否属于 held-out draw 的 reward top-4；
2. 反方向诊断：held-out draw 的第一个 `argmax` 索引，是否属于另外两个 draw 预测的 top-4。

主统计为 common/rare `0.953340/0.925719`；反方向诊断仅为 `0.856564/0.814962`。若强行要求命中 `torch.argmax` 返回的第一个并列索引，则为 `0.679295/0.601953`，这主要测量任意 tie-break，而不是 top-1 reward quality。审计保留了全部四种统计，没有删除不利结果。

最终不可覆盖的机制 gate：

```text
experiments/diffusiondrive/selector_lcpgrpo_v1/audit/
mechanism_v2_20260902/mechanism_gate.json
SHA256: f2a9a2e756727a3bb0de30862631ad40412787e281cf7d235c6ef0942c65bbf0
decision: AUTHORIZE_LCPGRPO_LOG_CV
```

该 gate 只确认问题与机制存在，没有训练或选择新方法。

## 3. LC-PGRPO 方法假设

### 3.1 Lineage-consistent leave-one-draw utility

对于 draw `d` 和 plan-anchor lineage `k`，不使用当前 draw 自己的 reward，而使用其余 draw：

```text
u[d,k] = mean_{d' != d} r[d',k]
A[d,k] = zscore_k(u[d,k])
```

这样训练输入仍是当前 draw 的候选特征，但监督方向来自同一 plan lineage 在独立 diffusion realization 中的重复表现。其目标是削弱 noise-specific winner，而不是简单做多 seed 数据增强。

### 3.2 V3-anchored proximal target

普通 Direct 直接最大化当前策略下的期望 advantage。Proximal 版本先围绕冻结 V3 构造目标分布：

```text
q[d,k] ∝ pi_V3[d,k] * exp(eta[d] * A[d,k])
KL(q[d] || pi_V3[d]) = delta
```

`eta[d]` 用逐 candidate-set、float64 二分求解。训练目标为：

```text
L = CE(q, pi_theta) + 1e-3 * KL(pi_theta || pi_V3)
```

若 V3 incumbent 到 lineage oracle 的 headroom `<=0.005`，reward 标准差 `<=1e-6`，或跨 draw lineage 不完整，则令 `q=pi_V3`，显式保留 V3，而不是让无信息集合产生任意梯度。

这两个机制分别回答：

- lineage：应该信任哪一种 candidate quality 估计；
- proximal：一次 post-training update 应该把概率移动多远，以及什么时候不应移动。

## 4. 四臂因果消融

四臂组成严格的 `2 × 2`，其余数据、初始化、模型、预算与评测完全一致。

| Arm | Utility | Update | 论文作用 |
|---|---|---|---|
| A0 `direct` | 当前 draw reward | 原 Direct exact GRPO | 精确代码路径基线 |
| A1 `lineage_direct` | leave-one-draw lineage utility | exact GRPO | 单独验证 lineage |
| A2 `per_draw_proximal` | 当前 draw reward | V3-anchored proximal | 单独验证更新约束 |
| A3 `lineage_proximal` | leave-one-draw lineage utility | V3-anchored proximal | 完整 LC-PGRPO |

A0 直接调用此前已审计的 Direct objective，不做“数学等价”的重新实现。A2/A3 的 `delta` 网格固定为 `{0.01, 0.03, 0.10}`。

只消费 official scalar PDM。禁止读取 `candidate_reward_components`；冻结 generator、perception、plan anchors、原 selector 与全部候选生成过程。

## 5. 数据、预算与模型选择

### 5.1 五折 log-disjoint CV

训练集合的完整 log 用下式固定分折：

```text
fold = int(sha256("20260902:" + log_name), 16) mod 5
```

每次用 4 folds 训练，在剩余 fold 的同 token、新 diffusion noise `9/10/11` 上评测。不能把同一 log 或 token 拆进不同 fold。

固定训练合同：

- V3 anchor：`d19c9d...d86d133`；
- 8 epochs，checkpoint `{1,2,4,8}`；
- 每 epoch 6,339 examples，rare/common 1:1；
- outer batch 64；AdamW，lr `3e-5`，weight decay `1e-4`；
- temperature `1.0`，KL weight `1e-3`；
- train noise `0/1/2`，CV evaluation noise `9/10/11`；
- 所有 arm 使用完全相同的 optimizer-step 与 view-example 预算。

总计 8 个 arm/configuration × 5 folds = 40 次训练；每次评测 4 个 checkpoint，共 160 个 held-out evaluation。runner 最多使用 8 张 H100/H200 并发调度，不在不同 arm 间复用挑选结果。

### 5.2 每个 configuration 的资格 gate

必须同时满足：

- equal-stratum hard gain `>=0.002`；
- paired-log bootstrap 95% lower `>0`；
- common gain `>=0` 且 rare gain `>=0`；
- 每个 noise seed 的 equal-stratum gain `>=-0.001`；
- V3-solved subset gain `>=-0.001`；
- V3-suppressed recoverable subset 的 oracle-regret reduction `>=0.005`；
- 5 folds 中至少 4 folds gain `>0`；
- mean `KL(pi_theta || pi_V3) <=0.9`。

### 5.3 归因与 winner 规则

- A1 必须相对同 epoch A0 达到 paired mean `>=0.001` 且 log-bootstrap lower `>0`；
- A3 同样必须胜过 matched A0；
- A3 若还以 `>=0.001`、lower `>0` 胜过最佳 A1，则选 A3；
- 若 A3 对 A1 没有达到该 margin，而 A1 通过归因 gate，则选 A1；
- 若只有 A2 通过，将其标为 proximal engineering baseline，停止 LC-PGRPO 论文 claim；
- 若 lineage arms 均不通过，不进入 development，转 proposal-history 信息审计。

最多只有一个 winner 可以进入 development。

## 6. Lineage 必需的负对照

若 A1 或 A3 被五折选中，必须用相同配置再做五折 `independent_shuffle`：每个 draw 内独立打乱 reward/valid 的 candidate index，同时保持每个 draw 的 reward multiset 不变。

真实 lineage 必须相对 shuffled lineage：

- paired mean PDM `>=0.001`；
- paired-log bootstrap lower `>0`。

否则不能声称收益来自 diffusion candidate lineage，停止并转 temporal 信息审计。

## 7. Development 与 certification

负对照通过后：

1. 只把已选 winner 在全部 train logs 上训练一次；
2. 只在 development seed `3/4/5` 上评测一次；
3. 使用第 5.2 节同一 efficacy gate，不得基于 development 改超参；
4. development 通过后，才允许一次 blind certification seed `6/7/8`；
5. certification 后禁止继续选择或回调方法。

Development 失败的结论是保留 V3，并分析信息瓶颈；不是围绕失败指标再加一个临时 loss。

## 8. 当前执行命令

在 8×H100/H200 实例中：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-rapg-v1

MECHANISM_GATE=experiments/diffusiondrive/selector_lcpgrpo_v1/audit/mechanism_v2_20260902/mechanism_gate.json
RUN_ID=lcpgrpo_v1_20260902a

# 先做一次 1-step A3 plumbing smoke；它不能用于方法选择。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_diffusiondrive_selector_lcpgrpo_v1.sh smoke "${MECHANISM_GATE}" "${RUN_ID}"

# smoke PASS 后再启动正式 40-run CV。

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_diffusiondrive_selector_lcpgrpo_v1.sh cv "${MECHANISM_GATE}" "${RUN_ID}"
```

CV 输出：

```text
experiments/diffusiondrive/selector_lcpgrpo_v1/runs/
lcpgrpo_v1_20260902a/cv/cv_gate.json
```

只有 `cv_gate.json` 的 decision 为 A1/A3 negative-control authorization 时才运行：

```bash
CV_GATE=experiments/diffusiondrive/selector_lcpgrpo_v1/runs/lcpgrpo_v1_20260902a/cv/cv_gate.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_diffusiondrive_selector_lcpgrpo_v1.sh control "${CV_GATE}" "${RUN_ID}"
```

Control 通过后：

```bash
CONTROL_GATE=experiments/diffusiondrive/selector_lcpgrpo_v1/runs/lcpgrpo_v1_20260902a/control/control_gate.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_diffusiondrive_selector_lcpgrpo_v1.sh development "${CONTROL_GATE}" "${RUN_ID}"
```

Development 明确授权后，才运行：

```bash
DEVELOPMENT_GATE=experiments/diffusiondrive/selector_lcpgrpo_v1/runs/lcpgrpo_v1_20260902a/development/development_gate.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_diffusiondrive_selector_lcpgrpo_v1.sh certification "${DEVELOPMENT_GATE}" "${RUN_ID}"
```

所有 stage 的根目录不可覆盖。若某个 `RUN_ID` 已存在，不删除旧证据，改用新的 run id。

## 9. 失败后的零新增标签后备路线

Proposal-history 不是与 LC-PGRPO 同时搜索的第五个 arm，只在以下任一事件后启动：

- 五折没有 lineage winner；
- real lineage 没有胜过 independent shuffle；
- 唯一 winner 在 development 失败。

信息审计只使用当前/前一时刻已经存在的 proposal bank、V3 logits/probabilities、V3 selected trajectory 和 ego pose；前序 PDM、reward component 或任何 outcome 都不能作为输入。当前 token 的 scalar PDM 只作为 probe label。

审计首先在 V3 logits 锁定的 top-5 内构造 challenger-vs-incumbent
边界任务；只有锁定候选后才读取 scalar PDM 作为 probe label。一般随机
candidate-pair 排序仅作为辅助诊断。两个任务都比较相同输入维度与训练
预算的三臂：

- current-only + zero history；
- real immediate-predecessor history；
- same-stratum、cross-log shuffled history。

只有主边界任务中 real 相对 shuffled 的 AUC mean >=0.001 且
paired-log lower >0，同时 real 相对 current mean >=0.01、lower
>0，且没有 fold/stratum 低于 -0.01 时，才授权构建 causal memory
selector。辅助随机 pairwise AUC 不参与授权。运行入口：

```bash
./run_diffusiondrive_selector_proposal_history_audit_v1.sh history_audit_20260902a
```

若该信息审计也失败，当前数据支持的结论是：单帧 selector 可观测信息不足，应该保留 V3 并重新寻找真正的新信息源；不能继续通过改变 common/rare 比例或堆结构化 reward supervision 刷分。

## 10. 论文归因边界

若最终 A1/A3 通过全部阶段，可以主张：

1. fixed-generator diffusion candidate selection 存在跨 noise 的 plan-lineage 结构；
2. 逐 draw Direct GRPO 的监督含有 realization-specific noise；
3. 利用重复 diffusion draw 的 lineage-consistent reward estimate 能提高 selector post-training 的 log-level transfer；
4. 若 A3 胜出，还可主张 V3-anchored KL-budgeted update 能缓解有效性与保留性冲突。

不能主张：增加了 generator 能力、增加了候选数、使用了新工况标签、依赖 reward components，或仅凭内部五折结果就已经得到最终闭环论文结论。最终仍需回到会议计划中的 OpenLoop-navtest、OpenLoop-rare、CL-NonReactive、CL-Reactive 与 Success Rate 三种子完整评测。
