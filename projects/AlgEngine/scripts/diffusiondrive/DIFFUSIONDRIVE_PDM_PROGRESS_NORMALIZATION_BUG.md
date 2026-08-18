# DiffusionDrive PDM Progress 归一化顺序问题

## 状态

- 发现日期：2026-08-18
- 修复日期：2026-08-18
- 状态：已修复；CPU 单元回归已通过，真实候选 CPU/CUDA/official parity
  已纳入单 H100 和正式 8 H100 preflight
- 修复分支：`diffusiondrive-selector-grpo-v3-progress-normalization-fix`
- 影响模块：`mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py`
- 影响函数：`pairwise_official_scores`
- 范围：仅记录 PDM reward 的 progress 计算问题；GRPO 组内优势和 cached training 的 `B x 20` 批量逻辑不在此问题范围内。

## 问题摘要

当前实现先把 collision、drivable-area 等乘法指标组成的
`multiplicative` 乘到 raw progress 上，再对 PDM reference 和 candidate
做 pairwise progress 归一化。

NAVSIM PDM 的聚合顺序是：

1. 使用 reference 和 candidate 的 **raw progress** 计算归一化分母；
2. 得到 normalized progress；
3. 再将 candidate 的 `multiplicative` 施加到 normalized progress 上。

两种顺序在所有 multiplicative 都为 `1` 时等价；当 reference 或
candidate 的 multiplicative 小于 `1` 时可能产生不同 reward。当前顺序还可能让
candidate 的安全惩罚进入分母，并被归一化部分或完全抵消。

## 修复前实现

相关代码位于：

```text
projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/
diffusiondrive_online_pdm_reward.py:59-86
```

修复前逻辑可以简化为：

```python
multiplicative = multi.prod(axis=0)
gated_progress = progress_raw * multiplicative
reference_progress = gated_progress[0]

candidate_progress = gated_progress[proposal_index]
denominator = max(reference_progress, candidate_progress)
normalized_progress = candidate_progress / denominator
```

设：

- `p_ref` 为 reference 的 raw progress；
- `p_i` 为 candidate `i` 的 raw progress；
- `m_ref`、`m_i` 为相应 proposal 的 multiplicative；

则当前实现计算的是：

```text
progress_current(i) = (p_i * m_i) / max(p_ref * m_ref, p_i * m_i)
```

距离较小时还有 threshold fallback，但同样是在 gated progress 上判断。

## 预期的 official pairwise 语义

仓库内 PDM scorer 的聚合实现位于：

```text
projects/SimEngine/worldengine/components/agents/policy/pdm_planner/
scoring/pdm_scorer.py:176-195
```

它先使用 raw progress 归一化，随后再乘 multiplicative：

```text
denominator = max(p_ref, p_i)
normalized_raw(i) = p_i / denominator
progress_official(i) = normalized_raw(i) * m_i
```

当 `denominator` 不超过 `progress_distance_threshold` 时，
`normalized_raw(i)` 应取 `1.0`，之后仍然乘 `m_i`。

## 可复现反例

### 反例一：reference 被 multiplicative 门控

```text
reference: p_ref = 10, m_ref = 0
candidate: p_i   = 5,  m_i   = 1
```

预期 official 结果：

```text
progress_official = (5 / max(10, 5)) * 1 = 0.5
```

当前结果：

```text
gated_ref = 10 * 0 = 0
gated_i   = 5 * 1  = 5
progress_current = 5 / max(0, 5) = 1.0
```

candidate 只达到 reference 一半的 raw progress，却得到满分 progress。

### 反例二：candidate 有部分 multiplicative 惩罚

```text
reference: p_ref = 10, m_ref = 1
candidate: p_i   = 20, m_i   = 0.5
```

预期 official 结果：

```text
progress_official = (20 / max(10, 20)) * 0.5 = 0.5
```

当前结果：

```text
gated_ref = 10 * 1   = 10
gated_i   = 20 * 0.5 = 10
progress_current = 10 / max(10, 10) = 1.0
```

candidate 的 `0.5` 惩罚被归一化分母抵消。

## 影响

`progress` 是 PDM weighted metrics 的组成部分，因此错误可能进一步改变：

- `candidate_rewards`；
- `candidate_reward_components[..., ego_progress]`；
- 组内 reward 均值、标准差和 advantage；
- selector 最终学习到的候选排序。

下列样本不受该顺序问题影响：

- reference 和 candidate 的 multiplicative 均为 `1`；
- 或虽然中间 progress 不同，但 candidate 最终被零 multiplicative 完全门控，导致最终
  reward 均为零。

风险主要集中在 reference 或 candidate 的 multiplicative 不为 `1`，且差异能够传播到
最终 weighted score 的样本。

正式 cached training 不会重新计算 reward，而是直接读取 cache 中的
`candidate_rewards`。因此，如果 cache 已由当前实现生成，修复代码本身不会自动修复已有
cache，可能需要重新评分或重新生成。

## 已实施修复

`pairwise_official_scores` 已改为：

```python
multiplicative = multi.prod(axis=0)
reference_progress = progress_raw[0]

for candidate_index in range(num_candidates):
    proposal_index = candidate_index + 1
    candidate_progress = progress_raw[proposal_index]
    progress_denominator = max(reference_progress, candidate_progress)

    if progress_denominator > threshold:
        normalized_progress = candidate_progress / progress_denominator
    else:
        normalized_progress = 1.0

    normalized_progress *= multiplicative[proposal_index]
    weighted[WeightedMetricIndex.PROGRESS, proposal_index] = normalized_progress
```

最终 PDM score 外层已有的 multiplicative 聚合应保持不变，以匹配 scorer 的原始聚合
语义。

## 回归验证

以下测试已落地：

1. reference 和 candidate 的 multiplicative 都为 `1`，确认修复前后结果一致；
2. reference multiplicative 为 `0`，覆盖反例一；
3. candidate multiplicative 介于 `0` 和 `1`，覆盖反例二；
4. raw progress 不超过 threshold，确认先产生 `1.0` 再施加 candidate gate；
5. 将 20 个 candidates 打包计算的结果与 20 次独立
   `[reference, candidate]` official scorer 调用逐项对齐；
6. 分别验证 CPU `PDMSimulator` 和 CUDA `TorchSimulator` 路径；
7. 验证输出候选顺序以及 `(B, 20)`、`(B, 20, 6)`、`(B, 20)` 三个张量的对齐关系。

CPU 公式回归位于：

```text
projects/AlgEngine/scripts/tests/test_navsim_online_pdm_sampling.py
```

真实模型候选的独立 official、CPU `PDMSimulator` 和 CUDA
`TorchSimulator` 一致性验证位于：

```text
projects/AlgEngine/scripts/diffusiondrive/
validate_grpo_selector_pdm_progress_fix.py
```

该验证器要求实际样本中至少出现一个 multiplicative 不为 `1` 的候选，避免只在
“修复前后本来就等价”的 easy case 上误报 PASS。

## 已有 cache 的处理

修复前 V3 cache 中已经固化了错误 reward，不能继续训练。新的 cache manifest 会记录
reward、planning head、scene selector、NavFormer、cache extractor 和 helper 的 SHA256；
cache runner 复用前会逐项核对。旧 manifest 缺少这些 provenance 字段，因此会被自动
拒绝并重新生成。

修复实验使用独立目录：

```text
experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1
```

不会覆盖原始 V3 的 cache、训练 checkpoint 或正式评测结果。最终汇总会按相同 eval
seed 0/1/2 比较 epoch-100、V3 修复前和 V3 修复后三组结果。
