# DiffusionDrive PDM progress normalization bug

## 结论

旧实现把 21 条 proposal（1 条 PDM reference + 20 条候选）的
`progress_raw` 先乘 collision / drivable-area multiplicative gate，再逐候选做
pairwise progress normalization。这个顺序不等价于 NAVSIM 官方对每条候选分别
调用一次 `pdm_score`。

官方顺序是：

1. 对 `[reference, candidate]` 的 **raw progress** 求 pairwise denominator；
2. 得到 candidate normalized progress；
3. 再把 candidate multiplicative gate 施加到 normalized progress；
4. 计算 weighted score，并在最外层再次施加 multiplicative gate。

修复后的核心逻辑是：

```python
reference_progress = progress_raw[0]
candidate_progress = progress_raw[proposal_index]
denominator = max(reference_progress, candidate_progress)
normalized_progress = (
    candidate_progress / denominator
    if denominator > progress_distance_threshold
    else 1.0
)
normalized_progress *= multiplicative[proposal_index]
```

## 旧逻辑为什么可能让分数看起来更好

旧逻辑先 gate：

```python
gated_progress = progress_raw * multiplicative
denominator = max(gated_reference_progress, gated_candidate_progress)
```

因此 gate 不只影响 candidate 最终得分，还错误地改变了 progress 的归一化
分母。某些 reference 或 candidate 被 gate 后，短但快、或安全 gate 异常的轨迹
可能得到不符合官方独立评分的相对 progress。selector 随后学习的是这个有偏的
排序目标；修复后离线/闭环指标下降并不说明修复错误，而说明旧目标中的偏差曾经
偶然有利于某些评测指标。

## 影响范围

- 已经生成的候选轨迹和候选特征没有错；
- 历史 cache 中的 `candidate_rewards`、reward components 受影响；
- 用历史错误 reward 训练出的 V2/V3 selector 不能被称为“修复后模型”；
- checkpoint 的正式 NAVSIM open-loop / closed-loop 评测本身仍是官方评分，
  但仅重新评测旧 checkpoint 无法验证训练目标修复。

因此 V2 的正确复现实验必须是：固定历史 candidates/features，只重算 reward，
重新做 calibration sweep，重新训练三个 seed，再做四块正式评测。

## 验证

- 单元回归覆盖 reference gate、candidate gate、threshold fallback 和 20 候选顺序；
- `validate_grpo_selector_pdm_progress_fix.py` 用真实候选比较：
  - CPU corrected batched reward；
  - CUDA corrected batched reward；
  - 每候选独立官方 `pdm_score`；
- 容差固定为 `1e-5`；
- corrected cache merge 会证明 trajectories/features/logits/baseline selector
  bit-identical，只允许 reward、components、valid mask 被替换。
