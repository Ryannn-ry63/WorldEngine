# PAF-GRPO V1: locked experiment and decision record

## Question

The paper question is whether direct transplantation of exact-group GRPO is a
poor selector objective after a diffusion generator has already produced and
scored a finite candidate set.  It is not whether another classifier,
structured label, or reward component can improve the selector.

For exact GRPO the logit ascent direction is
`u_i = pi_i * (A_i - E_pi[A])`.  A high-reward proposal with tiny current
selector probability can therefore receive an almost zero update.  This is the
probability-suppression hypothesis.  PAF-GRPO uses all already-observed scalar
rewards and conditions departure from frozen V3 on recoverable oracle headroom.
It adds no reward component, scene label, evaluator network, data type, or
inference-time module.

## Fixed method

- `pi0`: frozen V3 selector distribution.
- `A`: within-set population-zscore of official scalar PDM.
- `q = softmax(A)` over valid candidates.
- `b = argmax(pi0)` and `h = max(reward) - reward[b]`.
- `lambda = clip((h - 0.005) / (0.1 - 0.005), 0, 1)`.
- `target = (1 - lambda) * pi0 + lambda * q`.
- PAF policy loss: `-sum(target * log(pi_theta))`.
- Every arm uses `1e-3 * KL(pi_theta || pi0)`.

The four locked attribution arms are `direct_grpo`, `full_feedback`,
`opportunity_grpo`, and `paf_grpo`.  All start from V3 and use train caches
0/1/2, rare/common 50/50, AdamW, learning rate `3e-5`, batch size 64, 6339
outer groups per epoch, and 8 epochs.  Epochs 1/2/4 are diagnostic only; epoch
8 is the only search checkpoint.

## Stages and hard stops

1. Existing-cache mechanism audit on noise seeds 0/1/2.
2. Only after it passes, construct fresh-noise caches 9/10/11 for the same train
   tokens and repeat the mechanism audit.
3. Only after the fresh audit passes, train the four equal-budget arms and score
   them on fresh noise 9/10/11.
4. Only an authorized winner may proceed to optimizer-seed replicas and the
   untouched development split 3/4/5.
5. Certification and formal NAVSIM remain prohibited until method lock.

Mechanism authorization requires, in both common and rare strata: recoverable
fraction lower 95% log-bootstrap bound above 0.25; suppressed-oracle fraction
among recoverable views lower bound above 0.75; exact-GRPO oracle-vs-incumbent
boundary-positive fraction upper bound below 0.75; and mixed solved/missed token
fraction lower bound above 0.10.  Primary headroom is 0.005; 0 and 0.01 are
reported only as sensitivity checks.

Search efficacy requires equal-stratum PDM gain at least 0.002, nonnegative
common and rare gains, every fresh-noise seed at least -0.001, positive
paired-log bootstrap lower bound, suppressed-recoverable regret reduction at
least 0.005, and solved-subset gain at least -0.001.

Full PAF attribution additionally requires PAF to beat each comparator by at
least 0.001 with a positive paired-log bootstrap lower bound.  If full feedback
passes and beats direct GRPO while PAF adds no significant margin, the claim is
reduced to full-feedback decoupling.  If only opportunity GRPO wins, or no arm
passes, this objective direction stops and V3 is retained.

## Commands

From this worktree root, run the first gate:

```bash
./run_diffusiondrive_selector_paf_grpo_v1_8hopper.sh audit-existing
```

If it authorizes fresh noise, the remaining stages can run serially:

```bash
./run_diffusiondrive_selector_paf_grpo_v1_8hopper.sh fresh-and-search \
  experiments/diffusiondrive/selector_paf_grpo_v1/audit/AUDIT_ID/mechanism_gate.json
```

For one queued job from the beginning, run `all`.  Fresh caches can instead be
built on three allocations with `cache-fresh-seed {9|10|11} EXISTING_GATE`,
then `audit-fresh` and `search`.  A failed gate terminates the script and never
silently advances to development.
