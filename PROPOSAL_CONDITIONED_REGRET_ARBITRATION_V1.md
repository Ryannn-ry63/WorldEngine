# Proposal-Conditioned Regret Arbitration V1

## Paper claim

DiffusionDrive already exhibits the target failure mode: candidate generation is strong, while deployed top-1 selection is weak. PCRA therefore does not introduce another general-purpose selector or another reward decomposition. It isolates the only downstream decision that matters:

> Given the reference incumbent and the single top-1 proposal exposed by a frozen strong generator-selector, should the system accept that proposal?

This restores the paper route to “generation strong, selection weak.” The contribution is a deployment-aligned arbitration objective, not a larger selector.

## Method

Let the frozen scalar-preference proposal choose incumbent b from the reference scores and proposal p from reference plus proposal residual. The arbiter observes only the realized pair (p, b), their frozen token/relation features, and two score gaps. The context arm additionally receives reference/proposal top-1 margins and normalized entropies.

The target uses only official scalar PDM:

- gain Δ = PDM(p) - PDM(b)
- target y = 1 when Δ > 0, otherwise 0
- main loss = mean over active samples of abs(Δ) times BCEWithLogits(evidence, y)

Active samples require p != b and abs(Δ) > 1e-8. The normalization is deliberately by active sample count, not by total regret weight: high-cost mistakes must retain larger gradients. The causal control replaces abs(Δ) with one.

Deployment accepts iff p != b and evidence > 0. The threshold is fixed at zero and is never tuned post hoc. At the population optimum, regret-weighted BCE with a zero threshold accepts exactly when conditional expected scalar gain is positive.

## Frozen provenance

The proposal is immutable:

- checkpoint: experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z/proposal_compute32/epoch_32_scene_selector.pt
- SHA256: 562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6
- training: relational trajectory-set reasoner, scalar preference weight 1.0, 32 epochs, train split, seed 0

PCRA loads this state strictly. Every proposal/encoder parameter is frozen; only regret_arbiter_head parameters are trainable. Training aborts if any frozen parameter changes.

## Fixed search

All arms use train split, seed 0, LR 1e-4, batch size 64, and final-epoch selection.

| Arm | Budget | Role | Promotable |
|---|---:|---|---|
| proposal32 | 32 epochs | frozen source proposal | baseline |
| proposal48 | 48 epochs | equal-total-compute proposal-only control | baseline |
| legacy-all | proposal32 + 16 epochs | candidate-wise IVPS, margin 0 | no |
| pair-sign | proposal32 + 16 epochs | actual-pair unweighted BCE | no |
| pair-regret | proposal32 + 16 epochs | actual-pair regret BCE | yes |
| pair-regret-context | proposal32 + 16 epochs | regret BCE plus four decision scalars | yes |

No margin, threshold, loss coefficient, epoch, or seed search is allowed. If both regret arms pass and their equal-weight development scores differ by at most 0.001, pair-regret is selected.

## Offline development gate

Evaluation is log-disjoint and uses equal weight over common, real-rare, and synthetic strata. A promotable regret arm must satisfy every condition:

1. equal-weight selected PDM is at least max(proposal32, proposal48) + 0.005;
2. common PDM is at least its own reference PDM minus 0.005;
3. real-rare and synthetic PDM are each within 0.020 of the best proposal arm on that stratum;
4. common degraded fraction is at most 0.10;
5. proposal SHA matches the immutable proposal32 and override threshold is exactly zero.

The evaluator additionally reports override coverage/precision, positive oracle-gain recovery, net accepted scalar gain, false-accept regret, and false-reject regret. These are diagnostics, not tunable gate thresholds.

If the offline gate fails, the PCRA route stops. It must not consume closed-loop or certification data.

## Closed-loop development gate

Only the arm locked by the offline gate may enter three-seed closed-loop development. For each seed, closed-loop score is the mean of non-reactive and reactive PDM. Relative to the stronger of proposal48 and CQR, the candidate must:

1. improve mean closed-loop score by at least 0.005;
2. be non-negative on at least two of three seeds;
3. have worst-seed delta at least -0.020;
4. preserve mean success rate within 0.020 of CQR.

Only a pass permits one certification run. Certification must reuse the same checkpoint, zero threshold, and offline gate without reselection.

## Commands

From the isolated worktree root:

~~~bash
./run_diffusiondrive_selector_pcra_v1_smoke.sh
./run_diffusiondrive_selector_pcra_v1_search_8hopper.sh
~~~

The search prints a root such as:

~~~text
experiments/diffusiondrive/selector_pcra_v1/search/search_YYYYMMDDTHHMMSSZ
~~~

Run log-disjoint development and the frozen offline gate with that exact root:

~~~bash
./run_diffusiondrive_selector_pcra_v1_development_8hopper.sh SEARCH_ROOT
~~~

The command exits nonzero when the offline gate fails. On a pass, development_gate.json contains the single locked arm and checkpoint.

After collecting proposal48, CQR, and locked-candidate three-seed metrics, apply the closed-loop gate:

~~~bash
projects/AlgEngine/scripts/diffusiondrive/select_pcra_closed_loop_development.py --offline-gate SEARCH_ROOT/development_gate.json --proposal48-baseline PROPOSAL48_METRICS.json --cqr-baseline CQR_METRICS.json --candidate LOCKED_LABEL=LOCKED_CANDIDATE_METRICS.json --output SEARCH_ROOT/closed_loop_development_gate.json
~~~

## Implementation status

As of 2026-08-29:

- model, realized-pair loss, frozen-proposal trainer, evaluator, materializer, search, smoke, offline gate, and closed-loop gate are implemented;
- focused PCRA tests pass;
- the combined RAPG/IVPS/PCRA regression suite passes;
- the real proposal32 checkpoint passes strict loading and arbiter-only trainability audit;
- the new Hopper search has not yet been launched, so experimental results remain pending.
