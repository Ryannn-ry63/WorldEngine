# CCV-Sweep v1 Sentinel Amendment

Status: **FROZEN AFTER SENTINEL AND BEFORE FORMAL ARMS**
Date: 2026-09-04 UTC

## Scope

This amendment corrects a verifier condition exposed by sentinel calibration.
It does not change candidate treatments, target scenes, formal decision gates,
or any observed or unobserved 33x20 arm outcome.

## Sentinel evidence

All three frozen smoke8 treatments reproduced their prior causal-pilot runs:

- Eight of eight scenes were present in every treatment.
- Candidate trajectories and selector logits passed the `1e-5` context audit.
- Maximum action parity error was below `7.68e-6` versus the `1e-4` limit.
- All score, NC, DAC, EP, and success outcomes matched exactly (`0.0` error).

The only failed verifier term was a repeated computation of candidate reward
against the already frozen causal-pilot reward vector:

- Maximum scalar difference was `0.2580066919326782`.
- Five reward components matched exactly; only `ego_progress` changed.
- Frozen and recomputed reward-oracle indices agreed on 6 of 8 scenes.
- The changed reward did not select any sentinel treatment or deployed action.

## Interpretation

Candidate treatments are frozen indices and do not depend on rerun reward.
The formal analysis uses the immutable local reward vector from the original
causal-pilot records, where target reward repeatability was audited as zero. It
uses the new runs only to estimate closed-loop candidate causal value.

The rerun drift is therefore retained as evidence that PDM's replanned
reference-progress normalization is sensitive even when the candidate set and
behavior are fixed. It is diagnostic evidence, not behavior non-reproduction.

## Frozen decision impact

- The behavior sentinel gate remains outcome, success, context, and action parity.
- Reward recomputation exactness is reported but is not a sentinel gate.
- The 33x20 regret decomposition and all predeclared thresholds are unchanged.
- No causal-value result is authorized for training by this amendment.
- No formal CCV arm had been launched when this amendment was frozen.

This amendment and its SHA256 must be embedded in the passing sentinel gate and
validated again by the final analyzer.
