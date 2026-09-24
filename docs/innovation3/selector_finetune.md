# Offline V3 → online selector continuation

Set `online_parameterization=v3_initialized_selector_finetune` in a separate
settings file. The default remains `frozen_v3_plus_zero_residual` for historical
reproduction. Results from these two modes are different experiments.

| Mode | Current score | Trainable module | Checkpoint |
| --- | --- | --- | --- |
| Legacy | `b + frozen_offline_selector + correction` | Complete correction, final layer initially zero | schema 2 |
| Continuation | `b + online_selector` | Complete selector initialized from offline V3 | schema 3 |

In continuation mode the frozen offline copy supplies only the KL reference
`b + offline_selector`. Perception, candidate generation and the original base
classifier remain frozen. The optimizer contains all selector parameters,
including encoders and attention. The method increases the probability of
selecting good candidates; it does not train the candidate generator.

Both modes preserve strict full20 causal H1 feedback, categorical T=1 training,
no AdamW step on globally tied rewards, and frozen argmax evaluation. Initial
no-grad logits/probabilities/argmax reproduce offline V3 exactly. The separate
autograd forward retains the existing 1e-5/1e-4 numerical guard.

Schema 3 saves the initialization file SHA (or explicit tensor fingerprint for
CPU fixtures), complete constructor config, selector/reference weights,
optimizer, private RNG backend/state, version and attempts. Restore stages
validation before changing the live learner. Different sources, architectures,
parameterizations, corrupt reference/optimizer/RNG and schema 1 are rejected.
CPU/CUDA RNG states are not interchangeable; a CUDA training resume needs CUDA.
Schema 2 keeps its historical fields and scoring arithmetic. Resident reuse
also checks model settings and initialization identity before temporal reset.

`online_probe`, `smoke`, `h1_protocol_probe`, `ddp_h1_probe` and `ddp_online_probe`
read the mode from settings. `innovation3_runtime.py` forwards those settings
unchanged. Reports distinguish the frozen input model and KL reference from the
trainable online selector. `frozen_generator_and_v3_unchanged` remains a legacy
compatibility field describing the input visual model's offline copy.

The initialized engineering manifest uses purpose
`distinct_scene_engineering_pilot_v3_initialized_selector` and binds both mode
and settings SHA. `ONLINE_SETTINGS` and `ONLINE_MANIFEST` override the H100
wrappers together. A new settings file cannot reuse a legacy manifest.

## Export

`innovation3.export_online_selector_v3` accepts only disposable strict-online
schema 3 checkpoints. It exports one ordinary `scene_selector_state`, validates
the offline file and frozen reference, and records the online method under
`online_export.method=scene_conditioned_causal_h1_online_finetune`. The outer
V3 schema/method is a loader compatibility field. Online LR/KL and T=1 are
recorded; offline epoch/seed are not relabeled as online training metadata.

Load the exported state into a single `SceneConditionedTrajectorySetSelector`
and add its output to `b`. Do not also add the offline selector. The ordinary
`materialize_grpo_selector_v3.py` preserves `online_export` and its disposable
markers in both the full model and manifest. The frozen-baseline audit remains
applicable. Unit tests exercise post-update logits/argmax equivalence, normal
materialization and unchanged baseline tensors.

## Acceptance boundary

CPU coverage includes nonzero offline final-layer fixtures, complete-selector
updates, no-signal skips, atomic failed restores, serialized optimizer/RNG replay,
resident/rebuild two-episode equivalence and real two-process Gloo mixed-signal
updates/global skips/mixed-mode rejection. A registered offline V3 CPU smoke
also passes. These are not real visual closed-loop or NCCL/H100 acceptance.

Next: target H100 preflight and learner smoke, then fresh-tag resident/rebuild
two-episode comparison, then staged two/eight-GPU acceptance. Long runs remain
pending. Parent residency does not imply persistent communication or workers.
A→B→A real scene switches, full simulator recovery, repeated efficiency trials,
train/development exclusions and a formal training budget remain separate gates.
The historical 16-episode resident run is legacy evidence; its rebuild counterpart
was interrupted and does not establish a complete efficiency comparison.
