# Causal H1 reward adapter v1

This is the next **reward acceptance** gate after the one-step H100 visual bridge.
It is not a production training command, a visual learner, or official full-episode
closed-loop PDMS. Contract name: `causal_h1_pdm_components_v1`. Freeze this contract
before performance/development/test runs; any change needs a new named condition.

## Inputs and exclusions

The scorer accepts detached current ego/actor states and actual executed history,
plus the registered nuPlan map and the route metadata present at reset. No logged
future trajectory, metric cache, candidate score, rare label or future spawn
schedule is accepted by the scoring API. Actor types and map origin are static
metadata. SimEngine itself retains the upstream traffic route/spawn priors: this
is a simulator assumption, not extra reward input.

Ego pose is the rear axle in nuPlan global coordinates. Velocity/acceleration use
SimEngine's rear-vehicle properties transformed to the body frame; steering and
angular derivatives are live values. Full Pacifica footprint and live actor boxes
are used, not point collisions. Dimensions/unknown types/nonfinite states fail.
The original converter is not used because it reads track dictionaries and sets
acceleration and steering to zero. Upstream physics and official evaluator are unchanged.

## Windows and definitions

- H1 is one **actual** 0.5-second SimEngine transition. NC and DAC apply the existing
  PDM classification/footprint routines independently to both endpoints and take
  the minimum. No substep/swept-collision guarantee is claimed. Collisions are not
  silently excluded because they occurred in earlier groups; the H1 window has
  empty prior-collision exclusions (same as a fresh PDM scoring window).
- TTC applies upstream `_calculate_ttc_optimized` to the **executed endpoint**,
  predicting both ego and current actors for 1 second at offsets 0, 0.5, 1 second
  using constant velocity/heading. It never reads logged future or another rollout.
  Predictions are for TTC only: they do not count as executed comfort/history or
  NC/DAC. The upstream 0.5-second direct-scoring call would perform no TTC checks,
  so an explicit endpoint sampling adapter is required.
- Comfort uses upstream `ego_is_comfortable` thresholds/filtering on 15 actual
  states at 0.5-second spacing (7 seconds trailing window). Missing/discontinuous
  history fails; no padding and no warm-start with logged future. At reset the
  probe performs 13 engineering warmup transitions, giving 14 states before the
  first H1 execution produces the 15th. These diagnostic actions are not training.
- Progress is center displacement along one map-derived route centerline shared
  by the canonical and every branch. Route/centerline come from upstream lane-graph
  helpers, not the logged ego trajectory. Reference is a 0.5-second constant-
  velocity/heading projection of the **pre-action ego state**, shared by all20.
  Upstream batch aggregation normalizes branch progress by max(reference, branch),
  with its unchanged 0.1m threshold. This is explicitly a causal CV reference, not
  the evaluator's expert future and not the best candidate in the group.
- Direction uses the same 15 actual states and upstream route-area thresholds.
  Lane keeping is also computed for aggregation compatibility. Their existing
  weights are zero. We do not add a direction multiplier or tune component weights.
- Aggregation calls the existing PDM routine unchanged: NC and DAC multipliers;
  progress:TTC:comfort weights 5:5:2. A reward tie is a measured no-signal group,
  not permission to invent a shaped score or take an AdamW step.

The window/reference/forecast differences are intentional and must stay visible
in experiment descriptions. Formal paired evaluation still uses the original
full evaluator. This contract is an engineering candidate for the online signal;
its usefulness on real generated candidates remains to be measured.

## State and acceptance

`RewardHistory.snapshot/restore` carries detached actual ego/actor history outside
`SnapshotCodec` (which continues to reject unadapted managers). `RewardSession.group`
executes and scores the canonical transition **before** branch replay, restores
exactly the same physics snapshot for all20 through the existing simulator, and
scores each endpoint against the same independent history snapshot/map/reference.
It checks selected state hash, every reward component, and the next history hash
against independently computed canonical results. Branches do not render or update
reward history. After success only the canonical history is committed. Failed
groups cannot be used for training.

The current probe is CPU-only and serial, with analytic candidates. It reports
all20 component ranges, variance, finite rewards, IDM fallbacks and NR/R parity.
There is no optimizer in this stage and no claim of visual/metric full-system
snapshot or production resume. Complete world/renderer/learner recovery is later work.

```bash
python projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py \
  --settings /absolute/private/online_runtime.json \
  --output /absolute/private/fresh_reward_report.json --steps 8 reward-probe
```

Expected acceptance: `PASS_CAUSAL_H1_REWARD_ADAPTER_ONLY`; 1 pinned engineering scene,
NR/R each 8 groups, 320 actual candidate branches, separate main rewards and no
IDM fallback. A pass proves the adapter gate, not real-generator reward quality,
wide scene coverage, log-disjointness, learner updates, DDP or formal readiness.

Next: repeat this gate on the target instance, then use the same adapter with live
generated candidates/history, audit signal, and connect the already-tested residual
learner and causal protocol. No-signal attempts must remain distinct from updates.
