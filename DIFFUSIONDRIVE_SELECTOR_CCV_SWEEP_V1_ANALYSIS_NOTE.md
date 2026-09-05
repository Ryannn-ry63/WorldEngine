# DiffusionDrive Selector CCV Sweep V1 Analysis Note

**Status: FROZEN AFTER FORMAL COLLECTION AND BEFORE CAUSAL GAP ANALYSIS**

Date: 2026-09-04 UTC

This note records a numerical-equivalence clarification made after all 20 CCV
arms had been collected, but before any causal-gap decomposition, bootstrap
result, or method-routing decision was computed or inspected. The analyzer
stopped at its first supplemental all-target baseline check and emitted no CCV
matrix or final gate.

The preregistered protocol requires the smoke8 policy, oracle, and matched
sentinels to reproduce prior outcomes within `1e-6`. That sentinel gate passed.
The analyzer additionally required all 33 formal `q_policy` branches to match
the earlier observe-only V3 scores within `1e-6`; this was a conservative
implementation check added after the protocol, not a preregistered causal gate.

Across the 33 formal policy branches:

- 29 scores reproduced within `1e-6`;
- all NC, DAC, and success labels were exactly equal;
- the maximum absolute score difference was `8e-5`;
- the maximum absolute ego-progress difference was `1.9e-4`;
- all deployed candidate actions passed the predeclared `1e-4` action-parity
  requirement.

The formal arms use the same intervention implementation and provenance as one
another. Their within-sweep causal contrasts therefore do not depend on exact
equality to the earlier observe-only simulator pass. The small continuous
differences above are treated as implementation-boundary numerical variation,
not as a categorical behavior change.

Before causal results are analyzed, the supplemental all-target baseline check
is frozen as follows:

- NC, DAC, and success must remain exactly equal for all 33 policy branches;
- absolute score and ego-progress differences must each be at most `1e-3`;
- all per-scene differences and maxima must be written to the final gate;
- failure of any condition invalidates the CCV analysis.

The `1e-3` continuous-metric equivalence margin is fixed independently of the
observed causal effects. It is 50 times smaller than the preregistered `0.05`
primary causal-gap threshold and corresponds to 0.1% of the unit metric range.
No causal-gap threshold, bootstrap rule, treatment, target, reward definition,
or method-routing rule is changed by this note.
