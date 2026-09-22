# Online simulation numeric runtime

Use `innovation3_runtime.py ... snapshot-probe` for the headless acceptance gate.
It sets `OPENBLAS_CORETYPE=Prescott` and BLAS/OMP threads to 1 **before** the
SimEngine subprocess imports NumPy. Spawned workers inherit these settings.
The same policy must be applied to both rendered main and branch processes when
that integration is implemented, including the frozen V3 paired baseline.
Do not set this variable after NumPy imports or assume an inherited shell value
is safe. Do not change the shared environment's installed packages for this fix.

In the installed NumPy 1.23.0 / OpenBLAS 0.3.20 runtime, explicitly selecting
Cooperlake reproduces the H100 failure from its saved prestate and action on our
local host, including the first failed canonical state's exact hash. Independent
40/64-dimensional SVD and pseudoinverse identities also fail grossly on that
path. Prescott and Haswell pass these checks; Prescott is the conservative pinned
reference for this stage. This is evidence about this installed runtime, not a
claim that all OpenBLAS builds or all Cooperlake CPUs are defective.

`worldengine.online.numerics.check_numeric_runtime` checks float32/float64 Gram,
SVD-reconstruction and pseudoinverse identities with non-BLAS oracle products.
It uses its own random generator. HeadlessSimulator calls it before initializing
an engine; snapshot-probe records its result before loading the scene dataset.
An unsafe direct invocation fails with a restart instruction. The tolerances
apply only to mathematical health checks: snapshot parity still requires exact
structural hash equality, including hidden controller and RNG state.

The fix changes process numeric dispatch, not LQR formulas, physical models,
actions, rewards or snapshot comparison. No clipping is added to conceal bad
control outputs. Prescott can sacrifice CPU linear-algebra throughput; correctness
comes first. Any faster kernel/library migration requires matched numeric-health,
full20 same-state, reverse-order, spawned-process and physical-reference checks
on target hardware before changing the shared protocol.

Target H100 acceptance, rendered-main parity, reward integration and real online
learning remain separate gates. A headless diagnostic pass is not an online
training or paper-results claim.
