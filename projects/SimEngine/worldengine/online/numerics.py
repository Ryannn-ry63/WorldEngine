"""Fail closed on a broken numeric runtime before online simulation starts.

These are mathematical health checks, not tolerances on snapshot equality.
All main and branch processes must use the same pre-import BLAS settings.
"""
import numpy as np


def check_numeric_runtime():
    # Local RNG: the check must not advance the simulation's random streams.
    rng = np.random.RandomState(0)
    checks = []
    for dtype, tolerance in ((np.float32, 1e-4), (np.float64, 1e-10)):
        for size in (40, 64):
            a = rng.normal(size=(2 * size, size)).astype(dtype)
            gram = a.T @ a
            reference = np.einsum('ki,kj->ij', a, a, optimize=False)
            inverse = np.linalg.pinv(gram)
            u, singular, vh = np.linalg.svd(gram, full_matrices=False)
            # The oracle products deliberately avoid BLAS and its dispatch.
            errors = dict(
                gram_relative=float(np.max(np.abs(gram-reference)) / np.max(np.abs(reference))),
                pinv_identity=float(np.max(np.abs(
                    np.einsum('ik,kj->ij', inverse, gram, optimize=False) - np.eye(size)))),
                svd_relative=float(np.max(np.abs(
                    np.einsum('ik,k,kj->ij', u, singular, vh, optimize=False)-gram)) / np.max(np.abs(gram))))
            checks.append(dict(dtype=np.dtype(dtype).name, size=size, tolerance=tolerance, **errors))
            if not all(np.isfinite(v) and v <= tolerance for v in errors.values()):
                raise RuntimeError(
                    'Online numeric runtime self-check failed: ' + repr(checks[-1]) +
                    '. Restart with OPENBLAS_CORETYPE=Prescott set BEFORE importing NumPy '
                    '(innovation3_runtime.py configures this for snapshot-probe). '
                    'Do not weaken snapshot parity or train on these dynamics.')
    return dict(status='PASS', checks=checks)
