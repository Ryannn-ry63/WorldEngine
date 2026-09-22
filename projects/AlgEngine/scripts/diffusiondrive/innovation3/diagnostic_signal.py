"""Explicit plumbing signal only. NOT PDM, driving reward, or training data.

Negative squared change in observed ego velocity creates a measured signal for
checking feedback/update causality. No collision/DAC/TTC/comfort names are used.
The formal online reward adapter must be implemented and accepted separately.
"""
import numpy as np

CONTRACT = "diagnostic_negative_velocity_change_squared_v1"


def transition_signal(before, after):
    values = [np.asarray(s["ego"]["velocity"], dtype=np.float64) for s in (before, after)]
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in values):
        raise ValueError("Missing/non-finite ego velocity in diagnostic feedback")
    result = -float(np.square(values[1] - values[0]).sum())
    if not np.isfinite(result):
        raise ValueError("Non-finite diagnostic transition signal")
    return result
