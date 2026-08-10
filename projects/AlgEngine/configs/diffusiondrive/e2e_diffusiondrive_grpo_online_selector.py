"""Deprecated compatibility alias for the formal selector-GRPO config.

New experiments must use e2e_diffusiondrive_grpo_selector.py directly. This
filename remains only so archived smoke/pilot commands still load for artifact
inspection; it no longer enables an iteration-limited training schedule.
"""

_base_ = ["./e2e_diffusiondrive_grpo_selector.py"]

deprecated_online_selector_config = True
