"""DiffusionDrive-only IterBasedRunner compatibility adapter.

The shared WorldEngine training entry still validates ``total_epochs`` against
``cfg.runner.max_epochs`` before building a runner. Online selector GRPO is
iteration-based, so this adapter accepts the mirrored compatibility field but
only forwards ``max_iters`` to MMCV's IterBasedRunner.
"""

from mmcv.runner import IterBasedRunner
from mmcv.runner.builder import RUNNERS


@RUNNERS.register_module()
class DiffusionGRPOIterBasedRunner(IterBasedRunner):
    """Run by iteration while tolerating WorldEngine's epoch compatibility key."""

    def __init__(self, *args, max_iters=None, max_epochs=None, **kwargs):
        if max_iters is None:
            raise ValueError("DiffusionGRPOIterBasedRunner requires max_iters")
        if max_epochs is not None and int(max_epochs) != int(max_iters):
            raise ValueError(
                "compatibility max_epochs must mirror max_iters for "
                "DiffusionGRPOIterBasedRunner"
            )
        super().__init__(*args, max_iters=max_iters, **kwargs)
