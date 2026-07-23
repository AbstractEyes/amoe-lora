"""aleph_mnist — the co-training dial on a 4-block linear MNIST trunk.

Reads the boundary between two measured poles: the address-bottleneck
prior pays when trunk and head co-train, and costs on a frozen
substrate. See experiments/README.md for the preregistered forks.

    from aleph_mnist import RunConfig, sweep, build_bed
    rows = sweep(seeds=(0, 1))
"""
from .data import DATASET_PIXELS, Bed, build_bed
from .heads import MODES, PatchHead, build_heads, super_fibonacci_s3
from .runner import (ARMS, DATASETS, DIAL, DIMS_CLIMB, RunConfig,
                     append_ledger, grid, pretrain, run, save_anchor,
                     smoke, sweep)
from .trunk import TinyConfig, TinyTrunk, build_trunk

__all__ = ["Bed", "build_bed", "DATASET_PIXELS", "MODES", "PatchHead",
           "build_heads", "super_fibonacci_s3", "ARMS", "DATASETS", "DIAL",
           "DIMS_CLIMB", "RunConfig", "append_ledger", "grid", "pretrain",
           "run", "save_anchor", "smoke", "sweep", "TinyConfig",
           "TinyTrunk", "build_trunk"]
