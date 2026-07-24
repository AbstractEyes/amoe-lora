"""aleph_mnist — the aleph co-training dial on a linear vision trunk.

Reads the boundary between two measured poles: the address-bottleneck prior
PAYS when trunk and head co-train (exp012) and COSTS on a frozen substrate
(exp013-A). See experiments/README.md for the preregistered forks.

    from aleph_mnist import RunConfig, run_sweep, run_climb
    rows = run_sweep(RunConfig(dataset="mnist", d=64))

Install (Colab, one line, no restart):
    pip install "amoe-lora[experiment] @ git+https://github.com/AbstractEyes/amoe-lora@experimental"

`import aleph_mnist` is deliberately LIGHT — it needs only torch + amoe.
torchvision / datasets / numpy are imported lazily by the loaders, and
matplotlib only inside `aleph_mnist.diagnostics.plots`.
"""
__version__ = "0.9.0"

from .api import (make_bed, publish, run_climb, run_conv, run_scratch,
                  run_sweep, smoke)
from .config import RunConfig, resolve_device
from .data import Bed, build_bed, get_spec
from .data import DATASET_CHANNELS, DATASET_CLASSES, DATASET_PIXELS
from .data import DATASETS as DATASET_SPECS      # the registry (name -> spec)
from .data import DatasetSpec
from .diagnostics import probes, vitals          # plots stays lazy on purpose
from .model import (MODES, PatchHead, TinyConfig, TinyTrunk, anchor_state,
                    build_heads, build_model, build_trunk, set_adapters,
                    super_fibonacci_s3)
from .train import (ARMS, DATASETS, DEFAULT_REPO, DIAL, DIMS_CLIMB,
                    append_ledger, grid, hf_token, ledger_path, pretrain,
                    results_root, run, save_anchor, sweep)

__all__ = [
    "__version__",
    # config + the public interface
    "RunConfig", "resolve_device", "run_sweep", "run_climb", "run_scratch",
    "run_conv", "smoke", "make_bed", "publish",
    # data
    "Bed", "build_bed", "DatasetSpec", "get_spec", "DATASET_SPECS",
    "DATASET_PIXELS", "DATASET_CLASSES", "DATASET_CHANNELS",
    # model
    "build_model", "build_trunk", "TinyTrunk", "TinyConfig", "PatchHead",
    "MODES", "build_heads", "set_adapters", "anchor_state",
    "super_fibonacci_s3",
    # train
    "run", "pretrain", "sweep", "grid", "save_anchor", "append_ledger",
    "ledger_path", "results_root", "hf_token", "DEFAULT_REPO",
    "DIAL", "ARMS", "DIMS_CLIMB", "DATASETS",
    # diagnostics
    "probes", "vitals",
]
