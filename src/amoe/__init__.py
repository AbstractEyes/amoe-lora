"""amoe — aleph mixture-of-experts adapters (train/attach/align/detach).

The productized adapter system from the geolip-aleph research line.
"LoRA-style" refers to the attach/detach usage pattern, not the math:
these are aleph-addressed patch heads, not low-rank matrices.
"""
from .core.adapter import AdapterSpec, RelayPatchwork, BlockWithAdapter
from .core.address import AlephAddress
from .core.dispatch import AnchorDispatch, BlockWithDispatch, set_mask
from .io.checkpoint import (AnchorCheckpoint, DispatchCheckpoint,
                            load_anchor, load_dispatch,
                            import_legacy_keys)
from .runtime.attach import attach, detach, AttachHandle
from .train.config import TrainConfig, AlignConfig
from .train.trainer import train
from .train.aligner import align
from . import laws

__version__ = "0.2.4"

# The diffusion subsystem (amoe.diffusion) is imported lazily — its verbs
# live under `import amoe.diffusion as ad`. Core is pure torch; diffusers
# is only touched by bindings/samplers/data that need it.
