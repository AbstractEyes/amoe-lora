"""The model fileset: trunk + trigram stem + arm heads + the validated
builder.

`build_model(bed, cfg)` is the supported constructor — it derives every
shape from the dataset and validates parity. `build_trunk` remains as the
low-level primitive it delegates to.
"""
from .build import READOUT_MAX, build_model
from .heads import (MODES, PatchHead, anchor_state, build_heads, set_adapters,
                    spec_for, super_fibonacci_s3)
from .routed_attention import AlephRoutedAttention, RoutedAttnConfig
from .stem import TRIGRAM_ORDER, VALID_CHANNELS, TrigramStem
from .trunk import (AlephRoutedBlock, LinearBlock, PatchStem2D, SquaredReLU,
                    TinyConfig, TinyTrunk, TrunkOutput, build_trunk)

__all__ = ["build_model", "READOUT_MAX", "MODES", "PatchHead", "anchor_state",
           "build_heads", "set_adapters", "spec_for", "super_fibonacci_s3",
           "TrigramStem", "TRIGRAM_ORDER", "VALID_CHANNELS", "TinyConfig",
           "TinyTrunk", "TrunkOutput", "LinearBlock", "SquaredReLU",
           "build_trunk", "AlephRoutedAttention", "RoutedAttnConfig",
           "AlephRoutedBlock", "PatchStem2D"]
