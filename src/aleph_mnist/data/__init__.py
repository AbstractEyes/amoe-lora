"""The dataloader fileset: registry -> loaders -> bed, plus trigram binning.

Nothing here imports torchvision / datasets / numpy at module level; those
are pulled lazily inside the loaders, so `import aleph_mnist` stays light.
"""
from .bed import Bed, build_bed
from .binning import assert_channel_major, bin_to_bytes
from .loaders import balanced_subset, load
from .registry import (DATASET_CHANNELS, DATASET_CLASSES, DATASET_PIXELS,
                       DATASETS, DatasetSpec, get_spec)

__all__ = ["Bed", "build_bed", "assert_channel_major", "bin_to_bytes",
           "balanced_subset", "load", "DATASETS", "DatasetSpec", "get_spec",
           "DATASET_PIXELS", "DATASET_CLASSES", "DATASET_CHANNELS"]
