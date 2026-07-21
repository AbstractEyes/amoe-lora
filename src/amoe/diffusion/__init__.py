"""amoe.diffusion — the aleph adapter verbs on diffusion trunks (0.2).

Same five-verb grammar as the text line, three certified deviations
(see .laws): the dtype law, structural banding instead of comparative
routing (align() is a grounded negative), and the declared multi-GPU
split (native train = single-GPU/DDP; production multi-GPU = the
diffusion-pipe fork).

Quickstart:
    import amoe.diffusion as ad
    h = ad.attach(pipe.unet, "mb3_s0.safetensors")
    img = ad.sample(pipe, h, "a lighthouse at dusk", seed=7)
    with h.lesion_band(2): ...
    base = h.detach()          # bit-exact or raises
"""
from ..io.checkpoint import (DiffusionAnchorCheckpoint,
                             load_diffusion_anchor)
from .core.cond import AlephCondAdapter
from .core.multiband import (BandBlockWrap, MultibandDelta, band_of,
                             band_weights)
from .core.relay import BlockWithRelay, RelayPatch2D
from .runtime.attach import DiffusionAttachHandle, attach, detach
from .runtime.sampler import StepGatedSampler, sample
from .train.aligner import align
from .train.config import DiffusionTrainConfig, RelaySpec
from .train.config import DiffusionTrainConfig as TrainConfig
from .train.trainer import ConditioningLawWarning, train
from . import laws
from .train import data
