"""The public interface — one config in, ledger rows out.

    from aleph_mnist import RunConfig, run_sweep, run_climb

    rows = run_sweep(RunConfig(dataset="mnist", d=64))          # the dial
    rows = run_climb(RunConfig(input_mode="trigram"), dims=(64, 256))

Everything the CLI can do is one of these three calls with the same
`RunConfig`, so the API and the CLI cannot drift apart. The lower-level
surface (`build_bed`, `build_model`, `build_heads`, `run`, `pretrain`,
`sweep`, `grid`, `save_anchor`) stays public for power users.
"""
from __future__ import annotations

from .config import RunConfig, resolve_device
from .data import Bed, build_bed
from .train import ARMS, DATASETS, DIAL, DIMS_CLIMB
from .train import grid as _grid
from .train import smoke as _smoke
from .train import sweep as _sweep


def make_bed(cfg: RunConfig | None = None) -> Bed:
    """Build the bed a config describes, already on the resolved device."""
    cfg = cfg or RunConfig()
    return build_bed(cfg.dataset, cfg.train_n, seed=cfg.seed, root=cfg.root,
                     synthetic=cfg.synthetic).to(resolve_device(cfg.device))


def run_sweep(cfg: RunConfig | None = None, *, seeds=(0, 1), dial=DIAL,
              arms=ARMS, out: str | None = None,
              include_scratch: bool = True, bed: Bed | None = None
              ) -> list[dict]:
    """One dial sweep on one dataset: arms x dial positions x seeds."""
    return _sweep(seeds=seeds, dial=dial, arms=arms, base=cfg or RunConfig(),
                  bed=bed, ledger=out, include_scratch=include_scratch)


def run_climb(cfg: RunConfig | None = None, *, datasets=DATASETS,
              dims=DIMS_CLIMB, seeds=(0, 1), out: str | None = None,
              include_scratch: bool = False) -> list[dict]:
    """The substrate climb: the dial walked up a width ladder across
    datasets, a fresh phase-0 trunk at every width."""
    return _grid(datasets=datasets, dims=dims, seeds=seeds,
                 base=cfg or RunConfig(), ledger=out,
                 include_scratch=include_scratch)


def smoke(device: str = "cpu", out: str | None = None) -> list[dict]:
    """Shapes/parse only, CPU, synthetic data — never a result."""
    return _smoke(device=device, ledger=out)
