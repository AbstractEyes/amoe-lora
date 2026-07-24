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
from .train import ARMS, DATASETS, DEFAULT_REPO, DIAL, DIMS_CLIMB
from .train import grid as _grid
from .train import publish as _publish
from .train import scratch as _scratch
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


def run_scratch(cfg: RunConfig | None = None, *, seeds=(0,), arms=ARMS,
                out: str | None = None, steps: int | None = None,
                bed: Bed | None = None) -> list[dict]:
    """ONLY the from-scratch co-training rows — trunk and head move together
    from step 0, no phase-0 pretrain. Re-run these on their own without
    redoing the pretrained dial.

    By default each cell trains for `pretrain_steps + steps` (the same total
    the pretrained dial's trunk saw) — pass `steps=` to override. This is the
    fix for scratch cells losing on compute rather than on the address.

        run_scratch(RunConfig(dataset="mnist", input_mode="patch",
                              steps=400, pretrain_steps=600))   # 1000 steps
    """
    return _scratch(seeds=seeds, arms=arms, base=cfg or RunConfig(), bed=bed,
                    ledger=out, steps=steps)


def smoke(device: str | None = None, out: str | None = None) -> list[dict]:
    """Shapes/parse only, synthetic data — never a result. Uses CUDA when
    present (device=None), CPU on a GPU-less runner."""
    return _smoke(device=device, ledger=out)


def publish(repo_id: str = DEFAULT_REPO, results_dir: str | None = None,
            preset: str | None = None, *, private: bool = True,
            dry_run: bool = False, **kw) -> dict:
    """Push a campaign's evidence (ledger, anchors, figures, config, model
    card) to a HuggingFace repo. Token from HF_TOKEN / Colab secret / cached
    login — never a hardcoded value. `dry_run=True` stages and returns the
    manifest without touching the network.

        from aleph_mnist import run_sweep, publish
        run_sweep(RunConfig(dataset="mnist", input_mode="patch"))
        publish()                       # -> AbstractPhil/geolip-amoe-classification
    """
    return _publish(repo_id, results_dir, preset, private=private,
                    dry_run=dry_run, **kw)
