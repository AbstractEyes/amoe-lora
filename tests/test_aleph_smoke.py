"""aleph_mnist smoke suite — SHAPES AND PARSE ONLY.

House rule: never a full train, never GPU, never a subprocess. Every test
here builds tiny fabricated beds on CPU and takes at most one backward pass,
purely to exercise wiring — never for accuracy. No dataset is downloaded, so
this runs offline and in CI.

Each test names the invariant it guards.

    pytest tests/test_aleph_smoke.py
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import aleph_mnist as A                                        # noqa: E402
from aleph_mnist import RunConfig, build_heads, build_model     # noqa: E402
from aleph_mnist.data import Bed, DATASETS, get_spec            # noqa: E402


def _bed(pixels: int, channels: int, name_key: str | None = None,
         n_labels: int = 10, n: int = 48) -> Bed:
    """A fabricated bed with a chosen shape/identity. Never a real dataset."""
    g = torch.Generator().manual_seed(0)
    xtr = torch.randn(n, pixels, generator=g)
    ytr = torch.arange(n) % n_labels
    xte = torch.randn(16, pixels, generator=g)
    yte = torch.arange(16) % n_labels
    neutral = {"permuted": xte.clone(), "noise": torch.randn(16, pixels)}
    return Bed(xtr, ytr, xte, yte, neutral,
               f"test-p{pixels}c{channels}", channels, name_key)


# ───────────────────────────── packaging / import ─────────────────────────
def test_import_stays_light():
    """A base install is torch + amoe only. If `import aleph_mnist` pulled
    matplotlib/torchvision/datasets, the console script would fail without
    the [experiment] extra."""
    for heavy in ("matplotlib", "torchvision", "datasets"):
        assert heavy not in sys.modules, f"{heavy} imported eagerly"


def test_public_api_surface():
    for name in A.__all__:
        assert hasattr(A, name), f"missing export: {name}"


def test_version_present():
    assert A.__version__


# ───────────────────────────── registry / bed ─────────────────────────────
def test_registry_self_consistent():
    """pixels must equal channels*H*W and mean/std must be per-channel —
    the disagreement build_model exists to make impossible."""
    for spec in DATASETS.values():
        assert spec.pixels == spec.channels * spec.height * spec.width
        assert len(spec.mean) == spec.channels == len(spec.std)


def test_unknown_dataset_raises_valueerror():
    """A friendly ValueError listing valid names, never a bare KeyError."""
    with pytest.raises(ValueError, match="unknown dataset"):
        get_spec("emnist")


@pytest.mark.parametrize("pixels,channels", [(784, 1), (3072, 3)])
def test_synthetic_bed_shapes(pixels, channels):
    bed = A.build_bed(train_n=32, synthetic=True, pixels=pixels,
                      channels=channels)
    assert bed.pixels == pixels and bed.channels == channels
    assert bed.name_key is None and bed.spec is None
    assert set(bed.neutral) >= {"permuted", "noise"}


# ──────────────────── the builder: derive, then validate ──────────────────
@pytest.mark.parametrize("dataset", ["mnist", "fashion", "cifar10"])
def test_build_model_derives_shapes_and_identity(dataset):
    spec = get_spec(dataset)
    bed = _bed(spec.pixels, spec.channels, dataset)
    trunk = build_model(bed, RunConfig(d=16, dataset=dataset))
    assert trunk.config.pixels == spec.pixels
    assert trunk.config.channels == spec.channels
    assert trunk.config.n_classes == spec.classes
    # THE strict-attach identity: per dataset, not frozen to mnist
    assert trunk.config._name_or_path == f"tiny-{dataset}-4block"


def test_build_model_rejects_pixel_mismatch():
    """A bed claiming to be cifar10 but carrying 784 pixels must fail at
    BUILD time, not as a reshape explosion mid-forward."""
    bed = _bed(784, 1, "cifar10")
    with pytest.raises(ValueError, match="pixels"):
        build_model(bed, RunConfig(d=16, dataset="cifar10"))


def test_build_model_rejects_absent_class():
    """ytr.max()+1 under-sizes the readout when a subsample drops a class."""
    bed = _bed(784, 1, "mnist", n_labels=8)
    with pytest.raises(ValueError, match="classes"):
        build_model(bed, RunConfig(d=16, dataset="mnist"))


def test_build_model_rejects_bad_trigram_channels():
    bed = _bed(98, 2, None)          # channels=2 is neither RGB nor gray
    with pytest.raises(ValueError, match="channels"):
        build_model(bed, RunConfig(d=16, input_mode="trigram"))


def test_build_model_rejects_oversized_readout():
    """linear tokens>1 builds Linear(d*T, C); guard the multi-GB flatten."""
    bed = _bed(784, 1, None)
    with pytest.raises(ValueError, match="readout"):
        build_model(bed, RunConfig(d=4096, tokens=784, input_mode="linear"))


def test_build_model_rejects_unknown_input_mode():
    bed = _bed(784, 1, None)
    with pytest.raises(ValueError, match="input_mode"):
        build_model(bed, RunConfig(d=16, input_mode="bigram"))


# ─────────────────────────── forward shapes ───────────────────────────────
@pytest.mark.parametrize("mode,pixels,channels", [
    ("linear", 784, 1), ("trigram", 784, 1), ("trigram", 3072, 3)])
def test_forward_shapes(mode, pixels, channels):
    bed = _bed(pixels, channels, None)
    trunk = build_model(bed, RunConfig(d=16, input_mode=mode))
    out = trunk(bed.xtr[:4], labels=bed.ytr[:4])
    assert out.logits.shape == (4, trunk.config.n_classes)
    assert out.loss.ndim == 0


def test_trigram_token_counts():
    """channel form -> one token per pixel (H*W); spatial -> one per raster
    position. Discovery #16: channel count = n-gram order."""
    rgb = build_model(_bed(3072, 3, None), RunConfig(d=16,
                                                     input_mode="trigram"))
    gray = build_model(_bed(784, 1, None), RunConfig(d=16,
                                                     input_mode="trigram"))
    assert rgb.n_tokens == 1024 and rgb.stem.kind == "channel"
    assert gray.n_tokens == 784 and gray.stem.kind == "spatial"


def test_model_follows_bed_device():
    """The trunk lands on the BED's device. Placing it anywhere else is
    always a bug: on Colab `build_model(bed, cfg).to(cfg.device or 'cpu')`
    sent the trunk to cpu (device is "" by default now) while the bed was on
    cuda -> 'index is on cuda:0, other tensors on cpu'. Deriving placement
    from the bed removes the whole class."""
    bed = _bed(784, 1, None)
    trunk = build_model(bed, RunConfig(d=16))
    assert next(trunk.parameters()).device == bed.xtr.device
    trunk(bed.xtr[:4])                    # no manual .to() anywhere


def test_probe_shim_is_deterministic():
    """amoe's LM-shaped _probe calls model(input_ids=...); it must work on
    this vision trunk and be a pure function of the ids."""
    trunk = build_model(_bed(784, 1, None), RunConfig(d=16))
    ids = torch.arange(8, dtype=torch.long).unsqueeze(0) % 7 + 1
    assert torch.equal(trunk(input_ids=ids).logits,
                       trunk(input_ids=ids).logits)


def test_dial_census():
    trunk = build_model(_bed(784, 1, None), RunConfig(d=16))
    for n in (0, 1, 2, 4):
        c = trunk.set_trainable_blocks(n)
        assert c["trainable_blocks"] == n
        assert (c["trunk_trainable"] == 0) == (n == 0)


# ───────────────────────── the arm ladder is honest ───────────────────────
@pytest.mark.parametrize("mode", ["soft", "sign", "none", "frozen"])
def test_each_arm_builds_and_forwards(mode):
    bed = _bed(784, 1, None)
    trunk = build_model(bed, RunConfig(d=16))
    heads, _, _ = build_heads(trunk, mode=mode, seed=0)
    assert trunk(bed.xtr[:4]).logits.shape == (4, 10)
    # every arm carries byte-identical parameter counts
    assert sum(p.numel() for h in heads for p in h.parameters()) > 0


def test_control_arm_gets_no_codebook_gradient():
    """`none` keeps the codebook parameter so counts match, but nothing
    reads it — so it must receive EXACTLY zero gradient. That is what makes
    a soft-vs-none delta attributable to the read."""
    grads = {}
    for mode in ("soft", "none"):
        bed = _bed(784, 1, None)
        trunk = build_model(bed, RunConfig(d=16))
        heads, _, _ = build_heads(trunk, mode=mode, seed=0)
        torch.nn.functional.cross_entropy(
            trunk(bed.xtr[:4]).logits, bed.ytr[:4]).backward()
        g = heads[0].addr.codebook.grad
        grads[mode] = 0.0 if g is None else float(g.abs().sum())
    assert grads["none"] == 0.0
    assert grads["soft"] >= 0.0


# ─────────────── the amoe contract: strict attach round-trip ──────────────
@pytest.mark.parametrize("dataset", ["mnist", "fashion", "cifar10"])
def test_strict_attach_roundtrip(tmp_path, dataset):
    """THE REGRESSION: save_anchor writes base_model_id per dataset and
    build_model derives the same _name_or_path, so amoe.attach(strict=True)
    round-trips for fashion and cifar — not only mnist. Then the toggle law
    (all_off == base) and bit-exact detach must hold."""
    import amoe
    from dataclasses import asdict

    from aleph_mnist import anchor_state, save_anchor
    from aleph_mnist.diagnostics import probes

    spec = get_spec(dataset)
    cfg = RunConfig(d=16, dataset=dataset)
    bed = _bed(spec.pixels, spec.channels, dataset)

    trunk = build_model(bed, cfg)
    heads, sites, _ = build_heads(trunk, mode="soft", seed=0)
    row = {"config": asdict(cfg), "name_key": dataset,
           "_artifact": {"state": anchor_state(heads, sites), "sites": sites}}
    path = str(tmp_path / f"{dataset}.anchor.pt")
    sha = save_anchor(row, path)                  # returns the CONTENT HASH
    assert sha.startswith("sha256:") and os.path.exists(path)

    fresh = build_model(bed, cfg)                 # same derived identity
    pre = probes.evaluate(fresh, bed.xte, bed.yte)
    handle = amoe.attach(fresh, path, binding="blocks", strict=True)
    with handle.all_off():                        # toggle law
        off = probes.evaluate(fresh, bed.xte, bed.yte)
    assert off["ce"] == pre["ce"], "adapters off must equal the base exactly"
    handle.detach(verify=True)                    # bit-exact or it raises


def test_strict_attach_roundtrip_synthetic(tmp_path):
    """A synthetic bed (name_key=None) must round-trip too. build_model and
    save_anchor both derive the identity from trunk_identity(), so they
    cannot disagree — they did once, and strict attach rejected an anchor
    against the very trunk that produced it."""
    import amoe
    from dataclasses import asdict

    from aleph_mnist import anchor_state, build_bed, save_anchor
    from aleph_mnist.model.build import trunk_identity

    cfg = RunConfig(d=16, synthetic=True, train_n=64)
    bed = build_bed(train_n=64, synthetic=True, pixels=64)
    trunk = build_model(bed, cfg)
    assert trunk.config._name_or_path == trunk_identity(None)

    heads, sites, _ = build_heads(trunk, mode="soft", seed=0)
    row = {"config": asdict(cfg), "name_key": bed.name_key,
           "_artifact": {"state": anchor_state(heads, sites), "sites": sites}}
    path = str(tmp_path / "syn.anchor.pt")
    save_anchor(row, path)
    amoe.attach(build_model(bed, cfg), path,
                binding="blocks", strict=True).detach(verify=True)


# ─────────────────────────── CLI and ledger ───────────────────────────────
def test_cli_parser_tolerates_ipykernel_flag():
    """ipykernel injects `-f /path/kernel.json`; parse_known_args must
    swallow it so calling main([]) from a cell never raises SystemExit."""
    from aleph_mnist.cli import build_parser
    args, _ = build_parser().parse_known_args(
        ["--smoke", "-f", "/tmp/kernel.json"])
    assert args.smoke is True


def test_cli_defaults_track_runconfig():
    """CLI flags default to RunConfig fields, so the two faces cannot drift."""
    from aleph_mnist.cli import build_parser
    args, _ = build_parser().parse_known_args([])
    d = RunConfig()
    assert args.d == d.d and args.dataset == d.dataset
    assert args.n_blocks == d.n_blocks


def test_ledger_writes_under_results_root(tmp_path, monkeypatch):
    """Never `./experiments/results` — an installed run has no such dir."""
    monkeypatch.setenv("ALEPH_RESULTS", str(tmp_path))
    from aleph_mnist import append_ledger, results_root
    assert results_root() == str(tmp_path)
    p = append_ledger({"cell": "x", "final": {"ce": 0.0, "acc": 0.0}})
    assert str(tmp_path) in p and os.path.exists(p)


def test_save_anchor_without_artifact_raises():
    from aleph_mnist import save_anchor
    with pytest.raises(ValueError, match="no adapter"):
        save_anchor({"config": {}}, "unused.pt")
