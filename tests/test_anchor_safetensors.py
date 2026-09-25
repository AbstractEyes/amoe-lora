"""Block anchors (amoe.anchor): a .safetensors path gets a real safetensors
file, and load_anchor reads either format by content, from paths and from
open handles. Adds to amoe's own suite."""
import pytest
import torch

pytest.importorskip("safetensors")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from amoe.io.checkpoint import AnchorCheckpoint, load_anchor  # noqa: E402
from amoe.io.convert import convert_block_anchor  # noqa: E402
from amoe.io.safetensors_io import is_safetensors  # noqa: E402


def _anchor():
    g = torch.Generator().manual_seed(0)
    ad = {}
    for b in range(2):
        ad[f"{b}.addr.home"] = torch.randn(4, 8, generator=g)
        ad[f"{b}.consume.0.weight"] = torch.randn(8, 8, generator=g)
        ad[f"{b}.gate"] = torch.randn(1, generator=g).to(torch.bfloat16)
    meta = {"name": "s1_demo", "base_model_id": "demo/core@step10",
            "phase": "stage_1", "lambda": 2.0,
            "spec": {"n_slots": 16, "K": 16, "D": 8, "hidden": 256}}
    return AnchorCheckpoint(ad, meta)


def _torch_file_under_safetensors_name(ck, p, **meta_extra):
    """What amoe < 0.2.6 wrote for a path ending in .safetensors."""
    torch.save({"format": "amoe.anchor", "version": 1,
                "meta": dict(ck.meta, **meta_extra),
                "adapters": ck.adapters}, str(p))


def _same(a, b):
    assert set(a.adapters) == set(b.adapters)
    for k in a.adapters:
        assert a.adapters[k].dtype == b.adapters[k].dtype, k
        assert torch.equal(a.adapters[k], b.adapters[k]), k


def test_safetensors_path_writes_safetensors(tmp_path):
    ck = _anchor()
    p = tmp_path / "arm_step10.safetensors"
    h = ck.save(str(p))
    raw = p.read_bytes()
    assert raw[:2] != b"PK" and raw[8:9] == b"{"
    assert is_safetensors(str(p))
    tensors = load_file(str(p))            # any safetensors reader opens it
    assert set(tensors) == {f"blocks.{k}" for k in ck.adapters}
    with safe_open(str(p), framework="pt") as f:
        md = f.metadata()
    assert md["format"] == "amoe.anchor" and md["name"] == "s1_demo"
    back = load_anchor(str(p))
    _same(ck, back)
    assert back.meta["spec"] == ck.meta["spec"]
    assert back.meta["content_hash"] == h
    assert back.meta["content_hash_v2"].startswith("sha256v2:")


def test_content_hash_is_the_same_for_both_formats(tmp_path):
    ck = _anchor()
    assert ck.save(str(tmp_path / "a.pt")) == \
        ck.save(str(tmp_path / "a.safetensors"))


def test_pt_path_keeps_the_torch_format(tmp_path):
    ck = _anchor()
    p = tmp_path / "arm.pt"
    ck.save(str(p))
    assert p.read_bytes()[:2] == b"PK" and not is_safetensors(str(p))
    _same(ck, load_anchor(str(p)))


def test_torch_file_under_a_safetensors_name_still_loads(tmp_path):
    ck = _anchor()
    p = tmp_path / "old.safetensors"
    _torch_file_under_safetensors_name(ck, p)
    assert not is_safetensors(str(p))
    _same(ck, load_anchor(str(p)))


def test_open_handles_load_both_formats(tmp_path):
    ck = _anchor()
    old = tmp_path / "old.safetensors"
    _torch_file_under_safetensors_name(ck, old)
    for p in (old, tmp_path / "new.safetensors", tmp_path / "new.pt"):
        if not p.exists():
            ck.save(str(p))
        with open(p, "rb") as fh:
            assert is_safetensors(fh) == (p.name == "new.safetensors")
            assert fh.tell() == 0
            _same(ck, load_anchor(fh))


def test_convert_rewrites_in_place_and_keeps_the_meta(tmp_path):
    ck = _anchor()
    p = tmp_path / "old.safetensors"
    _torch_file_under_safetensors_name(ck, p, content_hash="sha256:kept")
    assert convert_block_anchor(str(p)) == str(p)
    assert is_safetensors(str(p))
    back = load_anchor(str(p))
    _same(ck, back)
    assert back.meta["content_hash"] == "sha256:kept"
    assert not list(tmp_path.glob("*.tmp"))
    assert convert_block_anchor(str(p)) == str(p)      # second call: no-op


def test_convert_a_pt_file_to_a_companion(tmp_path):
    ck = _anchor()
    p = tmp_path / "arm.pt"
    ck.save(str(p))
    out = convert_block_anchor(str(p))
    assert out == str(tmp_path / "arm.safetensors") and p.exists()
    _same(ck, load_anchor(out))


def test_a_changed_tensor_fails_the_hash(tmp_path):
    ck = _anchor()
    p = tmp_path / "t.safetensors"
    ck.save(str(p))
    with safe_open(str(p), framework="pt") as f:
        md = f.metadata()
        ts = {k: f.get_tensor(k) for k in f.keys()}
    ts["blocks.0.consume.0.weight"] = ts["blocks.0.consume.0.weight"] + 1
    save_file(ts, str(p), metadata=md)
    with pytest.raises(ValueError, match="content hash"):
        load_anchor(str(p))


def test_meta_that_is_not_json_is_refused_for_safetensors(tmp_path):
    ck = _anchor()
    ck.meta["tensor_in_meta"] = torch.zeros(1)
    with pytest.raises(ValueError, match="JSON"):
        ck.save(str(tmp_path / "bad.safetensors"))
    assert not (tmp_path / "bad.safetensors").exists()


def test_a_diffusion_file_is_not_a_block_anchor(tmp_path):
    p = tmp_path / "d.safetensors"
    save_file({"blocks.0.addr.home": torch.zeros(2)}, str(p),
              metadata={"format": "amoe.diffusion.anchor", "version": "1"})
    with pytest.raises(ValueError, match="not an amoe.anchor"):
        load_anchor(str(p))
