"""Publish a campaign's evidence to a HuggingFace repo.

`geolip-amoe-classification` — the `geolip` prefix marks this as the aleph /
geometric-addressing experimental line (same family as geolip-sd-trainer,
geolip-aleph-diffusion); `amoe-classification` says what the bed is.

Everything this bed produces IS the evidence — the JSONL ledger, the shipped
`*.anchor.pt` adapters, the figure set, the exact `RunConfig` — so publishing
means uploading *those*, not a black-box weights blob. The layout, per run:

    runs/{preset}-{timestamp}/
      ledger.jsonl        the cells (the evidence figures are derived from)
      anchors/*.pt        shipped amoe anchors (soft rows are stock anchors)
      figures/*.png       the rendered figure set, if matplotlib is present
      verdict.txt         the one-line-per-cell table
      config.json         the RunConfig the campaign ran
      README.md           model card (Apache-2.0) with the verdict inlined

TOKEN. Read from `HF_TOKEN` / `HUGGINGFACE_TOKEN`, Colab `userdata`, or the
cached `huggingface_hub` login — never hardcoded, never a function argument
that gets logged. Nothing here runs at import time.

The heavy imports (`huggingface_hub`, `matplotlib`) are lazy, so
`import aleph_mnist` stays light and this module only costs anything when you
actually call `publish`.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import asdict

from .ledger import results_root

DEFAULT_REPO = "AbstractPhil/geolip-amoe-classification"


# --------------------------------------------------------------- token
def hf_token(explicit: str | None = None) -> str | None:
    """Resolve a token WITHOUT ever putting it in a signature a caller might
    log: explicit arg (for tests) -> Colab userdata -> env -> cached login."""
    if explicit:
        return explicit
    try:                                   # Colab secret (key icon)
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if env:
        return env
    try:                                   # a prior `huggingface-cli login`
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _require_hub():
    try:
        import huggingface_hub as hh
        return hh
    except ImportError as e:                       # pragma: no cover
        raise ImportError(
            "publishing needs huggingface_hub. Install the extra:\n"
            "    pip install 'amoe-lora[hf]'\n"
            "(the notebook's [experiment,experiment-hf] install already "
            "brings it in via datasets)") from e


# --------------------------------------------------------------- staging
def _load_ledger(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _model_card(repo_id: str, preset: str, stamp: str, run_path: str,
                rows: list[dict], license: str) -> str:
    from ..diagnostics.plots import verdict_table
    from .. import __version__

    cfg = rows[0]["config"] if rows else {}
    n_cells = len(rows)
    datasets = sorted({r.get("name_key") or r["config"].get("dataset")
                       for r in rows if r})
    modes = sorted({r["config"]["mode"] for r in rows})
    table = verdict_table(rows) if rows else "(no cells)"
    return f"""---
license: {license}
tags:
- aleph
- amoe
- geolip
- classification
- geometric-addressing
library_name: amoe-lora
---

# {preset} — aleph co-training dial

Evidence from the `aleph_mnist` bed in
[amoe-lora](https://github.com/AbstractEyes/amoe-lora) (v{__version__}).
The RelayPatchwork adapter rides a 2D-patch trunk whose token mixer is the
**aleph router** (signed-projective addresses as a linear-attention kernel,
not softmax); the dial co-trains trunk and adapter and the `soft`/`sign`/
`none` arms are parameter-identical, so a delta is attributable to the read.

- run: `{stamp}`  ·  cells: {n_cells}  ·  datasets: {', '.join(datasets)}
- arms: {', '.join(modes)}  ·  input_mode: `{cfg.get('input_mode')}`
- d={cfg.get('d')}, blocks={cfg.get('n_blocks')}, patch_size={cfg.get('patch_size')}

## Verdict (one line per cell)

```
{table}
```

## Layout

```
{run_path}/
  ledger.jsonl   the cells — every figure is derived from this, never hand-typed
  anchors/       shipped amoe anchors (soft rows load with amoe.attach unmodified)
  figures/       the rendered figure set
  verdict.txt    the table above
  config.json    the RunConfig this campaign ran
```
"""


def _stage(staging: str, ledger_path: str, rows: list[dict],
           anchors: list[str] | None, make_figures: bool) -> dict:
    """Assemble the run folder. Returns a manifest of what was collected."""
    manifest: dict[str, list[str]] = {"ledger": [], "anchors": [],
                                      "figures": [], "meta": []}
    os.makedirs(staging, exist_ok=True)

    dst_ledger = os.path.join(staging, "ledger.jsonl")
    shutil.copy2(ledger_path, dst_ledger)
    manifest["ledger"].append("ledger.jsonl")

    # config.json — the RunConfig this campaign ran (from the first cell)
    if rows:
        with open(os.path.join(staging, "config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(rows[0]["config"], f, indent=2)
        manifest["meta"].append("config.json")

    # verdict.txt
    try:
        from ..diagnostics.plots import verdict_table
        with open(os.path.join(staging, "verdict.txt"), "w",
                  encoding="utf-8") as f:
            f.write(verdict_table(rows) + "\n")
        manifest["meta"].append("verdict.txt")
    except Exception as e:                             # pragma: no cover
        print(f"[publish] verdict table skipped ({e})", flush=True)

    # anchors
    adir = os.path.join(staging, "anchors")
    picks = anchors if anchors is not None else glob.glob(
        os.path.join(os.path.dirname(ledger_path) or ".", "**", "*.anchor.pt"),
        recursive=True)
    if picks:
        os.makedirs(adir, exist_ok=True)
        for a in picks:
            if os.path.exists(a):
                shutil.copy2(a, os.path.join(adir, os.path.basename(a)))
                manifest["anchors"].append(f"anchors/{os.path.basename(a)}")

    # figures (best-effort — never let a plotting failure block the upload)
    if make_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from ..diagnostics.plots import figure_set
            fdir = os.path.join(staging, "figures")
            os.makedirs(fdir, exist_ok=True)
            fig = figure_set(rows, path=os.path.join(fdir, "figure_set.png"))
            manifest["figures"].append("figures/figure_set.png")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"[publish] figures skipped ({e})", flush=True)

    return manifest


# --------------------------------------------------------------- publish
def publish(repo_id: str = DEFAULT_REPO, results_dir: str | None = None,
            preset: str | None = None, *, stamp: str | None = None,
            anchors: list[str] | None = None, private: bool = True,
            make_figures: bool = True, license: str = "apache-2.0",
            token: str | None = None, dry_run: bool = False) -> dict:
    """Upload one campaign's evidence to `repo_id` under a timestamped run.

    `results_dir` defaults to `results_root()` and must hold a `ledger.jsonl`.
    `preset` labels the run (defaults to the dataset+input_mode of cell 0).
    `private=True` by default — a results repo is private until you choose to
    share it. `dry_run=True` stages everything and returns the manifest
    WITHOUT touching the network (used by the smoke suite; also a safe
    'what would upload?' check).

    Returns a dict: {url, run_path, repo_id, manifest, uploaded}.
    """
    results_dir = results_dir or results_root()
    ledger_path = os.path.join(results_dir, "ledger.jsonl")
    if not os.path.exists(ledger_path):
        raise FileNotFoundError(
            f"no ledger at {ledger_path} — run a sweep first, or pass "
            "results_dir=... pointing at the run's output")
    rows = _load_ledger(ledger_path)

    if preset is None:
        c = rows[0]["config"] if rows else {}
        preset = f"{c.get('dataset', 'run')}-{c.get('input_mode', 'x')}"
    stamp = stamp or _timestamp()
    run_path = f"runs/{preset}-{stamp}"

    staging = os.path.join(
        os.environ.get("TEMP", results_dir), f"_hf_stage_{preset}_{stamp}")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        manifest = _stage(staging, ledger_path, rows, anchors, make_figures)
        with open(os.path.join(staging, "README.md"), "w",
                  encoding="utf-8") as f:
            f.write(_model_card(repo_id, preset, stamp, run_path, rows,
                                license))
        manifest["meta"].append("README.md")

        if dry_run:
            files = sorted(os.path.relpath(os.path.join(dp, fn), staging)
                           for dp, _, fns in os.walk(staging) for fn in fns)
            return {"url": None, "run_path": run_path, "repo_id": repo_id,
                    "manifest": manifest, "files": files, "uploaded": False,
                    "dry_run": True}

        hh = _require_hub()
        tok = hf_token(token)
        if not tok:
            raise ValueError(
                "no HF token found. Set HF_TOKEN (env), add it to Colab "
                "secrets, or `huggingface-cli login`. The token is never "
                "passed positionally or logged.")
        hh.create_repo(repo_id, token=tok, private=private, exist_ok=True)
        hh.upload_folder(
            folder_path=staging, repo_id=repo_id, path_in_repo=run_path,
            token=tok, commit_message=f"campaign {preset} @ {stamp}")
        url = f"https://huggingface.co/{repo_id}/tree/main/{run_path}"
        print(f"[publish] {url}", flush=True)
        return {"url": url, "run_path": run_path, "repo_id": repo_id,
                "manifest": manifest, "uploaded": True, "dry_run": False}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _timestamp() -> str:
    """UTC stamp for the run folder. Imported here (not at module top) so an
    `import aleph_mnist` never pays for it."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
