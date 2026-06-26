#!/usr/bin/env python3
"""Rebuild colab/Lab22_DPO_T4.ipynb and Lab22_DPO_T4.ipynb from notebook sources."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO / "notebooks"
OUT_COLAB = REPO / "colab" / "Lab22_DPO_T4.ipynb"
OUT_ROOT = REPO / "Lab22_DPO_T4.ipynb"
COMPAT = NOTEBOOKS / "colab_compat.py"

STAGES = [
    "01_sft_mini.py",
    "02_preference_data.py",
    "03_dpo_train.py",
    "04_compare_and_eval.py",
    "05_merge_deploy_gguf.py",
    "06_benchmark.py",
]


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": _lines(text.rstrip() + "\n"),
        "outputs": [],
        "execution_count": None,
    }


def _lines(text: str) -> list[str]:
    return [line if line.endswith("\n") else line + "\n" for line in text.splitlines()]


def _split_jupytext(py_path: Path) -> list[dict]:
    """Use jupytext so markdown cells keep headings (fixes broken # display in Colab)."""
    import jupytext

    nb = jupytext.read(py_path)
    cells: list[dict] = []
    for cell in nb.cells:
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if cell.cell_type == "markdown":
            cells.append(_md(src.rstrip("\n")))
        elif cell.cell_type == "code":
            cells.append(_code(src))
    return cells


def _colab_header() -> list[dict]:
    return [
        _md(
            """# Lab 22 — DPO/ORPO Alignment (T4 tier)

**Track 3 · Day 22 · VinUni AICB program**

Single-file Colab notebook — **core graded path: NB1 → NB4** (~45 min T4).

| Stage | Required for 100 pts? | ~Time (T4) |
|-------|----------------------|------------|
| NB1 SFT-mini | **Yes** | ~10 min |
| NB2 Preference data | **Yes** | ~2 min |
| NB3 DPO train | **Yes** | ~15 min |
| NB4 Compare & eval | **Yes** | ~10 min |
| NB5 GGUF deploy | Bonus (+6 rigor) | ~15 min |
| NB6 Benchmark | Bonus (+8 rigor) | ~30–60 min |

**Tier:** `T4` — Qwen2.5-3B + 2k UltraFeedback + 1k VN Alpaca SFT

> **Before running:** Runtime → Change runtime type → **T4 GPU**.
> After NB4 you may **stop** if you only need the core grade; run NB5/NB6 for bonus."""
        ),
        _md("## A. Colab setup — install deps + workspace\n(Skip if running `make pipeline` from a local clone.)"),
        _code(
            """import os
os.environ["COMPUTE_TIER"] = "T4"
os.environ["XFORMERS_DISABLED"] = "1"
print(f"COMPUTE_TIER = {os.environ['COMPUTE_TIER']}")"""
        ),
        _code(
            """!pip install -q \\
  "unsloth>=2025.10,<2026.5" "trl>=0.12,<0.20" "peft>=0.13,<1.0" \\
  "bitsandbytes>=0.44,<1.0" "datasets>=3.1,<4.0" "accelerate>=1.1,<2.0" \\
  "llama-cpp-python>=0.3,<1.0" "lm-eval[ifeval,math]>=0.4.5,<1.0" \\
  "matplotlib>=3.9,<4.0" "pandas>=2.2,<3.0" "pyarrow>=17,<22" \\
  "openai>=1.55,<2.0" "anthropic>=0.40,<1.0"

# Text-only lab — torchcodec breaks on Colab PyTorch 2.10
!pip uninstall -y torchcodec

# T4 (sm_75): xformers FA backward needs sm_80+ — use PyTorch SDPA instead
!pip uninstall -y xformers

# Vietnamese glyphs in matplotlib tables (NB4 screenshot)
!apt-get -qq install -y fonts-noto-core > /dev/null"""
        ),
        _code(
            """import torch
assert torch.cuda.is_available(), "Runtime → Change runtime type → T4 GPU"
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name}  ({gpu.total_memory / 1e9:.1f} GB)")
# Screenshot for submission: save as submission/screenshots/01-setup-gpu.png"""
        ),
        _code(
            f'''from pathlib import Path
import os

WORK = Path("/content/lab22")
WORK.mkdir(exist_ok=True)
for sub in [
    "notebooks", "data/pref", "data/eval",
    "adapters/sft-mini", "adapters/dpo", "adapters/merged-fp16",
    "gguf", "submission/screenshots",
]:
    (WORK / sub).mkdir(parents=True, exist_ok=True)

# T4/Colab helpers (attention, chat template, dataset default)
(WORK / "notebooks" / "colab_compat.py").write_text(
    {json.dumps(COMPAT.read_text(encoding="utf-8"))},
    encoding="utf-8",
)
os.chdir(WORK / "notebooks")
print(f"Working dir: {{Path.cwd()}}")

from colab_compat import setup_matplotlib_vn, configure_t4_attention
setup_matplotlib_vn(refresh_font_cache=True)
configure_t4_attention()
import colab_compat as _cc
print(f"colab_compat OK — export_gguf_colab={{hasattr(_cc, 'export_gguf_colab')}}")'''
        ),
        _md(
            """---
## Core pipeline (NB1–NB4) — run in order
If you OOM on DPO, restart runtime and rerun from section A.
**Optional bonus:** continue to NB5 (GGUF +6) and NB6 (benchmark +8) after NB4.
---"""
        ),
    ]


def _stage_banner(nb_file: str) -> list[dict]:
    return [
        _md(f"---\n# ⏵ Stage: `notebooks/{nb_file}`\n---"),
    ]


def _core_stop_banner() -> dict:
    return _md(
        """---
## ✅ Core lab complete (NB1–NB4)

You have everything required for **100 core points** + `make verify` (except REFLECTION + screenshots).

**Next (optional bonus rigor):**
- **NB5** — GGUF merge + llama.cpp smoke (+6 pts)
- **NB6** — IFEval/GSM8K/MMLU/AlpacaEval-lite (+8 pts)
- **β-sweep** — see NB3 vibe-coding callout (+6 pts)

Fill `submission/REFLECTION.md` and push a **public** GitHub repo for LMS submission.
---"""
    )


def build() -> list[dict]:
    cells = _colab_header()
    for i, nb in enumerate(STAGES):
        cells.extend(_stage_banner(nb))
        cells.extend(_split_jupytext(NOTEBOOKS / nb))
        if nb == "04_compare_and_eval.py":
            cells.append(_core_stop_banner())
    return cells


def main() -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "cells": build(),
    }
    OUT_COLAB.parent.mkdir(parents=True, exist_ok=True)
    OUT_COLAB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy2(OUT_COLAB, OUT_ROOT)
    print(f"Wrote {OUT_COLAB} ({len(nb['cells'])} cells)")
    print(f"Wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
