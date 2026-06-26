"""Colab / T4 compatibility helpers for Lab 22 (import from notebook cells)."""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

# HF dataset that replaced the removed 5CD-AI/Vietnamese-alpaca-cleaned hub entry.
DEFAULT_SFT_DATASET = "tsdocode/vi_alpaca_clean"

_VN_FONTPROP = None


def disable_torchcodec() -> None:
    """Lab is text-only; torchcodec breaks on Colab PyTorch 2.10."""
    os.environ["HF_DATASETS_DISABLE_TORCHCODEC"] = "1"


def _is_t4_or_older() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0)


def _disable_unsloth_xformers() -> None:
    """Unsloth still routes training through xformers on T4; FA backward needs sm_80+."""
    for name in ("unsloth.utils.attention_dispatch", "unsloth.models._utils"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        if hasattr(mod, "HAS_XFORMERS"):
            mod.HAS_XFORMERS = False
        if hasattr(mod, "xformers"):
            mod.xformers = None
        if hasattr(mod, "xformers_attention"):
            mod.xformers_attention = None


def configure_t4_attention() -> None:
    """Use SDPA/math attention on Turing T4 (sm_75) — xformers FA needs sm_80+."""
    import torch

    if not _is_t4_or_older():
        return
    major, minor = torch.cuda.get_device_capability()
    os.environ["XFORMERS_DISABLED"] = "1"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    _disable_unsloth_xformers()
    print(f"T4 attention: SDPA/math mode, xformers off (GPU sm_{major}{minor})")


def attn_implementation_for_gpu() -> str | None:
    """Return 'sdpa' on T4; None lets Unsloth pick on Ampere+."""
    if _is_t4_or_older():
        return "sdpa"
    return None


def apply_attn_config(model) -> None:
    impl = attn_implementation_for_gpu() or "sdpa"
    configs = []
    if hasattr(model, "config"):
        configs.append(model.config)
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "config"):
        configs.append(base.config)
    for cfg in configs:
        cfg._attn_implementation = impl
        if hasattr(cfg, "attn_implementation"):
            cfg.attn_implementation = impl
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = impl


def setup_matplotlib_vn(*, refresh_font_cache: bool = False):
    """Pick a font with Vietnamese glyphs (fixes □□□ in NB4 table plots)."""
    global _VN_FONTPROP
    import matplotlib
    import matplotlib.font_manager as fm

    if refresh_font_cache:
        fm._load_fontmanager(try_read_cache=False)

    available = {f.name for f in fm.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for cand in (
        "Noto Sans",
        "Noto Sans Display",
        "Arial Unicode MS",
        "Segoe UI",
        "Tahoma",
        "Liberation Sans",
    ):
        if cand in available:
            chosen = cand
            break
        partial = next((n for n in available if cand.lower() in n.lower()), None)
        if partial:
            chosen = partial
            break

    matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    _VN_FONTPROP = fm.FontProperties(family=chosen)
    print(f"matplotlib font: {chosen}")
    return _VN_FONTPROP


def get_vn_fontprop():
    return _VN_FONTPROP or setup_matplotlib_vn()


def style_matplotlib_table(table) -> None:
    """Apply VN-capable font to every cell in ax.table()."""
    prop = get_vn_fontprop()
    for cell in table.get_celld().values():
        cell.set_text_props(fontproperties=prop)


def render_side_by_side_table(rows: list[dict], screenshot_path: Path) -> None:
    """NB4 deliverable — readable VN text, wrapped cells, saved PNG."""
    import matplotlib.pyplot as plt

    setup_matplotlib_vn()

    def wrap_cell(text: str, width: int) -> str:
        return textwrap.fill(str(text), width=width, break_long_words=False)

    table_data = [["#", "Category", "Prompt", "SFT-only", "SFT+DPO"]]
    for r in rows:
        table_data.append([
            r["id"],
            r["category"],
            wrap_cell(r["prompt"], 42),
            wrap_cell(r["SFT-only"], 58),
            wrap_cell(r["SFT+DPO"], 58),
        ])

    n_rows = len(table_data)
    fig_h = max(6.0, 0.55 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="left",
        colWidths=[0.03, 0.08, 0.27, 0.31, 0.31],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    style_matplotlib_table(table)
    table.scale(1.0, 1.85)

    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor("#2e548a")
        table[(0, j)].set_text_props(color="white", weight="bold", fontproperties=get_vn_fontprop())
    for i in range(1, n_rows):
        if table_data[i][1] == "safety":
            table[(i, 1)].set_facecolor("#fce4e4")

    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(screenshot_path, dpi=140, bbox_inches="tight", pad_inches=0.25)
    plt.show()


def ensure_chat_template(tokenizer, ref_model: str = "Qwen/Qwen2.5-3B", *, adapter_path: Path | None = None):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer
    if adapter_path is not None:
        from transformers import AutoTokenizer

        try:
            saved = AutoTokenizer.from_pretrained(str(adapter_path))
            if getattr(saved, "chat_template", None):
                tokenizer.chat_template = saved.chat_template
                print(f"✓ chat_template copied from {adapter_path}")
                return tokenizer
        except OSError:
            pass
    from transformers import AutoTokenizer

    ref = AutoTokenizer.from_pretrained(ref_model)
    tokenizer.chat_template = ref.chat_template
    print(f"✓ chat_template copied from {ref_model}")
    return tokenizer


def export_gguf_colab(
    *,
    base_model: str,
    max_len: int,
    sft_path: Path,
    dpo_path: Path,
    gguf_dir: Path,
    quantization_method: str = "q4_k_m",
):
    """NB5 — try multiple load paths for Unsloth GGUF on T4.

    After merge_and_unload on a 4-bit base, Unsloth often fails to detect PEFT and
    Transformers 5.5 cannot reverse quantize (reverse_op NotImplementedError).
  """
    import gc

    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    configure_t4_attention()
    gguf_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    def _load_kw(*, load_in_4bit: bool, dtype=None) -> dict:
        kw = dict(
            model_name=base_model,
            max_seq_length=max_len,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        attn = attn_implementation_for_gpu()
        if attn:
            kw["attn_implementation"] = attn
        return kw

    def _prep_tokenizer(tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _free(model=None, tokenizer=None) -> None:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def _export(model, tokenizer, label: str):
        print(f"\n=== {label} ===")
        print(f"Model type: {type(model).__name__}")
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=quantization_method,
        )
        ggufs = sorted(gguf_dir.glob("*.gguf"))
        if not ggufs:
            raise RuntimeError("save_pretrained_gguf finished but no .gguf files found")
        print(f"✓ Wrote {len(ggufs)} GGUF file(s) to {gguf_dir}")
        for p in ggufs:
            print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
        return model, tokenizer

    model = tokenizer = None

    # A: DPO adapter on fresh 4-bit base (Unsloth PEFT path — same as NB4 SFT+DPO load)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**_load_kw(load_in_4bit=True))
        tokenizer = _prep_tokenizer(tokenizer)
        model = PeftModel.from_pretrained(model, str(dpo_path))
        return _export(model, tokenizer, "Method A — 4bit base + DPO adapter")
    except Exception as exc:
        errors.append(f"Method A: {exc}")
        _free(model, tokenizer)
        model = tokenizer = None

    # B: merge SFT into base, then DPO as single PeftModel
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**_load_kw(load_in_4bit=True))
        tokenizer = _prep_tokenizer(tokenizer)
        model = PeftModel.from_pretrained(model, str(sft_path))
        if hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False
        model = model.merge_and_unload()
        print("Merged SFT into base")
        model = PeftModel.from_pretrained(model, str(dpo_path))
        return _export(model, tokenizer, "Method B — merge SFT + DPO PeftModel")
    except Exception as exc:
        errors.append(f"Method B: {exc}")
        _free(model, tokenizer)
        model = tokenizer = None

    # C: FP16 full-precision load (fits ~3B on T4 16GB)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            **_load_kw(load_in_4bit=False, dtype=torch.float16)
        )
        tokenizer = _prep_tokenizer(tokenizer)
        model = PeftModel.from_pretrained(model, str(sft_path))
        model = model.merge_and_unload()
        print("Merged SFT into base (FP16)")
        model = PeftModel.from_pretrained(model, str(dpo_path))
        return _export(model, tokenizer, "Method C — FP16 + DPO PeftModel")
    except Exception as exc:
        errors.append(f"Method C: {exc}")
        _free(model, tokenizer)

    raise RuntimeError(
        "All GGUF export methods failed on this Colab stack:\n"
        + "\n".join(f"  • {e}" for e in errors)
        + "\n\nWorkarounds:\n"
        "  1. Restart runtime, pin Transformers 4.x BEFORE importing unsloth:\n"
        '     !pip install "transformers>=4.46,<5.0" --force-reinstall\n'
        "  2. Skip NB5 — core lab (NB1–NB4) is sufficient for 100 pts."
    )


def format_alpaca_to_chat(row, tokenizer=None) -> dict[str, str]:
    """Alpaca row → Qwen2.5 ChatML text (template or manual fallback)."""
    prompt = row.get("instruction") or ""
    if row.get("input"):
        prompt += "\n\n" + row["input"]
    output = row.get("output") or ""

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        if output:
            messages.append({"role": "assistant", "content": output})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    else:
        text = (
            f"<|im_start|>user\n{prompt}\n"
            f"<|im_start|>assistant\n{output}\n"
        )
    return {"text": text}
