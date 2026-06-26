# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Merge + Deploy + GGUF  (OPTIONAL / BONUS)
#
# > **Optional (bonus).** Core lab = NB1--NB4. GGUF export builds llama.cpp at
# > runtime and is the most fragile step --- skip on free Colab T4 if short on time.
#
# **Stack:** Unsloth `save_pretrained_gguf(quantization='q4_k_m')` + llama-cpp-python smoke test.
# Maps to deck §7.1 lab brief: "merge adapter, quantize GGUF, serve với vLLM".
#
# > **Mục tiêu:** export the SFT+DPO adapter as a deployable GGUF Q4_K_M file
# > (~1.5 GB on 3B / ~4 GB on 7B), then smoke-test it through llama-cpp-python.
#
# > **Colab T4 note:** `save_pretrained_merged` / `merge_and_unload` + `save_pretrained`
# > often fail on Transformers 5.5 + 4-bit (`reverse_op NotImplementedError`).
# > Section 1 uses `export_gguf_colab()` which tries three load paths automatically.

# %% [markdown]
# ## 0. Setup
#
# > If `ImportError: export_gguf_colab` — re-run **Section A workspace cell** (writes
# > latest `colab_compat.py`). Section 1 also has an inline fallback.

# %%
import os
import json
import gc
from pathlib import Path

import torch

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = (
    "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4"
    else "unsloth/Qwen2.5-7B-bnb-4bit"
)
MAX_LEN = 512 if COMPUTE_TIER == "T4" else 1024

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
GGUF_DIR = REPO_ROOT / "gguf"
GGUF_DIR.mkdir(parents=True, exist_ok=True)

assert SFT_PATH.exists(), f"NB1 must run first — {SFT_PATH} missing"
assert DPO_PATH.exists(), f"NB3 must run first — {DPO_PATH} missing"
assert torch.cuda.is_available(), "Need CUDA GPU"

print(f"COMPUTE_TIER:  {COMPUTE_TIER}")
print(f"BASE_MODEL:    {BASE_MODEL}")
print(f"SFT adapter:   {SFT_PATH}")
print(f"DPO adapter:   {DPO_PATH}")
print(f"GGUF output:   {GGUF_DIR}")

# %% [markdown]
# ## 1. Load adapters + export GGUF Q4_K_M
#
# First llama.cpp compile on Colab takes ~3 min; quantize ~30 s after that.
# **Do not** call `save_pretrained_merged` on T4 — it hits Transformers 5.5 + 4-bit bugs.

# %%
try:
    from colab_compat import export_gguf_colab
except ImportError:
    # Stale colab_compat on Colab (Section A not re-run) — inline fallback
    from peft import PeftModel
    from unsloth import FastLanguageModel
    from colab_compat import configure_t4_attention, attn_implementation_for_gpu

    def export_gguf_colab(
        *,
        base_model: str,
        max_len: int,
        sft_path: Path,
        dpo_path: Path,
        gguf_dir: Path,
        quantization_method: str = "q4_k_m",
    ):
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
                str(gguf_dir), tokenizer, quantization_method=quantization_method
            )
            ggufs = sorted(gguf_dir.glob("*.gguf"))
            if not ggufs:
                raise RuntimeError("save_pretrained_gguf finished but no .gguf files found")
            print(f"✓ Wrote {len(ggufs)} GGUF file(s) to {gguf_dir}")
            for p in ggufs:
                print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
            return model, tokenizer

        model = tokenizer = None

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(**_load_kw(load_in_4bit=True))
            tokenizer = _prep_tokenizer(tokenizer)
            model = PeftModel.from_pretrained(model, str(dpo_path))
            return _export(model, tokenizer, "Method A — 4bit base + DPO adapter")
        except Exception as exc:
            errors.append(f"Method A: {exc}")
            _free(model, tokenizer)
            model = tokenizer = None

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
            "All GGUF export methods failed:\n"
            + "\n".join(f"  • {e}" for e in errors)
            + '\nTry: !pip install "transformers>=4.46,<5.0" --force-reinstall then restart runtime.'
        )

model, tokenizer = export_gguf_colab(
    base_model=BASE_MODEL,
    max_len=MAX_LEN,
    sft_path=SFT_PATH,
    dpo_path=DPO_PATH,
    gguf_dir=GGUF_DIR,
    quantization_method="q4_k_m",
)

# %% [markdown]
# ### 1a. Optional — additional quantization tiers (for the +3 rigor add-on)

# %%
# Uncomment for Q5_K_M + Q8_0 (~2× disk). Re-run export_gguf_colab or call save_pretrained_gguf again.
#
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q5_k_m")
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q8_0")

# %%
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
print("GPU memory cleared.")

# %% [markdown]
# ## 2. Smoke test with llama-cpp-python

# %%
from llama_cpp import Llama

gguf_files = (
    list(GGUF_DIR.glob("*Q4_K_M*.gguf"))
    + list(GGUF_DIR.glob("*q4_k_m*.gguf"))
    + list(GGUF_DIR.glob("*.gguf"))
)
assert gguf_files, f"No GGUF in {GGUF_DIR} — section 1 failed"
gguf_path = sorted(gguf_files, key=lambda p: p.stat().st_size)[-1]
print(f"Loading: {gguf_path.name} ({gguf_path.stat().st_size / 1e6:.1f} MB)")

llm = Llama(
    model_path=str(gguf_path),
    n_ctx=MAX_LEN,
    n_gpu_layers=-1,
    verbose=False,
)
print("Loaded.")

# %% [markdown]
# ### 2a. Smoke prompt + response (deliverable: `06-gguf-smoke.png`)

# %%
SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": SMOKE_PROMPT}],
    max_tokens=200,
    temperature=0.0,
)

print(f"PROMPT:\n  {SMOKE_PROMPT}\n")
print(f"RESPONSE (Q4_K_M GGUF, llama-cpp-python):\n  {response['choices'][0]['message']['content']}")
print(f"\nTokens used: {response['usage']}")

# %% [markdown]
# ## 3. Optional — vLLM serving (BigGPU only)
#
# vLLM provides production-grade OpenAI-compatible serving. **Requires CUDA GPU
# with ≥ 16 GB VRAM** and `vllm` installed (see `requirements-biggpu.txt`).
# On T4 tier this cell will OOM. Skip on T4.
#
# Run in a SEPARATE terminal (NOT in the notebook — vLLM blocks until killed):
#
# ```bash
# pip install vllm                         # once
# vllm serve adapters/merged-fp16 \
#   --port 8000 \
#   --max-model-len 1024 \
#   --gpu-memory-utilization 0.9
# ```
#
# For the lab, llama-cpp-python in step 2 is the graded artifact.

# %% [markdown]
# ## 4. Save deployment metadata

# %%
deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "gguf_path": str(gguf_path),
    "gguf_size_mb": round(gguf_path.stat().st_size / 1e6, 1),
    "quantization": "q4_k_m",
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_response": response["choices"][0]["message"]["content"],
}
eval_dir = REPO_ROOT / "data" / "eval"
eval_dir.mkdir(parents=True, exist_ok=True)
(eval_dir / "deploy_meta.json").write_text(
    json.dumps(deploy_meta, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"Saved {eval_dir / 'deploy_meta.json'}")

# %% [markdown]
# ## 5. Submission checklist
#
# Bạn vừa hoàn thành core lab. Trước khi submit:
#
# 1. **Run** `make verify` — gatekeeper sẽ list missing artifacts.
# 2. **Take screenshots** vào `submission/screenshots/`.
# 3. **Fill** `submission/REFLECTION.md`.
# 4. **(Optional)** Pick a rigor add-on từ rubric.md (β-sweep, HF push, GGUF release).
#
# Push public repo + paste URL vào VinUni LMS Day-22 box.
