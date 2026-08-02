# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Overview

This is the official **Qwen3-VL** repository — the vision-language (multimodal) model family from the Qwen team. It is not a single application but a collection of loosely-coupled components: a preprocessing utility package, a fine-tuning framework, per-benchmark evaluation harnesses, a Gradio web demo, and inference cookbooks. Each component has its own dependencies and is used independently.

Model variants: `Qwen/Qwen3-VL-{4B,8B,30B-A3B,32B,235B-A22B}-{Instruct,Thinking}` (dense and MoE; the `A` in the name, e.g. `30B-A3B`, denotes a MoE model with active params). Also supports the older Qwen2-VL and Qwen2.5-VL families in the training framework.

## Repository Layout

- **`qwen-vl-utils/`** — Standalone pip package (`qwen-vl-utils`, published to PyPI). The single important entry point is `process_vision_info()` in `src/qwen_vl_utils/vision_process.py`, which resolves images/videos from file paths, URLs, base64, or PIL objects into the tensors the processor expects. This is the shared preprocessing layer used by the web demo, cookbooks, and any inference code.
- **`qwen-vl-finetune/`** — SFT/LoRA training framework (see Architecture below).
- **`evaluation/`** — Independent per-benchmark harnesses (MMMU, MathVision, ODinW-13, RealWorldQA, etc.). Each subdir has the same layout: `run_<bench>.py` with `infer`/`eval` subcommands, `infer_{instruct,think}.sh`, `eval_{instruct,think}.sh`, and its own `requirements.txt`. Inference uses vLLM; evaluation of some benchmarks calls an external judge model API.
- **`cookbooks/`** — Standalone Jupyter notebooks demonstrating capabilities (grounding, OCR, computer/mobile agents, video understanding, spatial reasoning). Reference material, not importable code.
- **`web_demo_mm.py`** — Gradio multimodal chat demo, supports `transformers` or `vllm` backends.
- **`docker/`** — `Dockerfile-qwen3vl-cu128` and `docker_web_demo.sh` (pulls `qwenllm/qwenvl:qwen3vl-cu128`, serves the web demo).
- **`test_qwen3vl_4b.py`** — Local offline vLLM inference smoke test (not part of upstream; see Local Environment Notes).

Note the components pin **different, incompatible dependency versions** (e.g. finetune wants `transformers==4.57.0.dev0`, web demo pins a specific transformers git commit, deployment wants `transformers>=4.57.0` + `vllm>=0.11.0`). Do not assume one environment works for all components — treat each as a separate setup.

## Common Commands

### qwen-vl-utils (the only component with lint/build tooling)
```bash
cd qwen-vl-utils
rye sync                    # or: pip install -e . (rye-managed project)
ruff check src/             # lint (config in pyproject.toml; line-length 119)
ruff format src/            # format (double quotes, space indent)
```
There is no test suite in this repo — do not look for `pytest`/`tox`. Validation is done via the cookbooks and the offline inference examples in the README.

### Fine-tuning
Training is launched via `torchrun` through bash wrapper scripts in `qwen-vl-finetune/scripts/`. Always run from the `qwen-vl-finetune/` directory (the entry point appends the project root to `sys.path`).
```bash
cd qwen-vl-finetune
# Pick the script matching the target model; e.g. Qwen3-VL-4B:
NPROC_PER_NODE=<num_gpus> bash scripts/sft_qwen3_4b.sh
# Others: sft_7b.sh, sft_32b.sh, sft_30a3b.sh, sft_30a3b_lora.sh
```
The entry point is `qwenvl/train/train_qwen.py`; scripts pass HF-style flags to it. DeepSpeed configs live in `scripts/{zero2,zero3,zero3_offload}.json`.

Data prep helpers:
```bash
python tools/check_image.py   # verify dataset media completeness
python tools/pack_data.py      # required before using --data_packing True
```

### Evaluation (per-benchmark, e.g. MMMU)
```bash
cd evaluation/mmmu
pip install -r requirements.txt
bash infer_instruct.sh   # runs run_mmmu.py infer via vLLM -> predictions.jsonl
bash eval_instruct.sh    # runs run_mmmu.py eval (may call an external judge model API)
# Use *_think.sh variants for Thinking-model checkpoints.
```
Edit the `--model-path`/`--data-dir`/`--output-file` placeholders inside the `.sh` scripts before running.

### Deployment / Serving
```bash
pip install accelerate qwen-vl-utils==0.0.14
uv pip install -U vllm          # requires vllm>=0.11.0 for Qwen3-VL support

# OpenAI-compatible server (adjust flags per GPU; the README example targets H100/H200):
vllm serve Qwen/Qwen3-VL-4B-Instruct --host 0.0.0.0 --port 22002
```

### Web Demo
```bash
pip install -r requirements_web_demo.txt
python web_demo_mm.py -c Qwen/Qwen3-VL-4B-Instruct --backend {transformers|vllm} \
    --server-port 7860 --server-name 127.0.0.1 [--flash-attn2] [--gpu-memory-utilization 0.9]
```

## Model Architecture

Qwen3-VL is a three-stage multimodal model. Pixels flow through a Vision Transformer, get projected into the LLM embedding space by a merger, and are interleaved with text tokens for the language model. The module names below are the exact attribute paths used by the HF model classes (verified in `qwenvl/train/train_qwen.py::set_model`), which is what the fine-tuning freeze flags target.

```
   image / video                      text prompt
        │                                  │
        ▼                                  │
┌──────────────────────┐                   │
│  model.visual  (ViT)  │                  │
│  ─ patch embed        │  ◄── DeepStack: fuses multi-level ViT
│  ─ transformer blocks │       features for fine-grained detail
└──────────┬───────────┘                   │
           │ multi-level visual features    │
           ▼                                │
┌──────────────────────────┐               │
│ model.visual.merger      │  ← vision→LLM projector (MLP)
│ (spatial merge, merge_size=2)             │
└──────────┬───────────────┘               │
           │ visual embeddings              │ text embeddings
           └──────────────┬─────────────────┘
                          ▼
        interleaved multimodal token sequence
        position ids via 3D MRoPE (time,H,W)  ← rope2d.py::get_rope_index_3
                          │                      (Interleaved-MRoPE;
                          ▼                        temporal_patch_size=2)
              ┌──────────────────────────┐
              │ model.language_model     │  (dense LLM, or MoE for *-A*B variants)
              │  ─ embed_tokens          │
              │  ─ decoder layers        │
              └──────────┬───────────────┘
                         ▼
                   model.lm_head  ──►  output tokens
```

The three README "Model Architecture Updates" map onto this diagram: **DeepStack** (multi-level ViT feature fusion into the merger), **Interleaved-MRoPE** (full-frequency time/H/W allocation in the 3D position ids), and **Text–Timestamp Alignment** (timestamp-grounded temporal position ids for video). MoE variants (`*-A*B`, e.g. `30B-A3B`) replace the dense `language_model` with a mixture-of-experts stack.

## Fine-tuning Architecture

The training framework wraps HuggingFace `Trainer` with multimodal-specific handling.

The three `--tune_mm_*` flags map directly onto the architecture stages above — each toggles `requires_grad` on one module subtree:

```
--tune_mm_vision ─► model.visual.*            (ViT; keep False when mixing image+video)
--tune_mm_mlp    ─► model.visual.merger.*     (vision→LLM projector)
--tune_mm_llm    ─► model.language_model.* + model.lm_head
--lora_enable    ─► freezes ALL of the above, injects LoRA adapters into
                    q_proj/k_proj/v_proj/o_proj (attention) instead
```

Key flow:

1. **`qwenvl/train/train_qwen.py`** — Entry point. Selects the model class by **string-matching the checkpoint name**: names containing `qwen3` + an `a` in the final path segment → `Qwen3VLMoeForConditionalGeneration`; `qwen3` → `Qwen3VLForConditionalGeneration`; `qwen2.5` → `Qwen2_5_VLForConditionalGeneration`; else `Qwen2VLForConditionalGeneration`. This is why checkpoint directory naming matters.

2. **Component-level freezing** (`set_model()`): three independent flags control which parts train — `--tune_mm_vision` (ViT / `model.visual`), `--tune_mm_mlp` (the vision→LLM projector / `model.visual.merger`), `--tune_mm_llm` (the language model + lm_head). When mixing image and video data, keep `tune_mm_vision=False`. LoRA (`--lora_enable`) is an alternative path that freezes everything and injects adapters into the attention projections (`q/k/v/o_proj`).

3. **`qwenvl/data/`** — Datasets are registered as dicts in `data/__init__.py` (`data_dict`), each with `annotation_path` + `data_path`. Selected at runtime via `--dataset_use "name%50,other%100"` where `%N` is a per-dataset sampling percentage (parsed by `data_list()`). `data_processor.py` holds `LazySupervisedDataset` and the collators; input JSON uses `conversations` with `from: human/gpt` turns and inline `<image>`/`<video>` tags that must correspond 1:1 to media files (tags must not appear in answers). `rope2d.py` provides the 2D RoPE position-id computation for the multimodal sequence.

4. **Sequence flattening/packing** — `--data_flatten` concatenates a batch into one sequence; `--data_packing` packs samples into even-length buckets (requires preprocessing with `tools/pack_data.py`). Either flag triggers `replace_qwen2_vl_attention_class()` (in `train/trainer.py`) to swap in a flash-attention variant that respects sample boundaries.

5. **Resolution control is critical for quality**: `--max_pixels`/`--min_pixels` (images) and `--video_max_pixels`/`--video_min_pixels`/`--video_fps`/`--video_max_frames` (videos). These are expressed as pixel counts (e.g. `50176` = `64*28*28`).

Constraint: **Qwen3-VL MoE models do not support DeepSpeed ZeRO-3**, and the HF implementation currently lacks MoE load-balancing loss.

## Inference Input Convention

All inference paths (transformers, vLLM, SGLang, web demo) share the same message format and preprocessing via `qwen-vl-utils`. For **Qwen3-VL specifically**, `process_vision_info()` must be called with `image_patch_size=processor.image_processor.patch_size` (or `16`), `return_video_kwargs=True`, and `return_video_metadata=True` — this differs from the Qwen2/2.5-VL call signature. See the README "New qwen-vl-utils Usage" section and `qwen-vl-utils/README.md` for the exact per-generation-model call patterns.

## Image → Token Accounting

How many context tokens an image consumes is fixed by `smart_resize()` in `qwen-vl-utils/src/qwen_vl_utils/vision_process.py` plus the model's `preprocessor_config.json`. The pipeline: decode → `to_rgb` → `smart_resize` (snap H,W to a `patch_factor` grid, clamped to min/max pixels) → resize → ViT patchify (`patch_size`) → 2×2 spatial merge in the merger → one token per merged cell.

For **Qwen3-VL-4B-Instruct** (`patch_size=16`, `merge_size=2`, so `patch_factor=32`):

```
tokens = (H_resized / 32) * (W_resized / 32)   ≈  H*W / 1024
```
Each final token ≈ a **32×32-pixel** cell. Bounds from `vision_process.py` constants: `IMAGE_MIN_TOKEN_NUM=4` (min 4 tokens, min_pixels = 4·32² = 4096) and `IMAGE_MAX_TOKEN_NUM=16384` (max 16,384 tokens = 4096×4096); aspect ratio must be < `MAX_RATIO=200`.

Examples: 512×512 → 256 tokens; 1024×768 → 768; 1920×1080 → ~2040; 4096×4096 → 16,384 (capped).

**Serving implication:** at `--max-model-len 8192` a single default-resolution image can exceed the whole context (16,384 > 8192). Constrain via the per-message `max_pixels`/`resized_height`+`resized_width` fields, or the training `--max_pixels` flag (e.g. `50176` = `64*28*28`). The `--limit-mm-per-prompt '{"image": N}'` serve flag caps image *count*, not size — it is a serving cap, not a model limit.

To change token cost per image: set `max_pixels` (per-message dict) or `resized_height`/`resized_width`; values are always snapped to the 32-grid.

Reference — **Kimi K2.5/K2.6** (also open-weight; `MoonshotAI/Kimi-K2.5` HF `preprocessor_config.json`) uses the same patchify→2×2-merge scheme but `patch_size=14`, so `patch_factor=28` → `tokens ≈ H*W / 784` (each token ≈ 28×28 px), same 16,384-token/image cap, 256K context.

## Local Environment Notes (this machine: WSL2 + RTX 5080)

- Use conda env `Qwen3VL` for local inference (torch 2.11.0+cu130 with Blackwell `sm_120`, vLLM 0.25.1, transformers 5.14.1). Do not reuse the separate `vllm` conda env — it is an editable dev install of a cloned vLLM source tree at `/mnt/e/GitHub/vllm` and must stay pristine.
- **WSL2 lacks UVA (managed memory)**. vLLM's newer V2 GPU model runner requires it and fails at engine startup with `RuntimeError: UVA is not available`. Set **`VLLM_USE_V2_MODEL_RUNNER=0`** to force the V1 runner (pinned memory) when running vLLM under WSL. `test_qwen3vl_4b.py` already sets this.
- 16 GB VRAM fits `Qwen3-VL-4B-Instruct` comfortably (fp16, `tensor_parallel_size=1`, `gpu_memory_utilization≈0.85`, `max_model_len` ~8192). Larger dense/MoE variants require quantization or multi-GPU.
