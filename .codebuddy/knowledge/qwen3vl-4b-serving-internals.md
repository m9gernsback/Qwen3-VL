# Qwen3-VL-4B-Instruct: Serving Internals — Architecture Loading, Attention & KV Cache

Findings from inspecting the local `Qwen/Qwen3-VL-4B-Instruct` checkpoint and the local
**vLLM 0.25.1** install (conda env `Qwen3VL`, WSL2 + RTX 5080). All numbers read directly from
files/code on disk. Companion to `qwen3vl-4b-weights-and-tokenizer.md`.

Config values used throughout (`config.json` → `text_config`):
```
num_hidden_layers   L   = 36
num_attention_heads Hq  = 32
num_key_value_heads Hkv = 8      (GQA: 32 Q heads share 8 KV heads → 4 Q per group)
head_dim            d   = 128
hidden_size             = 2560
vocab_size              = 151936
```
vLLM source root: `~/miniconda3/envs/Qwen3VL/lib/python3.12/site-packages/vllm/`

---

## 1. How vLLM "understands" the architecture (NOT from safetensors)

Key insight: **safetensors is passive data** (`{name → numbers}`); it carries no architecture.
The structure comes from code selected by config, then weights are matched in by NAME.

```
config.json ──▶ pick model class ──▶ instantiate EMPTY network (named nn.Modules, known shapes)
   (blueprint)   "Qwen3VLForConditionalGeneration"                     │ match by NAME
index.json ──▶ which shard holds each name ───────────────────────────▼
*.safetensors {name → BF16 numbers} ─────────▶ copy each tensor into its matching slot
```

- `config.json` field **`architectures: ["Qwen3VLForConditionalGeneration"]`** (`model_type: "qwen3_vl"`)
  → maps to a hard-coded vLLM class file `vllm/model_executor/models/qwen3_vl.py`. **That file IS the
  architecture** (ViT, merger, 36 decoder layers, attention, MRoPE, forward pass) — none of it from weights.
- `config.json` numeric fields fill in dimensions (36 layers, hidden 2560, 32/8 heads, etc.) → produces a
  fully-structured but **empty** network; every param has a name + shape but uninitialized values.
- `model.safetensors.index.json`: `weight_map` maps all **713 tensor names → shard file**
  (`total_size = 8,875,631,616` = 8.88 GB). Lets vLLM open the right shard per param.
- Load = for each param slot, look up the same name in safetensors, read bytes via header
  `data_offsets`, check shape, copy in. **Tensor names are the glue.**
- This mirrors CODEBUDDY.md's note that the finetune framework "selects the model class by
  string-matching the checkpoint name" — name/config picks the code; code defines architecture.
- You could NOT reconstruct architecture from safetensors alone (only a flat list of named tensors).

---

## 2. Fused / sharded weight loading (vLLM's load-time remapping)

Problem: disk layout ≠ runtime layout. Disk has separate `q/k/v_proj` and `gate/up_proj`;
vLLM wants them **fused** (fewer, bigger matmuls) and **sharded** across GPUs (tensor parallelism).
Solved on the fly at load time by name-remap + slicing; **disk never changes**.

### Fusion table (`qwen2.py:323`, reused by Qwen3-VL LM via `hf_to_vllm_mapper`)
```python
WeightsMapper(orig_to_new_stacked={
    ".q_proj":    (".qkv_proj",     "q"),
    ".k_proj":    (".qkv_proj",     "k"),
    ".v_proj":    (".qkv_proj",     "v"),
    ".gate_proj": (".gate_up_proj", 0),
    ".up_proj":   (".gate_up_proj", 1),
})
# packed_modules_mapping (qwen2.py:445):
#   "qkv_proj":     ["q_proj","k_proj","v_proj"]
#   "gate_up_proj": ["gate_proj","up_proj"]
```
There is **no `qkv_proj` tensor on disk**; vLLM creates it in memory and fills sub-regions from
the three separate disk tensors. Names not in the table (norms, embed, `o_proj`) pass through.

### Where each piece lands (`QKVParallelLinear.weight_loader`, `linear.py:1145`)
Fused buffer offsets (GQA-aware), for this model:
```
q region: rows 0    .. 4096   (32 heads × 128)
k region: rows 4096 .. 5120   ( 8 heads × 128)
v region: rows 5120 .. 6144   ( 8 heads × 128)
→ qkv_proj.weight is one [6144, 2560] buffer   (q_proj[4096,2560], k/v_proj[1024,2560] on disk)
```
`weight_loader` is called once per source tensor with the `loaded_shard_id` ("q"/"k"/"v") the
mapper attached; copies it into the right offset. (`loaded_shard_id is None` branch handles models
like Phi-3 that ship an already-fused qkv on disk.) `MergedColumnParallelLinear` does gate→first
half, up→second half.

### Sharding (tensor parallelism)
Same `weight_loader` does the multi-GPU split via `narrow(output_dim, tp_rank*per_gpu, per_gpu)`:
- **Column-parallel** (`ColumnParallelLinear`/`QKVParallelLinear`/`MergedColumnParallelLinear`):
  split OUTPUT rows → each GPU holds `total_heads/tp_size` heads.
- **Row-parallel** (`RowParallelLinear`, e.g. `o_proj`, `down_proj`): split INPUT columns, all-reduce
  after matmul.
- **`VocabParallelEmbedding`** shards embed/lm_head along the vocab dim (151,936 rows) — why vocab
  padding to a multiple of 128 matters (must divide across GPUs).
- `serve_qwen3vl.sh` uses `tensor_parallel_size=1` → tp_size=1 → each shard = whole tensor; same code path.

Load loop (`AutoWeightsLoader.load_weights(weights, mapper=...)`, `qwen3_vl.py:844`):
```
for (name, tensor) in safetensors:            # 713 tensors
    name, shard_id = hf_to_vllm_mapper(name)   # q_proj → qkv_proj + "q"
    param = model.get_parameter(name)          # fused, TP-sized buffer
    param.weight_loader(param, tensor, shard_id)
```
Payoff: fewer bigger matmuls; fit big models across GPUs; keep standard HF checkpoints (remap by
name, never rewrite disk); offset/size branches also handle quantized packed formats (Marlin,
bitsandbytes, FP8 block scales).

---

## 3. Attention head parameters (what they mean)

Attention: each token makes 3 vectors — **Q**uery ("what I look for"), **K**ey ("what I advertise"),
**V**alue ("my content"). Q·K similarity → softmax → weights → blend everyone's V.

| Param | Value | Meaning |
|---|---|---|
| `head_dim` | 128 | length of ONE Q/K/V vector; the space where Q·K dot-product happens (softmax scale `1/√128`) |
| `num_attention_heads` | 32 | how many parallel Query "lenses" run at once; each learns a different relationship |
| `num_key_value_heads` | 8 | how many **shared** K/V heads (GQA); 32 Q heads split into 8 groups of 4, each group shares one K/V |

Arithmetic ties to projection shapes:
```
q_proj: 32×128 = 4096 rows → [4096, 2560]
k_proj:  8×128 = 1024 rows → [1024, 2560]   (fewer, because GQA)
v_proj:  8×128 = 1024 rows → [1024, 2560]
o_proj: [2560, 4096]  (concat 32 heads × 128 = 4096, project back to 2560)
```
GQA spectrum: `Hkv==Hq` = classic Multi-Head; `Hkv==1` = Multi-Query; `Hkv=8` = middle ground.
Why GQA: shrinks the KV cache (below) ~4× with little quality loss. Q is NOT cached; only K,V are.

One-liner: head_dim = how *wide* each view is; num_attention_heads = how *many* views; num_key_value_heads = how many views *share* their K/V memory.

---

## 4. KV cache growth during generation

Attention needs every past token's K,V; generating token N without a cache would recompute
1..N-1 each step (O(n²)). Fix: compute each token's K,V once, **store & reuse** = KV cache.
Grows with every token. **Q not cached** (only current token's Q needed each step).

Per-token cost (this model, BF16 = 2 bytes), uses **Hkv=8 NOT Hq=32** (the GQA payoff):
```
bytes/token = 2(K,V) × L(36) × Hkv(8) × d(128) × 2 = 147,456 bytes = 144 KiB/token
full MHA (Hkv=32) would be 576 KiB/token → 4× more
```
Total = 144 KiB × context_length. Strictly **linear, append-only** (nothing freed until request ends).
Two phases: **prefill** (whole prompt at once → jumps to prompt_len×144 KiB), then **decode**
(each new token appends exactly 144 KiB and reads the whole cache).

| Context | KV (GQA, Hkv=8) | If MHA (Hkv=32) |
|---:|---:|---:|
| 256 | 36 MiB | 144 MiB |
| 1,024 | 144 MiB | 576 MiB |
| 4,096 | 576 MiB | 2.25 GiB |
| 8,192 (this setup's `--max-model-len`) | 1.13 GiB | 4.5 GiB |

16 GB budget: weights ≈8.3 GB (the index.json total_size) + CUDA/activations ≈1–2 GB + rest → KV pool.
Concurrency (not one request) is the real limit; GQA's 4× cut is load-bearing for fitting long
context + multiple requests on 16 GB.

---

## 5. PagedAttention block allocation (vLLM 0.25.1)

Naive "one contiguous buffer per request sized to max-model-len" wastes unused slots (internal frag)
and fragments free memory (external frag). PagedAttention = OS-paging idea: fixed-size blocks,
allocated on demand, per-request **block table** maps logical→physical (scattered) blocks.

Real pieces (source):
- **Block = 16 tokens.** `DEFAULT_BLOCK_SIZE = 16` (`config/cache.py:47`).
  → one block = 16 × 144 KiB = **2,304 KiB = 2.25 MiB** of KV.
- **Global pool.** `BlockPool` builds `[KVCacheBlock(idx) for idx in range(num_gpu_blocks)]`
  (`block_pool.py:177`); `num_gpu_blocks` = the "# GPU blocks" startup log number. Each
  `KVCacheBlock` has `block_id` (physical slot) + `ref_cnt` (`kv_cache_utils.py:118`).
- **Free queue.** `free_block_queue`; allocate = `get_new_blocks(n)` → `popleft_n` (`:542`);
  free = `free_blocks(...)` (`:614`).
- **Per-request block table:** ordered list logical#i → physical block_id (physical IDs NOT contiguous).

Allocation rule: a new physical block is pulled **only when the current block fills = once every 16
tokens**; appends within a block are free. ("+144 KiB/token" = "+1 block (2.25 MiB) every 16 tokens".)
```
tokens :  1..16 | 17..32 | 33..48 | 49..64 | 65..
blocks : [--1--] [--2--]  [--3--]  [--4--]  [-5-]
          ▲ pop a block only when crossing a 16-token boundary
```
Attention kernel gets the block table; for token t: `logical=t//16`, `offset=t%16`,
`phys=block_table[logical]`, read K,V at `(phys, offset)` — exactly OS page-table translation.

Two payoffs:
1. **~0 fragmentation** → waste capped at <1 block/sequence (partial last block) → much higher concurrency.
2. **Copy-free sharing via `ref_cnt`:** multiple sequences point at same physical block.
   - Prefix caching: shared system prompt computed once (`get_cached_block`/`cache_full_blocks`,
     block keyed by hash of token contents, `block_pool.py:199/226`).
   - Parallel sampling (n>1): share prompt blocks, only diverging suffix is private.
   - Copy-on-write on divergence; block returns to free queue when `ref_cnt==0`.

### Tie-back to 16 GB setup
```
num_gpu_blocks ≈ (0.90×16GB − weights 8.3GB − overhead) / 2.25 MiB
≈ ~5.5 GiB left / 2.25 MiB ≈ ~2500 blocks ≈ ~40,000 tokens total KV capacity (shared, all requests)
```
- One 8192-token request = ceil(8192/16) = **512 blocks** (1.13 GiB) → fits easily.
- Pool (~2500 blocks) is the aggregate concurrency limit (~5 simultaneous 8k requests) before the
  scheduler **queues** new ones; can **preempt** (free blocks, recompute later) instead of OOM.
- `--max-model-len 8192` caps ONE request's logical length; the **block pool** caps the AGGREGATE —
  independent knobs. Per-message `max_pixels` (image token cost) + concurrency both feed pool usage.
