# Qwen3-VL-4B-Instruct: Weights, Tokenizer & the Text→Token Pipeline

Findings from inspecting the local `Qwen/Qwen3-VL-4B-Instruct` checkpoint on this machine
(WSL2 + RTX 5080, conda env `Qwen3VL`). All numbers below were read directly from the
files on disk, not from documentation.

Local snapshot:
```
~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17/
```
(`HF_HOME` / `HF_HUB_CACHE` unset → default cache path.)

---

## 1. Where the model lives

- `MODEL="Qwen/Qwen3-VL-4B-Instruct"` in `serve_qwen3vl.sh` is a **Hugging Face Hub repo ID**,
  not a filesystem path. vLLM resolves it through the HF cache (downloads if absent).
- On-disk weights are content-addressed **blobs** under `.../blobs/`, exposed via symlinks in
  `.../snapshots/<commit>/`.
- Already fully downloaded → loads from cache, no network fetch.
- Override: `MODEL=/path/to/local/dir bash serve_qwen3vl.sh`, or `export HF_HOME=/mnt/e/hf-cache`.

---

## 2. What the `.safetensors` files consist of

Physical layout (3 parts):
```
┌──────────────┬─────────────────────────────┬──────────────────────────┐
│ 8 bytes      │  JSON header                │  raw tensor byte-buffer  │
│ header size  │  {name:{dtype,shape,        │  (contiguous weights,    │
│ (u64 LE)     │   data_offsets}, ...}       │   no code)               │
└──────────────┴─────────────────────────────┴──────────────────────────┘
```
- 8-byte little-endian `u64` = JSON header length.
- JSON header: one entry per tensor → `dtype`, `shape`, `data_offsets` (byte range). Optional
  `__metadata__` (here `{"format":"pt"}` = PyTorch origin).
- Data buffer: raw tensor bytes back-to-back → loading is `mmap` + slice = **zero-copy, no code
  execution** (the security advantage over pickle).

This checkpoint (measured):
- 2 shards; shard 1 = 4.97 GB, 231 tensors; **713 tensors total, ALL BF16** (2 bytes/param).
- Contents = pure weights, named by module path (matches CODEBUDDY.md architecture):
  `model.language_model.embed_tokens.weight` `[151936, 2560]`, per-layer
  `input_layernorm` / `mlp.{gate,up,down}_proj` / attention `q/k/v/o_proj`, plus (shard 2)
  the vision tower `model.visual.*`, the merger, and `lm_head`.
- Which tensor is in which shard → recorded in `model.safetensors.index.json`.
- Does NOT contain: config, tokenizer, chat template, optimizer state, or any executable code.

---

## 3. Codebook — NOT present (full precision)

- All 713 tensors are **BF16**; `config.json` has **no `quantization_config`**; scan for
  `codebook/scale/zero/qweight/qzeros/g_idx/absmax/lut` tensor names → **NONE**.
- A "codebook" exists only in **lookup-table (LUT) quantization** (AQLM, QuIP#, GGUF k-quants,
  SqueezeLLM): store a small dictionary of representative values + a compact per-weight index
  → `w ≈ codebook[index]`.
- Three quant families:
  | Family | Reconstruction | Codebook? | Examples |
  |---|---|---|---|
  | Uniform/affine | `w ≈ scale·(q − zero)` | No (scales+zeros) | GPTQ, AWQ, nf4 |
  | Vector/LUT | `w ≈ codebook[index]` | **Yes** | AQLM, QuIP#, GGUF, SqueezeLLM |
  | Full precision (this model) | stored as-is | No | BF16/FP16 |
- The 4B fits comfortably in BF16 (~8 GB weights) on 16 GB → no reason to quantize locally.
  Quantized variants are relevant for the 30B/32B models.
- **Do not confuse** the embedding lookup table (token→vector, a normal learned weight) with a
  quantization codebook (compressed-code→weight-value, a storage trick, absent here). Both are
  "lookup tables" but sit at opposite ends of the pipeline.

---

## 4. Text → token IDs → vectors (the full pipeline)

```
"你好" ──tokenizer──▶ [108386] ──embed lookup──▶ 2560-dim vector ──▶ Transformer layers ──▶ lm_head ──▶ next-token scores
text                 token IDs                  (row of weight matrix)
```

### Stage 1 — Tokenizer (BPE, byte-level)
- Qwen uses **byte-level BPE**. Not word/char splitting — a fixed dictionary of subword chunks
  learned by merging frequent adjacent byte-pairs. **Frequency in training = granularity.**
- Measured examples:
  | Input | IDs | Pieces | Note |
  |---|---|---|---|
  | `"hello"` | `[14990]` | `hello` | common → 1 token |
  | `"unbelievable"` | `[359,31798,23760]` | `un/belie/vable` | rare bare form → 3 pieces |
  | `" unbelievable"` (Ġ) | `[51129]` | `Ġunbelievable` | space-prefixed form IS 1 token |
  | `"cat"` vs `" cat"` | `[4616]` vs `[8251]` | `cat` vs `Ġcat` | leading space = DIFFERENT token |
  | `"你好"` | `[108386]` | 1 token | |
  | `"我爱编程"` | `[35946,99242,110569]` | `我/爱/编程` | 4 chars → 3 tokens |
- **Byte-level:** BPE operates on UTF-8 bytes, not characters. `你` = bytes `E4 BD A0`; each
  byte maps to a placeholder glyph → `你好` displays as mojibake `ä½łå¥½` but is exactly the
  original bytes. This is why ANY input encodes with no "unknown token" errors (byte fallback).

### Stage 2 — Embedding lookup
- `model.language_model.embed_tokens.weight` shape `[151936, 2560]` = a **lookup table**:
  one row per vocab slot, each row a **2560-number (hidden size) vector**.
- Embedding = just index the row at the token ID (no math). Row 14990 (`hello`) starts
  `[-0.0287, -0.0115, -0.0100, 0.0062, 0.0500, ...]` — learned during training.
- A sentence → stack of row-vectors: `[seq_len × 2560]`.

### Stage 3 — Transformer → output
- The `[seq_len × 2560]` matrix flows through `language_model.layers` (attention mixes tokens).
- **`lm_head`** `[151936 × 2560]` (reverse of embedding) → 151,936 scores (one per token) →
  softmax → probabilities → pick ID → decode to text → repeat.

---

## 5. Why 151,936 rows but only 151,643 tokens?

Three legitimately different numbers:
| Number | Meaning |
|---|---|
| **151,643** | `vocab_size` — the learned BPE subword vocabulary |
| **151,669** | base + **26 special tokens** (`len(tokenizer)`) |
| **151,936** | rows in `embed_tokens` / `lm_head` — the weight matrix (`config.vocab_size`) |

Chain: **151,643 → +26 specials → 151,669 → padded to → 151,936.**
- **+26 special tokens**: control/chat/reasoning markers appended at IDs 151643+
  (`<|endoftext|>`=151643, `<|im_start|>`=151644, `<|im_end|>`=151645, … `<think>`=151667,
  `</think>`=151668). Max special ID = 151668 → 151,669 usable entries.
- **Padding to 151,936**: `151936 = 128 × 1187` → clean multiple of 128 for **GPU tensor-core /
  tensor-parallel alignment**. The `151936 − 151669 = 267` extra rows map to **no token**; they
  are dead padding for matmul efficiency (model never emits them, scores ignored).

---

## 6. How BPE actually runs locally

Tokenizer defined entirely by **data files** (no code, no network) in the snapshot:
| File | Role |
|---|---|
| `vocab.json` | `{token_string → id}` — 151,643 pieces |
| `merges.txt` | ranked merge rules (priority order) — the heart of BPE |
| `tokenizer.json` | compiled fast (Rust) tokenizer, self-contained |
| `tokenizer_config.json` | class name, special-token roles, max length, chat template |

Algorithm (encode one word):
1. Split into individual bytes/chars.
2. Find the adjacent pair whose merge rule has the **lowest rank** (earliest in `merges.txt` =
   most frequent in training).
3. Merge every occurrence of that pair.
4. Repeat until no adjacent pair has a rule.
5. Look up surviving symbols in `vocab.json`.

`merges.txt` first lines (most frequent pairs): `Ġ Ġ`, `i n`, `Ġ t`, `e r`, `o n` …
Lower line number = higher priority.

Measured trace — `"Ġunbelievable"` (with leading space):
```
start:  Ġ u n b e l i e v a b l e
#1 l+e (rank 17) → ...a b le      #5 ab+le (rank 224) → ...v able
#3 u+n (rank 103) → Ġ un b el ... #7 Ġ+un (rank 393) → Ġun ...
...
#12 Ġunbelie+vable → Ġunbelievable   → single token id=51129
```
Trace — `"hello"`: `e+l`→`l+o`→`el+lo`→`h+ello` = `hello`, id=14990 (4 merges).
Intuition: frequent-enough strings have a full merge chain to a single vocab entry; rarer
strings run out of rules partway and stay fragmented. Chinese identical, on UTF-8 bytes.

---

## 7. Verification against official Qwen files (Hub)

Fresh `hf_hub_download` from `Qwen/Qwen3-VL-4B-Instruct`, SHA-256 compared:
- `merges.txt`: local `599bab54…e26f5e3` == official → **identical bytes**; 151,291 rules match
  exactly in the same order.
- `vocab.json`: local `ca10d7e9…b1440910` == official → **identical bytes**; 151,643 entries,
  full `{token→id}` dicts equal.
- Conclusion: local snapshot carries the **genuine, unmodified official Qwen tokenizer** — no
  corruption/tampering/drift.

Rule-count note: legacy `merges.txt` (strict parse, skipping `#version` header + blanks) =
**151,291** rules; `tokenizer.json`'s embedded `merges` = **151,387**. Known formatting
difference between the old split format and the fast tokenizer's internal representation; the
fast one is authoritative and both produce identical tokenization. (An earlier loose Python
parse miscounted 151,387 for merges.txt — the strict count is 151,291.)

---

## 8. `tokenizer.json` vs `tokenizer_config.json`

| | `tokenizer.json` | `tokenizer_config.json` |
|---|---|---|
| Role | the tokenizer **engine** (text ↔ IDs) | **settings + chat formatting policy** |
| Contains | vocab, merges, regex pre-tokenizer, normalizer, post_processor, decoder | class name, special-token roles, max length, **chat template** |
| Read by | Rust `tokenizers` (fast) | Python `AutoTokenizer` loader |
| Size | ~7 MB (holds vocab+merges) | small (settings + Jinja2 template) |
| Redundant with | `vocab.json` + `merges.txt` | partially `added_tokens` in tokenizer.json |

### `tokenizer.json` — self-contained engine, a pipeline:
```
text → normalizer → pre_tokenizer (regex split + byte-level map) → model (BPE: vocab+merges)
     → post_processor (special tokens) → decoder (IDs→bytes→text) → IDs
```
- `model.type = "BPE"`, embeds vocab (151,643) + merges (151,387) + BPE settings
  (`byte_fallback`, `unk_token`).
- `pre_tokenizer`: GPT-style regex `('s|'t|...|\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+...)` splits text
  into word/number/punct chunks BEFORE BPE, plus byte-level encoder (produces `Ġ`/`ä½ł`).
- `added_tokens`: the 26 special tokens with metadata.
- If you only had this file, tokenization works fully. `vocab.json`+`merges.txt` are the older
  split, human-readable form kept for backward compatibility.

### `tokenizer_config.json` — settings & policy (no vocab/merges):
- `tokenizer_class: "Qwen2Tokenizer"`, `eos_token: <|im_end|>` (NOT `<|endoftext|>`),
  `pad_token: <|endoftext|>`, `bos_token: None` + `add_bos_token: False` (Qwen adds no BOS),
  `unk_token: None` (byte-fallback), `model_max_length: 262144` (256K).
- `additional_special_tokens`: vision/box/tool markers (`<|vision_start|>`, `<|image_pad|>`,
  `<|video_pad|>`, `<|box_start|>`, …).
- **`chat_template`**: a 5,292-char Jinja2 program that renders a message list into the trained
  string, wrapping turns with `<|im_start|>role\n...<|im_end|>` and handling tool calls +
  `<think>`/`</think>`. Invoked by `tokenizer.apply_chat_template(messages)`. It lives here (not
  in tokenizer.json) because it is a usage convention, not part of the text↔ID algorithm.

Analogy: `tokenizer.json` = dictionary + grammar (how to read/write the language);
`tokenizer_config.json` = style guide (what the markers mean, how to lay out a conversation,
max length).
