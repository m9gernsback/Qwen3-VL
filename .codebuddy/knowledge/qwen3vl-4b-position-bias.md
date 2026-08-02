# Position (Primacy) Bias in VLM Comparison Prompts — Qwen3-VL-4B Finding & Fixes

Findings from the user's mesh part-importance comparison task (2026-08): judging which of two
sub-parts of a whole 3D mesh is more important, by feeding multi-view renders to Qwen3-VL and
asking for an A/B verdict. Symptom observed on `Qwen3-VL-4B-Instruct` (local vLLM offline setup,
conda env `Qwen3VL`); generalizes to small-VLM comparison/judgment prompts. Companion to the
serving/weights notes in this directory.

---

## 1. Symptom

The model's verdict **always favored the part presented first**: descriptions differed
("A is a small ..., B is a bigger ...") but the conclusion was invariant — "A is important" —
regardless of which part was actually more important.

---

## 2. Diagnosis methodology (reuse these tests)

Two cheap controls that separate the failure modes:

| Test | Procedure | Interpretation |
|---|---|---|
| **Swap test** | Swap presentation order (image order AND mention order), rerun | Verdict follows *position* → **order bias (primacy)**. Verdict sticks to the *label* ("A") regardless of order → **label prior**; model isn't using visual input at all |
| **Identical-part control** | Feed the same part as both options | Still confidently picks one → answer is decoupled from visual evidence |
| Grounding probe | Ask "what color/shape is part A?" | Wrong → the parts aren't visually marked well enough (fix rendering before debugging bias) |

Result here: verdict followed position → **order/position bias (primacy)**, the more common and
more fixable of the two biases.

---

## 3. Mechanism — why it happens

There is **no comparison operation inside the model**. The verdict is one autoregressive token,
and its logit decomposes roughly as:

```
logit("first part wins") = position_prior + visual_evidence
```

- **Position prior is strong**: SFT/RLHF comparison data is not order-balanced — first-mentioned
  options are disproportionately often correct/preferred, and RLHF reward models carry the same
  skew, distilling it into the policy. The model has learned "when in doubt, pick the first one."
- **Causal attention amplifies it**: the first-described item anchors the context (primacy /
  attention-sink effect); later items are encoded *relative* to it, so the first part's
  representation is more salient at the verdict position.
- **The evidence term scales with model capability**: at 4B, with multi-view images of subtle
  geometry, visual evidence is too weak/noisy to override the prior. (Low `max_pixels` starves it
  further — see Image → Token Accounting in CODEBUDDY.md.)
- **Greedy decoding makes a tilt look absolute**: a 55/45 logit preference becomes 100% of
  outputs under argmax. The underlying bias is usually weaker than it appears.

**Observed resolution: switching to a stronger (larger) model eliminated the bias** — the
evidence term grows enough to beat the prior. Treat in-context A/B verdicts from ~4B-class VLMs
as unreliable for this task.

---

## 4. Mitigations, ranked (when stuck with a small model)

1. **Independent scoring (best).** One call per part, scored against an explicit rubric
   (e.g. 1–10 on structural load / functional role / failure impact); compare numbers offline.
   With only one part in context there is no position to be biased toward.
2. **Permutation-averaged logprob margins.** Don't read the generated verdict — read
   probabilities. Constrain the answer to one token and average across both orders:
   ```
   score(X) = ½ · [ logp("X wins" | X first) + logp("X wins" | X second) ]
   ```
   Averaging cancels the position term, leaving the evidence term. With local offline vLLM:
   append each candidate completion (`... the more important part is A`) to the prompt and score
   it with `prompt_logprobs` — 4 forward passes per decision, no sampling. (Remember
   `VLLM_USE_V2_MODEL_RUNNER=0` under WSL2, per CODEBUDDY.md.)
3. **Ground labels perceptually.** Color-highlight the parts (red vs blue) in every render and
   refer to them **by color**, not letters — color binds to pixels, letters bind to the language
   prior. Keep both descriptions structurally parallel; interleave views A,B,A,B… instead of
   all-A-then-all-B to reduce chunk-level primacy.
4. **Evidence-before-verdict output format.** Require "describe part 1 → describe part 2 → then
   verdict", so the verdict conditions on generated observations. Verdict-first-then-justify is
   the worst structure.
5. **Order-balanced few-shot.** 2–4 example comparisons with the correct answer first half the
   time. Recalibrates the local prior; doesn't beat #1/#2.
6. **Fine-tune it away (recurring production use).** A few hundred comparison examples with
   **order-balanced labels** (each pair in both orders), LoRA SFT via `qwen-vl-finetune`
   (`--lora_enable`). Cancels the position term in the weights.

---

## 5. What does NOT work

- **Temperature / sampling** — samples the same biased distribution; only hides determinism.
- **"Answer without bias" instructions** — prompt injunctions don't move logit-level priors.

---

## 6. Validation hygiene

Keep a small **balanced test set** (~30 pairs; true answer equally often first/second) and track:

- **Swap-consistency**: fraction of pairs whose verdict is invariant under order swap.
- **Accuracy** on the balanced set (50% = the model is only reading the prior).

Any mitigation that doesn't move swap-consistency toward 1.0 isn't working — don't eyeball it.
