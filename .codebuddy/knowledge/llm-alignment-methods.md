# LLM Alignment Training Methods — SFT, RLHF, DPO, GRPO, PEFT

Reference notes on the post-training / alignment toolbox: what each method is, the mechanics and
math, why it exists, and its characteristic failure modes. Written in the context of diagnosing
position bias in VLM comparison prompts — companion to `qwen3vl-4b-position-bias.md`, whose core
lesson ("the bias lives in the data/reward signal, not the algorithm") is the thread connecting
all methods below.

Two categories:
- **Training objectives** (what is optimized): SFT, RLHF, DPO, GRPO
- **Training-efficiency method** (how cheaply it is optimized): PEFT (LoRA/QLoRA)

Pipeline position:
```
Pretrain → SFT (often via LoRA) → preference stage (DPO | RM+PPO | GRPO) → [domain LoRA adapters]
```

---

## 1. SFT — Supervised Fine-Tuning

**Role:** foundation stage; turns a raw pretrained model (text continuer) into an
instruction-following chat model.

**Mechanics:**
- Data: (instruction, ideal response) pairs rendered with the model's chat template
  (`<|im_start|>user ... <|im_end|>` for Qwen).
- Loss: next-token cross-entropy, **masked to response tokens only** (prompt tokens excluded):

```
L = − Σ_t log π_θ(y_t | x, y_<t)        (sum over response tokens only)
```

- **Teacher forcing**: during training the model always conditions on the *gold* prefix, never
  its own outputs.
- Recipe: 1–3 epochs, LR ~1e-5 (full FT) / ~1e-4 (LoRA). Data quality ≫ quantity — a few
  thousand excellent examples beat millions of noisy ones (LIMA, "Less Is More").

**Failure modes:**
- **Imitates everything**, including statistical artifacts — no notion of *why* an answer is
  right, only what answers look like. Skewed demonstrations → skewed policy (position bias
  enters here).
- **Exposure bias**: trained on gold prefixes, generates on its own (off-distribution) prefixes
  at inference → compounding errors.
- **Catastrophic forgetting**: aggressive narrow-domain SFT degrades general capability.

---

## 2. RLHF — Reinforcement Learning from Human Feedback

**Role:** alignment stage; optimizes *preferences* ("which response is better") instead of
imitation. Three-stage pipeline (Christiano et al.; InstructGPT).

**Stage 1 — SFT** (also frozen later as the reference model).

**Stage 2 — Reward Model (RM):**
- SFT model with LM head replaced by a scalar head → `r(x,y)` per response.
- Trained on human pairwise rankings with the **Bradley–Terry** loss:

```
L_RM = − log σ( r(x, y_chosen) − r(x, y_rejected) )
```

- The RM only sees *comparisons*, never absolute quality: anything correlating with winning —
  length, confidence, **option position** — becomes a predictive feature it will exploit.

**Stage 3 — PPO against the RM:**

```
maximize  E_y~π [ r_RM(x,y) ]  −  β · KL( π_θ ‖ π_ref )
```

- Roll out from current policy → score with RM → PPO clipped surrogate update; a **critic**
  (value network) provides the advantage baseline (GAE).
- The **KL penalty** to the frozen reference is load-bearing: without it the policy drifts into
  degenerate text that exploits RM blind spots.
- Cost: **4 models in memory** (policy, reference, RM, critic) + on-policy sampling → expensive,
  hyperparameter-fragile.

**Failure modes:**
- **Reward hacking / Goodhart**: optimize a proxy hard enough and it stops tracking the goal
  (verbosity exploitation).
- **Annotator-bias inheritance + amplification**: human comparison judges have documented primacy
  and verbosity biases → RM learns "first/longer tends to win" → PPO **amplifies** it far beyond
  the data's base rate. This is where small data skews become dominant policy behaviors.
- RM misgeneralization off-distribution; mode collapse; KL-coefficient sensitivity.

---

## 3. DPO — Direct Preference Optimization

**Role:** same goal as RLHF (learn from preference pairs) with the entire RL machinery deleted.

**Key derivation:** the KL-constrained RLHF objective has a closed-form optimal policy

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y)/β )
⟹  r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β·log Z(x)
```

Substituting into Bradley–Terry, the partition function `Z(x)` **cancels** between chosen and
rejected → a plain logistic loss on pairs:

```
L_DPO = − log σ( β [ log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x) ] )
```

**Intuition:** binary classifier whose "implicit reward" is how far the policy has moved from the
reference; gradient raises chosen's relative probability, lowers rejected's. `β` plays the KL
role (larger = stay closer to ref).

**Properties:** no RM, no critic, no rollouts, no PPO; 2 models in memory (policy + frozen ref);
trains like supervised learning; **offline** (fixed pair dataset, no exploration).

**Failure modes:**
- **Inherits pair skews directly, unbuffered** — no RM to inspect; order-skewed pairs become an
  order-skewed policy verbatim.
- **Length bias**: conflates verbosity with quality (implicit-reward parameterization artifact).
- **Distribution shift**: as π_θ moves off the data-collection policy the off-policy assumption
  degrades; can drive *both* chosen and rejected likelihoods down (rejected just falls faster).
- Variants: **IPO** (overfitting fix), **KTO** (binary good/bad labels, no pairs),
  **ORPO/SimPO** (no reference model).

---

## 4. GRPO — Group Relative Policy Optimization

**Role:** RL for reasoning (DeepSeekMath; DeepSeek-R1). PPO-style optimization with the **critic
deleted** — the most unreliable component for long reasoning chains.

**Mechanics:**
1. Per prompt `x`, sample a **group** of G responses (G ≈ 8–64) from the current policy.
2. Score each — ideally with **rule-based/verifiable rewards**: exact math-answer match, unit
   tests pass, format constraints.
3. Advantage = **within-group normalization** (group mean replaces the learned value baseline):

```
A_i = ( r_i − mean(r_1..G) ) / std(r_1..G)
```

4. PPO-style clipped token-level update + explicit KL to reference:

```
J = E[ (1/G) Σ_i (1/|y_i|) Σ_t min( ρ_t·A_i, clip(ρ_t, 1−ε, 1+ε)·A_i ) ] − β·KL(π_θ‖π_ref)
```

**Why it works:** no critic to train or fail; empirical unbiased baseline; verifiable rewards
can't be hacked like learned RMs. This combination enabled the emergent long chain-of-thought in
R1-Zero.

**Failure modes:**
- **Sampling-hungry**: G rollouts per prompt per step — cost moved from critic to generation.
- **Degenerate groups**: all G responses equally rewarded (e.g. all wrong on a hard problem) →
  advantage 0 → no gradient; needs reward variance / curriculum.
- Entropy collapse; gaming of rule-based format rewards.
- **Subjective tasks have no rule-based reward** → need an RM again → bias re-enters through it.
  *Exception:* if ground truth is programmatically computable (e.g. mesh part importance from
  structural simulation), GRPO becomes a structurally unbiased option.

---

## 5. PEFT — Parameter-Efficient Fine-Tuning

**Role:** not an objective — an answer to "run any of the above without 8×H100." Family of
methods adapting a model by training <1% of parameters.

**LoRA (dominant):**
- Freeze base weights; per target matrix `W ∈ R^{d×k}` learn a low-rank delta:

```
W' = W + ΔW = W + B·A,   B ∈ R^{d×r}, A ∈ R^{r×k},   r ≪ min(d,k)   (r = 8–64 typical)
```

- Init `A ~ Gaussian`, `B = 0` → ΔW = 0 at step 0 (starts exactly as base model); output scaled
  by `α/r`; at inference merge `W + BA` → **zero added latency**.
- Memory win: gradients + Adam states (2× params) exist only for tiny adapters → 4B fine-tune on
  one consumer GPU. **QLoRA**: base quantized to 4-bit NF4, adapters in BF16 → 65B on one 48 GB GPU.
- Siblings: **adapters** (bottleneck MLPs per layer), **prefix/prompt tuning** (learned virtual
  tokens), **(IA)³** (learned rescaling vectors).

**Properties / failure modes:**
- Objective-neutral: same SFT/DPO loss, fewer trainable params → inherits whatever the objective
  encodes, bias included. A vehicle, not a fix or a cause.
- Side benefit: less catastrophic forgetting (base untouched).
- Rank limits expressivity: fine for style/format/task adaptation; insufficient for fixes needing
  large weight shifts.
- In this repo: `--lora_enable` (qwen-vl-finetune) freezes everything and injects adapters into
  attention `q/k/v/o_proj` (see CODEBUDDY.md) — the vehicle for the order-balanced-pairs position
  bias fix at a few hundred pairs of data cost on a 16 GB GPU.

---

## 6. Comparison & composition

| | Data | Models in memory | On-policy? | Bias entry point |
|---|---|---|---|---|
| SFT | demonstrations | 1 | n/a | skewed demos |
| RLHF (PPO) | human comparisons | 4 | yes | annotator bias → RM → amplified |
| DPO | preference pairs | 2 | no | skewed pairs, learned unbuffered |
| GRPO | verifiable rewards | 2–3 | yes | unbiased *if reward is programmatic* |
| PEFT/LoRA | (any of the above) | −memory | n/a | neutral vehicle |

**Common thread:** every method is a different machine for fitting a preference signal. Feed an
order-skewed signal into any of them and it is learned — and RL-style optimization exaggerates
it. Durable fixes are order-balanced data and reward design, not switching algorithms.

**For the mesh part-importance task**, two clean paths:
1. **LoRA + order-balanced comparison pairs** (cheap, direct; `qwen-vl-finetune --lora_enable`).
2. **GRPO with a programmatic reward**, if importance is computable from simulation/geometry —
   structurally immune to position bias.
