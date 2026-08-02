# Alignment Update Rules — PPO vs DPO vs GRPO Gradients Compared

How the three preference-optimization methods actually move parameters. All three are the same
species — **a scalar weight times `∇log π`** — differing only in where the weight's baseline comes
from, its granularity, and where the brake is. Companion to `llm-alignment-math.md` (loss
functions/derivations) and `llm-alignment-methods.md` (method survey).

---

## 1. The three updates, written to look alike

**PPO** (token-level, as used in RLHF):
```
L_PPO = E_t [ min( ρ_t·A_t,  clip(ρ_t, 1−ε, 1+ε)·A_t ) ]
gradient ≈ E_t [ ρ_t · A_t · ∇log π_θ(y_t | x, y_<t) ]

ρ_t = π_θ/π_old   per-token importance ratio vs the behavior (old) policy
A_t               advantage from the critic + GAE (explicit, per-token)
```

**GRPO** (PPO machinery, critic deleted; group of G responses per prompt):
```
J = E [ (1/G) Σ_i (1/|y_i|) Σ_t min( ρ_{i,t}·A_i, clip(ρ_{i,t},1±ε)·A_i ) ] − β·KL(π_θ‖π_ref)
gradient ≈ (1/G) Σ_i A_i · [ (1/|y_i|) Σ_t ∇log π_θ(y_{i,t}|…) ] − β·∇KL

A_i = ( r_i − mean(r_1..G) ) / std(r_1..G)   sequence-level advantage from the GROUP MEAN
ρ_{i,t} per-token ratio + clip (kept from PPO); KL explicit in the loss (k3 estimator)
```

**DPO** (derivative of −log σ(m), m = implicit margin; see math file §6):
```
gradient = −β · σ(−m) · [ Σ_t ∇log π_θ(y_t^w|…) − Σ_t ∇log π_θ(y_t^l|…) ]

σ(−m) = P(pair currently ranked WRONG) — one weight for the whole pair
```

**Hybrid anatomy of GRPO:** advantage is sequence-level (like DPO), ratio/clip per-token (like
PPO), on-policy (like PPO — G fresh rollouts per prompt per step).

---

## 2. Three-way comparison

| | PPO | GRPO | DPO |
|---|---|---|---|
| **Weight** | `ρ_t·A_t`, per token | `ρ_{i,t}·A_i`, seq-level A + token-level ρ | `β·σ(−m)`, per sequence |
| **Baseline** | learned critic `V(s)` | **group mean** of G sampled rewards | **paired sibling** (the `y_l` term) |
| **Sign/magnitude source** | critic + GAE | reward variance *within the group* | BT margin confidence |
| **Credit assignment** | per-token (GAE) | none within sequence | none within sequence |
| **Trust region** | clip vs π_old + KL to ref | clip vs π_old + explicit KL term | implicit anchor to π_ref inside `m` |
| **Data** | on-policy rollouts | on-policy, G per prompt | offline pairs |
| **Reward source** | reward model | rule-based/verifiable ideally | implicit (log-ratio to ref) |
| **Gradient dies when…** | ratio leaves [1−ε,1+ε] | clip — **or zero reward variance in the group** (`std→0 ⇒ A_i=0`) | pair confidently ordered (`σ(−m)→0`) |
| **Models in memory** | 4 (policy, old/ref, RM, critic) | 2–3 (no critic; no RM if rule-based) | 2 |
| **Cost center** | everything | shifted to **generation** | two forward passes per pair |

---

## 3. Failure modes read straight off the gradient forms

**PPO:**
- Clip stops updates because the policy *moved*, not because it's *right* → over-optimizes the RM
  proxy (Goodhart/reward hacking).
- Pays for on-policy unbiasedness with single-sample-per-token noise → needs critic, GAE, many
  rollouts.

**GRPO:**
- **Degenerate groups**: all G responses equally rewarded (all wrong on hard prompt / all right on
  easy) → `A_i = 0` → zero gradient. Signal exists only where the policy is *uncertain* — a quiet
  built-in curriculum; motivates difficulty-filtered prompts.
- **std-normalization bias**: dividing by `std(r)` trains tiny-gap groups as hard as huge-gap
  groups (equal pressure regardless of true effect size). Dr. GRPO flags this + the `1/|y_i|`
  length term as optimization biases; recommends dropping the std division.
- **Verifiable-reward dependence**: rule-based reward ⇒ no RM to hack; subjective tasks (e.g. mesh
  part-importance) force an RM back in → bias re-enters through it.

**DPO:**
- **Gap-only constraint** (BT unidentifiability, math file §4/§6): the loss sees only the margin
  `m`; absolute likelihoods are free ⇒ empirically `log π(y_w)` and `log π(y_l)` often **both
  decrease** (rejected falls faster). PPO avoids this particular failure (real rewards + KL pin
  the scale) but has its own.
- `σ(−m)` stops on *confidence*, blind to how far π_θ drifted from the data distribution →
  overfits a fixed offline dataset (distribution shift).

---

## 4. The unified picture — it's all baseline shopping

Every update is `E[ w · ∇log π ]`; the design space is three knobs:

```
PPO:   A = r − V_critic(s)                baseline = a LEARNED NETWORK   (expensive, per-token)
DPO:   direction = ∇log π(y_w) − ∇log π(y_l)  baseline = the PAIRED SIBLING (free, offline, seq-level)
GRPO:  A_i = r_i − mean(group)            baseline = the GROUP MEAN       (G samples, on-policy, seq-level)

weight granularity:   per-token (PPO)  →  sequence-level (DPO, GRPO advantages)
brake placement:      ratio clip (PPO, GRPO)  vs  BT margin saturation (DPO)
```

**Family-resemblance check (G=2 corner case):** GRPO with two samples and rewards {1,0} gives
advantages {+1,−1} — push winner up, loser down, on-policy, with a clip = an **online,
trust-regioned cousin of DPO** (sibling baseline, minus offline pairs, plus rollouts, with clip
instead of the `σ(−m)` margin weight).

**One-line takeaway:** PPO buys its baseline with a critic, DPO gets it free from the pair, GRPO
computes it from the group — same `w·∇log π` skeleton; everything else is *where the baseline
comes from, what granularity the weight has, and where you put the brake.*
