# Alignment Math Primer — Softmax, Cross-Entropy, Bradley–Terry, KL, and the DPO Derivation

The math underneath the alignment toolbox: the loss functions, notation, and derivations that
`llm-alignment-methods.md` uses. Read alongside `qwen3vl-4b-position-bias.md` — the recurring
theme is that each objective **fits its signal faithfully, biases included**, which is where
small-VLM comparison bias enters. The update-rule mechanics (PPO vs DPO vs GRPO gradients) are in
`llm-alignment-gradients.md`.

---

## 1. Policy notation — what π_θ and π_ref are

Notation borrowed from RL: **π (pi) = policy** = a probability distribution over actions given a
state. A language model IS a policy:

| RL concept | LLM equivalent |
|---|---|
| state `s` | prompt + tokens so far: `(x, y_<t)` |
| action `a` | next token `y_t` |
| policy `π(a\|s)` | vocab softmax `π(y_t \| x, y_<t)` |
| trajectory | full response `y`, with `π(y\|x) = Π_t π(y_t \| x, y_<t)` |

The two policies in every alignment loss:

```
checkpoint after SFT ──┬──► π_θ   trainable — gradients update θ every step ("the model you're producing")
                       └──► π_ref  frozen snapshot of the SFT model, never updated
                                   ("the last model you trusted"; answers: what would the
                                    sensible pre-RL model have said?)
```

- The quantity that matters is the **ratio / log-ratio** `π_θ(y|x) / π_ref(y|x)`: how much more
  (or less) the new model likes a response than the old one did. RLHF taxes its divergence (KL
  penalty); DPO's entire implicit reward IS this log-ratio.
- In code: one checkpoint instantiated twice; π_ref in eval mode, no gradients, no optimizer
  states.
- Health metric: watch `KL(π_θ‖π_ref)` during a run — growing too fast → reward hacking; stuck at
  0 → nothing being learned.

---

## 2. Softmax — the shared read-out

**Job:** turn arbitrary real logits into a distribution (non-negative, sums to 1, monotonic,
differentiable):

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```
Two steps with separate jobs: **exponentiate** (positivity; additive gaps → multiplicative ratios)
then **normalize** (share exactly 1 unit of probability). "Softened argmax."

**The one key property — gaps become odds:**
```
p_i / p_j = exp(z_i − z_j)          1 logit unit = e ≈ 2.72× odds

logits  [2.0,  1.0,  0.1]  →  exp  [7.39, 2.72, 1.11]  →  probs [0.659, 0.242, 0.099]
```
Shared denominator ⇒ raising one logit pushes ALL others down (forced competition — why the CE
gradient is `q − p`).

**Three facts that explain everything downstream:**
1. **Shift invariance**: `softmax(z + c) = softmax(z)` — only *differences* exist. Explains: BT
   rewards identifiable only up to a constant; DPO's `Z(x)` cancellation; the `exp(z − max)`
   overflow trick (free, by shift invariance).
2. **Sigmoid = 2-class softmax**: `σ(r_i − r_j) = softmax over [r_i, r_j], class i`. Bradley–Terry
   isn't analogous to softmax — it IS softmax with vocabulary size 2.
3. **Softmax = ∇ log-sum-exp**: `∇_z [log Σ exp(z_j)] = softmax(z)`. LSE ≈ smooth max; its
   gradient ≈ smooth argmax. This is why the pair (LSE, softmax) appears everywhere.

**Temperature**: `q_i = exp(z_i/T)/Σ` scales the gaps ⇒ scales the odds: T=2 square-roots every
odds ratio. T→0 argmax, T→∞ uniform, T=1 default. Generation "temperature" is this knob.

**Physics reading (why exp, not something else):** softmax is the Boltzmann/Gibbs distribution —
logits are negative energies, the denominator is the partition function `Z` (the same `Z(x)` as in
the RLHF→DPO derivation). exp is the unique function turning additive quantities (energy,
evidence) into multiplicative ones (weight, odds).

**Where it appears:** attention weights `softmax(QKᵀ/√d)`; LM head (151,936-way); Bradley–Terry
(2-way); DPO (σ of implicit margin). One function: scores → exclusive normalized distribution.

---

## 3. Cross-entropy (CE) — and its fusion with softmax

**Information-theoretic origin:**
```
Entropy:       H(p)   = − Σ_i p_i log p_i      irreducible uncertainty of the data
Cross-entropy: H(p,q) = − Σ_i p_i log q_i      avg code length using q's code on p's data
Decomposition: H(p,q) = H(p) + D_KL(p ‖ q)      → min CE ⇔ min KL (H(p) is constant)
```

**Reduction to NLL:** one-hot target collapses the sum: `H(p,q) = −log q_{y*}`. CE = maximum
likelihood. Per-token LM loss (151,936-way for Qwen3-VL); `perplexity = exp(mean L)`.

**The fused form (softmax substituted into CE) collapses:**
```
L = −log( exp(z_{y*}) / Σ_j exp(z_j) ) = log-sum-exp(z) − z_{y*}
```
≈ (smooth max of all logits) − (true logit) — roughly **the margin by which the true class fails
to be on top**.

**Gradient — the whole reason for the pairing:**
```
∂L/∂z_i = q_i − p_i        (softmax output − one-hot target)

true class:  q_i − 1 ≤ 0  → pushes z_i UP
others:      q_i     ≥ 0  → pushes DOWN ∝ probability wasted there
Σ_i(q_i − p_i) = 0        → updates REDISTRIBUTE mass, never create it
```
- **No saturation**: gradient at full force exactly when confidently wrong. (MSE-on-softmax
  carries extra `q(1−q)` factors → vanishes when most wrong — why sigmoid+MSE died.)
- **Self-regulating**: confident-and-right ⇒ gradient ≈ 0.

**Worked example** (logits [2.0, 1.0, 0.1], true class 0):
```
q = [0.659, 0.242, 0.099]     L = −log 0.659 = 0.418
check: LSE = log 11.22 = 2.418;  L = 2.418 − 2.0 = 0.418 ✓
grad = [−0.341, +0.242, +0.099]  (sums to 0 ✓)
if true class were 2: L = 2.31, grad = [+0.659, +0.242, −0.901]  ← full-force correction
```

**Why a canonical pair, not a convention:** (1) CE is the NLL of the categorical distribution
softmax parameterizes (MLE); (2) softmax is the *canonical link* of the categorical exponential
family — pairing link with matching NLL makes the loss **convex in the logits**, collapsing the
Hessian to yield `q − p`; (3) max-entropy duality.

**Practical:** frameworks fuse via LSE with max-subtraction. `F.cross_entropy(logits, target)`
takes RAW logits; never compute `log(softmax(z))` yourself (underflow).

**Alignment relevance:** SFT = token-level CE; CE faithfully matches the model to *whatever
statistics the data has*, including skews. CE has no opinion about the data. Its calibration
sensitivity is also what makes logprobs/`prompt_logprobs` meaningful (basis of the
permutation-logprob debiasing recipe in `qwen3vl-4b-position-bias.md`).

---

## 4. Bradley–Terry (BT) pairwise loss

**Origin:** Bradley–Terry (1952), ranking chess players from match outcomes only — never absolute
strength, only who beat whom. Latent strength `r_i` per item:

```
P(i ≻ j) = exp(r_i) / ( exp(r_i) + exp(r_j) ) = σ( r_i − r_j )
```
2-class softmax over the two strengths (Section 2, fact 2). Only the *difference* identifiable
(shift invariance, fact 1).

**Fitting = binary CE** on the margin — directly from the Section-3 fused form with two logits:
```
L_BT = LSE([r_w, r_l]) − r_w = log(1 + exp(r_l − r_w)) = − log σ( r_w − r_l )
```

**Gradient = self-pacing curriculum** — magnitude equals the probability the model currently
assigns to the WRONG ordering (just `q − p` at vocab size 2):
```
∂L/∂(margin) = −σ( −(r_w − r_l) ) = −P(model prefers the loser)

margin      +3        +1        0       −1       −3
loss        0.049     0.313     0.693   1.313    3.049
grad weight 0.047     0.269     0.500   0.731    0.953   ← confident-wrong pairs dominate
```

**Why comparisons, not absolute scores:** humans are inconsistent at absolute ratings but reliable
at pairwise judgments (high inter-annotator agreement). BT converts the signal humans can provide
into a scalar reward.

**Structural consequences (explain real failure modes):**
1. **Only differences pinned down** — per-prompt rewards free up to an additive constant. Exactly
   why DPO's derivation works: `Z(x)` cancels between chosen and rejected (Section 6).
2. **Magnitude discarded** — "slightly better" and "vastly better" are the same binary label
   (ties usually dropped; Rao–Kupper variant models them).
3. **Transitivity assumed** — cyclic human preferences (A≻B≻C≻A) get flattened onto a 1-D scale.
4. **Feature absorption** — anything correlated with winning (length, confidence, **option
   position**) becomes a feature of `r`. The loss does its job perfectly; the data lies. This is
   the position-bias entry point into the reward model.

---

## 5. KL divergence

**Definition & three readings:**
```
D_KL(p ‖ q) = Σ_i p_i · log( p_i / q_i ) = E_p[ log p − log q ]
```
1. Coding: excess nats/symbol compressing `p`'s data with `q`'s code.
2. Expected log-likelihood ratio — evidence against `q` per sample.
3. `H(p,q) = H(p) + D_KL(p‖q)` — the part of cross-entropy the model can fix.

**Properties:** `≥ 0` (Gibbs, via Jensen); `= 0 ⇔ p = q`; **not symmetric**, not a metric; nats
(ln) or bits (log2).

**The asymmetry is the point.** Example `p=[0.5, 0.5, 0]`, `q=[0.25, 0.25, 0.5]`:
```
D_KL(p‖q) = 0.5·ln2 + 0.5·ln2 = 0.693 nats     (finite)
D_KL(q‖p) = ... + 0.5·ln(0.5/0) = ∞            (explodes)
```
Direction with `p` in front: any event `p` cares about that `q` ignores costs ∞; `q`'s mass where
`p=0` costs nothing. When `q` is being fit:

| | Penalizes | Behavior |
|---|---|---|
| `D_KL(p‖q)` forward | q missing any p-mass region | mode-covering (conservative) |
| `D_KL(q‖p)` reverse | q mass where p has none | mode-seeking (hunkers on subset) |

**Why RLHF penalizes `D_KL(π_θ ‖ π_ref)` in that direction:** the trainable policy is the first
argument → π is taxed for putting probability *where the reference has none* → the policy cannot
leave the reference's support. The RM is only trustworthy on the distribution it was trained on;
this KL term keeps the policy on that distribution — a soft trust region measured in nats, and the
reward-hacking defense.

**Estimators in practice:** per-token log-ratio `log π_θ − log π_ref` (PPO reward shaping); GRPO
uses the non-negative unbiased **k3 estimator** `(r − 1) − log r` with `r = π_ref/π_θ`.

**One intuition:** CE asks "how surprised is the model?"; KL asks "how much of that surprise is
the model's own fault?"

---

## 6. Deriving the DPO loss from Bradley–Terry

Three moves: solve the RLHF objective in closed form → invert to get the reward → feed to BT.

**Step 0 — the RLHF objective** (reward inside a trained RM; π_ref frozen SFT; β = KL tax):
```
max_π  E_{x, y~π} [ r(x,y) ] − β · D_KL( π(y|x) ‖ π_ref(y|x) )
```

**Step 1 — closed-form optimum.** Expand KL per-prompt and complete the distribution:
```
objective = −β · E_{y~π} [ log ( π(y) / (π_ref(y)·exp(r/β)) ) ]
Z(x)      = Σ_y π_ref(y|x)·exp(r(x,y)/β)                       (partition function)
π*(y|x)   = (1/Z(x)) · π_ref(y|x) · exp( r(x,y)/β )            (proper distribution)

objective = −β · D_KL( π ‖ π* ) + β·log Z(x)
```
`D_KL ≥ 0` with equality iff `π = π*` ⇒ **optimal policy = a reward-tilted reference** (reweight
SFT by exp(r/β); larger β → less tilt). PPO was only ever a clunky way to *find* this.

**Step 2 — invert: solve for the reward.**
```
r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```
Reward recoverable from the policy up to the per-prompt constant `β·log Z(x)` — matching BT's own
unidentifiability (Section 4, consequence 1).

**Step 3 — substitute into BT.** `P(y_w ≻ y_l|x) = σ(r_w − r_l)`:
```
r_w − r_l = β[ log π*(y_w)/π_ref(y_w) − log π*(y_l)/π_ref(y_l) ] + βlogZ(x) − βlogZ(x)
                                                                  └────────┬────────┘
                                                            cancels: same x; softmax
                                                            shift-invariance strikes again
P(y_w ≻ y_l | x) = σ( β·log[π*(y_w)/π_ref(y_w)] − β·log[π*(y_l)/π_ref(y_l)] )
```
Unknown reward gone; intractable normalizer gone. Only policies remain.

**Step 4 — MLE over the preference dataset** with π_θ standing in for π*:
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L_DPO = − E_D [ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x))                          │
│                      − β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]                      │
└──────────────────────────────────────────────────────────────────────────────┘
```
Since `log π(y|x) = Σ_t log π(y_t|…)`, the implicit reward is **β × the summed per-token
log-prob ratio vs the SFT model** — two forward passes to compute.

**Gradient** (m = implicit margin):
```
∇L = −β · E_D [ σ(−m) · ( ∇log π_θ(y_w|x) − ∇log π_θ(y_l|x) ) ]
```
`q − p` again: confidently-misranked pairs dominate; solved pairs fade out.

**Costs visible from the derivation:** everything is relative to π_ref (bad ref ⇒ bounded DPO);
offline (Step 1 assumed the TRUE reward fixed — the implicit one stops tracking it off-distribution);
implicit reward is a sum of per-token log-ratios ⇒ length/style shift it systematically, and
inherited skews (position bias included) pass through unbuffered.

---

## 7. The unifying picture

```
token CE:      softmax over 151,936 classes,  target = which token came next
Bradley–Terry: softmax over 2 classes,        target = which response humans preferred
DPO:           Bradley–Terry, reward reparameterized as β·log(π_θ/π_ref)
```

One formula — `−log(correct class's softmax probability)` — at three levels of the stack:
learning to talk (SFT), learning to judge (RM), learning to prefer without a judge (DPO).

And where KL sits across the methods:
```
SFT:   min H(data, π) = H(data) + D_KL(data ‖ π)   forward KL, mode-covering:
                                                     must cover ALL demo behaviors (incl. biases)
RLHF:  reward − β·D_KL(π_θ ‖ π_ref)                trust region on RM's known turf
DPO:   KL hidden in the math — π* ∝ π_ref·exp(r/β) is the KL-constrained optimum; same β
GRPO:  explicit −β·KL term via k3 estimator
```

Shared property: every objective fits its signal faithfully, **including the signal's biases** —
durable fixes live in data/reward design, not in swapping losses.
