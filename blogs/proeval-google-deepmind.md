# ProEval: Proactive Failure Discovery and Efficient Performance Estimation

Source: https://arxiv.org/abs/2604.23099  
Publisher: Google DeepMind  
Date: April 2026

---

## Problem they were solving

Evaluating generative AI models is expensive: slow inference, costly human raters, and an exploding number of models and benchmarks to cover. Current practice downsamples test data to save cost — but this misses rare failure cases and produces inaccurate performance estimates. ProEval asks: *can we choose which inputs to evaluate next, rather than sampling randomly?*

---

## Approach / engineering decisions

Two objectives, both addressed with the same Gaussian Process (GP) framework:

**1. Performance estimation** — formulated as **Bayesian quadrature (BQ)**
- A GP acts as a surrogate that maps inputs → error severity scores
- Actively selects the next input to evaluate using a variance-reduction acquisition function
- Result: estimate model performance within ±1% MAE using only 1–27 evaluated inputs (vs. hundreds with random sampling)

**2. Failure discovery** — formulated as **superlevel set sampling**
- Identify input regions where the model fails (error score above a threshold)
- Three escalating strategies:
  1. Superlevel set sampling from the GP posterior
  2. Generative synthesis — use an LLM (Gemini 3 Pro) to generate new failure-inducing inputs anchored on known failures
  3. Topic-aware exploration — multi-armed bandit to force semantic diversity across failure cases

**Transfer learning across models/benchmarks:**
- Instead of starting from scratch per model, build an informed GP prior using:
  - Score features from historical eval results on the same benchmark across other models
  - Learned text embeddings (pre-trained, with optimizable parameters) for cross-benchmark transfer
- Falls back gracefully when fewer than 3 similar models exist in history

---

## Metrics / eval methods used

**Performance estimation:**
- MAE between estimated and ground-truth performance
- Samples needed to reach ±1% MAE (primary efficiency metric)

**Failure discovery:**
- Cumulative failures found and failure rate
- Samples to First Failure (SFF)
- Diversity: log-determinant of embedding Gram matrix + topic entropy (Shannon entropy of topic distribution)
- Composite diversity score combining the above

Tested across 9 benchmarks (reasoning, knowledge, safety, visual) with 16 LLM/VLM models.

---

## Key results

- **8–65x fewer samples** to reach 1% estimation accuracy vs. baselines
- **2–5x more diverse** failure cases discovered vs. LLM-based generation baselines
- Works in three generalization scenarios: default (known models/benchmarks), new model, new benchmark
- Gemini 3 Pro used as the query generator; human verification showed 90% of generated questions were valid — so reported failure rates are lower bounds

---

## Limitations

- GP prior depends on historical data — degrades when fewer than 3 similar models exist
- Embedding quality bottleneck for new benchmarks (assumes off-the-shelf embeddings capture relevant semantics)
- Uses Gaussian observation model instead of Bernoulli/probit for computational tractability — may not be optimal for binary pass/fail outcomes
- Failure discovery quality is bounded by the LLM generator's capability

---

## Background concepts

### Gaussian Process (GP)

A GP is a probability distribution over functions — instead of predicting a single value, it predicts a **distribution** with a mean and uncertainty at every point.

Given a few evaluated inputs and scores:
```
input x₁ → score 0.8
input x₂ → score 0.2
input x₃ → score 0.6
```

The GP fits a function through these points and gives for any new input x:
- **Mean prediction** μ(x) — best guess of the score
- **Variance** σ²(x) — how uncertain it is (high where you haven't evaluated)

The outputs at multiple inputs are **jointly Gaussian** and correlated via a **kernel function** k(x_i, x_j) — same Gaussian RBF idea as CMMD. In ProEval, inputs are text prompts, so the kernel is built on text embeddings.

When you evaluate a new input and observe its score, the GP **updates its beliefs everywhere** via Bayes' rule (closed-form, no gradient descent). This is the posterior update:
```
prior (no data) → posterior (after n evaluations)
```

GP is a decades-old method from statistics (kriging, 1950s–60s); the canonical ML reference is Rasmussen & Williams (2006).

---

### Bayesian Quadrature (BQ)

BQ solves the problem of estimating an **average** over many inputs when evaluating each input is expensive:

```
μ = ∫ f(x) p(x) dx    ← true average performance over all prompts
```

**Standard Monte Carlo** approximates this by sampling randomly and averaging — needs ~100–1000 samples for 1% accuracy.

**BQ uses the GP surrogate** to do better: since the GP predicts f(x) everywhere, you can integrate it analytically over p(x):

```
μ_BQ = ∫ μ_GP(x) p(x) dx         ← integrate GP mean (closed form)
σ²_BQ = ∫∫ k(x,x') p(x)p(x') dx dx'  ← uncertainty on the estimate
```

This gives an accurate estimate with far fewer real evaluations because the GP fills in the gaps between evaluated points.

**Active selection (acquisition function):** once you have BQ, pick the next input to evaluate where GP uncertainty is highest — that's where a new observation reduces overall uncertainty the most:

```
x_next = argmax σ²_GP(x)
```

Repeat: evaluate → GP updates → pick next highest-uncertainty point → repeat. ProEval achieves ±1% MAE with 1–27 evaluations this way.

BQ also predates this paper (O'Hagan, 1991). ProEval's contribution is applying GP + BQ to generative AI eval with a transfer-learning warm start.

---

## What's reusable / what to remember

1. **Active evaluation is the right frame** — random sampling of test inputs is wasteful; treat eval as a sequential decision problem where each evaluation informs the next

2. **Bayesian quadrature for performance estimation** — a principled way to get accurate aggregate metrics with far fewer samples; applicable any time your eval is expensive (human raters, slow inference)

3. **Failure discovery ≠ performance estimation** — they require different objectives and metrics; don't conflate them. A model can have good average performance and many rare but severe failure modes simultaneously

4. **Diversity of failures matters** — finding 100 semantically identical failures is less useful than finding 10 diverse ones; topic entropy + embedding diversity are concrete ways to measure this

5. **Directly relevant to Gemini eval pipelines** — uses Gemini 3 Pro as generator, tests VLMs, and is from DeepMind; methodology is likely being used internally for Nano Banana / Imagen evaluation
