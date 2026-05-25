# Active / Efficient Evaluation

*Updated: 2026-05-21. Sources: ProEval (Google DeepMind, 2026)*

---

## The Problem

Standard eval practice: randomly sample N inputs, evaluate each, average the scores. This is wasteful — every sample is treated equally regardless of how much information it adds. At scale (slow inference, expensive human raters, hundreds of models to compare), this becomes a bottleneck.

Two distinct subproblems that require different approaches:
1. **Performance estimation** — get an accurate average score with as few evaluations as possible
2. **Failure discovery** — find diverse inputs where the model fails, not just measure how often it fails

---

## Performance Estimation via Bayesian Quadrature

### Core idea

Use a **Gaussian Process (GP)** as a surrogate for the score function f(x) (where x is an input prompt and f(x) is the score):
- GP predicts f(x) at unevaluated inputs with a mean and uncertainty
- Integrate the GP analytically over the input distribution → estimated average performance with uncertainty bounds
- This is **Bayesian Quadrature (BQ)**: replacing random sampling with GP-informed integration

### Why it's better than Monte Carlo

Monte Carlo estimates the average by sampling randomly and averaging — needs ~100–1000 samples for 1% accuracy. BQ uses the GP to fill in unevaluated regions analytically, achieving ±1% MAE with **1–27 samples** (ProEval result).

### Active selection

After each evaluation, pick the next input where GP uncertainty is highest — that's where a new observation reduces overall uncertainty the most:
```
x_next = argmax σ²_GP(x)
```
GP updates after each observation (posterior update via Bayes' rule, closed-form). Repeat until estimate is tight enough.

### Transfer learning warm-start

Instead of starting with a flat GP prior, warm-start using historical eval results from similar models on the same benchmark. Requires ≥3 similar models in history to be effective; falls back to uninformed prior otherwise.

**ProEval result:** 8–65x fewer samples than competitive baselines to reach 1% estimation accuracy.

---

## Failure Discovery

Finding where a model fails is different from measuring average performance — you want **diverse** failure cases, not just many failures.

### Three escalating strategies (ProEval)

1. **Superlevel set sampling** — use the GP posterior to identify input regions where f(x) > failure threshold; sample from those regions
2. **Generative synthesis** — use an LLM (Gemini 3 Pro in ProEval) to generate new failure-inducing prompts anchored on known failures
3. **Topic-aware exploration** — multi-armed bandit to force semantic diversity across failure cases, preventing clustering around one failure type

### Measuring failure diversity

Two concrete metrics:
- **Log-determinant of embedding Gram matrix** — measures how spread out failure cases are in embedding space
- **Topic entropy (Shannon entropy)** — measures how evenly failures are distributed across semantic topics

**ProEval result:** 2–5x more diverse failure cases than LLM-generation baselines.

---

## Key Engineering Takeaways

1. **Treat eval as a sequential decision problem** — each evaluation should inform which input to evaluate next; random sampling is the baseline to beat, not the default to use

2. **Performance estimation and failure discovery are separate objectives** — a model can have good average performance and many rare severe failures simultaneously; don't conflate the two

3. **Diversity of failures matters operationally** — finding 100 semantically identical failures gives you one bug; finding 10 diverse failures gives you 10 bugs. Measure diversity explicitly

4. **Active eval is especially valuable when eval is expensive** — human raters, slow inference (large diffusion models), or many models to compare; the more expensive each evaluation, the higher the ROI of active selection

5. **Directly applicable to Gemini/Nano Banana pipelines** — ProEval uses Gemini 3 Pro as the failure-synthesis generator; the methodology is likely used internally for Imagen eval

---

## Open Questions

- How does active eval interact with distributional shift? If the GP is warm-started on one model family but evaluated on a very different model, does active selection still work?
- Can failure synthesis be done without a strong LLM generator? ProEval's failure rate is a lower bound because the generator (Gemini 3 Pro) may not always produce valid failures
- What's the right failure threshold? ProEval treats it as a hyperparameter — in practice, who decides what counts as a "failure"?
