# HEIM: Holistic Evaluation of Text-to-Image Models

## Problem

Existing benchmarks focus narrowly on text-image alignment and image quality, ignoring critical dimensions like originality, aesthetics, toxicity, and bias. Automated metrics also correlate weakly with human judgment, creating a false sense of completeness.

## Proposed metric / method

HEIM is a multi-aspect evaluation framework with four components:
- **Aspects** — 12 evaluative dimensions: Alignment, Quality, Aesthetics, Originality, Reasoning, Knowledge, Bias, Toxicity, Fairness, Robustness, Multilinguality, Efficiency
- **Scenarios** — 62 specific use cases with curated prompt sets
- **Adaptations** — zero-shot prompting strategies
- **Metrics** — 25 human and automated measurement methods

## How it's computed

**Automated metrics by aspect:**
- Alignment: CLIPScore
- Quality: FID, Inception Score
- Aesthetics: LAION Aesthetics predictor, Fractal coefficient
- Toxicity: NSFW detector, NudeNet, watermark detection
- Bias: gender / skin tone classifiers
- Reasoning: object detection accuracy

**Human evaluation:**
- Crowdsourced 1–5 scale ratings for alignment, photorealism, aesthetics, originality
- Minimum 5 raters per image; 100+ samples per aspect
- Automated-to-human correlation is weak: 0.42 (alignment), 0.59 (quality), 0.39 (aesthetics) — key finding, not a footnote

---

## Metric Deep Dives

### FID (Fréchet Inception Distance)

**What it measures:** distributional similarity between real and generated images.

**How it works:**
1. Run both image sets through **Inception-v3** → each image becomes a **2048-dim vector** from the pooling layer
2. Fit a multivariate Gaussian to each set: compute mean μ (shape: 2048) and covariance Σ (shape: 2048×2048)
3. Compute Fréchet distance between the two Gaussians:

```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))
```

Lower = better (distributions are closer).

**Key assumptions and failure modes:**
- Inception-v3 trained on ImageNet → biased toward natural photos, undervalues art/illustration styles
- Needs **10k+ images** to estimate the 2048×2048 covariance reliably; small batches make Σ rank-deficient and the matrix square root numerically unstable
- Measures distribution-level similarity, not per-image quality
- `clean-fid` fixes a bug in the original: wrong resize filter (PIL BILINEAR vs. LANCZOS) caused systematic FID inflation

---

### Inception Score (IS)

**How it works:**
1. Run generated images through full Inception-v3 classifier → class probability p(y|x)
2. Good images should have: (a) sharp, confident per-image predictions (low entropy p(y|x)) and (b) diverse classes across the set (high entropy marginal p(y))
3. IS = exp(E[KL(p(y|x) || p(y))])

Higher = better.

**Why it's weaker than FID:** doesn't use real images at all; Inception-v3 bias is worse here; largely considered secondary.

---

### CLIPScore

**How it works:**
1. Encode image with CLIP vision encoder → image embedding
2. Encode prompt text with CLIP text encoder → text embedding
3. CLIPScore = max(100 × cosine_similarity(image_emb, text_emb), 0)

Higher = better alignment.

**How CLIP is trained:**
- Trained on 400M (image, text) pairs from the web using **contrastive learning**
- For a batch of N pairs, build an N×N similarity matrix S where S[i][j] = cosine_similarity(image_i, text_j)
- The diagonal (matching pairs) should be high; off-diagonal (mismatched) should be low
- Loss = cross-entropy applied in both directions (image→text and text→image), averaged:

```python
labels = [0, 1, 2, ..., N-1]  # correct answer is always the diagonal

loss_i2t = cross_entropy(S,   labels)
loss_t2i = cross_entropy(S.T, labels)
loss = (loss_i2t + loss_t2i) / 2
```

- Cross-entropy for row i: apply softmax over S[i], then take -log of the probability assigned to the correct column
- A learned **temperature τ** scales similarities before softmax (S / τ) — small τ sharpens the distribution and drives stronger gradients
- CLIP used batch size N=32,768 — the model must rank the correct caption #1 out of 32,768, which forces rich discriminative representations

**Compositional blindspot:**
- Both encoders produce a **single vector** — all relational information is lost in that bottleneck
- "A red cube on top of a blue sphere" → CLIP sees a bag of concepts (cube, red, blue, sphere), cannot verify spatial relations or attribute binding
- A wrong image (blue cube on red sphere) scores nearly identically → CLIPScore is unreliable for compositional prompts
- This is structural, not a training data issue — T2I-CompBench exists specifically to address this gap

**Human correlation: 0.42** (from HEIM) — reflects the compositional failure on a large fraction of their prompt set.

---

### LAION Aesthetics Predictor

**How it works:**
1. Extract CLIP image embedding (reuses the same vision encoder)
2. Pass through a small **linear regression head** trained on LAION's crowdsourced 1–10 aesthetic ratings
3. Output: score 1–10

**Key bias:** "aesthetic" = what LAION crowdworkers in 2022 rated as visually appealing — skews toward photorealistic, high-contrast, western art styles. Abstract, minimalist, or culturally specific aesthetics score poorly.

---

### Toxicity / Safety

Three separate detectors stacked — each catches different failure modes:
- **NSFW detector** — binary classifier on image features
- **NudeNet** — fine-grained nudity/exposure classifier
- **Watermark detection** — flags training data leakage artifacts

**Shared failure mode across all metrics:** FID, IS, CLIPScore, and LAION Aesthetics all reuse Inception-v3 or CLIP as backbone. Any systematic bias in those backbones propagates to every metric built on top — this is a structural reason why automated-human correlation stays low (0.39–0.59) even across different metric types.

## Compared against

26 models evaluated:
- **Diffusion**: Stable Diffusion v1-4, v1-5, v2, v2-1; Dreamlike Photoreal 2.0; Openjourney; Redshift; SafeStableDiffusion; DeepFloyd-IF; Vintedois
- **Autoregressive**: DALL-E 2 (3.5B), DALL-E mini (0.4B), DALL-E mega (2.6B), minDALL-E (1.3B), CogView2 (6B)
- **Other**: MultiFusion (13B), GigaGAN, Promptist + SD, Lexica Search

**Key findings:**
- DALL-E 2: best text-image alignment
- Dreamlike Photoreal 2.0: best photorealism and knowledge tasks
- Openjourney: highest aesthetics and originality
- minDALL-E / CogView2: lowest bias and toxicity
- No single model excels across all aspects — tradeoffs are inherent
- All models fail at reasoning (best: 47.2% accuracy)
- All models degrade on non-English prompts
- GigaGAN fastest (0.14s); autoregressive models ~2s slower than diffusion

## Limitations

- 12 aspects are not exhaustive
- Bias metrics only cover binary gender and skin tone
- Efficiency measured by wall-clock time, not energy consumption
- Subjective metrics rely on general crowdworkers, not domain experts
- Generated images include potentially harmful content

## Relevance to engineering practice

1. **Don't rely on one metric** — FID and CLIPScore alone are insufficient; automated-human correlation is 0.39–0.59 at best
2. **Budget for human eval** — it's non-negotiable for aesthetics, originality, and alignment quality
3. **Be explicit about tradeoffs** — alignment vs. aesthetics vs. safety cannot all be maximized; pick your priority
4. **Design for extensibility** — HEIM's modular structure (aspects → scenarios → metrics) is the right pattern for any serious eval system
5. **Test multilingual and demographic coverage** — document it; measure performance drops systematically
6. **Release eval artifacts** — generated images + scores enable reproducibility and comparison

## Source

https://arxiv.org/abs/2311.04287
