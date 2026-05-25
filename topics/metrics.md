# Image Generation Metrics — Synthesis

*Updated: 2026-05-21. Sources: HEIM (2023), cmmd-pytorch, Efficient Adjoint Matching (2026)*

---

## The Backbone Problem

Every major automated metric in image gen eval reuses one of two pretrained encoders:
- **Inception-v3** — FID, Inception Score
- **CLIP** — CLIPScore, LAION Aesthetics predictor

This means all automated metrics share the same failure modes: both encoders are biased toward natural photos (ImageNet / web image-caption pairs), and any systematic blind spot in the backbone propagates to every metric built on top. This is a core reason why automated-to-human correlation stays at 0.39–0.59 even across supposedly different metrics.

---

## Metric Categories

### Distribution-level Quality

**FID (Fréchet Inception Distance)**
- Runs images through Inception-v3 pooling layer → 2048-dim vectors
- Fits a multivariate Gaussian (μ: 2048-dim, Σ: 2048×2048) to both real and generated sets
- Computes Fréchet distance between the two Gaussians
- Lower = better; needs 10k+ samples for stable Σ estimation
- `clean-fid` fixes a resize filter bug that caused systematic inflation in the original

**Inception Score (IS)**
- Uses full Inception-v3 classifier (not just pooling layer)
- Measures sharpness (confident per-image predictions) + diversity (varied classes across set)
- Does not use real images → weaker than FID; largely secondary

### Text-Image Alignment

**CLIPScore**
- Cosine similarity between CLIP image embedding and CLIP text embedding
- CLIP trained via contrastive learning on 400M web pairs: N×N similarity matrix, diagonal should be high, cross-entropy loss in both directions, learned temperature τ
- Useful for simple prompts; **structurally fails on compositional prompts** (spatial relations, attribute binding, counting) — the single-vector bottleneck loses relational info
- Human correlation: 0.42 (HEIM)

### Aesthetics

**LAION Aesthetics Predictor**
- Linear regression head on CLIP image embeddings, trained on crowdsourced 1–10 ratings
- Biased toward photorealistic, high-contrast, western art styles
- Human correlation: 0.39 (HEIM)

### Safety / Toxicity

Three separate detectors (each catches different failure modes):
- NSFW detector — binary
- NudeNet — fine-grained nudity/exposure
- Watermark detection — training data leakage

---

## Key Engineering Takeaways

1. **Never use a single metric** — FID and CLIPScore together still only cover quality + coarse alignment; aesthetics, safety, and compositional accuracy need separate treatment
2. **Sample size matters for FID** — below ~5k images, covariance estimation breaks down; below 1k, results are noise
3. **CLIPScore is not a compositional metric** — for attribute binding, spatial relations, or counting prompts, use VQA-based eval (e.g., T2I-CompBench) instead
4. **Human eval is non-negotiable** for aesthetics and originality — no automated proxy exists with meaningful correlation
5. **All metrics share backbone bias** — Inception-v3 and CLIP both undervalue non-photographic styles; document this when reporting results on art/illustration datasets

---

### Distribution-level Quality (FID alternative)

**CMMD (CLIP Maximum Mean Discrepancy)**
- Replaces Inception-v3 with CLIP ViT-L/14-336 embeddings (768-dim)
- Replaces Fréchet distance (assumes Gaussian) with MMD using a Gaussian RBF kernel (σ=10)
- MMD formula: `1000 * (k_xx + k_yy - 2 * k_xy)` where each k is the mean kernel similarity within/across sets
- Lower = better; reliable at **~300 samples** vs FID's 10k minimum — biggest practical advantage
- Hardcoded σ=10 is tuned for CLIP embedding space; re-tune via median heuristic for different domains
- On Google stack: use Vertex AI Multimodal Embeddings (`multimodalembedding@001`) as drop-in; re-tune σ

### Human Preference (learned reward models)

Feature-based metrics (FID, CLIPScore) don't capture what humans actually prefer. Learned reward models close this gap by training on human preference data:

**PickScore**
- Backbone: CLIP ViT-H; trained on Pick-a-Pic dataset (~600k pairwise picks)
- Each training example: (prompt, image_A, image_B) → human pick
- Output: scalar score per (prompt, image) pair; higher = more likely preferred for that prompt
- Captures comparative preference conditioned on the prompt; trained on real user intent

**ImageReward**
- Backbone: BLIP; trained on ~137k (prompt, image) absolute human ratings
- Output: unbounded continuous scalar (e.g., -2.03 to +0.58); negative = bad, positive = good
- Captures holistic absolute quality, not just relative preference
- Outperforms CLIPScore by 38.6% and Aesthetics by 39.6% on human preference prediction

**HPSv2.1 (Human Preference Score)**
- Backbone: CLIP ViT-H; trained on ~800k pairwise comparisons across diverse styles (anime, photo, painting, concept art)
- Most style-robust of the three — diverse training set reduces distribution shift for non-photorealistic prompts
- v2.1 improves calibration over v2.0

**Key distinction from CLIPScore/Aesthetics:** learned reward models use human preference data as supervision; CLIPScore and Aesthetics use no preference signal. A model can score high on CLIPScore (semantically aligned) but low on PickScore (not visually preferred) — this gap is what preference metrics are designed to catch.

---

## Metric Comparison Table

| Metric | Backbone | Human pref signal | Text-aware | Min samples | Output |
|--------|----------|------------------|------------|-------------|--------|
| FID | Inception-v3 | No | No | ~10k | lower=better |
| CMMD | CLIP ViT-L | No | No | ~300 | lower=better |
| CLIPScore | CLIP ViT-L | No | Yes | 1 | 0–100 |
| LAION Aesthetics | CLIP ViT-L | Weak | No | 1 | 1–10 |
| PickScore | CLIP ViT-H | Yes (pairwise) | Yes | 1 | scalar |
| ImageReward | BLIP | Yes (absolute) | Yes | 1 | unbounded scalar |
| HPSv2.1 | CLIP ViT-H | Yes (pairwise) | Yes | 1 | scalar |

---

## Key Engineering Takeaways

1. **Never use a single metric** — FID and CLIPScore together still only cover quality + coarse alignment; aesthetics, safety, and compositional accuracy need separate treatment
2. **Sample size matters for FID** — below ~5k images, covariance estimation breaks down; below 1k, results are noise; use CMMD for small eval sets
3. **CLIPScore is not a compositional metric** — for attribute binding, spatial relations, or counting prompts, use VQA-based eval (e.g., T2I-CompBench) instead
4. **Human eval is non-negotiable for originality** — no automated proxy exists with meaningful correlation
5. **All metrics share backbone bias** — Inception-v3 and CLIP both undervalue non-photographic styles; document this when reporting results on art/illustration datasets
6. **Use learned reward models for preference-aligned eval** — PickScore or HPSv2.1 should replace CLIPScore as the primary alignment metric when human preference matters
7. **CLIPScore is still useful as a diagnostic** — low CLIPScore = fundamental alignment failure; high CLIPScore + low PickScore = aligned but not preferred → aesthetic/detail issue

---

## Open Questions

- What replaces CLIPScore for compositional eval? → see T2I-CompBench (in tracker)
- How do these metrics behave on video generation? → not covered by HEIM
- Can PickScore/ImageReward be used as training rewards without reward hacking? → see Power Reinforcement paper (in tracker)
