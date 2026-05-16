# Image Generation Metrics — Synthesis

*Updated: 2026-05-15. Sources: HEIM (2023)*

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

## Open Questions

- What replaces CLIPScore for compositional eval? → see T2I-CompBench (in tracker)
- Is there a better backbone than Inception-v3 for FID? → CMMD uses CLIP embeddings instead (in tracker)
- How do these metrics behave on video generation? → not covered by HEIM
