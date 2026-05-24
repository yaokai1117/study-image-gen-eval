# Human Preference Metrics — Notes from Efficient Adjoint Matching (2026)

Source: https://arxiv.org/abs/2605.11480

> This paper is primarily about efficient diffusion model fine-tuning (not eval). Its value here is that it evaluates against five human preference metrics simultaneously, making it a useful comparison point. Notes focus on the metrics, not the training method.

---

## Background concepts

### Learned reward models vs. feature-based metrics

The older metrics (FID, CLIPScore, LAION Aesthetics) compute scores directly from image/text features using fixed pretrained encoders — no human preference data involved.

**Learned reward models** go further: they train a neural network to predict human preference scores, using datasets of human ratings or pairwise comparisons as supervision. The network learns what humans actually prefer, not just what a CLIP encoder thinks is similar.

The key insight: humans care about things CLIP doesn't — fine-grained detail preservation, coherence, whether the image is actually pleasing vs. merely semantically correct. Learned reward models capture this signal; CLIP-based metrics don't.

### Pairwise comparison vs. absolute rating

Two common ways to collect human preference data:

- **Pairwise ("which do you prefer?")** — easier for raters, lower noise, but only captures relative preferences; can't say how good an image is in absolute terms
- **Absolute rating (1–5 scale)** — harder to calibrate across raters, more noise, but gives absolute quality signal

PickScore uses pairwise; ImageReward uses absolute ratings. Both approaches have been shown to produce useful reward signals.

---

## Human preference metrics

### PickScore

**Training data:** Pick-a-Pic dataset — collected via a web app where users typed their own prompts, two images were generated (by different models or settings), and users picked their favorite.

Each training example is a triple: **(prompt, image_A, image_B) → human pick**. The prompt is essential — without it, you can't train a model that understands text-image alignment. The user sees:
```
Prompt: "a golden retriever playing in autumn leaves"
Image A: [generated]    Image B: [generated]
Which do you prefer?
```

This means the dataset reflects **real user intent** — prompts people actually wanted to generate, not synthetic benchmark prompts. Preference is always judged relative to the prompt, not in isolation.

**How it's trained:**
- Built on top of CLIP (specifically `clip-vit-h-14`)
- Fine-tuned with a pairwise ranking loss: given (prompt, image_A, image_B), predict which image the user picked
- Output: a scalar score for a (prompt, image) pair — higher = more likely to be preferred for that prompt

**What it captures:** comparative human preference conditioned on the prompt — "would a user pick this image over an alternative, given what they asked for?" Captures aesthetic preference, detail quality, and prompt adherence simultaneously.

**Strengths:** trained on real user behavior (not paid annotators rating in isolation), large dataset, open weights  
**Weakness:** captures *average* user preference — may not reflect domain-specific taste (e.g., professional photography vs. casual users)

---

### ImageReward

**Training data:** ~137k (prompt, image, human rating) triples collected via expert annotation, with ratings on a 1–5 scale covering text-image alignment, visual fidelity, and overall quality.

**How it's trained:**
- BLIP (a vision-language model) as backbone, not CLIP
- Fine-tuned with a regression head to predict the absolute human rating
- Also uses a ranking loss over rated pairs to improve relative ordering

**Output:** a continuous unbounded scalar — can be negative (e.g., -2.03) or positive (e.g., 0.58). Higher = better. Not a 1–5 scale despite being trained on absolute ratings; the model learns real-valued scores where relative ordering matters more than the absolute number:
- Clearly bad images (wrong subject, blurry, misaligned prompt) → negative
- Good images → positive
- No hard ceiling or floor

Cannot be directly compared to PickScore numbers — different scales, different meanings. Use each to rank images within its own scoring system.

**What it captures:** holistic human judgment — not just "which is better" but "how good is this image, absolutely?" Considers text alignment and visual quality jointly.

**Strengths:** absolute scores (not just relative), considers multiple quality dimensions explicitly; outperforms CLIP by 38.6% and Aesthetics by 39.6% on human preference prediction  
**Weakness:** smaller training set than PickScore (~137k vs ~600k); BLIP backbone means different embedding space from CLIP-based metrics — less interoperable

---

### HPSv2.1 (Human Preference Score)

**Training data:** ~800k pairwise human comparisons from a diverse set of T2I models and prompts, spanning multiple styles (anime, photo, painting, concept art).

**How it's trained:**
- CLIP backbone, similar architecture to PickScore
- Trained with pairwise ranking loss across a much larger and more diverse dataset than PickScore
- v2.1 is a calibration improvement over v2.0 — better score distribution across styles

**What it captures:** broad human preference across diverse generation styles. The large, style-diverse training set makes HPSv2.1 more robust than PickScore to distribution shift (e.g., anime vs. photorealistic prompts).

**Strengths:** largest training set of the three; style-diverse; well-calibrated  
**Weakness:** CLIP backbone inherits CLIP biases; pairwise only (no absolute scores)

---

### CLIPScore (reference)

See [topics/metrics.md](../topics/metrics.md) for full deep dive.

**Summary here:** cosine similarity between CLIP image and text embeddings. No human preference signal — purely semantic. Fails on compositional prompts. Human correlation: 0.42 (HEIM). Included in this paper as a baseline to show the gap between CLIP-based and preference-based metrics.

---

### Aesthetics (reference)

Linear regression head on CLIP image embeddings, trained on LAION crowdsourced ratings. Text-independent — scores visual appeal only, ignoring prompt. Biased toward photorealism. See [topics/metrics.md](../topics/metrics.md).

---

## Comparing the five metrics

| Metric | Backbone | Training data | Measures | Text-aware? | Human pref signal? |
|--------|----------|--------------|----------|-------------|-------------------|
| PickScore | CLIP ViT-H | ~600k pairwise picks (Pick-a-Pic) | Comparative preference | Yes | Yes |
| ImageReward | BLIP | ~137k absolute ratings | Holistic quality | Yes | Yes |
| HPSv2.1 | CLIP ViT-H | ~800k pairwise (style-diverse) | Broad preference | Yes | Yes |
| CLIPScore | CLIP ViT-L | None (zero-shot) | Semantic alignment | Yes | No |
| Aesthetics | CLIP ViT-L | LAION ratings | Visual appeal | No | Weak |

**When they disagree:**
- High CLIPScore + low PickScore → semantically correct but not visually preferred (e.g., blurry, wrong style, lacks detail)
- High Aesthetics + low PickScore → visually striking but doesn't match the prompt
- High ImageReward + low HPSv2.1 → good absolute quality but doesn't compare well against alternatives in that style

**For a production eval pipeline:** use at least one learned reward model (PickScore is the most practical — open weights, CLIP-based, easy to run). CLIPScore alone is insufficient. Aesthetics alone is insufficient.

---

## Relevance to engineering practice

1. **PickScore is the most practical starting point** — open weights, CLIP-based, runs on CPU for small sets, good correlation with real user preference; viable on Google stack via Vertex AI CLIP embeddings + fine-tuned head

2. **HPSv2.1 is the most robust across styles** — if your eval set includes diverse styles (anime, photo, art), HPSv2.1's diverse training data makes it more reliable than PickScore

3. **Run multiple reward models, not just one** — PickScore + HPSv2.1 together cover more ground than either alone; they agree on clear wins/losses and disagree on edge cases worth investigating manually

4. **CLIPScore is still useful as a diagnostic** — if CLIPScore is low, something is fundamentally wrong with text alignment; if CLIPScore is high but PickScore is low, the model is aligned but not preferred — points to aesthetic/detail issues

5. **ImageReward is useful for absolute thresholds** — if you need to filter out clearly bad images (not just rank them), ImageReward's absolute score scale is more useful than pairwise-only metrics
