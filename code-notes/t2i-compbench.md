# T2I-CompBench(++) — Code Notes

*Source: [github.com/Karine-Huang/T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) | Paper: [2307.06350](https://arxiv.org/abs/2307.06350)*

---

## What it does

Benchmark for **compositional** text-to-image evaluation — tests whether models correctly bind attributes to objects, follow spatial instructions, and count correctly. CLIPScore fails structurally on all of these (single-vector bottleneck loses relational info); T2I-CompBench replaces it with task-specific metrics.

8,000 prompts across 8 categories. Each category uses a different metric because no single metric works for all compositional tasks.

---

## Category → Metric mapping

| Category | Metric | Human corr (Kendall τ / Spearman ρ) |
|----------|--------|--------------------------------------|
| Color binding | BLIP-VQA | 0.6297 / 0.7958 |
| Texture binding | BLIP-VQA | 0.5177 / 0.6995 |
| Shape binding | BLIP-VQA | 0.2707 / 0.3795 |
| 2D spatial | UniDet | 0.4756 / 0.5136 |
| 3D spatial | UniDet | 0.3126 / 0.4262 |
| Numeracy | UniDet | 0.4251 / 0.5273 |
| Non-spatial relations | CLIPScore / GPT-4V | GPT-4V best |
| Complex composition | GPT-4V | — |

**vs CLIPScore on color:** BLIP-VQA (0.6297) vs CLIP (0.1938) — 3× better human correlation. The gap makes the CLIP structural failure concrete and quantified.

---

## How the core metrics are computed

### BLIP-VQA (`BLIPvqa_eval/BLIP_vqa.py`)

Key insight: **decompose the prompt into independent per-attribute yes/no questions.**

Given prompt "a green bench and a red car":
- Q1: "Is there a green bench?" → P(yes)
- Q2: "Is there a red car?" → P(yes)
- Final score: P(yes_1) × P(yes_2)

Implementation:
1. Parse image filename as prompt text via spaCy
2. Extract noun chunks (noun + adjective pairs), filter out spatial terms
3. For each noun chunk, generate question `"{chunk}?"` and run BLIP VQA
4. Collect per-chunk probabilities into `reward[sample][chunk_index]`
5. Multiply across chunks: `reward_final *= reward[:, i]`
6. Average over the dataset

**Why multiplication:** treats each attribute binding as an independent condition. If either object is wrong, the overall score drops. This is stricter than averaging.

**Gotcha:** question generation depends on spaCy parse quality and noun chunk extraction. Unusual prompt structures (passive voice, unusual word order) can produce malformed or empty questions — empty questions get score 1 (neutral), which inflates scores for hard-to-parse prompts.

---

### UniDet — Spatial (`UniDet_eval/2D_spatial_eval.py`)

1. Run UniDet object detector → bounding boxes + confidence scores for all instances
2. For each (obj1, obj2, relation) triple from the prompt:
   - Find best-matching detected boxes for obj1 and obj2
   - Call `determine_position(box1, box2, relation)`:
     - Computes center-to-center distances (Δx, Δy) and IoU
     - Returns soft score 0–1 for whether the spatial relation holds
   - Combined score: `0.25 × conf(obj1) + 0.25 × conf(obj2) + 0.5 × spatial_score`
   - If combined score < 0.5, reset to 0 (hard threshold)
3. Average over all triples in the dataset

**Score breakdown:** detection quality contributes 50%, spatial correctness contributes 50%. A correctly detected but wrongly positioned pair can still score ~0.5.

---

### UniDet — Numeracy (`UniDet_eval/numeracy_eval.py`)

1. Parse prompt with NLP to extract `{object: expected_count}` pairs
   - Handles "a / one / two / three / ..." via `word2number` library
2. Run UniDet → raw detections
3. Deduplicate detections: remove boxes with IoU > 0.9 or >90% contained-in overlap
4. For each expected object:
   - **Presence score:** 0.5 × weight if any instance detected
   - **Count score:** +0.5 × weight if detected count == expected count
   - `weight = 1.0 / num_distinct_objects`
5. Sum over all objects → final score in [0, 1]

**Two-part scoring deliberately:** presence and accuracy are separate conditions. A model that generates "some dogs" when asked for "3 dogs" gets 0.5, not 0 or 1.

---

### CLIPScore (`CLIPScore_eval/CLIP_similarity.py`)

Standard CLIP cosine similarity — same as the metric in HEIM:
1. Encode image with CLIP image encoder
2. Encode prompt text with CLIP text encoder
3. Normalize both vectors, compute dot product

Only used for **non-spatial relationships** (e.g., "wearing", "holding", "eating") where the relationship can be captured holistically rather than requiring spatial parsing. Still structurally weak — this is the category where GPT-4V eventually does better.

---

## Interesting design decisions

1. **No single metric** — explicitly rejects the idea of one universal metric; each compositional skill needs a metric matched to that skill's detection requirements

2. **Multiplicative BLIP scoring** — multiplication vs averaging is a principled choice: compositional correctness requires all conditions to hold, not just some

3. **Soft scores everywhere** — UniDet scores are continuous (0–1), not binary pass/fail; this gives gradient signal for fine-tuning (the repo also includes GORS fine-tuning code that uses these scores as rewards)

4. **Filename-as-prompt convention** — all eval scripts parse prompts from image filenames (first segment before `_`). This is a tight coupling that makes the evaluation pipeline inflexible for images with arbitrary filenames.

5. **MLLM eval added in ++ version** — GPT-4V/ShareGPT4V added as a metric for complex compositions and non-spatial; shows the community shift toward LLM-as-judge for cases where detection-based approaches saturate

---

## Limitations / gotchas

- **UniDet is not a general-purpose detector** — fine-tuned on COCO-like categories; unusual objects, abstract entities, or style-specific content (anime, sketch) will have poor detection recall, making spatial and numeracy scores unreliable
- **BLIP-VQA question generation is brittle** — dependent on spaCy parse; complex prompt structures can fail silently (empty question → score 1)
- **MLLM metrics are expensive and non-deterministic** — GPT-4V scores vary run to run; ShareGPT4V gives less diverse ratings (bunches near the middle); neither is suitable for automated CI-style eval
- **Score threshold in spatial eval is a hyperparameter** — the 0.5 hard threshold in `2D_spatial_eval.py` is not justified in the paper; changing it shifts all model rankings
- **Filename convention coupling** — eval only works if images are named with the prompt as prefix; breaks for any standard image gen pipeline that uses its own naming

---

## Key model benchmark results

| Category | Best model | Score |
|----------|-----------|-------|
| Color | SD3 | 0.813 |
| Texture | SD3 | 0.733 |
| 2D Spatial | SD3 | 0.320 |
| 3D Spatial | FLUX.1 | 0.387 |
| Numeracy | FLUX.1 | 0.619 |
| Non-spatial | FLUX.1 | 0.921 |
| Complex | DALL-E 3 | 0.377 |

**Takeaway:** Even the best models score ~0.32–0.39 on spatial categories — compositional spatial understanding is the hardest unsolved dimension. Numeracy and non-spatial are much easier.

---

## Relevance to engineering practice

- **Replaces CLIPScore for compositional eval** — use BLIP-VQA for attribute binding eval pipelines; the 3× human correlation improvement over CLIP on color makes it the clear choice
- **Spatial + numeracy evals require an object detector** — UniDet or equivalent; this adds infra complexity and domain constraints (detector coverage)
- **On Google stack:** BLIP-VQA could be replaced by Gemini's VQA capabilities with the same multiplicative decomposition strategy; UniDet could be replaced by a Vertex Vision API detector
- **The multiplicative decomposition pattern** is reusable: decompose any multi-condition prompt into independent yes/no questions, multiply probabilities
