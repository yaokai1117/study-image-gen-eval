# Weekly Summary

---

## Week of 2026-05-25

### Intake (Monday)

New items added to backlog:
- **Repos:** OneIG-Bench/OneIG-Benchmark (120★, NeurIPS 2025 DB), yzc-ippl/LongT2IBench (6★, AAAI 2026 Oral)
- **Papers:** ImagenWorld (2603.27862, 2026) — explainable human eval with 3.6K condition sets across 6 tasks × 6 domains, 20K human annotations
- **Blogs:** none new beyond existing backlog

Carry-overs from last week: all backlog items still valid.

### This week focus
- **Tue 2026-05-27** — `Karine-Huang/T2I-CompBench`: VQA-based compositional eval — answers the persistent open question from weeks 1–2
- **Wed 2026-05-28** — "Image Generators are Generalist Vision Learners" (Google DeepMind, Apr 22): what does image generation training actually teach the model?
- **Thu 2026-05-29** — Power Reinforcement Post-Training (2605.10937): reward hacking question — can PickScore/ImageReward be used as training signals?

### What I covered

*(to be filled in as the week progresses)*

### Key insights / open questions
*(to be filled in)*

---

## Week of 2026-05-18

### Intake (Monday)

New items added to backlog:
- **Repos:** Awesome-Evaluation-of-Visual-Generation (437★), Awesome-IQA (1500★), GenEditEvalKit (43★, Mar 2026), T2I-CoReBench (53★, ICLR 2026)
- **Papers:** none new this week beyond last week's backlog
- **Blogs:** none new beyond last week's backlog

Carry-overs from last week: all `up-next` items still valid.

### This week focus
- **Tue 2026-05-19** — `sayakpaul/cmmd-pytorch`: how CMMD differs from FID (CLIP embeddings vs. Inception)
- **Wed 2026-05-20** — ProEval blog (Google DeepMind): proactive failure discovery methodology
- **Thu 2026-05-21** — Efficient Adjoint Matching paper: covers PickScore, ImageReward, HPSv2.1, CLIPScore

### What I covered

- **Tuesday (code)** — `sayakpaul/cmmd-pytorch`: CMMD metric internals (CLIP embeddings + MMD with Gaussian RBF kernel), O(n²d) complexity, σ=10 bandwidth, comparison with FID; added Google stack section (Vertex AI Multimodal Embeddings as drop-in, σ re-tuning)
- **Wednesday (blog)** — ProEval (Google DeepMind, Apr 2026): active eval via GP surrogates + Bayesian Quadrature for performance estimation, superlevel set sampling + LLM synthesis for failure discovery; learned GP and BQ from scratch as background
- **Thursday (paper)** — Efficient Adjoint Matching (2026): focused on human preference metrics — PickScore (pairwise CLIP-based, Pick-a-Pic dataset), ImageReward (BLIP-based, unbounded scalar), HPSv2.1 (style-diverse pairwise); comparison table and diagnostic guide

### Key insights

- **CMMD's practical advantage is sample size** — works at ~300 images vs FID's 10k; the non-parametric MMD avoids the Gaussian assumption that breaks down on small sets
- **Learned reward models are the right eval target for preference-aligned generation** — PickScore/ImageReward/HPSv2.1 all trained on human preference data; CLIPScore alone is insufficient and the gap is measurable
- **High CLIPScore + low PickScore = aligned but not preferred** — a model can be semantically correct but aesthetically displeasing; preference metrics catch this, CLIP doesn't
- **Active eval is a fundamentally different frame** — ProEval treats input selection as a sequential decision problem; random sampling is the baseline to beat, not the default to use; 8–65x more sample-efficient
- **Performance estimation ≠ failure discovery** — require different objectives, different metrics; conflating them misses rare severe failures

### Open questions / carry-overs

- How does VQA-based compositional eval work? → T2I-CompBench still in backlog
- Can PickScore/ImageReward be used as training rewards without reward hacking? → Power Reinforcement paper (in backlog)
- Does active eval degrade when GP prior diverges from the model being evaluated?

### Next week focus

*(to be set on Monday intake)*

---

## Week of 2026-05-11

### What I covered

- **Monday intake** — ran keyword searches across GitHub, arXiv, and engineering blogs; populated TRACKER.md with repos, blog posts, and papers; dropped repo-watchlist approach (not useful)
- **Thursday paper** — HEIM: Holistic Evaluation of Text-to-Image Models (Stanford 2023); deep dive into metric internals: FID, IS, CLIPScore (including CLIP contrastive training mechanics), LAION Aesthetics, toxicity detectors

### Key insights

- All major automated metrics (FID, IS, CLIPScore, LAION Aesthetics) share Inception-v3 or CLIP as backbone — systematic bias in those encoders propagates everywhere; this explains why automated-human correlation is stuck at 0.39–0.59
- CLIPScore's failure on compositional prompts is **structural**, not a data issue — the single-vector bottleneck loses relational information by design; VQA-based eval is the right alternative
- FID requires 10k+ images to estimate the 2048×2048 covariance matrix reliably; small-batch FID is noise
- No single model wins across all 12 HEIM aspects — alignment vs. aesthetics vs. safety tradeoffs are inherent, not solvable

### Open questions / carry-overs

- How does VQA-based compositional eval actually work? → T2I-CompBench is next
- CMMD (in tracker) claims to fix FID using CLIP embeddings instead of Inception — is that actually better?
- Bias metrics in HEIM only cover binary gender + skin tone — what does better bias eval look like?

### Next week focus

- **Tue** — `sayakpaul/cmmd-pytorch`: read core metric implementation, understand how CMMD differs from FID
- **Wed** — ProEval (Google DeepMind blog): proactive failure discovery for generative AI eval
- **Thu** — Efficient Adjoint Matching paper: covers PickScore, ImageReward, HPSv2.1, CLIPScore in one place
