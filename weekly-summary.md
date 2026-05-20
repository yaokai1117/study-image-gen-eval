# Weekly Summary

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

### Key insights / open questions
- *(to be filled in as the week progresses)*

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
