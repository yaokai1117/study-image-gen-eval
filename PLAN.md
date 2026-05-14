# Image Generation Eval — Study Plan

## Goal

Build applied engineering knowledge in image generation evaluation, in preparation for a new job. Focus: practical metrics, benchmarks, and engineering systems — not just theory.

---

## Tooling Constraint

Work requires Google model stack: **Gemini / Gemini Nano**. Any hands-on experiments or metric implementations must be compatible with this. Avoid building study dependencies on HuggingFace models.

---

## Information Sources

### Tier 1 — Engineering-first (highest priority)
- **GitHub** — read source code of eval frameworks, issues, and PRs; this is where real engineering decisions are debated
  - `clean-fid`
  - `torch-fidelity`
  - `t2i-compbench`
  - Google's internal eval tooling (whatever is public)
- **Engineering blogs from established companies/projects**
  - Google DeepMind / Google Research blog
  - Google AI blog (imagen, imagen video, etc.)
  - Meta AI Research blog
  - Adobe Research blog
  - Scale AI blog (human eval pipelines)
  - Stability AI blog
  - ML engineering blogs (e.g., Chip Huyen, Eugene Yan)

### Tier 2 — Papers (algorithm understanding)
- **arXiv cs.CV** — eval/benchmark papers via RSS, skim broadly, deep-read selectively
- Focus: understand the algorithm and its limitations, not necessarily to run the code

### Tier 3 — Lower priority
- **Hugging Face** — useful for benchmark/leaderboard context, but skip model-specific content (can't use those models at work anyway)
- Twitter/X — curated list of ~20 researchers, for real-time signal only

### Skip
- Reddit, YouTube courses — low signal-to-noise

---

## Key Papers to Read (in order)

1. **HEIM** (Stanford, 2023) — holistic survey of metrics, tradeoffs, and gaps. Read this first for the map.
2. FID (Fréchet Inception Distance) — foundational metric
3. CLIP Score — text-image alignment
4. T2I-CompBench — compositional text-to-image eval
5. GenAI-Bench
6. EvalCrafter

---

## Folder Structure

```
~/study/image-gen-eval/
  PLAN.md              # this file
  TRACKER.md           # running list of all items: repos, blogs, papers — with status
  code-notes/          # notes from reading eval framework source code (primary)
  blogs/               # one .md per engineering blog post: key ideas, engineering decisions
  papers/              # one .md per paper: problem, metric, computation, limitations
  topics/              # synthesis notes: metrics.md, human-eval.md, benchmarks.md
  experiments/         # toy experiments using Gemini / Gemini Nano
  weekly-summary.md    # rolling weekly log
```

> Note: citation counts for papers are fetched via Google Scholar at intake time. For brand-new papers (< 1 month old), citations won't be meaningful — use recency only. For backlog papers, citations are a strong signal of community adoption.

---

## Daily After-Work Schedule (~1.5 hrs)

**Rule:** never let intake days bleed into deep-read days.

### Monday — Intake + Planning

1. Skim intake sources (~45 min):
   - **GitHub keyword search** — search for new repos using terms: `image generation evaluation`, `text-to-image benchmark`, `image quality metric`, `image generation eval`; sort by stars and recently updated
   - **GitHub Trending** — Python + ML filter, weekly view
   - **arXiv keyword search** — search cs.CV for same keywords; sort candidates by citation count (via Semantic Scholar) + recency; prefer papers with both recent activity and meaningful citations
   - **Engineering blog keyword search** — search Google DeepMind, Meta AI, Adobe Research, Scale AI blogs directly for recent relevant posts (no RSS needed)
   - **Twitter/X curated list** — quick scroll for signal
2. Add new items to `TRACKER.md` as `backlog`
3. Review carry-overs from last week — still relevant? Deprioritize or drop if not
4. Promote 1 item to `up-next` for each of Tue / Wed / Thu
5. Write a brief plan for the week in `weekly-summary.md` (just a few bullets: what you'll focus on and why)

### Tuesday — Code

1. Pick the `up-next` repo/framework from tracker; mark `in-progress`
2. Clone locally into `experiments/` if needed; open in VS Code
3. Navigate to core metric implementation (skip tests and CLI entry points)
4. Use Claude to explain specific functions — paste code or point to file
5. Focus questions: *How is this computed? What assumptions does it make? What breaks at scale?*
6. Write notes in `code-notes/<repo-name>.md`:
   ```
   ## What it does
   ## How the core metric is computed (key functions)
   ## Interesting design decisions
   ## Limitations / gotchas
   ```
7. Update tracker status to `done`

### Wednesday — Blog

1. Pick the `up-next` blog post from tracker; mark `in-progress`
2. Give Claude the URL: ask to extract — main claim, engineering decisions, metrics used, limitations
3. Review and annotate the draft note
4. Save to `blogs/<slug>.md`:
   ```
   ## Problem they were solving
   ## Approach / engineering decisions
   ## Metrics / eval methods used
   ## What's reusable / what to remember
   ## Source: <url>
   ```
5. Update tracker status to `done`

### Thursday — Paper

1. Pick the `up-next` paper from tracker; mark `in-progress`
2. Find the ar5iv HTML version: `ar5iv.org/abs/<arxiv-id>` (easier for Claude to fetch than PDF)
3. Give Claude the URL: ask to extract — problem, proposed metric, how computed, known limitations, what it replaces
4. Review and annotate the draft note
5. Save to `papers/<slug>.md`:
   ```
   ## Problem
   ## Proposed metric / method
   ## How it's computed
   ## Compared against
   ## Limitations
   ## Relevance to engineering practice
   ## Source: <arxiv url>
   ```
6. Update tracker status to `done`

### Friday — Synthesis

1. Tell Claude which files were added this week; ask it to identify emerging themes
2. Update or create relevant `topics/<topic>.md` files (e.g., `metrics.md`, `human-eval.md`)
3. Write this week's entry in `weekly-summary.md`:
   ```
   ## Week of YYYY-MM-DD
   ### What I covered
   ### Key insights
   ### Open questions / carry-overs
   ### Next week focus
   ```
4. Mark all completed items `done` in tracker; move unfinished items back to `backlog` or keep as `up-next`

### Weekend — Experiment (optional)

1. Pick a metric or concept from the week
2. Implement or run it using Gemini / Gemini Nano on a toy example
3. Save code and observations in `experiments/`

### TRACKER.md structure

Status flow: `backlog` → `up-next` → `in-progress` → `done`

```markdown
## GitHub Projects
| Repo | Topic | Status | Notes |

## Blog Posts
| Title | Source | Date | Status | Notes |

## Papers
| Title | Year | Status | Notes |
```

---

## Working Pattern with Claude

- Fetch papers/posts via Claude (`WebFetch`) → extract structured notes → save to `papers/<slug>.md`
- Periodically ask Claude to synthesize across papers into `topics/<topic>.md`
- This file (`PLAN.md`) is the source of truth — update it as the plan evolves

---

## Status

- [ ] Set up folder structure
- [ ] Subscribe to arXiv cs.CV RSS
- [ ] Curate Twitter/X list of researchers
- [ ] Read HEIM paper
