# Study Tracker

Status flow: `backlog` → `up-next` → `in-progress` → `done`

---

## GitHub Projects

| Repo | Stars | Topic | Status | Notes |
|------|-------|-------|--------|-------|
| [sayakpaul/cmmd-pytorch](https://github.com/sayakpaul/cmmd-pytorch) | 166 | CMMD — CLIP Maximum Mean Discrepancy, FID alternative | **up-next** | Tue 2026-05-19; 451 citations on underlying paper; small readable codebase |
| [zai-org/ImageReward](https://github.com/zai-org/ImageReward) | 1672 | Human preference learning & eval for T2I | backlog | 1,443 citations — foundational, high priority |
| [Vchitect/Evaluation-Agent](https://github.com/Vchitect/Evaluation-Agent) | 125 | Eval image/video gen like humans — fast + explainable | backlog | ACL2025 Oral, 35 citations |
| [PKU-YuanGroup/WISE](https://github.com/PKU-YuanGroup/WISE) | 201 | World knowledge-informed semantic eval | backlog | ICML 2026, 129 citations, recent + gaining traction |
| [TIGER-AI-Lab/ImagenHub](https://github.com/TIGER-AI-Lab/ImagenHub) | 180 | Standardizes inference + eval across conditional image gen | backlog | |
| [CodeGoat24/UniGenBench](https://github.com/CodeGoat24/UniGenBench) | 134 | Unified semantic eval benchmark for T2I | backlog | Updated this week |
| [PicoTrex/GPT-ImgEval](https://github.com/PicoTrex/GPT-ImgEval) | 306 | Evaluates GPT-4o image generation | backlog | Practical comparison baseline |
| [clean-fid](https://github.com/GaParmar/clean-fid) | — | FID metric implementation | backlog | |
| [torch-fidelity](https://github.com/toshas/torch-fidelity) | — | IS, FID, KID metrics | backlog | v0.4.0 released 2026-02-17 |
| [Karine-Huang/T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) | 339 | Compositional text-to-image eval | backlog | |

---

## Blog Posts

| Title | Source | Date | Status | Notes |
|-------|--------|------|--------|-------|
| [ProEval: Proactive Failure Discovery for Generative AI Evaluation](https://deepmind.google/research/publications/) | Google DeepMind | Apr 25, 2026 | **up-next** | Wed 2026-05-20; directly about eval methodology |
| [Image Generators are Generalist Vision Learners](https://deepmind.google/research/publications/) | Google DeepMind | Apr 22, 2026 | backlog | Image gen capabilities angle |
| [Scaling How We Build and Test Our Most Advanced AI](https://ai.meta.com/blog/) | Meta AI | Apr 8, 2026 | backlog | Evaluation methodology at scale |

---

## Papers

| Title | arXiv | Year | Citations | Status | Notes |
|-------|-------|------|-----------|--------|-------|
| Efficient Adjoint Matching for Fine-tuning Diffusion Models | [2605.11480](https://arxiv.org/abs/2605.11480) | 2026 | — | **up-next** | Thu 2026-05-21; covers PickScore, ImageReward, HPSv2.1, CLIPScore in one paper |
| Power Reinforcement Post-Training of T2I Models | [2605.10937](https://arxiv.org/abs/2605.10937) | 2026 | — | backlog | GenEval + UniGenBench++ eval, reward hacking |
| AlphaGRPO | [2605.12495](https://arxiv.org/abs/2605.12495) | 2026 | — | backlog | Covers GenEval, TIIF-Bench, DPG-Bench, WISE |
| Context Matters: Auditing Gender Bias in T2I | [2605.13113](https://arxiv.org/abs/2605.13113) | 2026 | — | backlog | Consolidates bias eval metrics |
| HEIM: Holistic Evaluation of Text-to-Image Models | [2311.01516](https://arxiv.org/abs/2311.01516) | 2023 | — | backlog | Start here — maps the whole space |
| FID (Fréchet Inception Distance) | [1706.08500](https://arxiv.org/abs/1706.08500) | 2017 | — | backlog | Foundational metric |
| CLIP Score | [2104.08718](https://arxiv.org/abs/2104.08718) | 2021 | 2,956 | backlog | Text-image alignment; highly cited |
| T2I-CompBench | [2307.06350](https://arxiv.org/abs/2307.06350) | 2023 | — | backlog | Compositional eval |
| CMMD: Rethinking FID | — | 2024 | 451 | backlog | Underlying paper for cmmd-pytorch |
