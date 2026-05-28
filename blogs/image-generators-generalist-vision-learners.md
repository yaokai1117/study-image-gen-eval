# Image Generators are Generalist Vision Learners

*Google DeepMind, Apr 22, 2026 | [arXiv 2604.20329](https://arxiv.org/abs/2604.20329)*

---

## Background concepts

**Generative pretraining analogy to LLMs:** LLMs learn powerful representations as a byproduct of next-token prediction on large text corpora — the pretraining task is generation, but the learned representations transfer to classification, reasoning, QA, etc. This paper claims the same is true for image generation models: training to generate images forces the model to understand world structure, semantics, and spatial relationships, making those representations reusable for perception tasks.

**RGB parameterization:** The trick that makes vision tasks expressible as image generation. Instead of predicting a segmentation mask as a class label map or depth as a float array, encode the output as an RGB image where each pixel's color encodes the target value. The generation model then "generates" a colored visualization rather than a raw image — and the inverse mapping recovers the structured output.

---

## Problem they were solving

Perception tasks (segmentation, depth estimation, surface normals) have traditionally used specialist encoders (e.g., ViT pretrained on classification) as backbones. These are trained discriminatively — they learn features that distinguish classes. The question: can a generative image model's learned representations do the same job, without any specialist pretraining?

Secondary question: can a single model handle both generation *and* perception, or does specializing for one hurt the other?

---

## Approach / engineering decisions

**Model:** Vision Banana, built by instruction-tuning **Nano Banana Pro** (NBP) — an internal Google image generation model, not publicly released — on a mixture of its original generation training data + a small amount of vision task data.

**Key insight — reframe perception as generation:** Instead of adding a decoder head on top of frozen features, treat every vision task output as an RGB image and generate it:

- **Semantic segmentation:** generate an image where each pixel is colored by class (color-to-class mapping provided in the prompt as JSON/hex/RGB)
- **Metric depth:** encode depth as a false-color visualization using power transforms + interpolation along RGB cube edges (invertible mapping: depth value → color → depth value)
- **Surface normals:** direct channel encoding — `R=(1−x)/2, G=(1+y)/2, B=(1+z)/2` — so the normal vector is recoverable from the RGB pixel

**Why this works better than regression:** The paper argues that discriminative/regression approaches suffer from *blur* — when the training target is ambiguous (multiple valid depths for a pixel), MSE loss averages them, producing blurry outputs. Generative models are mode-seeking: they commit to a specific sharp prediction rather than averaging. This is the same argument used to explain why diffusion beats regression for dense prediction.

**Training data for depth:** zero real-world depth data — all synthetic. The model's pretraining world knowledge provides "stronger priors on object sizes and distances" that compensate.

---

## Metrics / eval methods used

| Task | Benchmark | Vision Banana | Best specialist |
|------|-----------|--------------|-----------------|
| Semantic segmentation | Cityscapes mIoU | 0.699 | SAM3: 0.652 |
| Referring segmentation | RefCOCOg cIoU | 0.738 | SAM3 Agent: 0.734 |
| Reasoning segmentation | ReasonSeg gIoU | 0.793 | SAM3 Agent: 0.770 |
| Metric depth | Average δ₁ | 0.929 | Depth Anything V3: 0.918 |
| Surface normals | Mean angle error | 18.93° | Lotus-2: 19.64° |
| Instance segmentation | SA-Co/Gold | 0.540 | DINO-X: 0.552 (**loses**) |

**Generation quality maintained:** 53.5% win rate on T2I and 47.8% on image editing vs. the base model (roughly parity — instruction-tuning on perception data doesn't noticeably hurt generation).

---

## What's reusable / what to remember

**The "generation = pretraining" thesis has empirical support now.** For years the community treated generation and perception as separate pipelines. This paper quantifies that the representations learned by image generation training transfer to SOTA perception — beating or matching SAM3 and Depth Anything with a single generalist model.

**The RGB parameterization pattern is a general technique.** Any structured output that can be encoded as a colored image can be turned into a generation task. This is not specific to Nano Banana Pro — the same trick could apply to any sufficiently capable image generation model.

**Implications for eval:** If generation models inherently learn strong perceptual representations, then generation training quality *is* a proxy for perceptual quality. A model that generates more accurately likely has better internal representations of depth, segmentation boundaries, object identity — which may explain why learned reward models (PickScore, ImageReward) trained on top of generation models transfer well to preference prediction.

**Practical limitation:** Vision Banana's computational overhead is "significantly higher" than specialist models. It's a research result showing the *ceiling* of what generation representations can do, not a deployment recipe.

**Nano Banana Pro is internal** — this work is not reproducible outside Google without an equivalent capable image generation backbone.

---

## Source

[https://arxiv.org/abs/2604.20329](https://arxiv.org/abs/2604.20329)
[https://deepmind.google/research/publications/240658/](https://deepmind.google/research/publications/240658/)
