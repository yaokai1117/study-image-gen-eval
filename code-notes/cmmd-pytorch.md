# cmmd-pytorch

Source: https://github.com/sayakpaul/cmmd-pytorch  
Paper: CMMD — Rethinking FID (Google Research, 2024)

---

## What it does

Computes **CMMD (CLIP Maximum Mean Discrepancy)** — a drop-in alternative to FID for measuring distributional similarity between real and generated image sets. Replaces Inception-v3 with CLIP embeddings and replaces Gaussian fitting + Fréchet distance with Maximum Mean Discrepancy (MMD).

---

## How the core metric is computed

### Step 1 — Embed (`embedding.py`)

Uses `openai/clip-vit-large-patch14-336` (via HuggingFace `CLIPVisionModelWithProjection`):

```python
image_embs = self._model(**inputs).image_embeds  # shape: (batch, 768)
image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)  # L2 normalize
```

Key detail: uses `CLIPVisionModelWithProjection` (not the base vision encoder) — this adds a linear projection head on top, giving a different embedding space than plain CLIP vision features. Images are resized with **bicubic** interpolation to 336×336.

### Step 2 — Compute MMD (`distance.py`)

Given two embedding matrices x (real, shape n×d) and y (generated, shape n×d):

```python
gamma = 1 / (2 * 10**2)   # bandwidth of Gaussian RBF kernel, σ=10

k_xx = mean(exp(-gamma * ||x_i - x_j||²))   # real vs real
k_yy = mean(exp(-gamma * ||y_i - y_j||²))   # gen vs gen
k_xy = mean(exp(-gamma * ||x_i - y_j||²))   # real vs gen

CMMD = 1000 * (k_xx + k_yy - 2 * k_xy)
```

The `||x_i - x_j||²` term is expanded as `||x_i||² + ||x_j||² - 2x_i·x_j` and computed via matrix multiply — memory-efficient for large n.

**Lower CMMD = better** (same direction as FID).

The ×1000 scale factor is cosmetic — makes the number more human-readable.

---

## How it differs from FID

| | FID | CMMD |
|--|-----|------|
| Backbone | Inception-v3 (ImageNet) | CLIP ViT-L/14-336 (web-scale) |
| Embedding dim | 2048 | 768 |
| Distribution model | Multivariate Gaussian | Non-parametric (kernel-based) |
| Distance function | Fréchet distance | Maximum Mean Discrepancy |
| Min sample size | ~10k (for stable Σ) | ~300 (no covariance to estimate) |
| Assumes Gaussian? | Yes | No |

The key claim: real image distributions are **not** Gaussian in embedding space. FID forces a Gaussian fit that introduces systematic error. MMD makes no distributional assumption — it directly tests whether two sets of points were drawn from the same distribution.

---

## Interesting design decisions

1. **`CLIPVisionModelWithProjection` not plain CLIP** — the projection head maps to a lower-dim space (768) that's explicitly trained for cross-modal alignment; hypothesis is this space captures semantic quality better than raw vision features

2. **σ=10 hardcoded** — bandwidth chosen via median heuristic on a reference dataset; not tuned per-run. This is a potential brittleness point if your image domain is very different from the reference distribution used to pick σ

3. **Pre-computed ref embeddings** — `ref_embed_file` arg lets you cache real image embeddings once and reuse them across eval runs; important for CI/eval pipelines where real set is fixed

4. **Biased MMD estimator** — uses the minimum-variance/biased version from Gretton et al. (2012); slightly underestimates true MMD but variance is lower, which matters more at practical sample sizes

---

## Limitations / gotchas

- **CLIP bias inherited** — still Inception-replaced-with-CLIP, not Inception-replaced-with-something-neutral; CLIP biases (western web images, photorealism) still apply
- **σ is fixed** — hardcoded bandwidth may not be optimal for all domains; no adaptive bandwidth selection
- **N×N kernel matrix** — computing k_xx requires an n×n matrix; at n=10k this is 100M entries × 4 bytes = 400MB per kernel, fits in memory but is worth knowing
- **Not directly comparable to FID scores** — different scale, different backbone; cannot mix CMMD and FID numbers across papers

---

## Relevance to engineering practice

- Better than FID for **small eval sets** (300–1k images) where FID's Gaussian assumption breaks down most severely
- Worth using alongside FID, not as a pure replacement — they measure slightly different things
- Pre-compute and cache ref embeddings if you're running evals repeatedly against a fixed reference set

---

## Running CMMD on Google Stack

The core dependency is a CLIP vision encoder. The cmmd-pytorch repo uses HuggingFace to load `openai/clip-vit-large-patch14-336`, but on the Google stack you have two practical paths:

### Option 1 — Vertex AI Multimodal Embeddings API

Google's `multimodalembedding@001` model on Vertex AI produces 1408-dim embeddings trained similarly to CLIP (image-text contrastive). You can use these as a drop-in replacement for the CLIP embeddings in the MMD computation.

```python
from google.cloud import aiplatform
from vertexai.vision_models import MultiModalEmbeddingModel

model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

def embed_images(image_paths: list[str]) -> np.ndarray:
    embeddings = []
    for path in image_paths:
        image = Image.load_from_file(path)
        result = model.get_embeddings(image=image)
        emb = np.array(result.image_embedding)
        emb /= np.linalg.norm(emb)  # L2 normalize, same as cmmd-pytorch
        embeddings.append(emb)
    return np.stack(embeddings)
```

Then feed the resulting embedding matrix directly into `distance.mmd()` from cmmd-pytorch — the MMD math doesn't care which encoder produced the vectors.

**Caveat:** the embedding dimension is 1408, not 768. The σ=10 bandwidth was tuned for CLIP's 768-dim L2-normalized space. You may want to re-tune σ via the median heuristic on your dataset:

```python
# Estimate median pairwise distance on a sample → use as σ
sample = ref_embs[:500]
dists = np.sqrt(((sample[:, None] - sample[None, :]) ** 2).sum(-1))
sigma = np.median(dists[dists > 0])
gamma = 1 / (2 * sigma ** 2)
```

### Option 2 — Load CLIP directly on a Vertex AI custom training job

If you want to use the exact same `openai/clip-vit-large-patch14-336` model as cmmd-pytorch (for comparability with published numbers), run it as a Vertex AI custom job with a GPU machine type:

```yaml
# job_spec.yaml
workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-8
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    containerSpec:
      imageUri: us-docker.pkg.dev/your-project/your-repo/cmmd-eval:latest
      args:
        - --ref_dir=gs://your-bucket/reference-images
        - --eval_dir=gs://your-bucket/generated-images
```

The container just wraps cmmd-pytorch's `main.py`, reading from GCS. This keeps the metric identical to the paper while staying fully within the Google infrastructure constraint.

### Option 3 — Gemini embeddings (experimental)

Gemini's embedding API (`text-embedding-004` and the multimodal variant) is newer and less characterized for image gen eval. Not recommended yet for CMMD — the σ=10 bandwidth assumption definitely doesn't hold, and there's no published baseline to compare against. Worth watching as it matures.

### Practical recommendation

For a production eval pipeline against Gemini / Nano Banana generated images:
1. Use **Vertex AI Multimodal Embeddings** for speed and no infrastructure overhead
2. Re-tune σ on your specific reference set (don't assume σ=10)
3. Cache reference embeddings in GCS — regenerate only when reference set changes
4. Run the MMD computation locally or in a lightweight Cloud Run job (it's just numpy/torch math, no GPU needed after embedding)
5. Report alongside CLIPScore (also computable via Vertex AI) so you have both distributional quality and alignment covered
