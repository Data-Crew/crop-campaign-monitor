# Clay Foundation Model — Architecture & Fine-tuning

> How the Vision Transformer inside Clay converts satellite imagery
> into semantic embeddings, how fine-tuning shapes those embeddings,
> and why the classification head is discarded before monitoring.

---

## Table of Contents

1. [From Pixels to Patches](#1-from-pixels-to-patches)
2. [Patch Embedding](#2-patch-embedding)
3. [The Transformer Encoder](#3-the-transformer-encoder)
4. [The CLS Token — Global Context](#4-the-cls-token--global-context)
5. [What Makes Clay Different](#5-what-makes-clay-different)
6. [Fine-tuning: Backbone + Head](#6-fine-tuning-backbone--head)
7. [Why the Head Is Auxiliary](#7-why-the-head-is-auxiliary)
8. [Exporting the Encoder](#8-exporting-the-encoder)
9. [From Embeddings to Anomaly Detection](#9-from-embeddings-to-anomaly-detection)

---

## 1. From Pixels to Patches

A chip is a multi-spectral image of shape `(10, 224, 224)` — 10 spectral
bands, each 224 × 224 pixels. The first thing the Vision Transformer
does is cut this image into non-overlapping square **patches** of
16 × 16 pixels.

### How many patches?

**Total number of patches:**

```text
(224 / 16) × (224 / 16) = 14 × 14 = 196 patches
```

The number of patches per dimension is:

```text
224 / 16 = 14
```

So the chip becomes a grid of:

```text
14 × 14 patches
```

```text
Chip: 10 bands × 224 px × 224 px

┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│  1 │  2 │  3 │  4 │  5 │  6 │  7 │  8 │  9 │ 10 │ 11 │ 12 │ 13 │ 14 │
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│ 15 │ 16 │ 17 │ 18 │ 19 │ 20 │ 21 │ 22 │ 23 │ 24 │ 25 │ 26 │ 27 │ 28 │
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│ .. │ .. │ .. │ .. │ .. │ .. │ .. │ .. │ .. │ .. │ .. │ .. │ .. │ .. │
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│183 │184 │185 │186 │187 │188 │189 │190 │191 │192 │193 │194 │195 │196 │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

14 × 14 = 196 patches
Each patch: 10 bands × 16 px × 16 px = 2,560 values
```

At 10 m/px, each patch covers a physical area of **160 m × 160 m**.

```text
┌──────────────────────────────┐
│ One patch                    │
│                              │
│ 16 px × 16 px                │
│ = 160 m × 160 m              │
│                              │
│ Contains:                    │
│ 10 bands × 256 pixels        │
│ = 2,560 raw values           │
└──────────────────────────────┘
```

---

## 2. Patch Embedding

Each patch is **flattened** into a 1D vector of 2,560 values and then
projected into the model's working dimension (`D = 768`) through a learned
linear transformation[^patch_embedding]:

```text
Patch (10 × 16 × 16)
        │
        ▼
[0.041, 0.059, 0.032, 0.321, 0.089, ..., 0.068]   ← 2,560 values
        │
        ▼
┌──────────────────────────────┐
│ Linear Projection            │
│                              │
│ W : (2560 × 768)             │
│ b : (768,)                   │
└──────────────────────────────┘
        │
        ▼
[0.234, -0.891, 0.456, 0.012, ..., 0.567]         ← 768 values
                                                    "patch token"
```

This is done simultaneously for all 196 patches, producing a **sequence
of 196 tokens**, each of dimension 768.

Positional information is added so the model knows *where* each patch was
located in the original image:

```text
token_42 = linear_proj(patch_42) + pos_embed[42]
```

---

## 3. The Transformer Encoder

The 196 patch tokens (plus the CLS token — see next section) pass through
a stack of **Transformer blocks**. Each block has two sub-layers:

```text
┌──────────────────────────────────────────────────────────────┐
│ Transformer Block                                            │
│                                                              │
│  1. Multi-Head Self-Attention                                │
│     - Every token attends to every other token               │
│     - Captures global context across the whole scene         │
│                                                              │
│  2. Add & Norm                                               │
│     - Residual connection + normalization                    │
│                                                              │
│  3. Feed-Forward Network (FFN)                               │
│     - 768 → 3072 → 768                                       │
│     - Two linear layers + GELU                               │
│     - Applied independently to each token                    │
│                                                              │
│  4. Add & Norm                                               │
│     - Residual connection + normalization                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       Next Transformer Block
```

(...) for `ViT` implementations see the Appendix C[^vit_impl].

### Self-Attention — the core mechanism

Self-Attention[^self_attention] allows each token to compute a weighted sum of all other
tokens, where the weights are learned based on content similarity:

```text
For token_i:

Query  = W_Q × token_i          ← "What am I looking for?"
Key    = W_K × token_j          ← "What do I contain?"
Value  = W_V × token_j          ← "What information do I provide?"

attention_weight(i → j) = softmax(Query_i · Key_j / √d)

output_i = Σ_j attention_weight(i → j) × Value_j
```

**Multi-Head** means this is done in parallel with `H = 12` different sets of
Q/K/V projections, each attending to different aspects (for example,
one head might attend to spatial neighbours, another to spectrally similar patches).

### What Self-Attention achieves spatially

Consider a 2.24 km × 2.24 km chip where the parcel occupies the centre
and is surrounded by other fields:

```text
┌──────────────────────────────────────────────────────┐
│ Patch A         Patch B         Patch C              │
│ soybean field   road            tree line            │
│                                                      │
│ Patch D         Patch E         Patch F              │
│ corn field      YOUR PARCEL     irrigation           │
│                (parcel centre)                       │
│                                                      │
│ Patch G         Patch H         Patch I              │
│ bare soil       wetland edge    wheat field          │
└──────────────────────────────────────────────────────┘
```

Through Self-Attention:

- **Patch E** (the parcel) learns it is surrounded by diverse agriculture,
  not urban land — this context matters for interpreting its spectral
  signature.
- **Patch D** (corn) and **Patch A** (soybeans) might develop similar
  NIR reflectance patterns but differ in red-edge bands — the attention
  heads learn to distinguish them.
- **Patch H** (wetland) helps contextualise soil moisture patterns in
  nearby patches.

After `N` layers of Self-Attention, each token carries not just its own
spectral information but a **contextualised understanding** of the
entire scene.

### Feed-Forward Network — expanding the representation

After Self-Attention mixes information across tokens, the Transformer
applies a **Feed-Forward Network (FFN)** to each token independently.

The FFN is a small **two-layer multilayer perceptron (MLP)** that
temporarily expands the dimensionality of the token representation:

```text
768 → 3072 → 768
```

In practice this is implemented as two learned linear projections with a
non-linear activation (typically GELU):

```text
FFN(x) = W₂ GELU(W₁ x + b₁) + b₂
```

where

```text
W₁ : (768 × 3072)
W₂ : (3072 × 768)
```

This expansion allows the model to learn richer combinations of the
features produced by Self-Attention.

Conceptually, the process works as follows:

1. **Self-Attention mixes information across patches**, allowing each
   token to incorporate context from the entire scene.

2. **The FFN then transforms the token's internal features**, learning
   new combinations of spectral, spatial, and contextual signals.

Unlike Self-Attention, the FFN **does not mix tokens with each other**.
Instead, it processes each token separately while using shared weights
across all tokens.

The result is a refined representation where the contextual information
gathered by attention is further transformed into higher-level features
useful for downstream tasks.

---

## 4. The CLS Token — Global Context

Before the patch tokens enter the Transformer, a special learnable token
called **CLS** (classification token) is prepended to the sequence:

```text
Input to Transformer

position:   [  0  ] [  1  ] [  2  ] [  3  ] ... [196]
token:      [ CLS ] [ P_1 ] [ P_2 ] [ P_3 ] ... [P_196]
               ↑
        learnable parameter

Total = 197 tokens
1 CLS + 196 patch tokens
Each token has dimension 768
```

The CLS token starts as a learned vector with no spatial meaning. Through
Self-Attention, it attends to **every patch** in every layer. After `N`
layers, it has aggregated a global summary of the entire image:

```text
After Transformer

position:   [  0  ] [  1  ] [  2  ] ... [196]
token:      [ CLS ] [ P_1 ] [ P_2 ] ... [P_196]
               │
               ▼
        GLOBAL SUMMARY VECTOR
             (768-dim)

This is the embedding.
It summarizes the full scene into a single representation.
```

This is exactly what `embed.py` extracts:

```python
encoded_patches = out[0]               # (B, 197, 768)
embeddings = encoded_patches[:, 0, :]  # (B, 768) ← CLS token
```

---

## 5. What Makes Clay Different

Clay is not a standard Vision Transformer trained on RGB photos. It was
designed specifically for Earth observation and incorporates several
domain-specific innovations:

### 5.1 — Multi-spectral input

While a standard ViT processes 3 channels (RGB), Clay processes **10+**
spectral bands covering the visible, near-infrared, and shortwave
infrared spectrum. The patch embedding layer is sized accordingly.

### 5.2 — Wavelength encoding

Clay receives the **centre wavelength** of each input band as a
continuous input. This allows the same model to process data from
different sensors (Sentinel-2, Landsat, etc.) without retraining:

```text
waves = [
  492.4, 559.8, 664.6, 704.1, 740.5,
  782.8, 832.8, 864.7, 1613.7, 2202.4
]
```

### 5.3 — Temporal encoding

The observation date is encoded as sinusoidal features of the day-of-year,
allowing the model to understand seasonality:

```text
time = [
  sin(2π × doy / 365),
  cos(2π × doy / 365),
  sin(hour),
  cos(hour)
]
```

A corn field looks very different in April (bare soil) versus July
(full canopy). The temporal encoding lets the model distinguish between
"corn in April" and "corn in July" as semantically different states.

### 5.4 — Geographic encoding

Latitude and longitude are encoded so the model accounts for geographic
variation (for example, corn in Iowa versus corn in Argentina may have
different reference signatures):

```text
latlon = [
  sin(lat_rad),
  cos(lat_rad),
  sin(lon_rad),
  cos(lon_rad)
]
```

### 5.5 — Pre-training via Masked Autoencoding

Clay was pre-trained by **masking random patches** and training the model
to reconstruct them from context. This forces the model to learn deep
geophysical relationships:

```text
Visible / masked patch pattern during pre-training

┌─────┬─────┬─────┬─────┬─────┐
│  ✓  │  ✗  │  ✓  │  ✗  │  ✓  │
├─────┼─────┼─────┼─────┼─────┤
│  ✗  │  ✓  │  ✗  │  ✓  │  ✗  │
├─────┼─────┼─────┼─────┼─────┤
│  ✓  │  ✗  │  ✓  │  ✗  │  ✓  │
└─────┴─────┴─────┴─────┴─────┘

✓ = visible to model
✗ = masked and must be reconstructed
```

This unsupervised pre-training on large-scale satellite imagery gives
Clay a strong foundation of geophysical knowledge before any crop-specific
fine-tuning.

---

## 6. Fine-tuning: Backbone + Head

Fine-tuning adds a **classification head** on top of the pre-trained
backbone and trains the combined model to classify crops.

The conceptual split between **backbone** and **head** is the same one
used in standard ViT implementations; see Appendix C[^vit_impl] for
minimal PyTorch and TensorFlow examples, and Appendix D[^clay_impl] for
the Clay-specific version of this pattern.

### The full architecture during training

```text
FULL MODEL (training)

Input
  chip (10, 224, 224)
  + time
  + latlon
  + waves
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│ BACKBONE (Clay ViT)                                      │
│                                                          │
│  patch embedding                                         │
│  + CLS token                                             │
│  + positional embeddings                                 │
│  + temporal / geographic / wavelength encodings          │
│  + transformer blocks                                    │
│                                                          │
│  output: 197 tokens × 768 dims                           │
│          └─ CLS token extracted as global embedding      │
└──────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│ HEAD (Linear Classifier)                                 │
│                                                          │
│  CLS embedding (768)                                     │
│      → Linear(768, N)                                    │
│      → logits / softmax                                  │
│                                                          │
│  Example output:                                         │
│  [P(corn)=0.91, P(soy)=0.06, P(wheat)=0.03]              │
└──────────────────────────────────────────────────────────┘
          │
          ▼
Cross-Entropy Loss against crop label
```

### Staged training strategy

Training proceeds in two phases to protect pre-trained knowledge:

```text
PHASE 1 — Head-only training
────────────────────────────────────────────────────────

Backbone : frozen   (no gradients)
Head     : trainable
LR       : typically higher for head

Purpose:
learn initial class boundaries using the existing embedding space
without perturbing the pre-trained encoder.

Typical duration:
~5 epochs


PHASE 2 — Full fine-tuning
────────────────────────────────────────────────────────

Backbone : trainable
Head     : trainable
LR       : lower for backbone than for head
           or uniformly small across the model

Purpose:
gently adapt the backbone so that embeddings become more
crop-discriminative for the target region and crop mix.

Typical duration:
remaining epochs
```

**Why two phases?** If you unfreeze the backbone from the start, the
large random gradients from the untrained head can damage the carefully
learned pre-trained weights. By training the head first, the gradients
that later reach the backbone are already more meaningful.

In Clay specifically, this means the Earth-observation encoder is first
used as a stable feature extractor and only later adapted end-to-end;
see Appendix D[^clay_impl] for a minimal Clay-style implementation.

---

## 7. Why the Head Is Auxiliary

This is the crucial conceptual point of the entire system.

**The monitor does not perform classification.** It does not use the
head to predict "this is corn" or "this is soybeans." The crop label
comes from the CDL in Phase 1.

The classification head serves a single purpose during training: **it
provides the loss signal** that forces the backbone to reorganise its
internal representations so that different crops produce different
embeddings.

```text
Before fine-tuning (conceptual embedding space)

corn      ×   ×      ○ soy
soy       ○ × ○      △ wheat
wheat     △ ○ △ ×

Clusters are mixed.
The encoder understands land cover broadly,
but crop types are not yet cleanly separated.


After fine-tuning with classification supervision

corn      × × × ×
          × × ×

soy               ○ ○ ○
                  ○ ○ ○

wheat                    △ △ △
                         △ △ △

Clusters become tighter and more separated.
```

After training, the head is **discarded**. What remains is the backbone
that learned to produce these well-structured, crop-discriminative
embeddings.

```text
┌──────────────────────────┐
│ Fine-tuned full model    │
│                          │
│  backbone  +  head       │
└─────────────┬────────────┘
              │
              ├── keep backbone
              │
              └── discard head

Result:
encoder-only checkpoint for embedding generation
```

---

## 8. Exporting the Encoder

The `export_encoder.py` script performs a surgical extraction:
it keeps only the backbone weights and removes the task-specific
classification head.

This mirrors the separation shown conceptually in Appendix C[^vit_impl]
and the Clay-specific training/export flow shown in Appendix D[^clay_impl].

```text
Fine-tuned checkpoint
────────────────────────────────────────────

model.backbone.patch_embed.weight
model.backbone.patch_embed.bias
model.backbone.blocks.0.attn.qkv.weight
model.backbone.blocks.0.attn.qkv.bias
model.backbone.blocks.0.attn.proj.weight
model.backbone.blocks.0.mlp.fc1.weight
...
model.backbone.cls_token
model.backbone.pos_embed

model.head.weight        ← discard
model.head.bias          ← discard
optimizer state          ← discard
scheduler state          ← discard


Exported encoder checkpoint
────────────────────────────────────────────

patch_embed.weight
patch_embed.bias
blocks.0.attn.qkv.weight
blocks.0.attn.qkv.bias
blocks.0.attn.proj.weight
...
cls_token
pos_embed
```

This encoder-only checkpoint is what `embed.py` later loads for
inference-time embedding generation.

---

## 9. From Embeddings to Anomaly Detection

The monitor pipeline (Phase 3) uses the fine-tuned encoder **without any
classification layer**. Instead, it leverages the geometric structure
of the embedding space for anomaly detection.

### The core idea

If the backbone was trained to cluster crops tightly, then a **healthy
parcel** will produce embeddings close to the cluster centre of its crop
type, while a **stressed parcel** will drift away from that centre.

```text
Reference trajectory (median crop profile)

April ─── May ─── June ─── July ─── August
  ●        ●        ●        ●        ●     ← reference profile

Healthy parcel
  ●        ●        ●        ●        ●     ← stays close

Stressed parcel
  ●        ●         ╲      ╱         ●      ← deviates mid-season
                      ○────○                   ○ = stressed state
                                              
```

### The scoring mechanism

For each observation date, cosine distance measures the angular separation
between a parcel embedding and the crop reference profile:

```text
reference vector
      │
      │\
      │ \
      │  \   θ
      │   \
      │    \
      └─────► parcel embedding

distance = 1 - cos(θ)
```
Interpretation:

```text
θ ≈ 0°    → distance ≈ 0.00   → very similar
θ ≈ 25°   → distance ≈ 0.09   → mild deviation
θ ≈ 45°   → distance ≈ 0.29   → significant deviation
θ ≈ 90°   → distance ≈ 1.00   → extreme mismatch
```
```text
SMALL ANGLE (similar crops)              LARGE ANGLE (anomaly)

parcel vector                            parcel vector
             ↗                                                                               ↑
     / ) θ                                     │ ) θ
    /  )                                       │  )
   /   )                                       │  )
origin ●────────────► reference vector   origin ●────────────► reference vector
  in                                       in
embedding                               embedding 
 space                                    space
                        
distance ≈ 0.05                          distance ≈ 0.7
```
### Complete scoring flow

```text
For each parcel

1. Load all dated embeddings
   {2024-04-10: vec₁, 2024-04-25: vec₂, ...}

2. Load the reference profile for its crop
   ref["corn"] = {2024-04-07: ref₁, 2024-04-21: ref₂, ...}

3. Match each parcel date to the nearest reference date
   within the allowed temporal window

4. Compute cosine distances
   dist₁, dist₂, dist₃, ...

5. Aggregate distances with a weighted average
   health_score = weighted_avg(distances, weights)

6. Assign status
   score < 0.10   → GREEN
   score < 0.25   → YELLOW
   score ≥ 0.25   → RED
   insufficient data → GRAY
```

### Summary: the role of each component

```text
TRAINING (Phase 2)
────────────────────────────────────────
Backbone → learns better embeddings
Head     → provides supervised loss
Head     → discarded after training


MONITORING (Phase 3)
────────────────────────────────────────
Backbone only → generates embeddings
Embeddings    → compared to crop reference profiles
Classification head is not used
```

The classification head is a **training-time tool**, not an inference-time
component. It enriches the backbone's representations by providing
crop-discriminative supervision, but the final monitoring system detects
anomalies purely through the **geometry of the embedding space**.

[^patch_embedding]: See **Appendix A — How the Linear Projection Produces a Patch Embedding** for a detailed explanation of the linear projection operation.

---

## Appendix A — How the Linear Projection Produces a Patch Embedding

The linear projection converts the raw pixel values of a patch into the
internal feature representation used by the transformer.

Each patch extracted from the chip has shape:

```text
10 bands × 16 × 16 pixels
```

Flattening this tensor produces a vector:

```text
x ∈ ℝ^2560
```

where:

```text
2560 = 10 × 16 × 16
```

The Vision Transformer does not operate directly on these raw values.
Instead, the flattened patch is mapped into the model's embedding space
through a learned linear transformation:

```text
y = xW + b
```

where:

```text
x : (1 × 2560)   flattened patch vector
W : (2560 × 768) learned weight matrix
b : (768,)       bias vector
y : (1 × 768)    patch embedding
```

Conceptually, the weight matrix can be viewed as follows:

```text
                    embedding dimensions
         ─────────────────────────────────────────────
              e1       e2       e3       ...    e768

v1        [  w1,1    w1,2    w1,3    ...   w1,768  ]
v2        [  w2,1    w2,2    w2,3    ...   w2,768  ]
v3        [  w3,1    w3,2    w3,3    ...   w3,768  ]
...
v2560     [ w2560,1  ...     ...     ...  w2560,768 ]
```

Each **row** corresponds to one value from the flattened patch.
Each **column** contributes to one dimension of the patch embedding.

The *j-th* embedding dimension is computed as:

```text
y_j = x1·w1,j + x2·w2,j + x3·w3,j + ... + x2560·w2560,j + b_j
```

This means that every embedding dimension is a learned weighted
combination of **all** pixel values in the patch.

The resulting vector

```text
y ∈ ℝ^768
```

is called the **patch embedding** (or **patch token**) and becomes the
representation processed by the transformer.

Each of the 768 values can be interpreted as a learned feature of the
patch, potentially capturing combinations of:

- spectral relationships between bands
- local texture patterns
- vegetation signatures
- spatial structure within the patch

The value **768** is a design choice of the transformer architecture
(the embedding dimension **D**). In many Vision Transformer models, this
matches the standard **ViT-Base** configuration, where the embedding is
split across multiple attention heads (for example, **12 heads × 64
dimensions**).

After projection, positional information is added so the model can
preserve the original spatial location of each patch:

```text
token_i = linear_proj(patch_i) + pos_embed_i
```

This produces the final sequence of tokens that enters the transformer
encoder.

[^self_attention]: See **Appendix B — How Self-Attention Computes Interactions Between Tokens** for a detailed explanation of the similarity computation.

---

## Appendix B — How Self-Attention Computes Interactions Between Tokens

After patch embedding, the model receives a sequence of tokens:

```text
197 tokens × 768 dimensions
```

(196 patch tokens plus the CLS token).

Each token is a vector produced by the patch embedding stage.
These vectors enter the Transformer encoder, where **Self-Attention**
allows tokens to exchange information.

### Step 1 — Project tokens into Query, Key, and Value spaces

The tokens are not compared directly in their original 768-dimensional
space. Instead, they are **linearly projected into three different
representations**:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

where

```text
X   : (197 × 768) token matrix
W_Q : (768 × d)
W_K : (768 × d)
W_V : (768 × d)
```

These matrices are **learned parameters of the model**, trained through
backpropagation just like any other neural network weights.

For a single attention head in ViT-Base:

```text
d = 64
```

So the projections produce:

```text
Q, K, V ∈ ℝ^(197 × 64)
```

Each token therefore generates three vectors:

```text
Query vector
Key vector
Value vector
```

### Step 2 — Compute similarity between tokens

For a given token *i*, the model measures how strongly it should attend
to every other token *j* by computing the similarity between their
Query and Key vectors:

```text
score(i, j) = (Query_i · Key_j) / √d
```

This dot product measures how compatible the two tokens are.

If two patches contain similar information (for example, similar
vegetation signatures), their score will be higher.

### Step 3 — Convert scores into attention weights

The raw scores are converted into normalized attention weights using
the softmax function:

```text
attention_weight(i → j) = softmax(score(i, j))
```

After this step, the weights satisfy:

```text
0 ≤ attention_weight(i → j) ≤ 1
Σ_j attention_weight(i → j) = 1
```

These values represent **how much token i attends to token j**.

Example:

```text
Attention distribution for token i

token j :      1      2      3      4      5
score    :    1.8    0.9    2.1    0.4    0.8
softmax  :   0.30   0.12   0.40   0.05   0.13
```

Interpretation:

```text
token i distributes its attention as follows

token 1 → 30%
token 2 → 12%
token 3 → 40%
token 4 →  5%
token 5 → 13%
```

So the updated representation of token *i* becomes:

```text
output_i =
0.30 × Value_1 +
0.12 × Value_2 +
0.40 × Value_3 +
0.05 × Value_4 +
0.13 × Value_5
```

Each token therefore becomes a **weighted combination of information
coming from all other tokens in the sequence**.

### Step 4 — Combine information from other tokens

The final representation of token *i* is computed as a weighted sum of
the Value vectors:

```text
output_i = Σ_j attention_weight(i → j) × Value_j
```

This means that each token becomes a **mixture of information from all
other tokens**, with the mixture determined by the attention weights.

### Multi-Head Attention

In practice, this process runs **multiple times in parallel**.

For example, ViT-Base uses:

```text
12 attention heads
```

Each head performs its own Q/K/V projections and computes its own
attention pattern. The results are then concatenated and projected
back to the model dimension (768).

This allows different heads to learn different types of relationships,
such as:

- spatial proximity between patches
- spectral similarity
- field boundaries
- contextual land-cover patterns

### Result

After Self-Attention, each token no longer represents only its original
patch. Instead, it contains information aggregated from the entire
image.

In matrix form:

```text
input tokens    : (197 × 768)
after attention : (197 × 768)
```

The dimensionality remains the same, but the tokens now encode a
**contextualized representation of the whole scene** rather than
isolated patches.

[^vit_impl]: See **Appendix C — ViT Implementations** for minimal PyTorch and TensorFlow examples showing how Vision Transformers separate the backbone encoder from the task-specific classification head.

---

## Appendix C — ViT Implementations

Modern Vision Transformer implementations typically separate the model
into two conceptual components:

```text
BACKBONE
patch embedding + transformer encoder
↓
embedding vector

HEAD
task-specific prediction layer
```

The **backbone** produces a fixed-dimensional representation of the
input image (an embedding).
The **head** converts that embedding into predictions for a specific task.

```text
input image
    │
    ▼
┌─────────────────────────────┐
│ ViT Backbone                │
│                             │
│ patch embedding             │
│ + CLS token                 │
│ + positional encoding       │
│ + transformer stack         │
└──────────────┬──────────────┘
               │
               ▼
        embedding vector
               │
               ▼
┌─────────────────────────────┐
│ Task-specific head          │
│ Linear / Dense              │
│ Softmax or task output      │
└──────────────┬──────────────┘
               │
               ▼
           prediction
```

This separation allows the backbone to later be reused as a standalone
**feature encoder**, while the head is only required during supervised
training.

### Minimal PyTorch example

```python
import torch
import torch.nn as nn


class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels=10,
            out_channels=768,
            kernel_size=16,
            stride=16
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(768)

    def forward(self, x):
        x = self.patch_embed(x)           # (B, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)

        x = self.encoder(x)
        x = self.norm(x)

        return x.mean(dim=1)              # embedding


class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = ViTBackbone()
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits
```

### Minimal Keras / TensorFlow example

```python
import tensorflow as tf
from tensorflow.keras import layers, Model


def transformer_block(x):
    x1 = layers.LayerNormalization()(x)

    attn = layers.MultiHeadAttention(
        num_heads=12,
        key_dim=64
    )(x1, x1)

    x = layers.Add()([x, attn])

    x2 = layers.LayerNormalization()(x)

    ffn = layers.Dense(3072, activation="gelu")(x2)
    ffn = layers.Dense(768)(ffn)

    x = layers.Add()([x, ffn])

    return x


def build_vit_backbone():
    inputs = layers.Input(shape=(224, 224, 10))

    x = layers.Conv2D(
        768,
        kernel_size=16,
        strides=16
    )(inputs)

    x = layers.Reshape((196, 768))(x)

    for _ in range(4):
        x = transformer_block(x)

    x = layers.LayerNormalization()(x)

    embedding = layers.GlobalAveragePooling1D()(x)

    return Model(inputs, embedding, name="vit_backbone")


def build_vit_classifier(num_classes):
    backbone = build_vit_backbone()

    inputs = backbone.input
    features = backbone(inputs)

    outputs = layers.Dense(num_classes)(features)

    return Model(inputs, outputs)
```

In both frameworks the architecture clearly separates:

```text
backbone → encoder that produces embeddings
head     → task-specific classifier
```

The head can later be removed when the model is used as a **feature encoder**.

[^clay_impl]: See **Appendix D — Clay Encoder Implementation** for a simplified Clay-specific example showing how the backbone and classification head are structured during fine-tuning and how the encoder is later used by TerraTorch during monitoring.

---

## Appendix D — Clay Encoder Implementation

Clay follows the same architectural separation described in Appendix C,
but its Vision Transformer backbone is specialized for Earth observation.

During training, the full model contains two components:

```text
chip + metadata
(time, lat/lon, wavelengths)
            │
            ▼
┌──────────────────────────────┐
│ Clay Backbone                │
│                              │
│ multispectral patch embed    │
│ + CLS token                  │
│ + positional embeddings      │
│ + temporal encoding          │
│ + geographic encoding        │
│ + wavelength encoding        │
│ + transformer blocks         │
└──────────────┬───────────────┘
               │
               ▼
        embedding (768)
               │
               ▼
┌──────────────────────────────┐
│ Classification Head          │
│ Linear(768 → N)              │
│ Softmax / logits             │
└──────────────┬───────────────┘
               │
               ▼
           crop class
```

The **backbone** is the Clay Vision Transformer encoder and contains
almost all of the model parameters.

The **classification head** is a small linear layer used only during
fine-tuning to provide a supervised training signal.

### Minimal Clay-style training example

```python
import torch
import torch.nn as nn


class ClayBackbone(nn.Module):
    def __init__(self, clay_encoder):
        super().__init__()
        self.encoder = clay_encoder

    def forward(self, chip, time, latlon, waves):
        encoded_tokens = self.encoder(
            chip=chip,
            time=time,
            latlon=latlon,
            waves=waves
        )

        cls_embedding = encoded_tokens[:, 0, :]   # (B, 768)
        return cls_embedding


class ClayClassifier(nn.Module):
    def __init__(self, clay_encoder, num_classes):
        super().__init__()

        self.backbone = ClayBackbone(clay_encoder)
        self.head = nn.Linear(768, num_classes)

    def forward(self, chip, time, latlon, waves):
        features = self.backbone(
            chip,
            time,
            latlon,
            waves
        )

        logits = self.head(features)
        return logits
```

In this setup:

```text
backbone → Clay encoder producing the 768-dim embedding
head     → linear classifier used during fine-tuning
```

### Exporting the encoder after fine-tuning

After training, the classification head is removed and only the backbone
weights are exported for monitoring.

```text
clay-finetuned-crops.ckpt
        │
        ▼
remove classification head
        │
        ▼
clay-crop-encoder.ckpt
```

Example export step:

```python
import torch

model = ClayClassifier(clay_encoder, num_classes=5)

backbone_state = model.backbone.state_dict()
torch.save(backbone_state, "clay-crop-encoder.ckpt")
```

This exported checkpoint contains only the encoder parameters needed to
produce embeddings.

### TerraTorch integration

During monitoring, the encoder is loaded through **TerraTorch**, which
provides a standardized interface for Earth observation foundation
models.

```text
clay-crop-encoder.ckpt
        │
        ▼
TerraTorch embedder
        │
        ▼
CLS embedding (768)
```

Minimal example:

```python
import torch

embedder = ClayBackbone(clay_encoder)
embedder.load_state_dict(torch.load("clay-crop-encoder.ckpt"))
embedder.eval()

with torch.no_grad():
    embeddings = embedder(
        chip=chip_batch,
        time=time_batch,
        latlon=latlon_batch,
        waves=waves_batch
    )   # (B, 768)
```

At this stage, the model no longer performs classification.

Instead, it acts purely as a **feature encoder** that maps satellite
imagery and metadata into a structured embedding space.

```text
satellite chip + metadata
        │
        ▼
   Clay backbone
        │
        ▼
 embedding (768)
        │
        ▼
distance to crop reference profile
```

These embeddings are then compared against crop reference profiles using
cosine distance or related similarity metrics.

In other words:

```text
training   → backbone + head
monitoring → backbone only
```
