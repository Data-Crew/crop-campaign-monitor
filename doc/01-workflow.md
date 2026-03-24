# Crop Campaign Monitor — Workflow Guide

> Step-by-step explanation of the full pipeline, from raw field
> boundaries to per-parcel health scores.

---

## Overview

The system transforms **satellite imagery** of agricultural parcels into
**health scores** by building a semantic representation of what each field
looks like across the growing season and comparing it against a reference
profile for its crop type.

```
  GeoJSON parcels    Sentinel-2 imagery    Crop labels (CDL)
       │                    │                     │
       └────────┬───────────┘                     │
                ▼                                 │
    ┌─────────────────────┐                       │
    │  Phase 1: Data Prep │◄──────────────────────┘
    │  ingest → fetch →   │
    │  chip               │
    └────────┬────────────┘
             │  chips (.npz)  +  parcels_labeled.parquet
             ├──────────────────────────────┐
             ▼                              ▼
    ┌─────────────────────┐     ┌───────────────────────┐
    │  Phase 2: Training  │     │  Phase 3: Monitor          │
    │  prepare → finetune │     │  embed → profile →         │
    │  → export_encoder   │     │  score → report → index    │
    └────────┬────────────┘     └───────────┬────────────────┘
             │  clay-crop-encoder.ckpt      │
             └──────────────────────────────┘
                                            │
                                  parcel_scores.geojson
                                  campaign_report.json
                                  parcels.faiss
                                  parcels_index_meta.parquet
                                            │
                                     ┌──────▼──────┐
                                     │  Streamlit  │
                                     │  Dashboard  │
                                     │  + Similar  │
                                     │    Parcels  │
                                     └─────────────┘
```

---

## Phase 1 — Data Preparation

Phase 1 produces the shared inputs consumed by both Training (Phase 2) and
Monitoring (Phase 3).  It has three sequential steps.

---

### Step 1 — Ingest (`src/ingest.py`)

**What it does:**  reads parcel geometries from GeoJSON files in
`data/fields/`, reprojects them to a common CRS, computes centroids and
areas, and assigns a crop label to each parcel by cross-referencing with
the USDA Cropland Data Layer (CDL).

**Input:**

```
data/fields/
  16tgk/
    16tgk_fragment_00_00.geojson
    16tgk_fragment_01_00.geojson
    ...
```

Each GeoJSON contains polygon features representing individual
agricultural parcels.

**Output — `data/output/parcels_labeled.parquet`:**

| parcel_id | crop_code | crop_name | centroid_lat | centroid_lon | area_ha | tile_id | fragment_id | geometry |
|-----------|-----------|-----------|-------------|-------------|---------|---------|-------------|----------|
| P001 | 1 | corn | 41.2300 | -93.4500 | 12.4 | 16tgk | 00_00 | POLYGON(…) |
| P002 | 5 | soybeans | 41.1900 | -93.5100 | 8.7 | 16tgk | 00_00 | POLYGON(…) |
| P003 | 22 | wheat | 41.2700 | -93.3800 | 15.2 | 16tgk | 01_00 | POLYGON(…) |
| P004 | 1 | corn | 41.3100 | -93.4200 | 22.1 | 16tgk | 01_00 | POLYGON(…) |
| P005 | 5 | soybeans | 41.2500 | -93.4900 | 9.8 | 16tgk | 00_00 | POLYGON(…) |

This table is the single source of truth for parcel identities and labels
throughout the rest of the pipeline.

---

### Step 2 — Fetch (`src/fetch.py`)

**What it does:**  searches the [Earth Search STAC catalog](https://earth-search.aws.element84.com/v1)
for Sentinel-2 L2A scenes that cover the parcel bounding box during the
configured growing season.  It does **not** download full tiles — only
metadata and COG (Cloud-Optimized GeoTIFF) asset URLs.

**Output — `data/tiles/tile_index.parquet`:**

| date | tile_id | cloud_pct | asset_urls |
|------|---------|-----------|-----------|
| 2024-04-10 | 16tgk | 3.2 | `{"B02": "https://…/B02.tif", "B03": "…", …}` |
| 2024-04-25 | 16tgk | 8.1 | `{"B02": "https://…", …}` |
| 2024-05-05 | 16tgk | 1.4 | `{"B02": "https://…", …}` |
| 2024-05-21 | 16tgk | 12.7 | `{"B02": "https://…", …}` |
| … | … | … | … |

Each row is a single Sentinel-2 pass.  The `asset_urls` column is a JSON
dict mapping band IDs (B02, B03, B04, B08, …) to their COG URLs on S3.

---

### Step 3 — Chip (`src/chip.py`)

**What it does:**  for each **parcel × date** combination, reads a
**224 × 224 pixel window** from the Sentinel-2 COGs, centred on the
parcel's centroid.  At 10 m/px resolution this covers an area of
**2.24 km × 2.24 km**.

#### What is a chip?

A chip is a small, square image patch cut from a satellite scene around
the centre point of a parcel.  Think of it as a standardised "portrait"
of a field at a specific date.

```
  Sentinel-2 tile (~110 km × 110 km)
  ╔══════════════════════════════════════════════════════════╗
  ║                                                          ║
  ║       ┌─────────────────────┐                            ║
  ║       │                     │                            ║
  ║       │       chip          │ 224 px × 224 px            ║
  ║       │     (2.24 km)       │ = 2.24 km × 2.24 km       ║
  ║       │                     │                            ║
  ║       │        ·  centroid  │                            ║
  ║       │       ╱╲            │                            ║
  ║       │      ╱  ╲           │                            ║
  ║       │     ╱    ╲  parcel  │                            ║
  ║       │     ╲    ╱  ~450 m  │                            ║
  ║       │      ╲  ╱           │                            ║
  ║       │       ╲╱            │                            ║
  ║       │                     │                            ║
  ║       └─────────────────────┘                            ║
  ║                                                          ║
  ║                        ┌────────────┐                    ║
  ║                        │ other chip │                    ║
  ║                        │     ·      │                    ║
  ║                        └────────────┘                    ║
  ║                                                          ║
  ╚══════════════════════════════════════════════════════════╝
```

**Scale reference:**

| Element | Physical size |
|---------|--------------|
| 1 Sentinel-2 pixel | 10 m × 10 m |
| 1 chip | 224 px = **2.24 km × 2.24 km** |
| Typical parcel (20 ha) | ~450 m × 450 m ≈ 20% of the chip |
| Full Sentinel-2 tile | ~110 km × 110 km |

**Important:** the chip is always a fixed-size square centred on the
centroid — it does not follow the parcel's irregular boundary.  The
surrounding context (neighbouring fields, roads, waterways) is
intentionally included, as the model learns from spatial context.

#### Number of chips

The total number of chips equals **parcels × valid dates**.  If there are
5 parcels and 2 cloud-free Sentinel-2 passes, the system produces
5 × 2 = **10 chips**.  Chips with excessive cloud cover or nodata are
discarded.

#### Chip file structure

Each chip is saved as a compressed NumPy archive (`.npz`):

```
data/chips/
  P001/
    2024-04-10.npz
    2024-04-25.npz
    2024-05-05.npz
  P002/
    2024-04-10.npz
    2024-04-25.npz
  ...
```

**Contents of a single `.npz` file:**

| field | dtype | shape / value | description |
|-------|-------|--------------|-------------|
| `pixels` | float32 | `(10, 224, 224)` | 10 spectral bands, 224×224 px |
| `parcel_id` | string | `"P001"` | Parcel identifier |
| `date` | string | `"2024-04-10"` | Observation date |
| `lat` | float64 | `41.23` | Centroid latitude |
| `lon` | float64 | `-93.45` | Centroid longitude |
| `gsd` | float64 | `10.0` | Ground sample distance (m) |
| `bands` | string[] | `["B02","B03",…]` | Band names in order |

**The 10 spectral bands** (one per channel in the `pixels` array):

| Channel | Band | Name | Wavelength (nm) | What it captures |
|---------|------|------|-----------------|-----------------|
| 0 | B02 | Blue | 492 | Water, aerosols |
| 1 | B03 | Green | 560 | Green reflectance peak |
| 2 | B04 | Red | 665 | Chlorophyll absorption |
| 3 | B08 | NIR | 833 | Vegetation structure |
| 4 | B05 | Red Edge 1 | 704 | Chlorophyll transition |
| 5 | B06 | Red Edge 2 | 741 | Canopy structure |
| 6 | B07 | Red Edge 3 | 783 | Leaf area index |
| 7 | B8A | NIR 08 | 865 | Extended near-infrared |
| 8 | B11 | SWIR 16 | 1614 | Moisture content |
| 9 | B12 | SWIR 22 | 2202 | Soil / dry vegetation |

**Example — pixel values at the chip centre (112, 112):**

| Band | Raw value (DN) | Normalised (÷ 10 000) |
|------|---------------|----------------------|
| B02 (Blue) | 412 | 0.0412 |
| B03 (Green) | 589 | 0.0589 |
| B04 (Red) | 320 | 0.0320 |
| B08 (NIR) | 3210 | 0.3210 |
| B05 (RedEdge1) | 890 | 0.0890 |
| B06 (RedEdge2) | 1450 | 0.1450 |
| B07 (RedEdge3) | 2100 | 0.2100 |
| B8A (NIR08) | 3050 | 0.3050 |
| B11 (SWIR16) | 1200 | 0.1200 |
| B12 (SWIR22) | 680 | 0.0680 |

The normalisation (division by 10 000) converts Sentinel-2 digital
numbers into surface reflectance values.  This is performed at embedding
time, not during chip extraction.

---

## Phase 2 — Training (optional)

Phase 2 fine-tunes the Clay Foundation Model to produce
crop-discriminative embeddings.  It is **optional** — the monitor can run
with a pre-trained encoder, but fine-tuning significantly improves
sensitivity to regional crop patterns.

> For a deep dive into the model architecture, see
> [02-architecture.md](02-architecture.md).

---

### Step 1 — Prepare Dataset (`train/prepare_dataset.py`)

**What it does:**

1. Reads `parcels_labeled.parquet` from Phase 1.
2. Optionally re-labels parcels from a different CDL year.
3. Filters out crop classes with fewer samples than `min_parcels_per_class`.
4. For each valid parcel, lists all its `.npz` chips and assigns the
   crop class ID.
5. Splits the result into **train / val / test** sets with stratified
   sampling (equal class proportions in each split).

**Output — `data/training/train_manifest.csv`** (same format for val and test):

| chip_path | crop_class_id | crop_name | parcel_id | date |
|-----------|--------------|-----------|-----------|------|
| data/chips/P001/2024-04-10.npz | 0 | corn | P001 | 2024-04-10 |
| data/chips/P001/2024-04-25.npz | 0 | corn | P001 | 2024-04-25 |
| data/chips/P001/2024-05-05.npz | 0 | corn | P001 | 2024-05-05 |
| data/chips/P002/2024-04-10.npz | 1 | soybeans | P002 | 2024-04-10 |
| data/chips/P002/2024-04-25.npz | 1 | soybeans | P002 | 2024-04-25 |
| data/chips/P003/2024-04-10.npz | 2 | wheat | P003 | 2024-04-10 |

Each row is one training example: a chip path and its numeric class label.

**Output — `data/training/class_mapping.json`:**

```json
{
  "0": "corn",
  "1": "soybeans",
  "2": "wheat"
}
```

---

### Step 2 — Fine-tune (`train/finetune.py`)

**What it does:** trains a crop classifier using the Clay Foundation Model
as a backbone, via the TerraTorch framework.  The training is **staged**:

| Phase | What trains | What is frozen | Learning rate | Purpose |
|-------|------------|----------------|--------------|---------|
| 1 (head-only) | Linear classification head (768 → N classes) | Clay ViT backbone | `1e-4` | Learn crop decision boundaries without damaging pre-trained weights |
| 2 (full) | Both backbone and head | Nothing | `1e-5` (lower) | Gently adapt the backbone's representations to regional crop signatures |

**Key insight:** the classifier head itself is not the goal.  The
classification task acts as a *supervision signal* that forces the backbone
to learn crop-discriminative representations.  It is the **backbone's
output** — the embedding — that gets used downstream in the monitor.

**Output:**

```
data/model/
  clay-finetuned-crops.ckpt    ← full checkpoint (backbone + head + optimizer)

data/training/logs/
  metrics.json                 ← training metrics
```

> For a detailed explanation of how fine-tuning works and why the head is
> auxiliary, see [02-architecture.md](02-architecture.md).

---

### Step 3 — Export Encoder (`train/export_encoder.py`)

**What it does:** opens the full fine-tuned checkpoint, extracts **only
the backbone** weights (discarding the classification head and optimizer
state), and saves them as a standalone file that the monitor can load.

```
clay-finetuned-crops.ckpt (full model)
  ├── model.backbone.*    ← backbone weights      ──► KEPT
  ├── model.head.*        ← classification head    ──► DISCARDED
  └── optimizer.*         ← optimizer state         ──► DISCARDED

                    ↓

clay-crop-encoder.ckpt (encoder only, ~300M params)
```

This step also runs a verification pass: it loads the exported weights
into a fresh `Embedder` instance and checks the output shape.

---

## Phase 3 — Monitor

Phase 3 uses the fine-tuned encoder to generate embeddings, build
reference profiles, and score parcels.

---

### Step 4 — Embed (`src/embed.py`)

**What it does:** loads each `.npz` chip, passes it through the Clay
encoder, and saves the resulting **768-dimensional embedding vector** as a
`.npy` file.

The encoder is loaded from a cascade of sources (first match wins):

1. Fine-tuned encoder (`clay-crop-encoder.ckpt`) — best quality.
2. Pre-trained base Clay (`clay-v1-base.ckpt`) — generic representations.
3. Mock encoder — deterministic random vectors for pipeline testing only.

**Input per chip:**

The encoder receives a `datacube` dict with five components:

| Key | Shape | Content |
|-----|-------|---------|
| `pixels` | `(B, 10, 224, 224)` | Normalised reflectance values (÷ 10 000) |
| `time` | `(B, 4)` | `[sin(week), cos(week), sin(hour), cos(hour)]` |
| `latlon` | `(B, 4)` | `[sin(lat), cos(lat), sin(lon), cos(lon)]` |
| `waves` | `(10,)` | Centre wavelength (nm) of each band |
| `gsd` | scalar | Ground sample distance (10.0 m) |

**Output:**

The encoder returns a tensor of shape `(B, 197, 768)` — one 768-dim
vector per patch token plus the CLS token.  The CLS token (position 0) is
extracted as the embedding:

```
encoder output → (B, 1+196, 768)
                        ↑
                   CLS token    → (B, 768) → saved as .npy
```

**File structure:**

```
data/embeddings/
  P001/
    2024-04-10.npy    ← array shape (768,)
    2024-04-25.npy
    2024-05-05.npy
  P002/
    2024-04-10.npy
    2024-04-25.npy
  ...
```

**Example embedding (first 8 of 768 dimensions):**

```
P001 / 2024-04-10:  [ 0.234, -0.891, 0.456, 0.012, -0.334, 0.789, -0.123, 0.567, …]
P001 / 2024-05-05:  [ 0.241, -0.878, 0.461, 0.019, -0.341, 0.801, -0.118, 0.559, …]
P002 / 2024-04-10:  [-0.102,  0.543, 0.871, 0.234,  0.112,-0.445,  0.667,-0.321, …]
```

The numbers are not human-interpretable.  What matters is that fields with
similar crops and conditions produce vectors that are **geometrically
close** in 768-dimensional space, while anomalous fields produce vectors
that are **far away** from their peers.

---

### Step 5 — Profile (`src/profile.py`)

**What it does:** builds a **reference embedding trajectory** per crop type
— "what does a healthy field of this crop look like across the season?"

**Algorithm:**

1. Divide the growing season into **biweekly (14-day) bins**.
2. For each crop type, collect all embeddings from all parcels of that
   crop that fall within each bin.
3. Compute the **component-wise median** across all collected vectors in
   each bin.  The median is robust to outliers (a few stressed parcels
   do not distort the reference).

**Concrete example — corn profile (season Apr–Sep, 3 parcels):**

For Bin 1 (01–14 April), there are three corn parcels with embeddings
available in that window:

```
P001: [0.23, -0.89, 0.45, …]
P007: [0.25, -0.87, 0.47, …]
P015: [0.22, -0.90, 0.44, …]
                ↓ median
Ref:  [0.23, -0.89, 0.45, …]  ← reference embedding for corn, bin 1
```

Repeating across all bins produces the full trajectory:

| Bin centre | Median embedding (768d, truncated) | Growth stage |
|------------|-----------------------------------|-------------|
| 2024-04-07 | `[0.23, -0.89, 0.45, …]` | Emergence |
| 2024-04-21 | `[0.31, -0.75, 0.58, …]` | Early vegetative |
| 2024-05-05 | `[0.52, -0.45, 0.71, …]` | Vegetative development |
| 2024-05-19 | `[0.68, -0.22, 0.83, …]` | Canopy closure |
| 2024-06-02 | `[0.71, -0.15, 0.86, …]` | Tasselling |
| 2024-06-16 | `[0.65, -0.25, 0.79, …]` | Grain fill |
| 2024-06-30 | `[0.48, -0.51, 0.62, …]` | Late grain fill |
| 2024-07-14 | `[0.31, -0.72, 0.44, …]` | Maturity |
| 2024-07-28 | `[0.18, -0.88, 0.33, …]` | Senescence |
| 2024-08-11 | `[0.12, -0.91, 0.31, …]` | Dry-down |

**Output — `data/output/reference_profiles.pkl`:**

A Python dict mapping crop names to lists of `(bin_date, median_vector)`
tuples:

```python
{
  "corn":     [("2024-04-07", array(768,)), ("2024-04-21", array(768,)), …],
  "soybeans": [("2024-04-07", array(768,)), ("2024-04-21", array(768,)), …],
  "wheat":    [("2024-04-07", array(768,)), …],
}
```

---

### Step 6 — Score (`src/score.py`)

**What it does:** compares each parcel's embedding trajectory against the
reference profile for its crop type using **cosine distance**, producing
a health score and a traffic-light status.

#### What is cosine distance?

Cosine distance measures the angle between two vectors, ignoring their
magnitude.  Two vectors pointing in the same direction have
distance **0** (identical); perpendicular vectors have distance **1**.

```
Cosine distance = 1 - (A · B) / (‖A‖ × ‖B‖)

  0.00 → identical direction (healthy)
  0.10 → very similar
  0.25 → noticeable deviation
  0.50 → significant deviation
  1.00 → perpendicular (maximum practical divergence)
```

#### Scoring algorithm

For each observation date of a parcel:

1. Find the **closest reference bin** (within ±21 days tolerance).
2. Compute the cosine distance between the parcel's embedding and the
   reference embedding for that bin.
3. Assign a weight: observations in the **early season** (first 30%) get
   a higher weight (`urgency_weight_early`), because early deviations are
   more agronomically significant.
4. Compute the **weighted average** of all distances → `health_score`.

#### Concrete example — Parcel P999 (corn, possibly stressed)

| Date | Closest ref bin | Cosine distance | Weight | Season phase |
|------|----------------|----------------|--------|-------------|
| 2024-04-12 | 2024-04-07 | 0.008 | 2.0 | Early (< 30%) |
| 2024-04-26 | 2024-04-21 | 0.012 | 2.0 | Early |
| 2024-05-08 | 2024-05-05 | 0.071 | 1.0 | Mid-season |
| 2024-05-22 | 2024-05-19 | 0.380 | 1.0 | Mid-season |
| 2024-06-05 | 2024-06-02 | 0.440 | 1.0 | Mid-season |

```
health_score = (0.008×2 + 0.012×2 + 0.071×1 + 0.380×1 + 0.440×1)
             ÷ (2 + 2 + 1 + 1 + 1)

             = (0.016 + 0.024 + 0.071 + 0.380 + 0.440) / 7

             = 0.931 / 7 ≈ 0.133
```

#### Traffic-light classification

The system supports two methods for thresholding:

**Fixed thresholds** (configured in YAML):

| Health score | Status | Interpretation |
|-------------|--------|---------------|
| < 0.10 | **GREEN** | Behaving like the reference — healthy |
| 0.10 – 0.25 | **YELLOW** | Moderate deviation — monitor closely |
| ≥ 0.25 | **RED** | Significant anomaly — investigate |
| N/A | **GRAY** | Insufficient data to assess |

**Adaptive thresholds** (z-score based):  thresholds are computed from the
score distribution itself using configurable sigma multipliers.  This
adapts to the specific season and region.

For P999 with score 0.133 → **YELLOW**.  The date `2024-05-22`
(distance 0.380) is recorded as `max_deviation_date`.

**Output — `data/output/parcel_scores.parquet`** (+ `.geojson`):

| parcel_id | crop_name | health_score | status | n_observations | max_deviation_date | geometry |
|-----------|-----------|-------------|--------|---------------|-------------------|----------|
| P001 | corn | 0.0234 | GREEN | 8 | — | POLYGON(…) |
| P002 | soybeans | 0.0891 | GREEN | 7 | — | POLYGON(…) |
| P003 | wheat | 0.1456 | YELLOW | 6 | 2024-06-02 | POLYGON(…) |
| P999 | corn | 0.1330 | YELLOW | 5 | 2024-05-22 | POLYGON(…) |
| P042 | corn | 0.3812 | RED | 9 | 2024-05-19 | POLYGON(…) |
| P077 | soybeans | — | GRAY | 1 | — | POLYGON(…) |

---

### Step 7 — Report (`src/report.py`)

**What it does:** aggregates the scored parcels into a campaign-level
summary consumed by the Streamlit dashboard.

**Output — `data/output/campaign_report.json`:**

```json
{
  "total_parcels": 500,
  "status_counts": { "GREEN": 380, "YELLOW": 72, "RED": 28, "GRAY": 20 },
  "status_pct":    { "GREEN": 76.0, "YELLOW": 14.4, "RED": 5.6, "GRAY": 4.0 },
  "crop_breakdown": [
    {
      "crop_name": "corn",
      "total": 250,
      "green": 195,
      "yellow": 35,
      "red": 15,
      "gray": 5,
      "avg_health_score": 0.0712
    },
    {
      "crop_name": "soybeans",
      "total": 180,
      "green": 145,
      "yellow": 25,
      "red": 8,
      "gray": 2,
      "avg_health_score": 0.0634
    }
  ],
  "top10_worst": [
    {
      "parcel_id": "P042",
      "crop_name": "corn",
      "health_score": 0.3812,
      "status": "RED",
      "n_observations": 9,
      "max_deviation_date": "2024-05-19"
    }
  ],
  "temporal_coverage": {
    "mean_observations": 7.3,
    "min_observations": 1,
    "max_observations": 12,
    "median_observations": 8.0
  }
}
```

This JSON powers the Streamlit dashboard's summary cards, per-crop
tables, and worst-parcel alerts.

---

### Step 8 — Index (`src/index.py`)

**What it does:** builds a **FAISS similarity index** over all parcel
seasonal embeddings, enabling fast nearest-neighbour search across
thousands of parcels.  The index is used by the dashboard's **Similar
Parcels** panel and by the `src/query.py` CLI.

#### Why a similarity index?

The scoring step (Step 6) compares each parcel against a **crop
reference profile** — "is this corn parcel behaving like typical corn?"
The index asks a different question: **"which other parcels in the
campaign look most like this one?"**  This enables:

- **Anomaly confirmation:** if a RED parcel's nearest neighbours are
  also RED, the signal is likely real.  If neighbours are all GREEN, the
  alert may be noise.
- **Rotation / mislabel detection:** if a parcel labelled *corn* has
  nearest neighbours that are mostly *soybeans*, the label may be
  wrong or the field may have rotated since the CDL year.
- **Unknown crops:** parcels with no crop label (GRAY) can be
  characterised by what their neighbours are.

#### Algorithm

For each parcel that has at least one dated embedding under
`data/embeddings/<parcel_id>/`:

1. Load all `*.npy` files for that parcel.
2. Compute the **temporal mean** of all dated vectors → one
   768-dimensional seasonal vector per parcel.
3. **L2-normalise** the vector so that inner product equals cosine
   similarity.
4. Collect all normalised vectors into a matrix `(N, 768)`.
5. Build `faiss.IndexFlatIP(768)` (exact inner-product search).
6. Add all vectors in the same row order as the metadata table.

```
data/embeddings/
  P001/
    2024-04-10.npy  ─┐
    2024-04-25.npy   │ mean → v̄  → L2-norm → row i in FAISS
    2024-05-05.npy  ─┘

  P002/
    2024-04-10.npy  ─┐
    2024-04-25.npy   │ mean → v̄  → L2-norm → row i+1 in FAISS
    ...             ─┘
```

**FAISS and cosine similarity:**  IndexFlatIP computes the inner product
between vectors.  When all vectors are unit-length (L2-normalised),
inner product equals cosine similarity:

```
similarity = v̂_query · v̂_neighbor   (both L2-normalised)
           = cos(θ)                  (angle between them)

  1.00 → identical direction (same crop, same condition)
  0.90 → very similar
  0.70 → related but distinct
  0.00 → orthogonal (unrelated)
```

**Cosine distance** is then `1 − similarity`, consistent with Step 6's
health score metric.

#### Fallback — FAISS not installed

If `faiss-cpu` is not available, the step writes
`data/output/parcels_vectors.npz` (the raw normalised matrix) instead
of a `.faiss` file.  The query module detects this automatically and
falls back to brute-force matrix multiplication for search.

**Output:**

| File | Role |
|------|------|
| `data/output/parcels.faiss` | Binary FAISS index (`faiss.write_index`) |
| `data/output/parcels_index_meta.parquet` | Row-aligned metadata — one row per indexed parcel, in FAISS order |
| `data/output/parcels_vectors.npz` | Fallback: normalised vectors when FAISS is unavailable |

**Schema — `parcels_index_meta.parquet`:**

| column | type | description |
|--------|------|-------------|
| `parcel_id` | string | Parcel identifier |
| `crop_name` | string / null | Crop label from `parcels_labeled.parquet` |
| `geometry` | geometry | Parcel polygon (same CRS as `parcel_scores.parquet`) |
| `centroid_lat` | float | Latitude |
| `centroid_lon` | float | Longitude |
| `status` | string | Traffic-light status at time of indexing |
| `faiss_row` | int | Row index in the FAISS index (for debugging) |

FAISS row `i` always corresponds to metadata row `i`.  Parcels with no
embeddings are **excluded** from both files.

**CLI:**

```bash
python -m src.index --config config/default.yaml
# overrides follow the same key=value pattern as other steps:
python -m src.index --config config/default.yaml output.dir=data/output
```

---

### Step 9 — Query (`src/query.py`)

**What it does:** loads the FAISS index and provides two query modes:

1. **Nearest-neighbour lookup** — given a `parcel_id`, return the *k*
   most similar parcels in the campaign.
2. **Crop mislabel / rotation scan** — given a `crop_name`, find
   parcels labelled as that crop whose neighbours are predominantly a
   *different* crop.

#### Nearest-neighbour lookup

```
query parcel P-0042  →  seasonal mean embedding  →  L2-normalise
                                                        │
                                               FAISS.search(k+1)
                                                        │
                                           drop self  ──┤
                                                        │
                        [P-0107  soybeans  sim=0.97  dist=0.03]
                        [P-0831  soybeans  sim=0.96  dist=0.04]
                        [P-1204  corn      sim=0.91  dist=0.09]  ← different crop
                        [P-0558  soybeans  sim=0.89  dist=0.11]
                        ...
```

Each result exposes both **similarity** (inner product, 0–1) and
**cosine distance** (`1 − similarity`), consistent with Step 6's
health score metric.

**CLI:**

```bash
python -m src.query --parcel-id P-0042 --k 10
```

**Example output (JSON):**

```json
[
  { "parcel_id": "P-0107", "crop_name": "soybeans", "similarity": 0.974, "cosine_distance": 0.026 },
  { "parcel_id": "P-0831", "crop_name": "soybeans", "similarity": 0.963, "cosine_distance": 0.037 },
  { "parcel_id": "P-1204", "crop_name": "corn",     "similarity": 0.911, "cosine_distance": 0.089 },
  { "parcel_id": "P-0558", "crop_name": "soybeans", "similarity": 0.892, "cosine_distance": 0.108 }
]
```

#### Crop mislabel / rotation scan

```bash
python -m src.query --crop-name corn --k 10 --min-neighbor-diff-frac 0.6
```

For every parcel labelled *corn*, the scan:

1. Retrieves its *k* nearest neighbours (excluding self).
2. Counts how many neighbours have a **different** crop label (ignoring
   GRAY / unlabelled neighbours).
3. Computes `diff_fraction = different_crop_neighbors / labeled_neighbors`.
4. Returns parcels where `diff_fraction ≥ min_neighbor_diff_frac`,
   sorted descending.

**Interpretation:**

| `diff_fraction` | Interpretation |
|----------------|---------------|
| 1.0 | All *k* neighbours are non-corn — strong mislabel / rotation signal |
| 0.7–0.9 | Probable mislabel or mid-season crop rotation |
| 0.5–0.7 | Borderline — worth manual review |
| < 0.5 | Within normal variation, less suspicious |

**Example output:**

```json
[
  { "parcel_id": "P-0392", "neighbor_diff_fraction": 1.0, "diff_neighbors": 10, "labeled_neighbors": 10 },
  { "parcel_id": "P-1047", "neighbor_diff_fraction": 0.8, "diff_neighbors": 8,  "labeled_neighbors": 10 },
  { "parcel_id": "P-0215", "neighbor_diff_fraction": 0.7, "diff_neighbors": 7,  "labeled_neighbors": 10 }
]
```

#### Output dir resolution

`src/query.py` does **not** require a config file; it defaults to
`data/output/` for index files.  Pass `--config` to derive the path
from a YAML config, or `--output-dir` to point to a custom directory.

---

## Pipeline summary — all outputs

| Step | Module | Key output |
|------|--------|-----------|
| 1 — Ingest | `src/ingest.py` | `parcels_labeled.parquet` |
| 2 — Fetch | `src/fetch.py` | `tile_index.parquet` |
| 3 — Chip | `src/chip.py` | `data/chips/<pid>/<date>.npz` |
| 4 — Embed | `src/embed.py` | `data/embeddings/<pid>/<date>.npy` |
| 5 — Profile | `src/profile.py` | `reference_profiles.pkl` |
| 6 — Score | `src/score.py` | `parcel_scores.parquet`, `.geojson` |
| 7 — Report | `src/report.py` | `campaign_report.json` |
| 8 — Index | `src/index.py` | `parcels.faiss`, `parcels_index_meta.parquet` |
| 9 — Query | `src/query.py` | CLI / importable API (no file output) |
