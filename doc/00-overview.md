# Architecture

## System Overview

The Crop Campaign Monitor is organized into **three phases** that share common
infrastructure (config, GPU management, chip extraction, and field ingestion):

1. **Phase 1 — Data Preparation** (`src/ingest.py`, `src/fetch.py`,
   `src/chip.py`): Ingests parcel geometries, labels them with CDL crop codes,
   fetches Sentinel-2 imagery metadata, and extracts 224 × 224 px chips. This
   phase produces the shared inputs consumed by both Phase 2 and Phase 3.

2. **Phase 2 — Training** (`train/`): Fine-tunes the Clay Foundation Model
   backbone with CDL crop labels via TerraTorch. Produces an encoder checkpoint
   optimized for crop discrimination. This phase is **optional**.

3. **Phase 3 — Monitor** (`src/embed.py` … `src/report.py`): Uses the encoder
   to embed Sentinel-2 chips, builds per-crop reference profiles, and scores
   each parcel against its expected trajectory. An optional **LLM explainer**
   step (`src/explain.py`) can generate JSON narratives; an optional **Geo-RAG
   retrieval layer** (`src/retrieve.py`) may enrich those prompts with local
   FAISS and geographic context without moving anomaly detection into the LLM
   (see [doc/llm_explainer.md](llm_explainer.md)).

For step-by-step execution details see [01-workflow.md](01-workflow.md).
For Docker and GPU setup see [doc/03-runtime-stack.md](03-runtime-stack.md).

---

## Phase Dependencies

| Rule | Detail |
|------|--------|
| Phase 1 is a required bootstrap stage | It must run at least once for every region before either Phase 2 or Phase 3 can execute. |
| Phase 2 is optional | The monitor can run without a fine-tuned encoder. `src/embed.py` falls back to the pre-trained Clay base, or to a mock encoder if no weights exist (logged as WARNING). |
| Phase 3 requires Phase 1 outputs | `parcels_labeled.parquet` and `data/chips/` must exist. Phase 3 never runs without them. |
| Both Phase 2 and Phase 3 depend on Phase 1 | Phase 2 consumes `parcels_labeled.parquet` and `.npz` chips. Phase 3 consumes the same chips plus, optionally, the encoder from Phase 2. |
| Region / season / CDL changes invalidate Phase 1 outputs | If `region.tiles`, `season.start_date`, `season.end_date`, or `cdl.source` change in a way that produces different chips or labels, Phase 1 must be re-run before downstream phases. |

---

## Pipeline Steps

### Phase 1 — Data Preparation

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 1. Ingest | `src/ingest.py` | GeoJSON parcels + CDL | `parcels_labeled.parquet` |
| 2. Fetch | `src/fetch.py` | Parcel bounds + STAC | `tile_index.parquet` |
| 3. Chip | `src/chip.py` | COG URLs + parcels | `.npz` chips per parcel/date |

### Phase 2 — Training (optional)

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 1. Prepare | `train/prepare_dataset.py` | CDL raster + chips | Manifest CSVs |
| 2. Fine-tune | `train/finetune.py` | Manifests + Clay base | `clay-finetuned-crops.ckpt` |
| 3. Export | `train/export_encoder.py` | Fine-tuned checkpoint | `clay-crop-encoder.ckpt` |

### Phase 3 — Monitor

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 4. Embed | `src/embed.py` | `.npz` chips + Clay model | `.npy` embeddings |
| 5. Profile | `src/profile.py` | Embeddings + crop labels | `reference_profiles.pkl` |
| 6. Score | `src/score.py` | Embeddings + profiles | `parcel_scores.parquet/.geojson` |
| 7. Report | `src/report.py` | Scored parcels | `campaign_report.json` |

---

## Data Formats

### Chip (.npz)

```
pixels:    ndarray (C, 224, 224) — reflectance values (uint16)
parcel_id: str
date:      str "YYYY-MM-DD"
lat:       float — centroid latitude
lon:       float — centroid longitude
gsd:       float — ground sampling distance (10m)
bands:     list[str] — band names ["B02", "B03", ...]
```

### Embedding (.npy)

768-dimensional float32 vector per chip.

### Reference Profile (.pkl)

```python
{
    "corn": [
        ("2024-04-22", ndarray_768),
        ("2024-05-06", ndarray_768),
        ...
    ],
    "soybeans": [...],
}
```

### Parcel Scores (GeoJSON/Parquet)

| Column | Type | Description |
|--------|------|-------------|
| parcel_id | str | Unique parcel identifier |
| crop_name | str | CDL crop label |
| health_score | float | 0.0 (healthy) to 1.0 (severe deviation) |
| status | str | GREEN / YELLOW / RED / GRAY |
| n_observations | int | Cloud-free dates available |
| max_deviation_date | str | Date of worst deviation |
| distance_trajectory | str (JSON) | Per-observation cosine distances |
| geometry | Polygon | Parcel boundary |

---

## Model Loading Cascade

`src/embed.py` tries three sources in order:

1. **Fine-tuned encoder** (`clay-crop-encoder.ckpt`): Backbone trained with
   CDL supervision. Produces embeddings optimized for crop discrimination.
   Loaded directly as a Clay `Encoder` with `mask_ratio=0`.

2. **Pre-trained base** (`clay-v1-base.ckpt`): Generic Clay MAE checkpoint.
   Loaded via `ClayMAEModule`, encoder extracted. Works but embeddings are
   not crop-specific.

3. **Mock encoder**: Deterministic random 768-dim vectors seeded by parcel_id.
   Allows full pipeline testing without any model weights. Logs a WARNING.

---

## Scoring Algorithm

For each parcel:

1. Load its time series of embeddings.
2. Load the reference profile for its crop type.
3. Align observations to the nearest reference time bin (within 21 days).
4. Compute cosine distance at each aligned point.
5. Compute weighted mean distance:
   - Early-season observations (first 30% of season) receive `urgency_weight_early`
     multiplier (default 1.5x).
   - Late-season observations receive weight 1.0.
6. Classify by thresholds:
   - `health_score < 0.15` → GREEN
   - `0.15 <= health_score < 0.30` → YELLOW
   - `health_score >= 0.30` → RED
7. Edge cases → GRAY:
   - Fewer than `min_observations` cloud-free dates
   - Crop type has no reference profile (too few parcels)
   - No CDL label assigned

---

## COG Access Pattern

Sentinel-2 imagery is accessed as Cloud-Optimized GeoTIFFs via HTTP range
requests (`vsicurl`). No full tiles are downloaded. GDAL environment variables
are configured to enable efficient partial reads:

- `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`
- `CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.TIF`
- `VSI_CACHE=TRUE`
