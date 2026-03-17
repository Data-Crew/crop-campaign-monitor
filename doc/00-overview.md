# Architecture

## System Overview

The Crop Campaign Monitor is composed of two pipelines that share common
infrastructure (config, GPU management, chip extraction, and field ingestion):

1. **Training pipeline** (`train/`): Fine-tunes the Clay Foundation Model
   backbone with CDL crop labels via TerraTorch. Produces an encoder checkpoint
   optimized for crop discrimination.

2. **Monitor pipeline** (`src/`): Uses the encoder to embed Sentinel-2 chips,
   builds per-crop reference profiles, and scores each parcel against its
   expected trajectory.

## Pipeline Flow

### Monitor Pipeline (7 Steps)

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 1. Ingest | `src/ingest.py` | GeoJSON parcels + CDL | `parcels_labeled.parquet` |
| 2. Fetch | `src/fetch.py` | Parcel bounds + STAC | `tile_index.parquet` |
| 3. Chip | `src/chip.py` | COG URLs + parcels | `.npz` chips per parcel/date |
| 4. Embed | `src/embed.py` | `.npz` chips + Clay model | `.npy` embeddings |
| 5. Profile | `src/profile.py` | Embeddings + crop labels | `reference_profiles.pkl` |
| 6. Score | `src/score.py` | Embeddings + profiles | `parcel_scores.parquet/.geojson` |
| 7. Report | `src/report.py` | Scored parcels | `campaign_report.json` |

### Training Pipeline (3 Steps)

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 1. Prepare | `train/prepare_dataset.py` | CDL raster + chips | Manifest CSVs |
| 2. Fine-tune | `train/finetune.py` | Manifests + Clay base | `clay-finetuned-crops.ckpt` |
| 3. Export | `train/export_encoder.py` | Fine-tuned checkpoint | `clay-crop-encoder.ckpt` |

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

## COG Access Pattern

Sentinel-2 imagery is accessed as Cloud-Optimized GeoTIFFs via HTTP range
requests (`vsicurl`). No full tiles are downloaded. GDAL environment variables
are configured to enable efficient partial reads:

- `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`
- `CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.TIF`
- `VSI_CACHE=TRUE`
