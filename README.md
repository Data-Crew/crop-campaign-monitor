# Crop Campaign Monitor

A geospatial AI tool for precision agriculture that monitors crop field health
across a growing season. It processes Sentinel-2 satellite imagery through the
[Clay Foundation Model](https://github.com/Clay-foundation/model) to produce
per-parcel traffic-light scores (GREEN / YELLOW / RED) indicating whether each
field is developing normally or showing anomalies.

## Architecture — Three Phases

The system is organized into three sequential phases with explicit
dependencies:

```
 Phase 1: Data Preparation          Phase 2: Training (optional)
 ─────────────────────────          ────────────────────────────
 GeoJSON parcels                    parcels_labeled.parquet
      │                                  │
   ingest  ── assign crop labels    prepare_dataset ── map chips → labels
      │                                  │
    fetch   ── Sentinel-2 STAC       finetune ── Clay + TerraTorch
      │                                  │
    chip    ── extract 224×224 px     export_encoder
      │                                  │
 data/chips/                        clay-crop-encoder.ckpt ──┐
      │                                                      │
      └──────────────────────────────────────────────────────┘
                                                             │
                              Phase 3: Monitor               │
                              ────────────────               │
                              embed  ◄───────────────────────┘
                                │         (uses encoder)
                             profile
                                │
                              score
                                │
                              report ──► parcel_scores.geojson
                                              │
                                        ┌─────▼──────┐
                                        │  Streamlit  │
                                        │  Dashboard  │
                                        └────────────┘
```

**Phase 1** generates labeled parcels and satellite chips — shared input for
both training and monitoring.

**Phase 2** fine-tunes the Clay encoder with crop-specific supervision. This
is optional: the monitor can run with mock embeddings (with a warning) if no
trained encoder exists.

**Phase 3** generates real embeddings, builds reference profiles per crop type,
and scores each parcel against its expected trajectory.

## Quick Start

```bash
cd crop-campaign-monitor
docker compose up -d
docker exec -d <container> streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

Then open `http://localhost:8501`. The sidebar guides you through the three
phases with status indicators and run buttons.

### From the command line

```bash
# Phase 1 — Data Preparation (ingest + fetch + chip)
docker exec <container> bash scripts/run_data_prep.sh

# Phase 2 — Training (prepare + finetune + export)
docker exec <container> bash scripts/run_training.sh

# Phase 3 — Monitor (embed + profile + score + report)
docker exec <container> bash scripts/run_monitor.sh

# Or run all three in sequence:
docker exec <container> bash scripts/run_pipeline.sh
```

## Crop Label Sources (CDL)

The system needs to know what crop each parcel contains. There are **three
independent sources** — they do not fall back to each other:

| Source | Config | What it does | When to use |
|--------|--------|-------------|-------------|
| `embedded` | `cdl.source: "embedded"`, `cdl.year: 2022` | Reads a column `crops_2022` from your GeoJSON properties | You pre-labeled the parcels externally (e.g. zonal majority in GIS) |
| `usda` | `cdl.source: "usda"`, `cdl.year: 2024` | Downloads the CDL `.tif` from CropScape, runs zonal majority | You want fresh labels for a year your GeoJSONs don't cover |
| `local` | `cdl.source: "local"`, `cdl.path: "..."` | Reads a CSV with `parcel_id, crop_code, crop_name` | You have labels in a separate file |

### Which config controls what

- **`config/monitor.yaml` → `cdl` section**: Controls how Phase 1 (ingest)
  labels parcels. For parcels with pre-computed labels in GeoJSON, use
  `source: "embedded"`.

- **`config/train.yaml` → `data.cdl_source` / `data.cdl_year`**: Controls how
  Phase 2 (training) labels parcels. If set to `"usda"`, it re-labels from
  the CDL raster for the specified year, overriding whatever labels came
  from ingest. If set to `"inherit"`, it uses the labels from Phase 1 as-is.

### Example: your GeoJSONs have 2022 labels, but you want to train with 2024

```yaml
# monitor.yaml — Phase 1 uses embedded labels
cdl:
  source: "embedded"
  year: 2022

# train.yaml — Phase 2 downloads CDL 2024 and re-labels
data:
  cdl_source: "usda"
  cdl_year: 2024
```

The CDL `.tif` is cached in `data/cdl/cdl_<year>.tif`. If the file already
exists, it is not re-downloaded.

## GPU Selection

GPUs are selected **by name** (not by index) to avoid issues when device
ordering changes between reboots. The resolver matches a substring
case-insensitively.

```yaml
# monitor.yaml
gpu:
  default_device: "RTX 4070"
  smoke_test_device: "RTX 500"

# train.yaml
gpu:
  device: "RTX 4070"
```

From the command line, pass `GPU_NAME`:

```bash
GPU_NAME="RTX 4070" bash scripts/run_data_prep.sh
```

The Streamlit dashboard discovers GPUs at runtime and presents them by name
and VRAM.

## Growing Season

The growing season is defined in `config/monitor.yaml`:

```yaml
season:
  start_date: "2024-04-15"
  end_date: "2024-10-31"
  cadence_days: 16
```

This controls which Sentinel-2 scenes are fetched (Phase 1, step 2). Only
images within this date range are searched. `cadence_days` controls temporal
binning for reference profiles — embeddings are grouped into 16-day windows
to build the "expected" trajectory for each crop.

## Data Layout

```
data/
├── fields/              # Input: GeoJSON parcels
│   └── 16tgk/           #   One directory per tile
│       ├── 16tgk_fragment_00_00.geojson
│       ├── 16tgk_fragment_01_00.geojson
│       └── ...
├── cdl/                 # CDL rasters (auto-downloaded)
│   └── cdl_2024.tif
├── tiles/               # STAC tile index (auto-generated)
├── chips/               # Extracted image chips (auto-generated)
│   └── <parcel_id>/
│       └── YYYY-MM-DD.npz
├── embeddings/          # Clay embeddings (auto-generated)
│   └── <parcel_id>/
│       └── YYYY-MM-DD.npy
├── training/            # Training manifests (auto-generated)
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   ├── test_manifest.csv
│   └── class_mapping.json
├── model/               # Model checkpoints
│   ├── clay-v1-base.ckpt         # Base Clay (downloaded on first run)
│   ├── clay-finetuned-crops.ckpt # Fine-tuned checkpoint
│   └── clay-crop-encoder.ckpt   # Exported encoder for inference
└── output/              # Pipeline results
    ├── parcels_labeled.parquet
    ├── parcel_scores.parquet
    ├── parcel_scores.geojson
    ├── reference_profiles.pkl
    └── campaign_report.json
```

## Embedding Cache

The embed step caches `.npy` files in `data/embeddings/`. If the encoder
checkpoint changes (e.g. after re-training), stale embeddings are
**automatically invalidated** by comparing the encoder's modification time
against a stamp file. You can also clear the cache manually from the
Streamlit sidebar.

## Adding Your Own Region

1. Place GeoJSON files in `data/fields/<tile_id>/` named
   `<tile_id>_fragment_XX_YY.geojson`
2. Update `config/monitor.yaml`:
   - `region.tiles`: your tile ID(s)
   - `region.fragments`: fragment IDs to monitor (or `"all"`)
   - `season.start_date` / `season.end_date`: your growing season
   - `cdl.source` / `cdl.year`: how to label your parcels
3. Run Phase 1, then Phase 2 (optional), then Phase 3

## Running Individual Steps

```bash
# Phase 1: Data Preparation
bash scripts/run_step.sh ingest
bash scripts/run_step.sh fetch
bash scripts/run_step.sh chip

# Phase 2: Training
bash scripts/run_step.sh prepare config/train.yaml
bash scripts/run_step.sh finetune config/train.yaml
bash scripts/run_step.sh export  config/train.yaml

# Phase 3: Monitor
bash scripts/run_step.sh embed
bash scripts/run_step.sh profile
bash scripts/run_step.sh score
bash scripts/run_step.sh report
```
