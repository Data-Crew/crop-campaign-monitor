# Crop Campaign Monitor

A geospatial AI tool for precision agriculture that monitors crop field health
across a growing season. It processes Sentinel-2 satellite imagery through the
[Clay Foundation Model](https://github.com/Clay-foundation/model) to produce
per-parcel traffic-light scores (GREEN / YELLOW / RED) indicating whether each
field is developing normally or showing anomalies.

## Documentation Map

- [doc/00-overview.md](doc/00-overview.md) — phase architecture, pipeline tables, and phase dependency rules
- [doc/01-workflow.md](doc/01-workflow.md) — step-by-step workflow with inputs, outputs, and algorithms
- [doc/02-architecture.md](doc/02-architecture.md) — Clay Foundation Model internals and fine-tuning deep-dive
- [doc/03-runtime-stack.md](doc/03-runtime-stack.md) — Docker Compose setup, GPU passthrough, environment variables

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
                                        │  Streamlit │
                                        │  Dashboard │
                                        └────────────┘
```

**Phase 1** generates labeled parcels and satellite chips — shared input for
both training and monitoring. It is a required bootstrap stage and must run at
least once before either Phase 2 or Phase 3.

**Phase 2** fine-tunes the Clay encoder with crop-specific supervision. This
is optional: the monitor can run with mock embeddings (with a warning) if no
trained encoder exists.

**Phase 3** generates real embeddings, builds reference profiles per crop type,
and scores each parcel against its expected trajectory. It requires Phase 1
outputs and never runs without them.

See [doc/00-overview.md](doc/00-overview.md) for the full phase dependency
contract.

## Quick Start

```bash
cd crop-campaign-monitor

# Start the persistent workspace and dashboard
docker compose up -d

# Open the dashboard in your browser
# http://localhost:8501
```

The Streamlit dashboard starts automatically on port 8501. The sidebar guides
you through the three phases with status indicators and run buttons.

See [doc/03-runtime-stack.md](doc/03-runtime-stack.md) for the full runtime
model and GPU setup details.

## Execution

### Open a shell

```bash
docker compose exec workspace bash
```

### Run pipeline phases

Each phase runs as a one-shot job and exits when complete:

```bash
# Phase 1 — Data Preparation (required first)
docker compose run --rm data-prep

# Phase 2 — Training (optional)
docker compose run --rm training

# Phase 3 — Monitor
docker compose run --rm monitor

# All three phases in sequence
docker compose run --rm pipeline
```

### Run individual steps

Inside the workspace shell, or via exec:

```bash
# Phase 1: Data Preparation
docker compose exec workspace bash scripts/run_step.sh ingest
docker compose exec workspace bash scripts/run_step.sh fetch
docker compose exec workspace bash scripts/run_step.sh chip

# Phase 2: Training
docker compose exec workspace bash scripts/run_step.sh prepare config/train.yaml
docker compose exec workspace bash scripts/run_step.sh finetune config/train.yaml
docker compose exec workspace bash scripts/run_step.sh export config/train.yaml

# Phase 3: Monitor
docker compose exec workspace bash scripts/run_step.sh embed
docker compose exec workspace bash scripts/run_step.sh profile
docker compose exec workspace bash scripts/run_step.sh score
docker compose exec workspace bash scripts/run_step.sh report
```

## Configuration

All pipeline behavior is driven by YAML files in `config/`. Parameters
defined there control which region and season are processed, how parcels are
labeled, how the model is loaded, and how scores are thresholded.

### Config files

| File | Controls |
|------|----------|
| `config/monitor.yaml` | Phases 1 and 3: region, season, CDL labeling, chip extraction, model loading, scoring, GPU, LLM explanation, output. Default config for all Phase 1 and Phase 3 modules. |
| `config/train.yaml` | Phase 2: base model, CDL labeling for training, dataset splits, training hyperparameters, output paths, GPU |
| `config/default.yaml` | Identical content to `monitor.yaml`. Used only as the default config path by `src/index.py`. Pass `--config config/monitor.yaml` explicitly to `src.index` to avoid relying on this file. |

### Crop Label Sources (CDL)

The system needs to know what crop each parcel contains. There are **three
independent sources** — they do not fall back to each other:

| Source | Config | What it does | When to use |
|--------|--------|-------------|-------------|
| `embedded` | `cdl.source: "embedded"`, `cdl.year: 2022` | Reads a column `crops_2022` from your GeoJSON properties | You pre-labeled the parcels externally (e.g. zonal majority in GIS) |
| `usda` | `cdl.source: "usda"`, `cdl.year: 2024` | Downloads the CDL `.tif` from CropScape, runs zonal majority | You want fresh labels for a year your GeoJSONs don't cover |
| `local` | `cdl.source: "local"`, `cdl.path: "..."` | Reads a CSV with `parcel_id, crop_code, crop_name` | You have labels in a separate file |

**`config/monitor.yaml` → `cdl` section**: Controls how Phase 1 (ingest)
labels parcels. For parcels with pre-computed labels in GeoJSON, use
`source: "embedded"`.

**`config/train.yaml` → `data.cdl_source` / `data.cdl_year`**: Controls how
Phase 2 (training) labels parcels. If set to `"usda"`, it re-labels from
the CDL raster for the specified year, overriding whatever labels came
from ingest. If set to `"inherit"`, it uses the labels from Phase 1 as-is.

**Example: your GeoJSONs have 2022 labels, but you want to train with 2024**

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

### Growing Season

Defined in `config/monitor.yaml` under the `season` key:

```yaml
season:
  start_date: "2024-04-15"
  end_date: "2024-10-31"
  cadence_days: 16
```

`start_date` and `end_date` control which Sentinel-2 scenes are fetched
(Phase 1, step 2). Only images within this date range are searched.
`cadence_days` controls temporal binning for reference profiles — embeddings
are grouped into 16-day windows to build the expected trajectory for each
crop type.

### GPU Selection

GPUs are selected by name (not by index) to avoid ordering issues between
reboots. Set `gpu.default_device` in `config/monitor.yaml` or
`gpu.device` in `config/train.yaml`, or pass `GPU_NAME` on the command line:

```bash
GPU_NAME="RTX 4070" bash scripts/run_data_prep.sh
```

See [doc/03-runtime-stack.md](doc/03-runtime-stack.md) for Docker GPU
passthrough setup and the full name-matching explanation.

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

### Embedding Cache

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
3. Run Phase 1 (required), then optionally Phase 2, then Phase 3

## First-Run Semantics

- **Phase 1 must run first** for any new region or season. It produces the
  chips and labeled parcels that all downstream phases depend on.
- **Phase 2 is optional.** If skipped, Phase 3 falls back to the pre-trained
  Clay base encoder, or to a mock encoder if no weights are available (logged
  as a WARNING).
- **Phase 3 requires Phase 1 outputs.** It will fail or produce no results if
  chips and `parcels_labeled.parquet` do not exist.
- **Changing region, season dates, or CDL source** invalidates Phase 1
  outputs. Re-run Phase 1 to regenerate chips and labels, then re-run any
  downstream phases that depend on them.
- **Changing the encoder** (by re-running Phase 2) invalidates the embedding
  cache. The embed step detects this automatically via a modification-time
  stamp and regenerates affected embeddings.
