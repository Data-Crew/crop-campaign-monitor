"""Prepare the training dataset from pre-labeled parcels and pre-extracted chips.

Reads ``parcels_labeled.parquet`` (produced by Phase 1 — Data Preparation),
maps chips to crop labels, filters rare classes, and produces stratified
train/val/test manifest CSVs in ``data/training/``.

If the training config specifies ``cdl_source: usda`` with a different
``cdl_year``, labels are re-assigned from the CDL raster (overriding the
labels from ingest).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import get_config

log = logging.getLogger(__name__)

PARCELS_LABELED_PATH = "data/output/parcels_labeled.parquet"


def _relabel_from_cdl(
    parcels: gpd.GeoDataFrame,
    cdl_year: int,
    project_root: str,
) -> gpd.GeoDataFrame:
    """Re-assign crop labels from USDA CDL for a specific year."""
    from rasterstats import zonal_stats

    from src.ingest import CDL_CROP_NAMES, _download_cdl_raster

    bbox = tuple(parcels.to_crs(epsg=4326).total_bounds)
    cdl_dir = Path(project_root) / "data" / "cdl"
    cdl_path = cdl_dir / f"cdl_{cdl_year}.tif"

    _download_cdl_raster(cdl_year, bbox, cdl_path)

    stats = zonal_stats(
        parcels.to_crs(epsg=5070),
        str(cdl_path),
        stats=[],
        categorical=True,
        all_touched=True,
    )

    codes: list[int | None] = []
    for s in stats:
        if s:
            codes.append(int(max(s, key=s.get)))
        else:
            codes.append(None)

    parcels["crop_code"] = codes
    parcels["crop_name"] = parcels["crop_code"].map(
        lambda c: CDL_CROP_NAMES.get(c, "unknown") if pd.notna(c) else None
    )
    return parcels


def _map_chips_to_labels(
    parcels: gpd.GeoDataFrame,
    chips_dir: Path,
    target_classes: list[str],
    class_mapping: dict[str, int],
) -> pd.DataFrame:
    """Build a flat table of (chip_path, crop_class_id, crop_name, parcel_id, date)."""
    records: list[dict[str, Any]] = []

    for _, row in tqdm(parcels.iterrows(), total=len(parcels), desc="Mapping chips"):
        pid = str(row["parcel_id"])
        crop = row.get("crop_name")
        if pd.isna(crop) or crop not in target_classes:
            continue

        parcel_chip_dir = chips_dir / pid
        if not parcel_chip_dir.exists():
            continue

        class_id = class_mapping.get(str(crop))
        if class_id is None:
            continue

        for npz_file in sorted(parcel_chip_dir.glob("*.npz")):
            records.append(
                {
                    "chip_path": str(npz_file),
                    "crop_class_id": class_id,
                    "crop_name": crop,
                    "parcel_id": pid,
                    "date": npz_file.stem,
                }
            )

    return pd.DataFrame(records)


def prepare_dataset(cfg: dict) -> dict[str, int]:
    """Prepare the full training dataset from labeled parcels + chips."""
    data_cfg = cfg["data"]
    project_root = cfg["_meta"]["project_root"]
    dataset_dir = Path(data_cfg["dataset_dir"])
    dataset_dir.mkdir(parents=True, exist_ok=True)

    chips_dir = Path(data_cfg["chips_dir"])
    target_classes = data_cfg["target_classes"]
    min_per_class = data_cfg["min_parcels_per_class"]

    parcel_path = Path(project_root) / PARCELS_LABELED_PATH
    if not parcel_path.exists():
        raise FileNotFoundError(
            f"Labeled parcels not found at {parcel_path}. "
            "Run Phase 1 (Data Preparation) first: bash scripts/run_data_prep.sh"
        )
    parcels = gpd.read_parquet(parcel_path)
    log.info("Loaded %d labeled parcels from %s", len(parcels), parcel_path)

    cdl_source = data_cfg.get("cdl_source", "inherit")
    if cdl_source == "usda":
        cdl_year = data_cfg["cdl_year"]
        log.info("Re-labeling parcels from USDA CDL year %d", cdl_year)
        parcels = _relabel_from_cdl(parcels, cdl_year, project_root)
    elif cdl_source != "inherit":
        log.info("Using labels from parcels_labeled.parquet (cdl_source=%s)", cdl_source)

    crop_counts = parcels["crop_name"].value_counts()
    valid_crops = [c for c in target_classes if crop_counts.get(c, 0) >= min_per_class]
    dropped = [c for c in target_classes if c not in valid_crops]
    if dropped:
        log.warning("Dropped classes with < %d parcels: %s", min_per_class, dropped)

    class_mapping = {crop: idx for idx, crop in enumerate(valid_crops)}

    all_samples = _map_chips_to_labels(parcels, chips_dir, valid_crops, class_mapping)
    if all_samples.empty:
        raise RuntimeError(
            "No training samples found — ensure chips exist in data/chips/. "
            "Run Phase 1 (Data Preparation) first: bash scripts/run_data_prep.sh"
        )

    log.info("Total samples: %d across %d classes", len(all_samples), len(valid_crops))
    log.info("Class distribution:\n%s", all_samples["crop_name"].value_counts().to_string())

    train_val, test = train_test_split(
        all_samples,
        test_size=data_cfg["test_split"],
        stratify=all_samples["crop_class_id"],
        random_state=42,
    )
    relative_val = data_cfg["val_split"] / (data_cfg["train_split"] + data_cfg["val_split"])
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val["crop_class_id"],
        random_state=42,
    )

    train.to_csv(dataset_dir / "train_manifest.csv", index=False)
    val.to_csv(dataset_dir / "val_manifest.csv", index=False)
    test.to_csv(dataset_dir / "test_manifest.csv", index=False)

    with open(dataset_dir / "class_mapping.json", "w") as f:
        json.dump({str(v): k for k, v in class_mapping.items()}, f, indent=2)

    log.info("Splits: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    log.info("Saved manifests and class_mapping.json to %s", dataset_dir)

    return {"train": len(train), "val": len(val), "test": len(test)}


@click.command("prepare_dataset")
@click.option("--config", "config_path", default="config/train.yaml")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Prepare CDL-labeled training dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = get_config(config_path, list(overrides) if overrides else None)
    prepare_dataset(cfg)


if __name__ == "__main__":
    cli()
