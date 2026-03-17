"""Step 5 — Build reference embedding profiles per crop type.

Groups parcels by crop type and computes median embedding trajectories across
the season.  Parcels with too few observations or belonging to rare crop
classes are excluded.  Outputs ``data/output/reference_profiles.pkl``.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import get_config

log = logging.getLogger(__name__)

MIN_PARCELS_DEFAULT = 10


def _biweekly_bins(start_date: str, end_date: str) -> list[tuple[datetime, datetime]]:
    """Generate biweekly (14-day) time bins spanning the season."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    bins: list[tuple[datetime, datetime]] = []
    current = start
    while current < end:
        bin_end = min(current + timedelta(days=14), end)
        bins.append((current, bin_end))
        current = bin_end
    return bins


def _bin_center(bin_start: datetime, bin_end: datetime) -> str:
    """Return the midpoint date of a bin as ISO string."""
    mid = bin_start + (bin_end - bin_start) / 2
    return mid.strftime("%Y-%m-%d")


def profile(cfg: dict) -> dict[str, Any]:
    """Build reference profiles per crop type.

    Returns a dict mapping crop names to lists of ``(bin_center, median_embedding)``
    tuples.
    """
    project_root = Path(cfg["_meta"]["project_root"])
    embeddings_dir = project_root / "data" / "embeddings"
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    parcels_path = output_dir / "parcels_labeled.parquet"
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels not found: {parcels_path} — run ingest first")
    parcels = gpd.read_parquet(parcels_path)

    season = cfg["season"]
    bins = _biweekly_bins(season["start_date"], season["end_date"])
    scoring_cfg = cfg.get("scoring", {})
    min_parcels = scoring_cfg.get("min_parcels_for_profile", MIN_PARCELS_DEFAULT)

    crop_groups: dict[str, list[str]] = defaultdict(list)
    for _, row in parcels.iterrows():
        cn = row.get("crop_name")
        if pd.notna(cn) and cn:
            crop_groups[str(cn)].append(str(row["parcel_id"]))

    log.info("Crop groups: %s", {k: len(v) for k, v in crop_groups.items()})

    profiles: dict[str, list[tuple[str, np.ndarray]]] = {}

    for crop_name, parcel_ids in tqdm(crop_groups.items(), desc="Building profiles"):
        if len(parcel_ids) < min_parcels:
            log.info(
                "Skipping '%s' — only %d parcels (min %d)",
                crop_name,
                len(parcel_ids),
                min_parcels,
            )
            continue

        bin_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)

        for pid in parcel_ids:
            emb_dir = embeddings_dir / pid
            if not emb_dir.exists():
                continue
            for npy_file in emb_dir.glob("*.npy"):
                date_str = npy_file.stem
                try:
                    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue

                for bi, (bstart, bend) in enumerate(bins):
                    if bstart <= date_dt < bend:
                        emb = np.load(npy_file)
                        bin_embeddings[bi].append(emb)
                        break

        profile_entries: list[tuple[str, np.ndarray]] = []
        for bi, (bstart, bend) in enumerate(bins):
            vecs = bin_embeddings.get(bi, [])
            if len(vecs) < 2:
                continue
            median_emb = np.median(np.stack(vecs), axis=0)
            profile_entries.append((_bin_center(bstart, bend), median_emb))

        if profile_entries:
            profiles[crop_name] = profile_entries
            log.info(
                "Profile '%s': %d parcels, %d/%d bins with data",
                crop_name,
                len(parcel_ids),
                len(profile_entries),
                len(bins),
            )

    out_path = output_dir / "reference_profiles.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(profiles, f)
    log.info("Saved %d reference profiles to %s", len(profiles), out_path)

    return profiles


@click.command("profile")
@click.option("--config", "config_path", default="config/monitor.yaml")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Step 5: Build reference profiles per crop type."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = get_config(config_path, list(overrides) if overrides else None)
    profile(cfg)


if __name__ == "__main__":
    cli()
