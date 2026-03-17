"""Step 3 — Extract image chips from Sentinel-2 COGs.

For each parcel and scene date, reads a 224x224 pixel window (centred on the
parcel centroid) directly from Cloud-Optimized GeoTIFFs via ``vsicurl`` —
no full tile downloads.  Saves chips as ``.npz`` files under
``data/chips/<parcel_id>/<YYYY-MM-DD>.npz``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
from tqdm import tqdm

from src.config import get_config

log = logging.getLogger(__name__)

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
}


def _chip_window(
    centroid_lon: float,
    centroid_lat: float,
    size_px: int,
    resolution_m: float,
) -> tuple[float, float, float, float]:
    """Compute geographic bounds for a chip centred on the given point.

    Returns (west, south, east, north) in EPSG:4326 degrees.  Uses an
    approximate meters-to-degrees conversion at the given latitude.
    """
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(centroid_lat))
    half_extent_m = (size_px * resolution_m) / 2.0
    dlon = half_extent_m / m_per_deg_lon
    dlat = half_extent_m / m_per_deg_lat
    return (
        centroid_lon - dlon,
        centroid_lat - dlat,
        centroid_lon + dlon,
        centroid_lat + dlat,
    )


def _read_chip(
    asset_urls: dict[str, str],
    bands: list[str],
    chip_bounds: tuple[float, float, float, float],
    size_px: int,
) -> np.ndarray | None:
    """Read a multi-band chip from COG URLs.  Returns (C, H, W) or None on failure."""
    arrays: list[np.ndarray] = []
    west, south, east, north = chip_bounds

    for band in bands:
        url = asset_urls.get(band)
        if not url:
            return None
        try:
            with rasterio.Env(**GDAL_ENV):
                with rasterio.open(url) as src:
                    src_bounds = transform_bounds("EPSG:4326", src.crs, west, south, east, north)
                    win = window_from_bounds(*src_bounds, transform=src.transform)
                    data = src.read(
                        1,
                        window=win,
                        out_shape=(size_px, size_px),
                        resampling=rasterio.enums.Resampling.bilinear,
                    )
                    arrays.append(data)
        except Exception as exc:
            log.debug("Failed reading %s band %s: %s", url, band, exc)
            return None

    return np.stack(arrays, axis=0)  # (C, H, W)


def _nodata_fraction(chip: np.ndarray) -> float:
    """Fraction of pixels that are zero/nodata across all bands."""
    all_zero = np.all(chip == 0, axis=0)
    return float(all_zero.sum()) / (chip.shape[1] * chip.shape[2])


def chip(cfg: dict) -> dict[str, int]:
    """Extract chips for all parcels across all available scene dates.

    Returns a summary dict with counts of chips extracted / skipped.
    """
    chips_cfg = cfg["chips"]
    size_px = chips_cfg["size_px"]
    resolution_m = chips_cfg["resolution_m"]
    nodata_threshold = chips_cfg["nodata_threshold"]
    bands = cfg["stac"]["bands"]

    project_root = Path(cfg["_meta"]["project_root"])
    chips_dir = project_root / "data" / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)

    parcels_path = Path(cfg["output"]["dir"]) / "parcels_labeled.parquet"
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels not found: {parcels_path} — run ingest first")
    parcels = gpd.read_parquet(parcels_path)

    tile_index_path = project_root / "data" / "tiles" / "tile_index.parquet"
    if not tile_index_path.exists():
        raise FileNotFoundError(f"Tile index not found: {tile_index_path} — run fetch first")
    scenes = pd.read_parquet(tile_index_path)

    is_smoke = os.environ.get("SMOKE_TEST", "0") == "1"
    if is_smoke:
        parcels = parcels.head(5).copy()
        scenes = scenes.head(2).copy()
        log.info("SMOKE_TEST: %d parcels, %d scenes", len(parcels), len(scenes))

    max_parcels = chips_cfg.get("max_parcels")
    if max_parcels and len(parcels) > max_parcels:
        parcels = parcels.head(max_parcels).copy()
        log.info("max_parcels: limited to %d parcels for chip extraction", len(parcels))

    stats = {"extracted": 0, "skipped_nodata": 0, "skipped_error": 0}

    total = len(parcels) * len(scenes)
    with tqdm(total=total, desc="Extracting chips") as pbar:
        for _, parcel in parcels.iterrows():
            pid = parcel["parcel_id"]
            lat, lon = parcel["centroid_lat"], parcel["centroid_lon"]
            parcel_dir = chips_dir / str(pid)
            parcel_dir.mkdir(parents=True, exist_ok=True)

            chip_bounds = _chip_window(lon, lat, size_px, resolution_m)

            for _, scene in scenes.iterrows():
                date_str = scene["date"]
                out_path = parcel_dir / f"{date_str}.npz"
                pbar.update(1)

                if out_path.exists():
                    stats["extracted"] += 1
                    continue

                asset_urls = json.loads(scene["asset_urls"])
                pixels = _read_chip(asset_urls, bands, chip_bounds, size_px)
                if pixels is None:
                    stats["skipped_error"] += 1
                    continue

                ndf = _nodata_fraction(pixels)
                if ndf > nodata_threshold:
                    stats["skipped_nodata"] += 1
                    continue

                np.savez_compressed(
                    out_path,
                    pixels=pixels,
                    parcel_id=pid,
                    date=date_str,
                    lat=lat,
                    lon=lon,
                    gsd=resolution_m,
                    bands=bands,
                )
                stats["extracted"] += 1

    log.info(
        "Chip extraction: %d extracted, %d skipped (nodata), %d skipped (error)",
        stats["extracted"],
        stats["skipped_nodata"],
        stats["skipped_error"],
    )
    return stats


@click.command("chip")
@click.option("--config", "config_path", default="config/monitor.yaml")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Step 3: Extract 224x224 chips from Sentinel-2 COGs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = get_config(config_path, list(overrides) if overrides else None)
    chip(cfg)


if __name__ == "__main__":
    cli()
