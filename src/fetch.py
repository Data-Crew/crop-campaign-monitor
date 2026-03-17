"""Step 2 — Fetch Sentinel-2 scene metadata via STAC API.

Searches the Earth Search STAC catalog for Sentinel-2 L2A scenes covering
the spatial and temporal extent of the parcels.  Stores scene metadata
(including COG asset URLs) in ``data/tiles/tile_index.parquet`` — no full
tile downloads are performed.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import pandas as pd
from pystac_client import Client
from tqdm import tqdm

from src.config import get_config

log = logging.getLogger(__name__)

# Earth Search v1 uses descriptive asset keys instead of Sentinel-2 band IDs
_BAND_TO_ASSET_KEY: dict[str, str] = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
}


def _bbox_with_buffer(gdf: gpd.GeoDataFrame, buffer_deg: float = 0.01) -> list[float]:
    """Compute bounding box of all parcels with a small buffer."""
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    return [
        bounds[0] - buffer_deg,
        bounds[1] - buffer_deg,
        bounds[2] + buffer_deg,
        bounds[3] + buffer_deg,
    ]


def _search_stac(
    catalog_url: str,
    collection: str,
    bbox: list[float],
    start_date: str,
    end_date: str,
    max_cloud_cover: int,
    bands: list[str],
) -> pd.DataFrame:
    """Query STAC and return a DataFrame of usable scenes with asset URLs."""
    client = Client.open(catalog_url)

    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        max_items=500,
    )

    records: list[dict[str, Any]] = []
    for item in tqdm(search.items(), desc="Processing STAC items"):
        asset_urls: dict[str, str] = {}
        for band in bands:
            asset = item.assets.get(band) or item.assets.get(_BAND_TO_ASSET_KEY.get(band, ""))
            if asset:
                asset_urls[band] = asset.href

        if not asset_urls:
            continue

        props = item.properties
        records.append(
            {
                "scene_id": item.id,
                "date": pd.Timestamp(item.datetime).strftime("%Y-%m-%d"),
                "cloud_cover": props.get("eo:cloud_cover", None),
                "asset_urls": json.dumps(asset_urls),
                "bbox": json.dumps(list(item.bbox)) if item.bbox else None,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def _load_cache(cache_path: Path, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Load cached tile index if it exists and covers the requested date range."""
    if not cache_path.exists():
        return None
    df = pd.read_parquet(cache_path)
    if df.empty:
        return None
    cached_min = df["date"].min()
    cached_max = df["date"].max()
    if cached_min <= start_date and cached_max >= end_date:
        log.info(
            "Tile index cache hit (%d scenes, %s to %s)",
            len(df),
            cached_min,
            cached_max,
        )
        return df
    log.info("Tile index cache exists but does not cover full date range — re-searching")
    return None


def fetch(cfg: dict) -> pd.DataFrame:
    """Run the STAC search and return a scene index DataFrame."""
    stac_cfg = cfg["stac"]
    season = cfg["season"]
    output_dir = Path(cfg["_meta"]["project_root"]) / "data" / "tiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "tile_index.parquet"

    cached = _load_cache(cache_path, season["start_date"], season["end_date"])
    if cached is not None:
        return cached

    parcels_path = Path(cfg["output"]["dir"]) / "parcels_labeled.parquet"
    if not parcels_path.exists():
        raise FileNotFoundError(
            f"Parcels file not found: {parcels_path} — run ingest step first"
        )
    gdf = gpd.read_parquet(parcels_path)
    gdf = gdf.to_crs(epsg=4326)
    bbox = _bbox_with_buffer(gdf)

    log.info(
        "STAC search: bbox=%s, dates=%s to %s, cloud<%d%%",
        [round(b, 4) for b in bbox],
        season["start_date"],
        season["end_date"],
        stac_cfg["max_cloud_cover"],
    )

    df = _search_stac(
        catalog_url=stac_cfg["catalog_url"],
        collection=stac_cfg["collection"],
        bbox=bbox,
        start_date=season["start_date"],
        end_date=season["end_date"],
        max_cloud_cover=stac_cfg["max_cloud_cover"],
        bands=stac_cfg["bands"],
    )

    if df.empty:
        log.warning("No scenes found matching search criteria")
    else:
        is_smoke = os.environ.get("SMOKE_TEST", "0") == "1"
        if is_smoke:
            df = df.head(2).copy()
            log.info("SMOKE_TEST: limited to %d scenes", len(df))

        # Keep the best (lowest cloud) scene per date to avoid duplicates
        # from overlapping MGRS tiles
        df = df.sort_values("cloud_cover").drop_duplicates(subset="date", keep="first")
        df = df.sort_values("date").reset_index(drop=True)

        max_scenes = stac_cfg.get("max_scenes")
        if max_scenes and len(df) > max_scenes:
            df = df.head(max_scenes).copy()
            log.info("max_scenes: limited to %d unique dates", len(df))

        log.info(
            "Found %d scenes across %d unique dates (%.0f%% avg cloud cover)",
            len(df),
            len(df),
            df["cloud_cover"].mean() if "cloud_cover" in df.columns else 0,
        )

    df.to_parquet(cache_path)
    log.info("Tile index saved to %s", cache_path)
    return df


@click.command("fetch")
@click.option("--config", "config_path", default="config/monitor.yaml")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Step 2: Fetch Sentinel-2 imagery metadata via STAC."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = get_config(config_path, list(overrides) if overrides else None)
    fetch(cfg)


if __name__ == "__main__":
    cli()
