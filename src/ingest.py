"""Step 1 — Ingest parcels and assign crop labels from CDL.

Scans the ``data/fields/`` directory for GeoJSON fragments, merges them into a
single GeoDataFrame, and attaches USDA CDL crop labels.  Supports three label
sources:

- ``embedded``: reads crop labels already present in the GeoJSON properties
  (e.g. ``props.crops_2022``, ``props.crops_ids_2022``).
- ``usda``: downloads CDL raster from CropScape and runs zonal majority.
- ``local``: reads a pre-labeled CSV file.

Outputs ``data/output/parcels_labeled.parquet``.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.config import get_config

log = logging.getLogger(__name__)

# Subset of USDA CDL crop codes → readable names (covers the most common classes)
CDL_CROP_NAMES: dict[int, str] = {
    1: "corn",
    2: "cotton",
    3: "rice",
    4: "sorghum",
    5: "soybeans",
    6: "sunflower",
    10: "peanuts",
    11: "tobacco",
    12: "sweet_corn",
    13: "popcorn",
    21: "barley",
    22: "durum_wheat",
    23: "spring_wheat",
    24: "winter_wheat",
    25: "other_small_grains",
    26: "winter_wheat_soybeans",
    27: "rye",
    28: "oats",
    29: "millet",
    30: "speltz",
    31: "canola",
    32: "flaxseed",
    33: "safflower",
    34: "rape_seed",
    35: "mustard",
    36: "alfalfa",
    37: "other_hay",
    41: "sugarbeets",
    42: "dry_beans",
    43: "potatoes",
    44: "other_crops",
    45: "sugarcane",
    46: "sweet_potatoes",
    47: "misc_vegs_fruits",
    48: "watermelons",
    49: "onions",
    50: "cucumbers",
    51: "chick_peas",
    52: "lentils",
    53: "peas",
    54: "tomatoes",
    55: "caneberries",
    56: "hops",
    57: "herbs",
    58: "clover",
    59: "sod_grass_seed",
    60: "switchgrass",
    61: "fallow",
    62: "pasture_grass",
    63: "forest",
    64: "shrubland",
    65: "barren",
    66: "cherries",
    67: "peaches",
    68: "apples",
    69: "grapes",
    70: "christmas_trees",
    71: "other_tree_crops",
    72: "citrus",
    74: "pecans",
    75: "almonds",
    76: "walnuts",
    77: "pears",
    176: "grassland",
    190: "woody_wetlands",
    195: "herbaceous_wetlands",
}


def _discover_fragments(
    fields_dir: str,
    tiles: list[str],
    fragments: list[str] | str,
) -> list[Path]:
    """Return list of GeoJSON paths matching the tile/fragment filter."""
    fields_path = Path(fields_dir)
    result: list[Path] = []

    for tile_id in tiles:
        tile_dir = fields_path / tile_id
        if not tile_dir.is_dir():
            log.warning("Tile directory not found: %s", tile_dir)
            continue
        if fragments == "all":
            pattern = str(tile_dir / f"{tile_id}_fragment_*.geojson")
            result.extend(Path(p) for p in sorted(glob.glob(pattern)))
        else:
            for frag_id in fragments:
                fpath = tile_dir / f"{tile_id}_fragment_{frag_id}.geojson"
                if fpath.exists():
                    result.append(fpath)
                else:
                    log.warning("Fragment file not found: %s", fpath)

    return result


def _parse_fragment_id(path: Path) -> str:
    """Extract ``XX_YY`` fragment id from a filename like ``16tgk_fragment_00_01.geojson``."""
    stem = path.stem
    parts = stem.split("_fragment_")
    return parts[1] if len(parts) == 2 else stem


def _flatten_props(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Flatten a nested ``props`` dict column into top-level columns.

    The GeoJSON files use the structure ``{id, props: {area, center_lat, ...}}``.
    This expands ``props`` so each key becomes its own column, keeping ``id`` as
    the parcel identifier.
    """
    if "props" not in gdf.columns:
        return gdf

    props_col = gdf["props"]
    if props_col.isna().all():
        return gdf

    expanded = pd.json_normalize(props_col.apply(
        lambda x: x if isinstance(x, dict) else {}
    ))
    expanded.index = gdf.index

    gdf = gdf.drop(columns=["props"])
    for col in expanded.columns:
        if col not in gdf.columns:
            gdf[col] = expanded[col]

    return gdf


def _load_parcels(
    fragment_paths: list[Path],
    tile_ids: list[str],
) -> gpd.GeoDataFrame:
    """Read and concatenate all fragment GeoJSON files into one GeoDataFrame."""
    frames: list[gpd.GeoDataFrame] = []

    for fpath in tqdm(fragment_paths, desc="Loading GeoJSON fragments"):
        gdf = gpd.read_file(fpath)
        tile_id = [t for t in tile_ids if t in fpath.stem]
        tid = tile_id[0] if tile_id else fpath.parent.name
        frag_id = _parse_fragment_id(fpath)

        # Flatten nested props dict (e.g. {id, props: {area, crops_2022, ...}})
        gdf = _flatten_props(gdf)

        gdf["tile_id"] = tid
        gdf["fragment_id"] = frag_id

        # Use existing id field as parcel_id, or generate one
        if "parcel_id" not in gdf.columns:
            if "id" in gdf.columns:
                gdf["parcel_id"] = gdf["id"].astype(str)
            else:
                gdf["parcel_id"] = [
                    f"{tid}_{frag_id}_{i}" for i in range(len(gdf))
                ]

        # Use center_lat/center_lng from props if available, compute otherwise
        has_center = "center_lat" in gdf.columns and "center_lng" in gdf.columns
        if has_center:
            gdf = gdf.rename(columns={"center_lng": "centroid_lon"})
            gdf = gdf.rename(columns={"center_lat": "centroid_lat"})

        frames.append(gdf)

    if not frames:
        log.error("No parcel geometries loaded — check fields_dir and tile/fragment config")
        return gpd.GeoDataFrame()

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
    merged = merged.to_crs(epsg=4326)

    # Compute centroids if not already present from props
    if "centroid_lat" not in merged.columns:
        centroids = merged.geometry.centroid
        merged["centroid_lat"] = centroids.y
        merged["centroid_lon"] = centroids.x

    # Compute area in hectares from props.area (m²) or from geometry
    if "area" in merged.columns:
        merged["area_ha"] = merged["area"] / 10_000
    else:
        area_gdf = merged.to_crs(merged.estimate_utm_crs())
        merged["area_ha"] = area_gdf.geometry.area / 10_000

    log.info(
        "Loaded %d parcels from %d fragments across %d tile(s)",
        len(merged),
        len(fragment_paths),
        len(set(merged["tile_id"])),
    )
    return merged


def _download_cdl_raster(year: int, bbox: tuple[float, ...], out_path: Path) -> Path:
    """Download CDL raster for *year* covering *bbox* from USDA CropScape.

    *bbox* must be in EPSG:4326 (lon/lat).  The function reprojects to
    EPSG:5070 (CONUS Albers) internally because CropScape expects Albers
    coordinates.
    """
    if out_path.exists():
        log.info("CDL raster cache hit: %s", out_path)
        return out_path

    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    minx, miny, maxx, maxy = bbox
    ax1, ay1 = transformer.transform(minx, miny)
    ax2, ay2 = transformer.transform(maxx, maxy)
    albers_bbox = (min(ax1, ax2), min(ay1, ay2), max(ax1, ax2), max(ay1, ay2))

    url = (
        "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
        f"?year={year}&bbox={albers_bbox[0]:.0f},{albers_bbox[1]:.0f},"
        f"{albers_bbox[2]:.0f},{albers_bbox[3]:.0f}"
    )
    log.info("Requesting CDL raster from CropScape: year=%d ...", year)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    import xml.etree.ElementTree as ET

    root = ET.fromstring(resp.text)
    tif_url_elem = root.find(".//{https://nassgeodata.gmu.edu/}returnURL")
    if tif_url_elem is None:
        tif_url_elem = root.find(".//returnURL")
    if tif_url_elem is None:
        for elem in root.iter():
            if "url" in elem.tag.lower() or (elem.text and elem.text.startswith("http")):
                tif_url_elem = elem
                break

    if tif_url_elem is None or not tif_url_elem.text:
        raise RuntimeError(f"Could not parse CDL download URL from response: {resp.text[:500]}")

    tif_url = tif_url_elem.text.strip()
    log.info("Downloading CDL raster from %s", tif_url)
    tif_resp = requests.get(tif_url, timeout=300, stream=True)
    tif_resp.raise_for_status()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in tif_resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    log.info("CDL raster saved to %s", out_path)
    return out_path


def _label_from_cdl_raster(
    gdf: gpd.GeoDataFrame,
    year: int,
    project_root: str,
) -> gpd.GeoDataFrame:
    """Assign crop labels via zonal majority on CDL raster."""
    from rasterstats import zonal_stats

    gdf_wgs = gdf.to_crs(epsg=4326)
    bbox = tuple(gdf_wgs.total_bounds)
    cdl_dir = Path(project_root) / "data" / "cdl"
    cdl_path = cdl_dir / f"cdl_{year}.tif"

    _download_cdl_raster(year, bbox, cdl_path)

    log.info("Running zonal majority on %d parcels...", len(gdf))
    stats = zonal_stats(
        gdf.to_crs(epsg=5070),
        str(cdl_path),
        stats=[],
        categorical=True,
        all_touched=True,
    )

    crop_codes: list[int | None] = []
    for s in stats:
        if s:
            majority_code = max(s, key=s.get)
            crop_codes.append(int(majority_code))
        else:
            crop_codes.append(None)

    gdf["crop_code"] = crop_codes
    gdf["crop_name"] = gdf["crop_code"].map(
        lambda c: CDL_CROP_NAMES.get(c, "unknown") if pd.notna(c) else None
    )
    return gdf


def _label_from_embedded(
    gdf: gpd.GeoDataFrame,
    year: int,
) -> gpd.GeoDataFrame:
    """Read crop labels already present in the GeoJSON properties.

    Expects columns ``crops_<year>`` (crop name) and ``crops_ids_<year>``
    (CDL numeric code) produced by a prior CDL zonal-majority step.
    """
    name_col = f"crops_{year}"
    code_col = f"crops_ids_{year}"

    if name_col not in gdf.columns:
        available = sorted(c for c in gdf.columns if c.startswith("crops_") and not c.startswith("crops_ids"))
        raise ValueError(
            f"Column '{name_col}' not found for embedded CDL labels. "
            f"Available crop year columns: {available}"
        )

    gdf["crop_name"] = gdf[name_col].astype(str).str.lower().str.replace("/", "_").str.replace(" ", "_")
    if code_col in gdf.columns:
        gdf["crop_code"] = pd.to_numeric(gdf[code_col], errors="coerce").astype("Int64")
    else:
        gdf["crop_code"] = gdf["crop_name"].map(
            {v: k for k, v in CDL_CROP_NAMES.items()}
        )

    log.info("Read embedded CDL labels from column '%s' (year %d)", name_col, year)
    return gdf


def _label_from_csv(
    gdf: gpd.GeoDataFrame,
    csv_path: str,
) -> gpd.GeoDataFrame:
    """Assign crop labels from a local CSV (columns: parcel_id, crop_code, crop_name)."""
    labels = pd.read_csv(csv_path)
    required = {"parcel_id", "crop_code", "crop_name"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label CSV is missing columns: {missing}")

    gdf = gdf.merge(labels[["parcel_id", "crop_code", "crop_name"]], on="parcel_id", how="left")
    return gdf


def ingest(cfg: dict) -> gpd.GeoDataFrame:
    """Run the full ingest step: load parcels and attach crop labels."""
    region = cfg["region"]
    fragments = region.get("fragments", "all")
    tiles = region["tiles"]
    fields_dir = region["fields_dir"]

    fragment_paths = _discover_fragments(fields_dir, tiles, fragments)
    if not fragment_paths:
        raise FileNotFoundError(
            f"No GeoJSON fragments found in {fields_dir} for tiles={tiles}, fragments={fragments}"
        )

    gdf = _load_parcels(fragment_paths, tiles)
    if gdf.empty:
        raise RuntimeError("No parcel geometries loaded")

    is_smoke = os.environ.get("SMOKE_TEST", "0") == "1"
    if is_smoke:
        gdf = gdf.head(5).copy()
        log.info("SMOKE_TEST: limited to %d parcels", len(gdf))

    cdl_cfg = cfg.get("cdl", {})
    source = cdl_cfg.get("source", "embedded")
    year = cdl_cfg.get("year", 2022)
    project_root = cfg["_meta"]["project_root"]

    if source == "embedded":
        gdf = _label_from_embedded(gdf, year)
    elif source == "local":
        csv_path = cdl_cfg.get("path")
        if not csv_path:
            raise ValueError("cdl.source=local requires cdl.path to be set")
        gdf = _label_from_csv(gdf, csv_path)
    elif source == "usda":
        gdf = _label_from_cdl_raster(gdf, year, project_root)
    else:
        raise ValueError(f"Unknown cdl.source: '{source}'. Use 'embedded', 'usda', or 'local'.")

    labeled = gdf["crop_name"].notna().sum()
    unlabeled = len(gdf) - labeled
    log.info("Crop labels: %d labeled, %d unlabeled", labeled, unlabeled)

    if labeled > 0:
        dist = gdf["crop_name"].value_counts()
        log.info("Crop distribution:\n%s", dist.to_string())

    # Drop the many per-year columns to keep the parquet lean
    yearly_cols = [c for c in gdf.columns if any(
        c.startswith(p) for p in ("crops_", "crops_ids_", "crop_percentage_", "confidence_")
    )]
    gdf = gdf.drop(columns=yearly_cols, errors="ignore")

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "parcels_labeled.parquet"
    gdf.to_parquet(out_path)
    log.info("Saved labeled parcels to %s", out_path)

    return gdf


@click.command("ingest")
@click.option("--config", "config_path", default="config/monitor.yaml")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Step 1: Ingest parcels and CDL labels."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = get_config(config_path, list(overrides) if overrides else None)
    ingest(cfg)


if __name__ == "__main__":
    cli()
