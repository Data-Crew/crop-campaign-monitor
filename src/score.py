"""Step 6 — Score parcels against reference profiles (traffic light).

Computes cosine distance between each parcel's embedding trajectory and the
reference profile for its crop type.  Produces a health score (0 = identical
to reference, 1 = maximum deviation) and a status label:
GREEN / YELLOW / RED / GRAY.

Outputs ``parcel_scores.parquet``, ``parcel_scores.geojson``, and
``campaign_summary.csv``.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import get_config

log = logging.getLogger(__name__)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors (0 = identical, 2 = opposite)."""
    dot = np.dot(a, b)
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(1.0 - dot / norm)


def _load_parcel_embeddings(
    embeddings_dir: Path,
    parcel_id: str,
) -> list[tuple[str, np.ndarray]]:
    """Load all dated embeddings for a parcel."""
    emb_dir = embeddings_dir / parcel_id
    if not emb_dir.exists():
        return []
    result: list[tuple[str, np.ndarray]] = []
    for npy_file in sorted(emb_dir.glob("*.npy")):
        date_str = npy_file.stem
        emb = np.load(npy_file)
        result.append((date_str, emb))
    return result


def _align_and_score(
    parcel_embs: list[tuple[str, np.ndarray]],
    ref_profile: list[tuple[str, np.ndarray]],
    urgency_weight_early: float,
    season_start: str,
    season_end: str,
) -> tuple[float, list[float], str | None]:
    """Align parcel embeddings to reference profile bins and compute distances.

    Returns (health_score, distance_trajectory, max_deviation_date).
    """
    ref_dates = {d: emb for d, emb in ref_profile}
    ref_sorted = sorted(ref_dates.keys())

    start_dt = datetime.strptime(season_start, "%Y-%m-%d")
    end_dt = datetime.strptime(season_end, "%Y-%m-%d")
    season_len = (end_dt - start_dt).days or 1

    distances: list[float] = []
    weights: list[float] = []
    dates_matched: list[str] = []

    for p_date, p_emb in parcel_embs:
        closest_ref_date = min(
            ref_sorted,
            key=lambda rd: abs(
                (datetime.strptime(p_date, "%Y-%m-%d") - datetime.strptime(rd, "%Y-%m-%d")).days
            ),
        )
        day_diff = abs(
            (datetime.strptime(p_date, "%Y-%m-%d") - datetime.strptime(closest_ref_date, "%Y-%m-%d")).days
        )
        if day_diff > 21:
            continue

        ref_emb = ref_dates[closest_ref_date]
        dist = _cosine_distance(p_emb, ref_emb)
        distances.append(dist)
        dates_matched.append(p_date)

        # Weight: how early in the season is this observation?
        p_dt = datetime.strptime(p_date, "%Y-%m-%d")
        season_frac = max(0.0, (p_dt - start_dt).days) / season_len
        # Early season (frac < 0.3) gets urgency_weight_early, rest gets 1.0
        w = urgency_weight_early if season_frac < 0.3 else 1.0
        weights.append(w)

    if not distances:
        return 0.0, [], None

    weights_arr = np.array(weights)
    distances_arr = np.array(distances)
    health_score = float(np.average(distances_arr, weights=weights_arr))

    max_idx = int(np.argmax(distances_arr))
    max_dev_date = dates_matched[max_idx]

    return health_score, distances, max_dev_date


def _classify(
    health_score: float,
    thresholds: dict[str, float],
) -> str:
    """Classify a health score using absolute threshold boundaries."""
    green_t = thresholds["green"]
    red_t = thresholds["red"]
    if health_score < green_t:
        return "GREEN"
    elif health_score < red_t:
        return "YELLOW"
    else:
        return "RED"


def _classify_adaptive(
    health_scores: np.ndarray,
    sigma_yellow: float = 1.5,
    sigma_red: float = 2.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Classify using z-score thresholds relative to the score distribution.

    Returns an array of status labels and the computed thresholds.
    """
    valid = health_scores[~np.isnan(health_scores)]
    if len(valid) < 5:
        return np.full(len(health_scores), "GRAY", dtype=object), {"green": 0.0, "red": 0.0}

    mu = float(np.mean(valid))
    std = float(np.std(valid))
    green_t = mu + sigma_yellow * std
    red_t = mu + sigma_red * std

    log.info(
        "Adaptive thresholds: mean=%.4f, std=%.4f → green<%.4f, yellow<%.4f, red>=%.4f",
        mu, std, green_t, red_t, red_t,
    )

    statuses = np.full(len(health_scores), "GRAY", dtype=object)
    for i, hs in enumerate(health_scores):
        if np.isnan(hs):
            continue
        if hs < green_t:
            statuses[i] = "GREEN"
        elif hs < red_t:
            statuses[i] = "YELLOW"
        else:
            statuses[i] = "RED"

    return statuses, {"green": round(green_t, 6), "red": round(red_t, 6)}


def score(cfg: dict) -> gpd.GeoDataFrame:
    """Score all parcels against their crop reference profiles."""
    project_root = Path(cfg["_meta"]["project_root"])
    embeddings_dir = project_root / "data" / "embeddings"
    output_dir = Path(cfg["output"]["dir"])

    parcels_path = output_dir / "parcels_labeled.parquet"
    profiles_path = output_dir / "reference_profiles.pkl"

    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels not found: {parcels_path}")
    if not profiles_path.exists():
        raise FileNotFoundError(f"Reference profiles not found: {profiles_path}")

    parcels = gpd.read_parquet(parcels_path)
    with open(profiles_path, "rb") as f:
        profiles = pickle.load(f)

    scoring_cfg = cfg["scoring"]
    thresholds = scoring_cfg["thresholds"]
    min_obs = scoring_cfg["min_observations"]
    urgency = scoring_cfg["urgency_weight_early"]
    season = cfg["season"]
    threshold_method = scoring_cfg.get("threshold_method", "fixed")

    is_smoke = os.environ.get("SMOKE_TEST", "0") == "1"
    if is_smoke:
        parcels = parcels.head(5).copy()

    results: list[dict[str, Any]] = []

    for _, row in tqdm(parcels.iterrows(), total=len(parcels), desc="Scoring parcels"):
        pid = str(row["parcel_id"])
        crop = row.get("crop_name")
        rec: dict[str, Any] = {
            "parcel_id": pid,
            "crop_name": crop if pd.notna(crop) else None,
            "centroid_lat": row.get("centroid_lat"),
            "centroid_lon": row.get("centroid_lon"),
            "area_ha": row.get("area_ha"),
            "tile_id": row.get("tile_id"),
            "fragment_id": row.get("fragment_id"),
        }

        if pd.isna(crop) or not crop:
            rec.update(
                health_score=None,
                status="GRAY",
                n_observations=0,
                max_deviation_date=None,
                distance_trajectory=json.dumps([]),
            )
            results.append(rec)
            continue

        parcel_embs = _load_parcel_embeddings(embeddings_dir, pid)

        if len(parcel_embs) < min_obs:
            rec.update(
                health_score=None,
                status="GRAY",
                n_observations=len(parcel_embs),
                max_deviation_date=None,
                distance_trajectory=json.dumps([]),
            )
            results.append(rec)
            continue

        ref_profile = profiles.get(str(crop))

        if not ref_profile:
            rec.update(
                health_score=None,
                status="GRAY",
                n_observations=len(parcel_embs),
                max_deviation_date=None,
                distance_trajectory=json.dumps([]),
            )
            results.append(rec)
            continue

        health_score_val, dist_traj, max_dev_date = _align_and_score(
            parcel_embs,
            ref_profile,
            urgency,
            season["start_date"],
            season["end_date"],
        )

        if not dist_traj:
            status = "GRAY"
        elif threshold_method == "fixed":
            status = _classify(health_score_val, thresholds)
        else:
            status = "__deferred__"

        rec.update(
            health_score=round(health_score_val, 4),
            status=status,
            n_observations=len(parcel_embs),
            max_deviation_date=max_dev_date,
            distance_trajectory=json.dumps([round(d, 4) for d in dist_traj]),
        )
        results.append(rec)

    # Adaptive thresholds: classify after collecting all scores
    if threshold_method == "adaptive":
        sigma_yellow = scoring_cfg.get("sigma_yellow", 1.5)
        sigma_red = scoring_cfg.get("sigma_red", 2.5)
        health_scores = np.array([
            r["health_score"] if r["health_score"] is not None else np.nan
            for r in results
        ])
        adaptive_statuses, computed_thresholds = _classify_adaptive(
            health_scores, sigma_yellow, sigma_red,
        )
        for i, rec in enumerate(results):
            if rec["status"] == "__deferred__":
                rec["status"] = str(adaptive_statuses[i])
        thresholds = computed_thresholds

    results_df = pd.DataFrame(results)
    scored = parcels[["parcel_id", "geometry"]].merge(results_df, on="parcel_id", how="right")
    scored = gpd.GeoDataFrame(scored, geometry="geometry", crs=parcels.crs)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist effective thresholds for the dashboard
    effective_thresholds = {
        "method": threshold_method,
        "green": thresholds.get("green", thresholds.get("green")),
        "red": thresholds.get("red", thresholds.get("red")),
    }
    with open(output_dir / "scoring_thresholds.json", "w") as f:
        json.dump(effective_thresholds, f, indent=2)
    log.info("Effective thresholds: %s", effective_thresholds)

    if cfg["output"].get("parquet", True):
        scored.to_parquet(output_dir / "parcel_scores.parquet")
        log.info("Saved parcel_scores.parquet")

    if cfg["output"].get("geojson", True):
        scored.to_file(output_dir / "parcel_scores.geojson", driver="GeoJSON")
        log.info("Saved parcel_scores.geojson")

    # Campaign summary CSV
    if cfg["output"].get("csv_summary", True):
        status_counts = scored["status"].value_counts().to_dict()
        summary_rows = []
        for crop_name in scored["crop_name"].dropna().unique():
            crop_mask = scored["crop_name"] == crop_name
            crop_df = scored[crop_mask]
            valid_scores = crop_df["health_score"].dropna()
            summary_rows.append(
                {
                    "crop_name": crop_name,
                    "total_parcels": len(crop_df),
                    "green": int((crop_df["status"] == "GREEN").sum()),
                    "yellow": int((crop_df["status"] == "YELLOW").sum()),
                    "red": int((crop_df["status"] == "RED").sum()),
                    "gray": int((crop_df["status"] == "GRAY").sum()),
                    "avg_health_score": round(float(valid_scores.mean()), 4)
                    if len(valid_scores) > 0
                    else None,
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "campaign_summary.csv", index=False)
        log.info("Saved campaign_summary.csv")

    # Log results
    log.info("Score distribution: %s", scored["status"].value_counts().to_dict())
    red_parcels = scored[scored["status"] == "RED"]
    if not red_parcels.empty:
        log.warning(
            "RED alerts (%d parcels): %s",
            len(red_parcels),
            red_parcels["parcel_id"].tolist()[:10],
        )

    return scored


@click.command("score")
@click.option("--config", "config_path", default="config/monitor.yaml")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Step 6: Score parcels (traffic light classification)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = get_config(config_path, list(overrides) if overrides else None)
    score(cfg)


if __name__ == "__main__":
    cli()
