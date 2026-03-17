"""Step 7 — Generate campaign report.

Reads the scored parcels and produces a summary JSON
(``data/output/campaign_report.json``) consumed by the Streamlit dashboard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import pandas as pd

from src.config import get_config

log = logging.getLogger(__name__)


def report(cfg: dict) -> dict[str, Any]:
    """Generate the campaign summary report."""
    output_dir = Path(cfg["output"]["dir"])
    scores_path = output_dir / "parcel_scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found: {scores_path} — run score step first")

    scored = gpd.read_parquet(scores_path)
    total = len(scored)

    status_counts = scored["status"].value_counts().to_dict()
    status_pct = {k: round(v / total * 100, 1) for k, v in status_counts.items()}

    # Per-crop breakdown
    crop_stats: list[dict[str, Any]] = []
    for crop_name in sorted(scored["crop_name"].dropna().unique()):
        crop_df = scored[scored["crop_name"] == crop_name]
        valid = crop_df["health_score"].dropna()
        crop_stats.append(
            {
                "crop_name": crop_name,
                "total": len(crop_df),
                "green": int((crop_df["status"] == "GREEN").sum()),
                "yellow": int((crop_df["status"] == "YELLOW").sum()),
                "red": int((crop_df["status"] == "RED").sum()),
                "gray": int((crop_df["status"] == "GRAY").sum()),
                "avg_health_score": round(float(valid.mean()), 4) if len(valid) > 0 else None,
            }
        )

    # Top-10 worst parcels
    scored["health_score"] = pd.to_numeric(scored["health_score"], errors="coerce")
    valid_scored = scored[scored["health_score"].notna()].copy()
    worst = valid_scored.nlargest(10, "health_score") if not valid_scored.empty else valid_scored
    top10_worst = [
        {
            "parcel_id": row["parcel_id"],
            "crop_name": row["crop_name"],
            "health_score": round(float(row["health_score"]), 4),
            "status": row["status"],
            "n_observations": int(row["n_observations"]) if pd.notna(row.get("n_observations")) else 0,
            "max_deviation_date": row.get("max_deviation_date"),
        }
        for _, row in worst.iterrows()
    ]

    # Temporal coverage
    scored["n_observations"] = pd.to_numeric(scored["n_observations"], errors="coerce").fillna(0)
    obs_stats = scored["n_observations"].describe().to_dict()
    temporal = {
        "mean_observations": round(obs_stats.get("mean", 0), 1),
        "min_observations": int(obs_stats.get("min", 0)),
        "max_observations": int(obs_stats.get("max", 0)),
        "median_observations": round(obs_stats.get("50%", 0), 1),
    }

    scoring_cfg = cfg.get("scoring", {})
    summary: dict[str, Any] = {
        "total_parcels": total,
        "status_counts": status_counts,
        "status_pct": status_pct,
        "crop_breakdown": crop_stats,
        "top10_worst": top10_worst,
        "temporal_coverage": temporal,
        "season": cfg.get("season", {}),
        "region": cfg.get("region", {}).get("name", "unknown"),
        "scoring": {
            "threshold_method": scoring_cfg.get("threshold_method", "fixed"),
            "thresholds": scoring_cfg.get("thresholds", {}),
            "sigma_yellow": scoring_cfg.get("sigma_yellow"),
            "sigma_red": scoring_cfg.get("sigma_red"),
        },
    }

    out_path = output_dir / "campaign_report.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Campaign report saved to %s", out_path)
    log.info(
        "Summary: %d parcels — GREEN:%d  YELLOW:%d  RED:%d  GRAY:%d",
        total,
        status_counts.get("GREEN", 0),
        status_counts.get("YELLOW", 0),
        status_counts.get("RED", 0),
        status_counts.get("GRAY", 0),
    )

    return summary


@click.command("report")
@click.option("--config", "config_path", default="config/monitor.yaml")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Step 7: Generate campaign report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = get_config(config_path, list(overrides) if overrides else None)
    report(cfg)


if __name__ == "__main__":
    cli()
