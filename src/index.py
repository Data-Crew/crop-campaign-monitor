"""Build FAISS similarity index over parcel seasonal embeddings.

Aggregates per-parcel embeddings (temporal mean), L2-normalizes, and builds
IndexFlatIP (cosine similarity via inner product). Writes ``parcels.faiss`` and
``parcels_index_meta.parquet``. If FAISS is unavailable, writes
``parcels_vectors.npz`` for query-time fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import get_config

log = logging.getLogger(__name__)

FAISS_FILENAME = "parcels.faiss"
META_FILENAME = "parcels_index_meta.parquet"
VECTORS_FALLBACK = "parcels_vectors.npz"


def _load_parcel_embedding_mean(embeddings_dir: Path, parcel_id: str) -> np.ndarray | None:
    """Mean of all dated .npy embeddings for a parcel."""
    emb_dir = embeddings_dir / parcel_id
    if not emb_dir.exists():
        return None
    arrs: list[np.ndarray] = []
    for npy_file in sorted(emb_dir.glob("*.npy")):
        arrs.append(np.load(npy_file).astype(np.float32).ravel())
    if not arrs:
        return None
    return np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)


def _l2_normalize_rows(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(np.float32)


def build_index(cfg: dict) -> dict[str, Any]:
    """Build seasonal embedding index from parcel_scores + data/embeddings."""
    project_root = Path(cfg["_meta"]["project_root"])
    output_dir = Path(cfg["output"]["dir"])
    scores_path = output_dir / "parcel_scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Scores not found: {scores_path} — run score step first"
        )

    embeddings_dir = project_root / "data" / "embeddings"
    scored = gpd.read_parquet(scores_path)

    rows_meta: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []

    for _, row in scored.iterrows():
        pid = str(row["parcel_id"])
        mean_emb = _load_parcel_embedding_mean(embeddings_dir, pid)
        if mean_emb is None:
            continue
        rows_meta.append(
            {
                "parcel_id": pid,
                "crop_name": row.get("crop_name"),
                "geometry": row.geometry,
                "centroid_lat": row.get("centroid_lat"),
                "centroid_lon": row.get("centroid_lon"),
                "status": row.get("status"),
                "faiss_row": len(vectors),
            }
        )
        vectors.append(mean_emb)

    if not vectors:
        raise RuntimeError(
            f"No parcel embeddings under {embeddings_dir} — run embed step first"
        )

    mat = np.stack(vectors, axis=0).astype(np.float32)
    mat = _l2_normalize_rows(mat)
    dim = mat.shape[1]

    meta_gdf = gpd.GeoDataFrame(rows_meta, geometry="geometry", crs=scored.crs)
    meta_path = output_dir / META_FILENAME
    meta_gdf.to_parquet(meta_path)
    log.info("Saved %s (%d parcels)", meta_path, len(meta_gdf))

    result: dict[str, Any] = {
        "n_parcels": len(meta_gdf),
        "dim": dim,
        "faiss_path": None,
        "vectors_fallback": None,
    }

    try:
        import faiss

        index = faiss.IndexFlatIP(dim)
        index.add(mat)
        faiss_path = output_dir / FAISS_FILENAME
        faiss.write_index(index, str(faiss_path))
        result["faiss_path"] = str(faiss_path)
        log.info("Saved FAISS index %s", faiss_path)
    except ImportError:
        log.warning(
            "faiss not installed — writing %s for cosine search fallback",
            VECTORS_FALLBACK,
        )
        fb = output_dir / VECTORS_FALLBACK
        np.savez_compressed(fb, vectors=mat)
        result["vectors_fallback"] = str(fb)

    return result


def main(config_path: str | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Importable entry point for building the similarity index."""
    path = config_path or "config/default.yaml"
    cfg = get_config(path, overrides)
    return build_index(cfg)


@click.command("index")
@click.option("--config", "config_path", default="config/default.yaml")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Build FAISS index over seasonal parcel embeddings."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    main(config_path, list(overrides) if overrides else None)


if __name__ == "__main__":
    cli()
