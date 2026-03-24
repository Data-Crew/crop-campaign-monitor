"""Query parcel similarity index (FAISS or numpy fallback)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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


@dataclass
class NeighborResult:
    """One nearest neighbor."""

    parcel_id: str
    crop_name: str | None
    similarity: float
    cosine_distance: float


class ParcelIndex:
    """FAISS IndexFlatIP or L2-normalized matrix + inner-product search."""

    def __init__(
        self,
        meta: gpd.GeoDataFrame,
        parcel_to_row: dict[str, int],
        faiss_index: Any = None,
        vectors: np.ndarray | None = None,
    ) -> None:
        if faiss_index is None and vectors is None:
            raise ValueError("Provide faiss_index or vectors")
        self.meta = meta
        self.parcel_to_row = parcel_to_row
        self._faiss_index = faiss_index
        self._vectors = vectors.astype(np.float32) if vectors is not None else None

    def row_for_parcel(self, parcel_id: str) -> int:
        if parcel_id not in self.parcel_to_row:
            raise KeyError(f"Unknown parcel_id: {parcel_id}")
        return self.parcel_to_row[parcel_id]

    def nearest_neighbors(self, parcel_id: str, k: int) -> list[NeighborResult]:
        """Return k nearest neighbors (excluding self). Similarity = inner product (= cosine)."""
        row = self.row_for_parcel(parcel_id)
        n_total = len(self.meta)
        k_search = min(k + 1, n_total)

        if self._faiss_index is not None:
            q = self._faiss_index.reconstruct(row).astype(np.float32).reshape(1, -1)
            sims, idx = self._faiss_index.search(q, k_search)
            sims = sims[0]
            idx = idx[0]
        else:
            assert self._vectors is not None
            q = self._vectors[row : row + 1]
            sims_full = (self._vectors @ q.ravel()).ravel()
            order = np.argsort(-sims_full)[:k_search]
            sims = sims_full[order]
            idx = order

        out: list[NeighborResult] = []
        for sim, i in zip(sims, idx):
            if int(i) == row:
                continue
            m = self.meta.iloc[int(i)]
            pid = str(m["parcel_id"])
            crop = m.get("crop_name")
            crop_s = str(crop) if pd.notna(crop) else None
            s = float(sim)
            out.append(
                NeighborResult(
                    parcel_id=pid,
                    crop_name=crop_s,
                    similarity=s,
                    cosine_distance=1.0 - s,
                )
            )
            if len(out) >= k:
                break
        return out

    def scan_crop_mislabels(
        self,
        crop_name: str,
        k: int,
        min_diff_frac: float,
    ) -> list[dict[str, Any]]:
        """Parcels labeled *crop_name* whose k-NN are mostly other crops."""
        label = crop_name.strip()

        def _matches_crop(x: Any) -> bool:
            if pd.isna(x):
                return False
            return str(x).strip() == label

        mask = self.meta["crop_name"].apply(_matches_crop)
        subset = self.meta[mask]
        results: list[dict[str, Any]] = []
        for pid in subset["parcel_id"].astype(str):
            try:
                nns = self.nearest_neighbors(pid, k)
            except KeyError:
                continue
            if not nns:
                continue
            diff = 0
            labeled = 0
            for nn in nns:
                nc = nn.crop_name
                if nc is None or (isinstance(nc, float) and np.isnan(nc)):
                    continue
                labeled += 1
                if str(nc).strip() != label:
                    diff += 1
            if labeled == 0:
                continue
            frac = diff / labeled
            if frac >= min_diff_frac:
                results.append(
                    {
                        "parcel_id": pid,
                        "neighbor_diff_fraction": round(frac, 4),
                        "diff_neighbors": diff,
                        "labeled_neighbors": labeled,
                    }
                )
        results.sort(key=lambda r: r["neighbor_diff_fraction"], reverse=True)
        return results


def load_parcel_index(
    output_dir: str | Path | None = None,
    cfg: dict | None = None,
) -> ParcelIndex:
    """Load FAISS index (or vectors npz) and metadata from data/output."""
    if cfg is not None:
        out = Path(cfg["output"]["dir"])
    elif output_dir is not None:
        out = Path(output_dir)
    else:
        raise ValueError("Provide output_dir or cfg")

    meta_path = out / META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    meta = gpd.read_parquet(meta_path)
    parcel_to_row = {str(r["parcel_id"]): int(i) for i, r in meta.iterrows()}

    faiss_path = out / FAISS_FILENAME
    npz_path = out / VECTORS_FALLBACK

    if faiss_path.exists():
        import faiss

        index = faiss.read_index(str(faiss_path))
        return ParcelIndex(meta, parcel_to_row, faiss_index=index)

    if npz_path.exists():
        log.info("Loading vectors from %s (FAISS index not found)", npz_path)
        data = np.load(npz_path)
        vectors = data["vectors"].astype(np.float32)
        return ParcelIndex(meta, parcel_to_row, vectors=vectors)

    raise FileNotFoundError(
        f"Neither {faiss_path} nor {npz_path} found — run src.index first"
    )


def nearest_neighbors(
    parcel_id: str,
    k: int,
    output_dir: str | Path | None = None,
    cfg: dict | None = None,
) -> list[NeighborResult]:
    idx = load_parcel_index(output_dir=output_dir, cfg=cfg)
    return idx.nearest_neighbors(parcel_id, k)


def scan_crop_mislabels(
    crop_name: str,
    k: int,
    min_diff_frac: float = 0.5,
    output_dir: str | Path | None = None,
    cfg: dict | None = None,
) -> list[dict[str, Any]]:
    idx = load_parcel_index(output_dir=output_dir, cfg=cfg)
    return idx.scan_crop_mislabels(crop_name, k, min_diff_frac)


def _default_output_dir(config_path: str | None) -> Path:
    if config_path:
        cfg = get_config(config_path, None)
        return Path(cfg["output"]["dir"])
    return Path("data/output")


@click.command("query")
@click.option("--config", "config_path", default=None, help="YAML config (resolves output dir)")
@click.option("--output-dir", "output_dir", default=None, type=click.Path())
@click.option("--parcel-id", "parcel_id", default=None, help="Query k-NN for this parcel")
@click.option("--k", "k", default=10, show_default=True)
@click.option("--crop-name", "crop_name", default=None, help="Scan mislabels for this crop label")
@click.option(
    "--min-neighbor-diff-frac",
    "min_diff_frac",
    default=0.5,
    show_default=True,
    help="Min fraction of neighbors with a different crop (crop scan only)",
)
def cli(
    config_path: str | None,
    output_dir: str | None,
    parcel_id: str | None,
    k: int,
    crop_name: str | None,
    min_diff_frac: float,
) -> None:
    """Query similarity index: nearest neighbors or crop mislabel scan."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    out = Path(output_dir) if output_dir else _default_output_dir(config_path)
    cfg = get_config(config_path, None) if config_path else None

    idx = load_parcel_index(output_dir=out, cfg=cfg)

    if parcel_id:
        nns = idx.nearest_neighbors(parcel_id, k)
        rows = [
            {
                "parcel_id": n.parcel_id,
                "crop_name": n.crop_name,
                "similarity": round(n.similarity, 6),
                "cosine_distance": round(n.cosine_distance, 6),
            }
            for n in nns
        ]
        click.echo(json.dumps(rows, indent=2))
        return

    if crop_name:
        found = idx.scan_crop_mislabels(crop_name, k, min_diff_frac)
        click.echo(json.dumps(found, indent=2))
        return

    raise click.UsageError("Provide --parcel-id or --crop-name")


def main(
    parcel_id: str | None = None,
    crop_name: str | None = None,
    k: int = 10,
    min_neighbor_diff_frac: float = 0.5,
    config_path: str | None = None,
    output_dir: str | None = None,
) -> Any:
    """Importable entry for dashboard and tests."""
    out = Path(output_dir) if output_dir else _default_output_dir(config_path)
    cfg = get_config(config_path, None) if config_path else None
    idx = load_parcel_index(output_dir=out, cfg=cfg)
    if parcel_id:
        return idx.nearest_neighbors(parcel_id, k)
    if crop_name:
        return idx.scan_crop_mislabels(crop_name, k, min_neighbor_diff_frac)
    raise ValueError("Provide parcel_id or crop_name")


if __name__ == "__main__":
    cli()
