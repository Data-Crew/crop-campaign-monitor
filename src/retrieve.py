"""Geo-RAG retrieval layer: semantic (FAISS) and geographic neighbors for LLM explain.

This module is optional. It does not perform anomaly detection; it only gathers
supporting context from existing pipeline artifacts.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in kilometres."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def _row_lat_lon(row: pd.Series) -> tuple[float | None, float | None]:
    la = row.get("centroid_lat")
    lo = row.get("centroid_lon")
    if la is not None and lo is not None and pd.notna(la) and pd.notna(lo):
        return float(la), float(lo)
    geom = row.get("geometry")
    if geom is not None and not geom.is_empty:
        c = geom.centroid
        return float(c.y), float(c.x)
    return None, None


def retrieve_similar_parcels(
    parcel_id: str,
    cfg: dict,
    k: int,
    parcel_index: Any | None = None,
    status_lookup: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """FAISS / inner-product nearest neighbours (semantic similarity, not geography).

    Returns (list of serialisable dicts, notes).
    """
    notes: list[str] = []
    out: list[dict[str, Any]] = []
    if k <= 0:
        return out, notes

    try:
        from src.query import load_parcel_index

        idx = parcel_index
        if idx is None:
            idx = load_parcel_index(cfg=cfg)
        nns = idx.nearest_neighbors(parcel_id, k)
    except FileNotFoundError as e:
        log.warning("Geo-RAG semantic retrieval: %s", e)
        notes.append("similarity_index_missing")
        return out, notes
    except KeyError:
        log.warning(
            "Geo-RAG semantic retrieval: parcel %r not in similarity index (no embeddings?)",
            parcel_id,
        )
        notes.append("parcel_not_in_similarity_index")
        return out, notes
    except Exception as e:
        log.warning("Geo-RAG semantic retrieval failed: %s", e)
        notes.append(f"similarity_error:{type(e).__name__}")
        return out, notes

    for nn in nns:
        pid = nn.parcel_id
        st = status_lookup.get(pid) if status_lookup else None
        out.append(
            {
                "parcel_id": pid,
                "crop_name": nn.crop_name,
                "similarity": round(nn.similarity, 6),
                "cosine_distance": round(nn.cosine_distance, 6),
                "status": st,
            }
        )
    return out, notes


def retrieve_spatial_neighbors(
    target_row: pd.Series,
    scored_gdf: Any,
    cfg: dict,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Geographic proximity neighbours (not embedding similarity).

    Uses centroid_lat/lon or geometry centroids. Excludes the target parcel.
    """
    notes: list[str] = []
    rag = (cfg.get("llm") or {}).get("geo_rag") or {}
    k = int(rag.get("k_spatial", 5))
    max_km = rag.get("max_spatial_km")

    if k <= 0:
        return [], notes

    tid = str(target_row["parcel_id"])
    lat0, lon0 = _row_lat_lon(target_row)
    if lat0 is None or lon0 is None:
        notes.append("no_centroid_for_spatial")
        return [], notes

    dists: list[tuple[float, str]] = []
    for _, r in scored_gdf.iterrows():
        pid = str(r["parcel_id"])
        if pid == tid:
            continue
        la, lo = _row_lat_lon(r)
        if la is None or lo is None:
            continue
        d_km = _haversine_km(lat0, lon0, la, lo)
        dists.append((d_km, pid))

    dists.sort(key=lambda x: x[0])
    out: list[dict[str, Any]] = []
    for d_km, pid in dists:
        if max_km is not None and float(d_km) > float(max_km):
            continue
        row = scored_gdf[scored_gdf["parcel_id"].astype(str) == pid]
        if row.empty:
            continue
        rr = row.iloc[0]
        cn = rr.get("crop_name")
        cn_s = str(cn) if cn is not None and pd.notna(cn) else None
        out.append(
            {
                "parcel_id": pid,
                "distance_km": round(float(d_km), 4),
                "status": str(rr.get("status") or "GRAY"),
                "crop_name": cn_s,
            }
        )
        if len(out) >= k:
            break

    return out, notes


def retrieve_reference_context(crop_name: Any, cfg: dict, profiles_cache: Any) -> dict[str, Any] | None:
    """Summarise reference profile for a crop (metadata only — no embedding vectors)."""
    if crop_name is None or (isinstance(crop_name, float) and pd.isna(crop_name)):
        return None
    cn = str(crop_name).strip()
    if not cn:
        return None

    if profiles_cache is None:
        return None

    prof = profiles_cache.get(cn)
    if not prof:
        return None

    # profile.py: list[tuple[bin_center_iso_str, ndarray]]
    dates = [t[0] for t in prof if isinstance(t, tuple) and len(t) >= 1]
    n_bins = len(dates)
    sample = dates[:5] + (dates[-3:] if n_bins > 8 else [])
    # de-dup preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for d in sample:
        if d not in seen:
            seen.add(d)
            uniq.append(d)

    return {
        "crop_name": cn,
        "n_temporal_bins": n_bins,
        "bin_sample_dates": uniq[:8],
    }


def build_retrieved_context(
    parcel_id: str,
    row: pd.Series,
    scored_gdf: Any,
    cfg: dict,
    parcel_index: Any | None = None,
    profiles_cache: Any | None = None,
) -> dict[str, Any]:
    """Assemble Geo-RAG bundle: similar parcels (FAISS), spatial neighbours, reference summary."""
    rag = (cfg.get("llm") or {}).get("geo_rag") or {}
    k_sem = int(rag.get("k_semantic", 5))
    notes: list[str] = []

    status_lookup = {
        str(r["parcel_id"]): str(r.get("status") or "GRAY")
        for _, r in scored_gdf.iterrows()
    }

    similar, n1 = retrieve_similar_parcels(
        parcel_id, cfg, k_sem, parcel_index=parcel_index, status_lookup=status_lookup
    )
    notes.extend(n1)

    spatial, n2 = retrieve_spatial_neighbors(row, scored_gdf, cfg)
    notes.extend(n2)

    ref = retrieve_reference_context(row.get("crop_name"), cfg, profiles_cache)
    if (
        ref is None
        and profiles_cache is not None
        and row.get("crop_name") is not None
        and pd.notna(row.get("crop_name"))
    ):
        cn = str(row.get("crop_name")).strip()
        if cn and cn not in profiles_cache:
            notes.append("no_reference_profile_for_crop")

    return {
        "similar_parcels": similar,
        "spatial_neighbors": spatial,
        "reference_context": ref,
        "retrieval_notes": notes,
    }
