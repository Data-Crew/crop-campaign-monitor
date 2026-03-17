"""Step 4 — Generate embeddings via Clay Foundation Model.

Loads each ``.npz`` chip, passes it through the Clay encoder, and saves the
resulting 768-dim embedding vector as ``.npy``.

Supports three model sources (tried in order):
  1. Fine-tuned encoder (``clay-crop-encoder.ckpt``) — best quality.
  2. Pre-trained base Clay (``clay-v1-base.ckpt``) — generic.
  3. Mock encoder — deterministic random vectors for pipeline testing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from tqdm import tqdm

from src.config import get_config
from src.gpu import check_vram, get_device, log_gpu_status

log = logging.getLogger(__name__)

# Sentinel-2 L2A band centre wavelengths (nm) for the 10 bands used
S2_WAVELENGTHS: dict[str, float] = {
    "B02": 492.4,
    "B03": 559.8,
    "B04": 664.6,
    "B08": 832.8,
    "B05": 704.1,
    "B06": 740.5,
    "B07": 782.8,
    "B8A": 864.7,
    "B11": 1613.7,
    "B12": 2202.4,
}


class MockClayEncoder:
    """Deterministic mock encoder for pipeline testing without model weights.

    Returns a 768-dim vector seeded by parcel_id so results are reproducible.
    """

    def __init__(self, embedding_dim: int = 768) -> None:
        self.embedding_dim = embedding_dim
        log.warning(
            "Using MockClayEncoder — embeddings are random and NOT meaningful. "
            "Download Clay weights or run the training pipeline for real embeddings."
        )

    def to(self, device: Any) -> "MockClayEncoder":
        return self

    def eval(self) -> "MockClayEncoder":
        return self

    def __call__(self, datacube: dict[str, torch.Tensor]) -> torch.Tensor:
        B = datacube["pixels"].shape[0]
        seeds = datacube.get("_seeds")
        embeddings = []
        for i in range(B):
            seed = int(seeds[i]) if seeds is not None else i
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.embedding_dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            embeddings.append(torch.from_numpy(vec))
        return torch.stack(embeddings)


S2_TO_CLAY_BANDS: dict[str, str] = {
    "B02": "blue", "B03": "green", "B04": "red", "B08": "nir",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B8A": "nir08", "B11": "swir16", "B12": "swir22",
}


class _TerraTorchEncoder:
    """Wraps TerraTorch's ``Embedder`` to match the interface expected by the
    embedding loop (datacube dict in → CLS token out)."""

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder

    def to(self, device: Any) -> "_TerraTorchEncoder":
        self.embedder = self.embedder.to(device)
        return self

    def eval(self) -> "_TerraTorchEncoder":
        self.embedder.eval()
        return self

    def __call__(self, datacube: dict[str, torch.Tensor]) -> tuple[torch.Tensor]:
        features = self.embedder.forward_features(
            x=datacube["pixels"],
            time=datacube["time"],
            latlon=datacube["latlon"],
            waves=datacube["waves"],
            gsd=datacube["gsd"],
        )
        return (features[-1],)  # (B, 1+L, D) — same shape as claymodel


def _load_encoder(cfg: dict, device: torch.device) -> Any:
    """Try to load a Clay encoder; fall back through the cascade."""
    weights_path = cfg["model"].get("weights_path", "")
    fallback_path = cfg["model"].get("fallback_weights", "")
    embedding_dim = cfg["model"]["embedding_dim"]
    s2_bands = cfg["stac"]["bands"]
    clay_bands = [S2_TO_CLAY_BANDS[b] for b in s2_bands]
    chip_size = cfg.get("chips", {}).get("size_px", 224)

    # 1. Fine-tuned encoder via TerraTorch
    if weights_path and Path(weights_path).exists():
        log.info("Loading fine-tuned encoder from %s", weights_path)
        try:
            from terratorch.models.backbones.clay_v1.embedder import Embedder

            state_dict = torch.load(weights_path, map_location=device, weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            embedder = Embedder(img_size=chip_size, bands=clay_bands)
            embedder.load_state_dict(state_dict, strict=False)
            encoder = _TerraTorchEncoder(embedder).to(device).eval()
            log.info("Fine-tuned encoder loaded via TerraTorch")
            return encoder
        except Exception as exc:
            log.warning("Failed to load fine-tuned encoder: %s — trying fallback", exc)

    # 2. Pre-trained base Clay via TerraTorch
    if fallback_path and Path(fallback_path).exists():
        log.info("Loading base Clay model from %s", fallback_path)
        try:
            from terratorch.models.backbones.clay_v1.embedder import Embedder

            embedder = Embedder(
                img_size=chip_size, bands=clay_bands, ckpt_path=str(fallback_path),
            )
            encoder = _TerraTorchEncoder(embedder).to(device).eval()
            log.info("Base Clay encoder loaded via TerraTorch")
            return encoder
        except Exception as exc:
            log.warning("Failed to load base Clay: %s — falling back to mock", exc)

    # 3. Mock
    return MockClayEncoder(embedding_dim=embedding_dim)


def _encode_time(date_str: str) -> list[float]:
    """Encode a date string as [sin(week), cos(week), sin(hour), cos(hour)]."""
    from datetime import datetime

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    doy = dt.timetuple().tm_yday
    week_frac = doy / 365.25
    return [
        np.sin(2 * np.pi * week_frac),
        np.cos(2 * np.pi * week_frac),
        0.0,  # hour sin (noon assumed for daily composites)
        1.0,  # hour cos
    ]


def _encode_latlon(lat: float, lon: float) -> list[float]:
    """Encode lat/lon as [sin(lat), cos(lat), sin(lon), cos(lon)] in radians."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    return [np.sin(lat_r), np.cos(lat_r), np.sin(lon_r), np.cos(lon_r)]


def _parcel_seed(parcel_id: str) -> int:
    """Deterministic integer seed from a parcel ID string."""
    return int(hashlib.md5(str(parcel_id).encode()).hexdigest()[:8], 16)


ENCODER_STAMP = ".encoder_mtime"


def _invalidate_stale_embeddings(
    embeddings_dir: Path, weights_path: str,
) -> None:
    """Remove cached embeddings if the encoder checkpoint is newer."""
    ckpt = Path(weights_path)
    stamp_file = embeddings_dir / ENCODER_STAMP

    if not ckpt.exists() or not embeddings_dir.exists():
        return

    ckpt_mtime = ckpt.stat().st_mtime

    if stamp_file.exists():
        try:
            recorded = float(stamp_file.read_text().strip())
        except (ValueError, OSError):
            recorded = 0.0
        if recorded >= ckpt_mtime:
            return

    import shutil

    npy_count = sum(1 for _ in embeddings_dir.rglob("*.npy"))
    if npy_count > 0:
        log.info(
            "Encoder checkpoint newer than cached embeddings — "
            "clearing %d stale .npy files",
            npy_count,
        )
        for d in list(embeddings_dir.iterdir()):
            if d.is_dir():
                shutil.rmtree(d)

    stamp_file.write_text(str(ckpt_mtime))


def embed(cfg: dict) -> dict[str, int]:
    """Generate embeddings for all chips."""
    log_gpu_status()
    device = get_device()
    batch_size = cfg["model"]["batch_size"]
    embedding_dim = cfg["model"]["embedding_dim"]
    bands_list = cfg["stac"]["bands"]

    check_vram(batch_size, min_gb=2.0)

    project_root = Path(cfg["_meta"]["project_root"])
    chips_dir = project_root / "data" / "chips"
    embeddings_dir = project_root / "data" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    weights_path = cfg["model"].get("weights_path", "")
    _invalidate_stale_embeddings(embeddings_dir, weights_path)

    encoder = _load_encoder(cfg, device)
    is_mock = isinstance(encoder, MockClayEncoder)

    if not chips_dir.exists():
        raise FileNotFoundError(f"Chips dir not found: {chips_dir} — run chip step first")

    parcel_dirs = sorted([d for d in chips_dir.iterdir() if d.is_dir()])
    is_smoke = os.environ.get("SMOKE_TEST", "0") == "1"
    if is_smoke:
        parcel_dirs = parcel_dirs[:5]
        batch_size = min(batch_size, 2)

    chip_paths: list[Path] = []
    for pd_ in parcel_dirs:
        chip_paths.extend(sorted(pd_.glob("*.npz")))

    log.info("Embedding %d chips from %d parcels (batch_size=%d)", len(chip_paths), len(parcel_dirs), batch_size)

    waves = torch.tensor([S2_WAVELENGTHS.get(b, 0.0) for b in bands_list], dtype=torch.float32)

    stats = {"embedded": 0, "skipped": 0}
    t0 = time.time()

    for batch_start in tqdm(range(0, len(chip_paths), batch_size), desc="Embedding batches"):
        batch_paths = chip_paths[batch_start : batch_start + batch_size]

        pixels_list, times_list, latlons_list, seeds_list = [], [], [], []
        valid_paths: list[Path] = []

        for cp in batch_paths:
            try:
                data = np.load(cp, allow_pickle=True)
                px = data["pixels"].astype(np.float32) / 10_000.0  # reflectance scaling
                pid = str(data["parcel_id"])
                date_str = str(data["date"])
                lat = float(data["lat"])
                lon = float(data["lon"])
            except Exception as exc:
                log.debug("Skipping %s: %s", cp, exc)
                stats["skipped"] += 1
                continue

            out_dir = embeddings_dir / pid
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{date_str}.npy"
            if out_file.exists():
                stats["embedded"] += 1
                continue

            pixels_list.append(torch.from_numpy(px))
            times_list.append(torch.tensor(_encode_time(date_str), dtype=torch.float32))
            latlons_list.append(torch.tensor(_encode_latlon(lat, lon), dtype=torch.float32))
            seeds_list.append(_parcel_seed(pid))
            valid_paths.append(cp)

        if not pixels_list:
            continue

        datacube = {
            "pixels": torch.stack(pixels_list).to(device),
            "time": torch.stack(times_list).to(device),
            "latlon": torch.stack(latlons_list).to(device),
            "gsd": torch.tensor(10.0),
            "waves": waves.to(device),
        }

        if is_mock:
            datacube["_seeds"] = torch.tensor(seeds_list)
            embeddings = encoder(datacube)
        else:
            with torch.no_grad():
                out = encoder(datacube)
                if isinstance(out, tuple):
                    encoded_patches = out[0]  # (B, 1+L, D)
                    embeddings = encoded_patches[:, 0, :]  # CLS token
                else:
                    embeddings = out

        for i, cp in enumerate(valid_paths):
            data = np.load(cp, allow_pickle=True)
            pid = str(data["parcel_id"])
            date_str = str(data["date"])
            out_dir = embeddings_dir / pid
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"{date_str}.npy", embeddings[i].cpu().numpy())
            stats["embedded"] += 1

    elapsed = time.time() - t0
    throughput = stats["embedded"] / elapsed if elapsed > 0 else 0
    log.info(
        "Embedding complete: %d embedded, %d skipped — %.1f chips/sec (%.1fs total)",
        stats["embedded"],
        stats["skipped"],
        throughput,
        elapsed,
    )
    return stats


@click.command("embed")
@click.option("--config", "config_path", default="config/monitor.yaml")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Step 4: Generate Clay embeddings for all chips."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = get_config(config_path, list(overrides) if overrides else None)
    embed(cfg)


if __name__ == "__main__":
    cli()
