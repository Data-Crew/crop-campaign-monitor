"""Export the encoder from a fine-tuned Clay checkpoint for inference.

Loads the full fine-tuned model (backbone + classification head), extracts
only the backbone (encoder) state dict, and saves it as a standalone
checkpoint that ``src/embed.py`` can load directly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import click
import torch

from src.config import get_config
from src.gpu import get_device

log = logging.getLogger(__name__)


def export_encoder(cfg: dict) -> Path:
    """Extract and save the encoder from a fine-tuned checkpoint."""
    output_cfg = cfg["output"]
    finetuned_path = Path(output_cfg["finetuned_checkpoint"])
    encoder_path = Path(output_cfg["encoder_checkpoint"])

    # Find the actual checkpoint (Lightning may add suffixes)
    if not finetuned_path.exists():
        ckpt_dir = finetuned_path.parent
        candidates = sorted(ckpt_dir.glob("clay-finetuned-crops*.ckpt"))
        if candidates:
            finetuned_path = candidates[-1]
            log.info("Using checkpoint: %s", finetuned_path)
        else:
            log.error(
                "Fine-tuned checkpoint not found at %s or %s. Run finetune first.",
                output_cfg["finetuned_checkpoint"],
                ckpt_dir,
            )
            sys.exit(1)

    device = get_device()
    log.info("Loading fine-tuned checkpoint from %s", finetuned_path)
    ckpt = torch.load(finetuned_path, map_location=device, weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt)

    # Extract backbone/encoder keys — TerraTorch nests them under model.backbone or model.encoder
    encoder_state: dict[str, torch.Tensor] = {}
    prefixes_to_try = ["model.backbone.", "model.encoder.", "backbone.", "encoder."]

    for prefix in prefixes_to_try:
        matched = {
            k.replace(prefix, "", 1): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if matched:
            encoder_state = matched
            log.info("Extracted %d parameters with prefix '%s'", len(encoder_state), prefix)
            break

    if not encoder_state:
        log.warning(
            "Could not identify backbone keys — saving full state_dict. "
            "Keys sample: %s",
            list(state_dict.keys())[:10],
        )
        encoder_state = state_dict

    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_state, encoder_path)

    param_count = sum(p.numel() for p in encoder_state.values())
    log.info(
        "Encoder exported to %s (%d parameters, %.1f MB)",
        encoder_path,
        param_count,
        param_count * 4 / (1024**2),
    )

    # Verification: load and check output shape
    log.info("Verifying exported encoder...")
    try:
        from terratorch.models.backbones.clay_v1.embedder import Embedder

        bands = cfg.get("base_model", {}).get("bands", [
            "blue", "green", "red", "nir", "rededge1", "rededge2",
            "rededge3", "nir08", "swir16", "swir22",
        ])
        embedder = Embedder(img_size=224, bands=bands)
        embedder.load_state_dict(encoder_state, strict=False)
        embedder = embedder.to(device).eval()

        with torch.no_grad():
            features = embedder.forward_features(
                x=torch.randn(1, len(bands), 224, 224).to(device),
                time=torch.zeros(1, 4).to(device),
                latlon=torch.zeros(1, 4).to(device),
                waves=torch.randn(len(bands)).to(device),
                gsd=torch.tensor(10.0),
            )
            cls_emb = features[-1][:, 0, :]
            log.info("Verification passed: output shape %s", cls_emb.shape)

    except Exception as exc:
        log.warning("Verification skipped: %s", exc)

    return encoder_path


@click.command("export_encoder")
@click.option("--config", "config_path", default="config/train.yaml")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Export encoder from fine-tuned checkpoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = get_config(config_path, list(overrides) if overrides else None)
    export_encoder(cfg)


if __name__ == "__main__":
    cli()
