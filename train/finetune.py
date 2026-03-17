"""Fine-tune Clay Foundation Model for crop classification via TerraTorch.

Uses TerraTorch's ``ClassificationTask`` to wrap Clay as a backbone with a
linear classification head.  Supports staged training: first train only the
head with the backbone frozen, then unfreeze the backbone at a lower
learning rate.

Requires ``terratorch`` — exits with a clear error and installation
instructions if not available.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

from src.config import get_config
from src.gpu import get_device, log_gpu_status, resolve_gpu

log = logging.getLogger(__name__)


def _check_terratorch() -> None:
    """Verify TerraTorch is installed."""
    try:
        import terratorch  # noqa: F401
    except ImportError:
        log.error(
            "TerraTorch is required for fine-tuning but is not installed.\n"
            "Install it with:\n"
            "  pip install terratorch\n"
            "  OR\n"
            "  pip install git+https://github.com/terrastackai/terratorch.git\n"
            "  OR\n"
            "  conda install -c conda-forge terratorch\n\n"
            "Then re-run the training pipeline."
        )
        sys.exit(1)


def finetune(cfg: dict) -> None:
    """Run the fine-tuning training loop."""
    _check_terratorch()

    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from terratorch.tasks import ClassificationTask

    from train.datamodule import CropDataModule

    log_gpu_status()

    training_cfg = cfg["training"]
    base_model_cfg = cfg["base_model"]
    data_cfg = cfg["data"]
    output_cfg = cfg["output"]

    dataset_dir = Path(data_cfg["dataset_dir"])
    class_mapping_path = dataset_dir / "class_mapping.json"
    if not class_mapping_path.exists():
        raise FileNotFoundError(
            f"class_mapping.json not found at {class_mapping_path} — run prepare_dataset first"
        )
    with open(class_mapping_path) as f:
        class_map = json.load(f)
    class_names = [class_map[str(i)] for i in range(len(class_map))]
    num_classes = len(class_names)

    log.info("Fine-tuning Clay for %d classes: %s", num_classes, class_names)

    # DataModule
    dm = CropDataModule(
        dataset_dir=str(dataset_dir),
        batch_size=training_cfg["batch_size"],
        num_workers=cfg["model"]["num_workers"] if "model" in cfg else 4,
        augmentations=training_cfg.get("augmentations"),
    )

    # Build task via TerraTorch + ClayModelFactory
    bands = base_model_cfg.get("bands", [
        "blue", "green", "red", "nir",
        "rededge1", "rededge2", "rededge3", "nir08",
        "swir16", "swir22",
    ])

    weights_path = Path(base_model_cfg["weights_path"])
    use_pretrained = not weights_path.exists()
    if use_pretrained:
        log.info("Local weights not found at %s — downloading from HuggingFace", weights_path)

    chip_size = data_cfg.get("chip_size_px", 224)

    model_args = {
        "backbone": base_model_cfg["backbone"],
        "decoder": "IdentityDecoder",
        "in_channels": len(bands),
        "bands": bands,
        "num_classes": num_classes,
        "pretrained": use_pretrained,
        "backbone_img_size": chip_size,
    }
    if not use_pretrained:
        model_args["backbone_checkpoint_path"] = str(weights_path)

    task = ClassificationTask(
        model_args=model_args,
        model_factory="ClayModelFactory",
        loss="ce",
        lr=training_cfg["lr"],
        class_names=class_names,
        freeze_backbone=True,
    )

    # Callbacks
    checkpoint_dir = Path(output_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monitor_metric = training_cfg.get("checkpoint_monitor", "val/loss")

    callbacks = [
        EarlyStopping(
            monitor=monitor_metric,
            patience=training_cfg["early_stopping_patience"],
            mode="min",
        ),
        ModelCheckpoint(
            monitor=monitor_metric,
            dirpath=str(checkpoint_dir),
            filename="clay-finetuned-crops",
            save_top_k=1,
            mode="min",
        ),
    ]

    log_dir = Path(output_cfg.get("training_logs", "data/training/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_vis:
        devices = [0]
    else:
        gpu_identifier = cfg.get("gpu", {}).get("device", "0")
        devices = [resolve_gpu(gpu_identifier)]

    trainer = L.Trainer(
        max_epochs=training_cfg.get("freeze_backbone_epochs", 5),
        accelerator="gpu",
        devices=devices,
        callbacks=callbacks,
        log_every_n_steps=10,
        default_root_dir=str(log_dir),
    )

    # Phase 1: frozen backbone
    log.info("Phase 1: Training classification head (backbone frozen) for %d epochs",
             training_cfg.get("freeze_backbone_epochs", 5))
    trainer.fit(task, datamodule=dm)

    # Phase 2: unfreeze backbone
    log.info("Phase 2: Unfreezing backbone at lr=%s", training_cfg.get("unfreeze_backbone_lr", 1e-5))
    task.freeze_backbone = False
    for param in task.model.parameters():
        param.requires_grad = True

    unfreeze_lr = training_cfg.get("unfreeze_backbone_lr", 1e-5)
    remaining_epochs = training_cfg["epochs"] - training_cfg.get("freeze_backbone_epochs", 5)

    if remaining_epochs > 0:
        trainer2 = L.Trainer(
            max_epochs=remaining_epochs,
            accelerator="gpu",
            devices=devices,
            callbacks=callbacks,
            log_every_n_steps=10,
            default_root_dir=str(log_dir),
        )
        # Update lr for backbone params
        for pg in task.optimizers().param_groups:
            pg["lr"] = unfreeze_lr

        trainer2.fit(task, datamodule=dm)

    # Test
    log.info("Running test evaluation...")
    trainer.test(task, datamodule=dm)

    # Save metrics
    metrics = {
        "num_classes": num_classes,
        "class_names": class_names,
        "epochs_trained": training_cfg["epochs"],
    }
    with open(log_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    log.info("Fine-tuning complete. Checkpoint at %s", checkpoint_dir)


@click.command("finetune")
@click.option("--config", "config_path", default="config/train.yaml")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Fine-tune Clay backbone for crop classification."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = get_config(config_path, list(overrides) if overrides else None)
    finetune(cfg)


if __name__ == "__main__":
    cli()
