"""Configuration loader with YAML parsing, validation, and CLI overrides."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import click
import yaml

log = logging.getLogger(__name__)

REQUIRED_MONITOR_SECTIONS = {
    "region": ["name", "fields_dir", "tiles"],
    "season": ["start_date", "end_date"],
    "stac": ["catalog_url", "collection", "bands"],
    "chips": ["size_px", "resolution_m"],
    "model": ["embedding_dim", "batch_size"],
    "scoring": ["method", "thresholds"],
    "output": ["dir"],
}

REQUIRED_TRAIN_SECTIONS = {
    "base_model": ["backbone", "weights_path"],
    "data": ["fields_dir", "tiles", "chips_dir", "dataset_dir"],
    "training": ["task", "num_classes", "epochs", "batch_size", "lr"],
    "output": ["checkpoint_dir", "encoder_checkpoint"],
}


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """Set a nested key in a dict, creating intermediate dicts as needed."""
    import json as _json

    for k in keys[:-1]:
        d = d.setdefault(k, {})
    raw = value
    if isinstance(raw, str):
        if raw.startswith(("[", "{")):
            try:
                raw = _json.loads(raw)
            except (ValueError, _json.JSONDecodeError):
                pass
        elif raw.lower() in ("true", "false"):
            raw = raw.lower() == "true"
        elif raw.lower() in ("null", "none"):
            raw = None
        else:
            try:
                raw = int(raw)
            except ValueError:
                try:
                    raw = float(raw)
                except ValueError:
                    pass
    d[keys[-1]] = raw


def _resolve_paths(cfg: dict, base_dir: Path, keys: list[str] | None = None) -> None:
    """Resolve known relative path fields to absolute using *base_dir*."""
    path_fields = keys or [
        "region.fields_dir",
        "model.weights_path",
        "model.fallback_weights",
        "output.dir",
        "base_model.weights_path",
        "data.fields_dir",
        "data.chips_dir",
        "data.dataset_dir",
        "output.checkpoint_dir",
        "output.finetuned_checkpoint",
        "output.encoder_checkpoint",
        "output.training_logs",
    ]
    for dotpath in path_fields:
        parts = dotpath.split(".")
        node = cfg
        try:
            for p in parts[:-1]:
                node = node[p]
            val = node[parts[-1]]
        except (KeyError, TypeError):
            continue
        if isinstance(val, str) and not os.path.isabs(val):
            node[parts[-1]] = str(base_dir / val)


def _validate(cfg: dict, schema: dict[str, list[str]]) -> list[str]:
    errors: list[str] = []
    for section, fields in schema.items():
        if section not in cfg:
            errors.append(f"Missing top-level section: '{section}'")
            continue
        for field in fields:
            if field not in cfg[section]:
                errors.append(f"Missing field '{field}' in section '{section}'")
    return errors


def get_config(
    path: str,
    overrides: list[str] | None = None,
    resolve_paths: bool = True,
) -> dict:
    """Load a YAML config, validate, apply CLI overrides, and resolve paths.

    Parameters
    ----------
    path:
        Path to the YAML config file.
    overrides:
        Optional list of ``"key.subkey=value"`` strings that override config values.
    resolve_paths:
        If *True*, convert relative path fields to absolute paths based on the
        project root (parent of the config file's directory).
    """
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for override in overrides:
            if "=" not in override:
                log.warning("Ignoring malformed override (no '='): %s", override)
                continue
            key, val = override.lstrip("-").split("=", 1)
            _set_nested(cfg, key.split("."), val)

    is_train = "base_model" in cfg
    schema = REQUIRED_TRAIN_SECTIONS if is_train else REQUIRED_MONITOR_SECTIONS
    errors = _validate(cfg, schema)
    if errors:
        for e in errors:
            log.error("Config validation: %s", e)
        raise ValueError(
            f"Config validation failed with {len(errors)} error(s) — see log above"
        )

    project_root = cfg_path.parent.parent
    if resolve_paths:
        _resolve_paths(cfg, project_root)

    cfg["_meta"] = {
        "config_path": str(cfg_path),
        "project_root": str(project_root),
    }
    log.info("Loaded config from %s", cfg_path)
    return cfg


@click.command("config")
@click.option("--config", "config_path", default="config/monitor.yaml", help="Config file path")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, overrides: tuple[str, ...]) -> None:
    """Print resolved configuration."""
    import json

    logging.basicConfig(level=logging.INFO)
    cfg = get_config(config_path, list(overrides) if overrides else None)
    click.echo(json.dumps(cfg, indent=2, default=str))


if __name__ == "__main__":
    cli()
