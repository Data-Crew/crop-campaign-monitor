"""Step 6 (Monitor): LLM explanations for scored parcels."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import get_config
from src.explain_prompts import USER_PROMPT_TEMPLATE, build_system_prompt
from src.explain_schema import (
    Confidence,
    ConsistencyCheck,
    ExplainStatus,
    ParcelExplanation,
    fallback_explanation,
    validate_explanation,
)
from src.llm import generate_json_response, load_model_and_tokenizer, select_llm_profile

log = logging.getLogger(__name__)

OUTPUT_NAME = "parcel_explanations.parquet"


def _load_effective_thresholds(output_dir: Path, cfg: dict) -> dict[str, Any]:
    """Green/red thresholds: prefer scoring_thresholds.json from score step."""
    path = output_dir / "scoring_thresholds.json"
    scoring = cfg.get("scoring") or {}
    thresholds = scoring.get("thresholds") or {}
    green = float(thresholds.get("green", 0.15))
    red = float(thresholds.get("red", 0.30))
    method = scoring.get("threshold_method", "fixed")
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            green = float(data.get("green", green))
            red = float(data.get("red", red))
            method = data.get("method", method)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            log.warning("Could not read scoring_thresholds.json: %s", e)
    return {"green": green, "red": red, "method": method}


def build_payload(row: pd.Series, cfg: dict, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Structured JSON-serialisable input for the LLM."""
    traj_str = row.get("distance_trajectory", "[]")
    try:
        distances = json.loads(traj_str) if isinstance(traj_str, str) else traj_str
    except (json.JSONDecodeError, TypeError):
        distances = []
    if not isinstance(distances, list):
        distances = []

    season = cfg.get("season") or {}
    scoring = cfg.get("scoring") or {}
    thr_green = float(thresholds.get("green", scoring.get("thresholds", {}).get("green", 0.15)))
    thr_red = float(thresholds.get("red", scoring.get("thresholds", {}).get("red", 0.30)))

    trajectory_flags: list[str] = []
    if len(distances) >= 2:
        early_half = distances[: max(1, len(distances) // 3)]
        if early_half and max(early_half) > thr_green:
            trajectory_flags.append("early_deviation")
        if len(distances) >= 3 and all(d > thr_green for d in distances[-3:]):
            trajectory_flags.append("persistent_deviation")
        if len(distances) >= 3:
            diffs = [distances[i + 1] - distances[i] for i in range(len(distances) - 1)]
            if diffs and sum(1 for d in diffs if d > 0) > len(diffs) * 0.6:
                trajectory_flags.append("worsening_trend")

    min_obs = int(scoring.get("min_observations", 1))
    few_obs_threshold = max(3, min_obs)

    data_quality_flags: list[str] = []
    if len(distances) < few_obs_threshold:
        data_quality_flags.append("few_observations")
    if row.get("status") == "GRAY":
        data_quality_flags.append("insufficient_data")

    dist_summary: dict[str, Any] = {}
    if distances:
        arr = np.asarray(distances, dtype=np.float64)
        dist_summary = {
            "mean": round(float(np.mean(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "n_points": len(distances),
        }

    hs = row.get("health_score")
    return {
        "parcel_id": str(row["parcel_id"]),
        "crop_name": row.get("crop_name"),
        "health_score": float(hs) if hs is not None and pd.notna(hs) else None,
        "status": row.get("status"),
        "n_observations": int(row.get("n_observations") or 0),
        "max_deviation_date": row.get("max_deviation_date"),
        "distance_summary": dist_summary,
        "trajectory_flags": trajectory_flags,
        "data_quality_flags": data_quality_flags,
        "context": {
            "season_start": season.get("start_date"),
            "season_end": season.get("end_date"),
            "cadence_days": season.get("cadence_days", 16),
            "scoring_method": thresholds.get("method", scoring.get("threshold_method", "fixed")),
            "green_threshold": thr_green,
            "red_threshold": thr_red,
        },
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


def extract_json_object(text: str) -> str | None:
    """Strip markdown fences and extract the first JSON object substring."""
    s = text.strip()
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.I)
    s = s.replace("```", "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start : end + 1]


def _mock_explanation_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON for tests when llm.use_mock is true."""
    status = str(payload.get("status") or "GRAY")
    status_map = {
        "GREEN": ExplainStatus.normal,
        "YELLOW": ExplainStatus.warning,
        "RED": ExplainStatus.critical,
        "GRAY": ExplainStatus.insufficient_data,
    }
    st = status_map.get(status, ExplainStatus.insufficient_data)
    dq = payload.get("data_quality_flags") or []
    if "insufficient_data" in dq or "few_observations" in dq:
        st = ExplainStatus.insufficient_data
    summary = (
        "Mock explanation: scores reflect the structured payload only. "
        "Set llm.use_mock=false and install transformers for full LLM text."
    )
    if st == ExplainStatus.insufficient_data:
        summary = (
            "Mock explanation: insufficient data or too few observations for a reliable assessment."
        )
    exp = ParcelExplanation(
        status=st,
        summary=summary,
        possible_causes=["Mock hypothesis — replace with LLM when enabled"],
        confidence=Confidence.low,
        recommended_action="Review parcel metrics manually.",
        evidence_used=[f"parcel_id={payload.get('parcel_id')}", f"status={status}"],
        consistency_check=ConsistencyCheck.consistent,
    )
    return json.dumps(exp.model_dump())


def _run_llm_once(
    model: Any,
    tokenizer: Any,
    profile: dict[str, Any],
    llm_cfg: dict[str, Any],
    system_prompt: str,
    payload: dict[str, Any],
    temperature: float,
) -> str:
    user_text = USER_PROMPT_TEMPLATE.format(
        payload_json=json.dumps(payload, indent=2, default=str),
    )
    return generate_json_response(
        model,
        tokenizer,
        system_prompt,
        user_text,
        profile,
        llm_cfg,
        temperature=temperature,
    )


def explain(
    cfg: dict,
    parcel_ids: list[str] | None = None,
    force_mock: bool = False,
) -> dict[str, Any]:
    """Generate LLM explanations and write parcel_explanations.parquet."""
    llm_cfg = cfg.get("llm") or {}
    if not llm_cfg.get("enabled", False):
        log.info("LLM explanations disabled (llm.enabled=false); skipping explain step.")
        return {"skipped": True, "reason": "disabled", "n_written": 0}

    output_dir = Path(cfg["output"]["dir"])
    scores_path = output_dir / "parcel_scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found: {scores_path} — run score step first")

    scored = gpd.read_parquet(scores_path)
    if parcel_ids:
        pids = {str(p) for p in parcel_ids}
        scored = scored[scored["parcel_id"].astype(str).isin(pids)]

    max_parcels = llm_cfg.get("max_parcels")
    if max_parcels is not None:
        scored = scored.head(int(max_parcels))

    thresholds = _load_effective_thresholds(output_dir, cfg)
    out_path = output_dir / OUTPUT_NAME

    existing: pd.DataFrame | None = None
    if out_path.exists() and llm_cfg.get("skip_if_unchanged"):
        existing = pd.read_parquet(out_path)

    use_mock = bool(force_mock or llm_cfg.get("use_mock", False))
    model = None
    tokenizer = None
    profile_name = ""
    profile: dict[str, Any] = {}
    if not use_mock:
        try:
            profile_name, profile = select_llm_profile(cfg)
            model, tokenizer = load_model_and_tokenizer(profile)
        except Exception as e:
            log.warning("LLM load failed (%s); using mock explainer", e)
            use_mock = True

    system_prompt = build_system_prompt(include_few_shot=True)

    rows: list[dict[str, Any]] = []
    n_fail = 0

    for _, row in scored.iterrows():
        payload = build_payload(row, cfg, thresholds)
        phash = _payload_hash(payload)
        pid = str(row["parcel_id"])
        pst = str(row.get("status") or "GRAY")

        if existing is not None and llm_cfg.get("skip_if_unchanged"):
            prev = existing[existing["parcel_id"].astype(str) == pid]
            if not prev.empty and str(prev.iloc[0].get("payload_hash", "")) == phash:
                rows.append(prev.iloc[0].to_dict())
                continue

        exp: ParcelExplanation

        if use_mock:
            raw_out = _mock_explanation_json(payload)
            js = extract_json_object(raw_out) or raw_out
            validated = validate_explanation(js, pst)
            exp = validated if validated is not None else fallback_explanation(pst, "mock validation failed")
            rows.append(
                {
                    "parcel_id": pid,
                    "explanation_json": json.dumps(exp.model_dump()),
                    "llm_status": exp.status.value,
                    "llm_confidence": exp.confidence.value,
                    "llm_model": "mock",
                    "profile": "mock",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "payload_hash": phash,
                }
            )
            continue

        assert model is not None and tokenizer is not None
        t0 = float(llm_cfg.get("temperature", 0.1))
        raw_text = _run_llm_once(
            model, tokenizer, profile, llm_cfg, system_prompt, payload, t0,
        )
        js = extract_json_object(raw_text)
        validated = validate_explanation(js or "", pst) if js else None
        if validated is None:
            t1 = float(llm_cfg.get("retry_temperature", 0.35))
            log.info("Retrying LLM with temperature %.2f for parcel %s", t1, pid)
            raw_text = _run_llm_once(
                model, tokenizer, profile, llm_cfg, system_prompt, payload, t1,
            )
            js = extract_json_object(raw_text)
            validated = validate_explanation(js or "", pst) if js else None
        if validated is None:
            n_fail += 1
            exp = fallback_explanation(pst, "JSON validation failed after retry")
        else:
            exp = validated

        rows.append(
            {
                "parcel_id": pid,
                "explanation_json": json.dumps(exp.model_dump()),
                "llm_status": exp.status.value,
                "llm_confidence": exp.confidence.value,
                "llm_model": str(profile.get("model_id", "unknown")),
                "profile": profile_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "payload_hash": phash,
            }
        )

    if not rows:
        log.warning("No explanation rows produced")
        return {"skipped": False, "n_written": 0, "n_validation_failures": n_fail}

    out_df = pd.DataFrame(rows)
    if parcel_ids and out_path.exists():
        old = pd.read_parquet(out_path)
        drop = set(str(p) for p in parcel_ids)
        old = old[~old["parcel_id"].astype(str).isin(drop)]
        out_df = pd.concat([old, out_df], ignore_index=True)

    out_df.to_parquet(out_path, index=False)
    log.info(
        "Saved %s (%d parcels, %d validation fallbacks)",
        out_path,
        len(rows),
        n_fail,
    )
    return {
        "skipped": False,
        "n_written": len(rows),
        "n_validation_failures": n_fail,
        "path": str(out_path),
    }


def main(
    config_path: str | None = None,
    overrides: list[str] | None = None,
    parcel_id: str | None = None,
    force_mock: bool = False,
) -> dict[str, Any]:
    path = config_path or "config/monitor.yaml"
    cfg = get_config(path, overrides)
    pids = [parcel_id] if parcel_id else None
    return explain(cfg, parcel_ids=pids, force_mock=force_mock)


@click.command("explain")
@click.option("--config", "config_path", default="config/monitor.yaml")
@click.option("--parcel-id", "parcel_id", default=None, help="Explain only this parcel.")
@click.option("--mock", "force_mock", is_flag=True, help="Force mock explainer (no Transformers load).")
@click.argument("overrides", nargs=-1)
def cli(config_path: str, parcel_id: str | None, force_mock: bool, overrides: tuple[str, ...]) -> None:
    """Step 6 (Monitor): Generate LLM explanations for scored parcels."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = get_config(config_path, list(overrides) if overrides else None)
    pids = [parcel_id] if parcel_id else None
    explain(cfg, parcel_ids=pids, force_mock=force_mock)


if __name__ == "__main__":
    cli()
