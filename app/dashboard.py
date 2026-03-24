"""Crop Campaign Monitor — Streamlit Dashboard.

Provides a two-tab layout (Scripting / Outputs) with a config-only sidebar.
Scripting tab contains three-phase pipeline controls and execution history.
Outputs tab visualises scored parcels via Kepler.gl maps, Plotly charts, and
action tables, filtered by the selected region.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
FIELDS_DIR = DATA_DIR / "fields"
MODEL_DIR = DATA_DIR / "model"

st.set_page_config(page_title="Crop Campaign Monitor", layout="wide")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_APP_CSS = """
<style>
/* ---- App background ---- */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stAppViewContainer"] > section > .main,
[data-testid="stAppViewContainer"] > section > .main > .block-container {
    background-color: #f0f2f4 !important;
}
[data-testid="stSidebar"],
section[data-testid="stSidebar"] {
    background-color: #e8eaed !important;
}

/* ---- Dark console styling for expanders ---- */
[data-testid="stExpander"] {
    background-color: #1e2128 !important;
    border: 1px solid #3a3f4b !important;
    border-radius: 8px !important;
    overflow: hidden;
}
[data-testid="stExpander"] details {
    background-color: #1e2128 !important;
}
[data-testid="stExpander"] summary {
    background-color: #262b34 !important;
    padding: 0.5rem 1rem !important;
}
[data-testid="stExpander"] summary p {
    color: #c8cdd5 !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary svg {
    fill: #8b929e !important;
}
[data-testid="stExpander"] [data-testid="stVerticalBlock"] {
    gap: 0.3rem !important;
}
[data-testid="stExpander"] [data-testid="stCaptionContainer"] p {
    color: #8b929e !important;
}
[data-testid="stExpander"] [data-testid="stMetricLabel"] p {
    color: #8b929e !important;
}
[data-testid="stExpander"] [data-testid="stMetricValue"] {
    color: #e0e4ea !important;
}
[data-testid="stExpander"] label p,
[data-testid="stExpander"] label span {
    color: #b0b8c4 !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: #b0b8c4 !important;
}
/* Button text white in expanders so Run All / Run Step are clearly readable */
[data-testid="stExpander"] button p,
[data-testid="stExpander"] button span {
    color: #ffffff !important;
}
/* Space below the live log so it doesn't sit flush against the next expander */
.live-log-wrap { margin-bottom: 1rem; }
</style>
"""

_TERMINAL_CSS = """
<style>
.pipeline-terminal {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', 'Monaco', monospace;
    font-size: 12px;
    line-height: 1.45;
    padding: 14px 16px;
    border-radius: 8px;
    max-height: 70vh;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    border: 1px solid #333;
}
.pipeline-terminal .t-info  { color: #6a9955; }
.pipeline-terminal .t-warn  { color: #dcdcaa; }
.pipeline-terminal .t-err   { color: #f44747; }
.pipeline-terminal .t-head  { color: #569cd6; font-weight: bold; }
.pipeline-terminal .t-prog  { color: #ce9178; }
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def _discover_tiles() -> list[str]:
    if not FIELDS_DIR.exists():
        return []
    return sorted(
        d.name for d in FIELDS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


@st.cache_data(ttl=30, show_spinner=False)
def _discover_fragments(tile_id: str) -> list[str]:
    tile_dir = FIELDS_DIR / tile_id
    if not tile_dir.exists():
        return []
    frags: list[str] = []
    for f in sorted(tile_dir.glob(f"{tile_id}_fragment_*.geojson")):
        parts = f.stem.split("_fragment_")
        if len(parts) == 2:
            frags.append(parts[1])
    return frags


def _model_status() -> tuple[str, str]:
    """Return (color_key, description)."""
    encoder_path = MODEL_DIR / "clay-crop-encoder.ckpt"
    base_path = MODEL_DIR / "clay-v1-base.ckpt"
    if encoder_path.exists():
        mtime = datetime.fromtimestamp(encoder_path.stat().st_mtime).strftime("%Y-%m-%d")
        return "green", f"Fine-tuned encoder ({mtime})"
    if base_path.exists():
        return "yellow", "Base Clay model only"
    return "red", "No model \u2014 mock embeddings"


@st.cache_data(ttl=10, show_spinner=False)
def _chip_count() -> int:
    chips_dir = DATA_DIR / "chips"
    return sum(1 for _ in chips_dir.rglob("*.npz")) if chips_dir.exists() else 0


@st.cache_data(ttl=10, show_spinner=False)
def _parcels_with_chips_count() -> int:
    """Number of distinct parcels that have at least one chip (.npz)."""
    chips_dir = DATA_DIR / "chips"
    if not chips_dir.exists():
        return 0
    return sum(
        1 for d in chips_dir.iterdir()
        if d.is_dir() and any(d.glob("*.npz"))
    )


@st.cache_data(ttl=10, show_spinner=False)
def _scene_count() -> int:
    """Number of scene dates (from tile_index)."""
    path = DATA_DIR / "tiles" / "tile_index.parquet"
    if not path.exists():
        return 0
    try:
        df = pd.read_parquet(path)
        if "date" in df.columns:
            return int(df["date"].nunique())
        return len(df)
    except Exception:
        return 0


@st.cache_data(ttl=10, show_spinner=False)
def _embedding_count() -> int:
    emb_dir = DATA_DIR / "embeddings"
    return sum(1 for _ in emb_dir.rglob("*.npy")) if emb_dir.exists() else 0


@st.cache_data(ttl=10, show_spinner=False)
def _parcel_count() -> int:
    path = OUTPUT_DIR / "parcels_labeled.parquet"
    if path.exists():
        try:
            return len(gpd.read_parquet(path))
        except Exception:
            return 0
    return 0


@st.cache_data(ttl=10, show_spinner=False)
def _load_scores() -> gpd.GeoDataFrame | None:
    path = OUTPUT_DIR / "parcel_scores.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    return None


@st.cache_data(ttl=10, show_spinner=False)
def _load_labeled_parcels() -> gpd.GeoDataFrame | None:
    path = OUTPUT_DIR / "parcels_labeled.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    return None


def _load_report() -> dict[str, Any] | None:
    path = OUTPUT_DIR / "campaign_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _load_scoring_thresholds() -> dict[str, float]:
    path = OUTPUT_DIR / "scoring_thresholds.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"green": 0.15, "red": 0.30}


def _format_overrides(tiles: list[str], fragments: list[str]) -> str:
    """Build CLI override strings from tile/fragment selections.

    Each value is single-quoted so brackets survive bash glob expansion
    when the command runs via ``subprocess.Popen(shell=True)``.
    """
    import json as _json

    parts: list[str] = []
    if tiles:
        val = _json.dumps(tiles, separators=(",", ":"))
        parts.append(f"region.tiles='{val}'")
    if fragments:
        val = _json.dumps(fragments, separators=(",", ":"))
        parts.append(f"region.fragments='{val}'")
    return " ".join(parts)


def _filter_by_selection(
    gdf: gpd.GeoDataFrame, ctx: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Filter a GeoDataFrame by the tiles/fragments selected in the sidebar."""
    frags = ctx.get("fragments", [])
    if frags and "fragment_id" in gdf.columns:
        gdf = gdf[gdf["fragment_id"].isin(frags)]
    return gdf


# ---------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------
def _colorize_line(line: str) -> str:
    """Apply terminal-like colour classes to a log line."""
    import html

    escaped = html.escape(line.rstrip())
    if not escaped:
        return ""
    if escaped.startswith(("\u2554", "\u255a", "\u2551")):
        return f'<span class="t-head">{escaped}</span>'
    if "ERROR" in escaped or "Traceback" in escaped or "raise " in escaped:
        return f'<span class="t-err">{escaped}</span>'
    if "WARNING" in escaped:
        return f'<span class="t-warn">{escaped}</span>'
    if "INFO" in escaped or escaped.startswith("["):
        return f'<span class="t-info">{escaped}</span>'
    if "%" in escaped or "it/s" in escaped or "s/it" in escaped:
        return f'<span class="t-prog">{escaped}</span>'
    return escaped


def _run_script(
    script_cmd: str,
    step_name: str = "Pipeline",
    live_ph: "st.delta_generator.DeltaGenerator | None" = None,
) -> None:
    """Execute a bash command, streaming output into *live_ph* (full-width)."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    lines: list[str] = []
    ph = live_ph if live_ph is not None else st.empty()

    try:
        proc = subprocess.Popen(
            script_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        for line in proc.stdout:
            lines.append(line)
            tail = lines[-80:]
            colored = "\n".join(_colorize_line(l) for l in tail if l.strip())
            ph.markdown(
                _TERMINAL_CSS
                + f'<div class="live-log-wrap"><div class="pipeline-terminal">{colored}</div></div>',
                unsafe_allow_html=True,
            )
        proc.wait()

        status = "ok" if proc.returncode == 0 else "error"
        log_entry = {
            "step_name": step_name,
            "lines": lines,
            "status": status,
            "exit_code": proc.returncode,
            "timestamp": timestamp,
        }
        st.session_state.setdefault("pipeline_logs", []).append(log_entry)
        st.rerun()

    except Exception as exc:
        log_entry = {
            "step_name": step_name,
            "lines": lines + [f"Exception: {exc}"],
            "status": "error",
            "exit_code": -1,
            "timestamp": timestamp,
        }
        st.session_state.setdefault("pipeline_logs", []).append(log_entry)
        st.rerun()


def _render_log_entry(log_data: dict[str, Any]) -> None:
    """Render a single log entry as an expander."""
    step_name = log_data.get("step_name", "Pipeline")
    timestamp = log_data.get("timestamp", "")
    status = log_data["status"]

    label = f"{step_name} \u2014 {timestamp}"
    if status == "ok":
        label += " (completed)"
    elif status == "error":
        label += " (failed)"

    with st.expander(label, expanded=(status == "error")):
        lines = log_data["lines"]
        colored = "\n".join(_colorize_line(l) for l in lines if l.strip())
        st.markdown(
            _TERMINAL_CSS + f'<div class="pipeline-terminal">{colored}</div>',
            unsafe_allow_html=True,
        )
        if status == "ok":
            st.success("Completed successfully")
        elif status == "error":
            st.error(f"Failed (exit code {log_data.get('exit_code', '?')})")


def _render_execution_history() -> None:
    """Render all accumulated pipeline logs, newest first."""
    logs = st.session_state.get("pipeline_logs", [])
    if not logs:
        st.caption("No executions yet.")
        return

    for log_data in reversed(logs):
        _render_log_entry(log_data)

    if st.button("Clear History", key="clear_history"):
        st.session_state["pipeline_logs"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# Kepler.gl
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent


def _load_kepler_config(name: str) -> dict[str, Any]:
    path = _APP_DIR / name
    with open(path) as f:
        return json.load(f)


def _render_kepler_map(
    gdf: gpd.GeoDataFrame,
    config_name: str,
    height: int = 550,
    zoom: int = 11,
) -> None:
    """Render a Kepler.gl map inside Streamlit."""
    try:
        from keplergl import KeplerGl
        from streamlit_keplergl import keplergl_static

        gdf_wgs = gdf.to_crs(epsg=4326) if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf

        config = _load_kepler_config(config_name)
        bounds = gdf_wgs.total_bounds
        config["config"]["mapState"]["latitude"] = float((bounds[1] + bounds[3]) / 2)
        config["config"]["mapState"]["longitude"] = float((bounds[0] + bounds[2]) / 2)
        config["config"]["mapState"]["zoom"] = zoom

        kmap = KeplerGl(height=height, config=config)
        kmap.add_data(data=gdf_wgs, name="parcels")
        keplergl_static(kmap, center_map=True)
    except ImportError:
        st.warning(
            "Kepler.gl not available. Install `keplergl` and `streamlit-keplergl` "
            "for interactive maps."
        )
    except Exception as exc:
        st.warning(f"Kepler.gl rendering error: {exc}")


def _render_kepler_map_similarity(
    gdf: gpd.GeoDataFrame,
    height: int = 360,
    zoom: int = 12,
) -> None:
    """Kepler mini-map for query parcel vs neighbors (color by map_role)."""
    try:
        from keplergl import KeplerGl
        from streamlit_keplergl import keplergl_static

        gdf_wgs = gdf.to_crs(epsg=4326) if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf

        config = _load_kepler_config("kepler_similarity_config.json")
        bounds = gdf_wgs.total_bounds
        config["config"]["mapState"]["latitude"] = float((bounds[1] + bounds[3]) / 2)
        config["config"]["mapState"]["longitude"] = float((bounds[0] + bounds[2]) / 2)
        config["config"]["mapState"]["zoom"] = zoom

        kmap = KeplerGl(height=height, config=config)
        kmap.add_data(data=gdf_wgs, name="parcels")
        keplergl_static(kmap, center_map=True)
    except ImportError:
        st.warning(
            "Kepler.gl not available. Install `keplergl` and `streamlit-keplergl` "
            "for interactive maps."
        )
    except Exception as exc:
        st.warning(f"Kepler.gl rendering error: {exc}")


@st.cache_resource(show_spinner=False)
def _parcel_similarity_index(_meta_mtime: float) -> Any:
    """Load FAISS-backed index; None if files missing or unloadable."""
    from src.query import load_parcel_index

    try:
        return load_parcel_index(output_dir=OUTPUT_DIR)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Sidebar — configuration only
# ---------------------------------------------------------------------------
def sidebar() -> dict[str, Any]:
    st.sidebar.title("Crop Campaign Monitor")

    # -- Region selection --
    st.sidebar.subheader("Region")
    available_tiles = _discover_tiles()
    selected_tiles = st.sidebar.multiselect(
        "Tiles", available_tiles, default=available_tiles,
    )

    all_frags: list[str] = []
    for t in selected_tiles:
        all_frags.extend(_discover_fragments(t))
    all_frags = sorted(set(all_frags))

    select_all = st.sidebar.checkbox("All fragments", value=True)
    if select_all:
        selected_frags = all_frags
    else:
        selected_frags = st.sidebar.multiselect("Fragments", all_frags)

    overrides = _format_overrides(selected_tiles, selected_frags)

    # -- GPU --
    st.sidebar.markdown("---")
    st.sidebar.subheader("GPU")
    try:
        from src.gpu import discover_gpus
        gpus = discover_gpus()
    except Exception:
        gpus = []

    if gpus:
        gpu_labels = [f"{g['name']} ({g['vram_gb']} GB)" for g in gpus]
        gpu_names = [g["name"] for g in gpus]
        sel = st.sidebar.radio("GPU Device", gpu_labels, index=0)
        gpu_name = gpu_names[gpu_labels.index(sel)]
    else:
        gpu_name = ""
        st.sidebar.warning("No GPUs detected")

    gpu_env = f"GPU_NAME='{gpu_name}'" if gpu_name else ""

    # -- Cache --
    st.sidebar.markdown("---")
    st.sidebar.subheader("Cache")

    n_emb = _embedding_count()
    n_chips = _chip_count()
    col_emb, col_chips = st.sidebar.columns(2)
    col_emb.metric("Embeddings", n_emb)
    col_chips.metric("Chips", n_chips)

    clear_what = st.sidebar.radio(
        "Clear", ["Embeddings", "Embeddings + Chips"],
        horizontal=True, label_visibility="collapsed",
    )
    if st.sidebar.button("Clear Cache", type="secondary"):
        emb_dir = DATA_DIR / "embeddings"
        chips_dir = DATA_DIR / "chips"
        if emb_dir.exists():
            shutil.rmtree(emb_dir)
            emb_dir.mkdir()
        if clear_what == "Embeddings + Chips" and chips_dir.exists():
            shutil.rmtree(chips_dir)
            chips_dir.mkdir()
        stamp = emb_dir / ".encoder_mtime"
        if stamp.exists():
            stamp.unlink()
        st.sidebar.success(f"Cleared {clear_what.lower()}")
        st.rerun()

    also_clear_labeled = st.sidebar.checkbox(
        "Include parcel catalog",
        value=False,
        key="clear_also_labeled",
        help="Also deletes output/parcels_labeled.parquet",
    )
    if also_clear_labeled:
        st.sidebar.warning(
            "You will need to re-run **Phase 1 \u2192 1-Ingest** before running Phase 3."
        )
    if st.sidebar.button("Clear Results", type="secondary"):
        files_to_clear = [
            "parcel_scores.parquet",
            "campaign_report.json",
            "scoring_thresholds.json",
            "parcels.faiss",
            "parcels_index_meta.parquet",
            "parcels_vectors.npz",
        ]
        if also_clear_labeled:
            files_to_clear.append("parcels_labeled.parquet")
        for fname in files_to_clear:
            p = OUTPUT_DIR / fname
            if p.exists():
                p.unlink()
        st.sidebar.success("Cleared results")
        st.rerun()

    return {
        "tiles": selected_tiles,
        "fragments": selected_frags,
        "gpu_name": gpu_name,
        "gpu_env": gpu_env,
        "overrides": overrides,
    }


# ---------------------------------------------------------------------------
# Tab: Scripting
# ---------------------------------------------------------------------------
def _enqueue_run(cmd: str, step_name: str, phase: str) -> None:
    """Store a pipeline command in session state and trigger a rerun.

    *phase* identifies which placeholder to stream into (``"dp"``,
    ``"tr"``, or ``"mo"``).
    """
    st.session_state["pending_run"] = {
        "cmd": cmd, "step_name": step_name, "phase": phase,
    }
    st.rerun()


def tab_scripting(ctx: dict[str, Any]) -> None:
    gpu_env = ctx["gpu_env"]
    overrides = ctx["overrides"]

    n_parcels = _parcel_count()
    n_chips = _chip_count()
    n_parcels_with_chips = _parcels_with_chips_count()
    n_scenes = _scene_count()
    n_emb = _embedding_count()

    # -- Phase 1: Data Preparation --
    with st.expander("Phase 1: Data Preparation", expanded=False):
        st.caption("Load parcels, fetch satellite imagery, and extract image chips.")

        default_max = 1000 if (n_parcels and n_parcels >= 1000) else (min(200, n_parcels) if n_parcels and n_parcels > 0 else 200)
        c_input, c_cat, c_pwc, c_cf, c_sc = st.columns([2, 1, 1, 1, 1])
        with c_input:
            max_parcels = st.number_input(
                "Max parcels to chip",
                min_value=10,
                max_value=n_parcels or 100_000,
                value=default_max,
                step=50,
                key="dp_max_parcels",
                help="Maximum number of parcels for which chips (and later embeddings) will be produced. "
                     "The rest of the catalog will have no chips and will appear as GRAY in the monitor. "
                     "With \"All fragments\" the catalog can be very large; this limit controls how many are processed.",
            )
        c_cat.metric("Catalog", n_parcels, help="Total parcels in parcels_labeled.parquet (from last Ingest)")
        c_pwc.metric("Parcels w/ chips", n_parcels_with_chips, help="Parcels that have at least one chip")
        c_cf.metric("Chip files", n_chips, help="Total .npz files; chip files ÷ parcels w/ chips = effective dates per parcel")
        c_sc.metric("Scenes", n_scenes, help="Scene dates in tile index (possible max); not all yield a chip per parcel")
        # Dynamic caption: possible vs actual, and the intuitive formula
        if n_parcels_with_chips and n_chips:
            effective = n_chips / n_parcels_with_chips
            effective_str = f"{effective:.1f}" if effective != int(effective) else str(int(effective))
            parts = [
                f"Chip files ({n_chips:,}) ÷ parcels w/ chips ({n_parcels_with_chips:,}) = **{effective_str} dates per parcel** (actual). "
            ]
            if n_scenes:
                possible = n_parcels_with_chips * n_scenes
                parts.append(
                    f"Possible (parcels × scenes in index): {n_parcels_with_chips:,} × {n_scenes} = {possible:,}; "
                    f"you have {n_chips:,} chips (gaps from clouds, no coverage, etc.). "
                )
            parts.append("Only these parcels get embeddings and health scores in the monitor.")
            st.caption(" ".join(parts))
        parcels_override = f"chips.max_parcels={max_parcels}"

        dp_steps = {
            "All (ingest+fetch+chip)": "Run the full data preparation pipeline.",
            "1-Ingest": "Load GeoJSON parcels and assign crop labels.",
            "2-Fetch": "Search Sentinel-2 STAC catalog for scenes.",
            "3-Chip": f"Extract 224\u00d7224 px image chips for {max_parcels} parcels.",
        }
        c_sel, c_run, c_step = st.columns([2, 1, 1])
        with c_sel:
            dp_step = st.selectbox("Step", list(dp_steps.keys()), key="dp_step")
        with c_run:
            if st.button("Run All", type="primary", key="dp_run", use_container_width=True):
                _enqueue_run(
                    f"{gpu_env} bash scripts/run_data_prep.sh config/monitor.yaml "
                    f"{overrides} {parcels_override}",
                    "Data Prep", phase="dp",
                )
        with c_step:
            if st.button("Run Step", key="dp_run_step", use_container_width=True):
                dp_map = {
                    "All (ingest+fetch+chip)": (
                        f"{gpu_env} bash scripts/run_data_prep.sh config/monitor.yaml "
                        f"{overrides} {parcels_override}"
                    ),
                    "1-Ingest": f"{gpu_env} bash scripts/run_step.sh ingest config/monitor.yaml {overrides}",
                    "2-Fetch": f"{gpu_env} bash scripts/run_step.sh fetch config/monitor.yaml {overrides}",
                    "3-Chip": (
                        f"{gpu_env} bash scripts/run_step.sh chip config/monitor.yaml "
                        f"{overrides} {parcels_override}"
                    ),
                }
                _enqueue_run(dp_map[dp_step], f"Data Prep \u2192 {dp_step}", phase="dp")
        st.caption(dp_steps[dp_step])

    live_ph_dp = st.empty()

    # -- Phase 2: Training --
    with st.expander("Phase 2: Training", expanded=False):
        st.caption("Fine-tune the Clay encoder. Optional \u2014 skip if you already have a trained encoder.")

        status_color, status_desc = _model_status()
        status_icons = {"green": "\U0001f7e2", "yellow": "\U0001f7e1", "red": "\U0001f534"}
        st.markdown(f"{status_icons.get(status_color, '')} {status_desc}")

        if n_chips == 0:
            st.warning("No chips \u2014 run Phase 1 first")

        tr_steps = {
            "All (prepare+finetune+export)": "Run the full training pipeline.",
            "1-Prepare Dataset": "Map chips to CDL crop labels and split manifests.",
            "2-Fine-tune": "Train the Clay encoder with crop supervision.",
            "3-Export Encoder": "Extract the trained encoder from the checkpoint.",
        }
        c_sel, c_run, c_step = st.columns([2, 1, 1])
        with c_sel:
            tr_step = st.selectbox("Step", list(tr_steps.keys()), key="tr_step")
        with c_run:
            if st.button("Run All", type="primary", key="tr_run",
                         disabled=(n_chips == 0), use_container_width=True):
                _enqueue_run(
                    f"{gpu_env} bash scripts/run_training.sh config/train.yaml",
                    "Training", phase="tr",
                )
        with c_step:
            if st.button("Run Step", key="tr_run_step", use_container_width=True):
                tr_map = {
                    "All (prepare+finetune+export)": f"{gpu_env} bash scripts/run_training.sh config/train.yaml",
                    "1-Prepare Dataset": f"{gpu_env} bash scripts/run_step.sh prepare config/train.yaml",
                    "2-Fine-tune": f"{gpu_env} bash scripts/run_step.sh finetune config/train.yaml",
                    "3-Export Encoder": f"{gpu_env} bash scripts/run_step.sh export config/train.yaml",
                }
                _enqueue_run(tr_map[tr_step], f"Training \u2192 {tr_step}", phase="tr")
        st.caption(tr_steps[tr_step])

    live_ph_tr = st.empty()

    # -- Phase 3: Monitor --
    with st.expander("Phase 3: Monitor", expanded=False):
        st.caption("Generate embeddings, build crop profiles, and score parcel health.")

        c_emb, c_mode = st.columns([1, 2])
        c_emb.metric("Embeddings", n_emb)
        with c_mode:
            threshold_method = st.radio(
                "Scoring mode", ["adaptive", "fixed"], index=0,
                horizontal=True, key="mo_threshold_method",
                help="**Adaptive**: z-score outliers. **Fixed**: absolute thresholds.",
            )

        scoring_overrides = f"scoring.threshold_method={threshold_method}"
        if threshold_method == "adaptive":
            col_sy, col_sr, _ = st.columns([1, 1, 2])
            sigma_y = col_sy.number_input("\u03c3 yellow", 0.5, 4.0, 1.5, 0.25, key="sig_y")
            sigma_r = col_sr.number_input("\u03c3 red", 1.0, 5.0, 2.5, 0.25, key="sig_r")
            scoring_overrides += f" scoring.sigma_yellow={sigma_y} scoring.sigma_red={sigma_r}"
        else:
            col_tg, col_tr, _ = st.columns([1, 1, 2])
            t_green = col_tg.number_input("Green <", 0.001, 1.0, 0.15, 0.01, key="t_green", format="%.3f")
            t_red = col_tr.number_input("Red >=", 0.001, 1.0, 0.30, 0.01, key="t_red", format="%.3f")
            scoring_overrides += f" scoring.thresholds.green={t_green} scoring.thresholds.red={t_red}"

        status_color, _ = _model_status()
        if status_color == "red":
            st.warning("No trained encoder \u2014 will use mock embeddings")

        mo_steps = {
            "All (embed+profile+score+report+index)": "Run the full monitor pipeline.",
            "1-Embed": "Produce 768-dim embeddings from chips (GPU).",
            "2-Profile": "Build reference trajectory per crop type.",
            "3-Score": "Cosine distance against crop reference profile.",
            "4-Report": "Generate scored GeoJSON, CSV, and campaign report.",
            "5-Index": "Build FAISS similarity index (cosine neighbors on seasonal embeddings).",
        }

        labeled_exists = (OUTPUT_DIR / "parcels_labeled.parquet").exists()
        steps_needing_labeled = {
            "All (embed+profile+score+report+index)", "2-Profile", "3-Score", "4-Report", "5-Index",
        }

        c_sel, c_run, c_step = st.columns([2, 1, 1])
        with c_sel:
            mo_step = st.selectbox("Step", list(mo_steps.keys()), key="mo_step")
        with c_run:
            if st.button("Run All", type="primary", key="mo_run",
                         disabled=(n_chips == 0), use_container_width=True):
                if not labeled_exists:
                    st.error("**parcels_labeled.parquet** missing \u2014 run Phase 1 \u2192 Ingest first.")
                else:
                    _enqueue_run(
                        f"{gpu_env} bash scripts/run_monitor.sh config/monitor.yaml "
                        f"{overrides} {scoring_overrides}",
                        "Monitor (embed+profile+score+report+index)", phase="mo",
                    )
        with c_step:
            if st.button("Run Step", key="mo_run_step", use_container_width=True):
                if mo_step in steps_needing_labeled and not labeled_exists:
                    st.error("**parcels_labeled.parquet** missing \u2014 run Phase 1 \u2192 Ingest first.")
                else:
                    mo_map = {
                        "All (embed+profile+score+report+index)": (
                            f"{gpu_env} bash scripts/run_monitor.sh config/monitor.yaml "
                            f"{overrides} {scoring_overrides}"
                        ),
                        "1-Embed": f"{gpu_env} bash scripts/run_step.sh embed config/monitor.yaml {overrides}",
                        "2-Profile": f"{gpu_env} bash scripts/run_step.sh profile config/monitor.yaml {overrides}",
                        "3-Score": (
                            f"{gpu_env} bash scripts/run_step.sh score config/monitor.yaml "
                            f"{overrides} {scoring_overrides}"
                        ),
                        "4-Report": f"{gpu_env} bash scripts/run_step.sh report config/monitor.yaml {overrides}",
                        "5-Index": f"{gpu_env} bash scripts/run_step.sh index config/monitor.yaml {overrides}",
                    }
                    _enqueue_run(mo_map[mo_step], f"Monitor \u2192 {mo_step}", phase="mo")
        st.caption(mo_steps[mo_step])

    live_ph_mo = st.empty()

    # -- Execute pending run in the correct phase placeholder --
    pending = st.session_state.pop("pending_run", None)
    if pending:
        ph_map = {"dp": live_ph_dp, "tr": live_ph_tr, "mo": live_ph_mo}
        _run_script(
            pending["cmd"], pending["step_name"],
            live_ph=ph_map[pending["phase"]],
        )

    # -- Execution History --
    st.subheader("Execution History")
    _render_execution_history()


# ---------------------------------------------------------------------------
# Tab: Campaign Overview
# ---------------------------------------------------------------------------
def tab_campaign_overview(ctx: dict[str, Any]) -> None:
    scored = _load_scores()
    parcels = _load_labeled_parcels()

    if scored is not None:
        scored = _filter_by_selection(scored, ctx)
        _monitor_overview(scored)
    elif parcels is not None:
        parcels = _filter_by_selection(parcels, ctx)
        _training_overview(parcels)
    else:
        st.info(
            "No data yet. Run **Phase 1: Data Preparation** from the "
            "Scripting tab to get started."
        )


def _monitor_overview(scored: gpd.GeoDataFrame) -> None:
    total = len(scored)
    monitored = scored[scored["status"] != "GRAY"]
    monitored_n = len(monitored)
    gray_n = total - monitored_n

    green_n = int((scored["status"] == "GREEN").sum())
    yellow_n = int((scored["status"] == "YELLOW").sum())
    red_n = int((scored["status"] == "RED").sum())

    st.caption(
        f"**{monitored_n}** parcels monitored out of **{total}** total "
        f"({gray_n} without embeddings)"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GREEN", f"{green_n} ({green_n/monitored_n*100:.0f}%)" if monitored_n else "0")
    c2.metric("YELLOW", f"{yellow_n} ({yellow_n/monitored_n*100:.0f}%)" if monitored_n else "0")
    c3.metric("RED", f"{red_n} ({red_n/monitored_n*100:.0f}%)" if monitored_n else "0")
    c4.metric("GRAY", f"{gray_n}", help="Parcels without chips/embeddings \u2014 not monitored")

    st.subheader("Parcel Health Map")
    if monitored_n > 0:
        _render_kepler_map(monitored, "kepler_monitor_config.json")
    else:
        st.info("No monitored parcels to display. Run Phase 1 with more parcels.")

    st.subheader("Action List \u2014 Parcels Needing Attention")
    alerts = scored[scored["status"].isin(["RED", "YELLOW"])].copy()
    if not alerts.empty:
        display_cols = [
            "parcel_id", "fragment_id", "crop_name", "health_score",
            "status", "n_observations", "max_deviation_date", "area_ha",
        ]
        avail_cols = [c for c in display_cols if c in alerts.columns]
        alerts_sorted = alerts[avail_cols].sort_values("health_score", ascending=False)
        st.dataframe(
            alerts_sorted,
            use_container_width=True,
            column_config={
                "health_score": st.column_config.ProgressColumn(
                    "Health Score", min_value=0.0, max_value=1.0, format="%.3f",
                ),
                "status": st.column_config.TextColumn("Status"),
            },
        )
    else:
        st.success("No alerts \u2014 all parcels are GREEN or GRAY.")


def _training_overview(parcels: gpd.GeoDataFrame) -> None:
    total = len(parcels)
    crops = parcels["crop_name"].dropna().nunique()
    most_common = parcels["crop_name"].value_counts().index[0] if crops > 0 else "N/A"
    most_common_pct = (
        parcels["crop_name"].value_counts().iloc[0] / total * 100 if crops > 0 else 0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Parcels", total)
    c2.metric("Crop Classes", crops)
    c3.metric("Most Common", f"{most_common} ({most_common_pct:.0f}%)")

    st.subheader("Crop Label Map")
    _render_kepler_map(parcels, "kepler_training_config.json")

    st.subheader("Class Balance")
    class_counts = parcels["crop_name"].value_counts().reset_index()
    class_counts.columns = ["crop_name", "count"]
    class_counts["percentage"] = (class_counts["count"] / total * 100).round(1)
    st.dataframe(class_counts, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Parcel Detail
# ---------------------------------------------------------------------------
def tab_parcel_detail(ctx: dict[str, Any]) -> None:
    scored = _load_scores()
    if scored is None:
        st.info("No scored parcels available. Run the full pipeline first.")
        return

    scored = _filter_by_selection(scored, ctx)
    monitored = scored[scored["status"] != "GRAY"]
    if monitored.empty:
        st.info("No monitored parcels for the selected region.")
        return

    parcel_ids = sorted(monitored["parcel_id"].unique())
    selected_pid = st.selectbox("Select Parcel", parcel_ids)

    if not selected_pid:
        return

    row = scored[scored["parcel_id"] == selected_pid].iloc[0]

    col1, col2, col3 = st.columns(3)
    status = row.get("status", "GRAY")
    status_icons = {
        "GREEN": "\U0001f7e2", "YELLOW": "\U0001f7e1",
        "RED": "\U0001f534", "GRAY": "\u26aa",
    }
    icon = status_icons.get(status, "\u26aa")
    col1.markdown(f"### {icon} {status}")
    hs = row.get("health_score")
    col2.metric("Health Score", f"{hs:.4f}" if pd.notna(hs) else "N/A")
    col3.metric("Crop", row.get("crop_name", "Unknown"))

    parcel_gdf = scored[scored["parcel_id"] == selected_pid]
    _render_kepler_map(parcel_gdf, "kepler_monitor_config.json", zoom=14)

    st.subheader("Similar Parcels")
    meta_sim = OUTPUT_DIR / "parcels_index_meta.parquet"
    if not meta_sim.exists():
        st.info(
            "Similarity index not built. Run Phase 3 \u2192 **5-Index** or **Run All** in the monitor section."
        )
    else:
        idx = _parcel_similarity_index(meta_sim.stat().st_mtime)
        if idx is None:
            st.info("Could not load similarity index (see logs).")
        else:
            k_nn = st.slider("Neighbors (k)", 3, 30, 10, key="parcel_sim_k")

            def _norm_crop(x: Any) -> str | None:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return None
                if pd.isna(x):
                    return None
                s = str(x).strip()
                return s if s else None

            try:
                nns = idx.nearest_neighbors(str(selected_pid), k_nn)
            except KeyError:
                st.info(
                    "This parcel is not in the similarity index (no seasonal embeddings). "
                    "Run **1-Embed** then **5-Index**."
                )
            else:
                if nns:
                    q_crop = row.get("crop_name")
                    q_crop_s = str(q_crop).strip() if pd.notna(q_crop) and q_crop else None
                    qc_ref = _norm_crop(q_crop)

                    q_rows: list[dict[str, Any]] = []
                    qm = idx.meta[idx.meta["parcel_id"].astype(str) == str(selected_pid)]
                    if not qm.empty:
                        qr = qm.iloc[0]
                        q_rows.append(
                            {
                                "parcel_id": str(selected_pid),
                                "geometry": qr.geometry,
                                "crop_name": q_crop_s,
                                "map_role": "query",
                                "similarity_note": "Selected parcel",
                            }
                        )
                    for nn in nns:
                        nm = idx.meta[idx.meta["parcel_id"].astype(str) == nn.parcel_id]
                        if nm.empty:
                            continue
                        nr = nm.iloc[0]
                        nc = _norm_crop(nn.crop_name)
                        if qc_ref and nc and nc != qc_ref:
                            role = "neighbor_diff"
                            note = (
                                f"cos_sim={nn.similarity:.3f} \u2014 different crop "
                                "(rotation/mislabel signal)"
                            )
                        else:
                            role = "neighbor_same"
                            note = f"cos_sim={nn.similarity:.3f}"
                        q_rows.append(
                            {
                                "parcel_id": nn.parcel_id,
                                "geometry": nr.geometry,
                                "crop_name": nn.crop_name,
                                "map_role": role,
                                "similarity_note": note,
                            }
                        )
                    sim_gdf = gpd.GeoDataFrame(q_rows, geometry="geometry", crs=scored.crs)
                    _render_kepler_map_similarity(sim_gdf, height=380, zoom=13)
                    tbl = pd.DataFrame(
                        [
                            {
                                "parcel_id": nn.parcel_id,
                                "crop": nn.crop_name,
                                "cosine_sim": round(nn.similarity, 4),
                                "cosine_dist": round(nn.cosine_distance, 4),
                                "diff_crop": bool(
                                    qc_ref
                                    and _norm_crop(nn.crop_name)
                                    and _norm_crop(nn.crop_name) != qc_ref
                                ),
                            }
                            for nn in nns
                        ]
                    )
                    st.dataframe(tbl, use_container_width=True)
                else:
                    st.caption("No neighbors returned.")

    st.subheader("Temporal Deviation")
    traj_str = row.get("distance_trajectory", "[]")
    try:
        traj = json.loads(traj_str) if isinstance(traj_str, str) else traj_str
    except (json.JSONDecodeError, TypeError):
        traj = []

    if traj:
        emb_dir = DATA_DIR / "embeddings" / str(selected_pid)
        dates = sorted(f.stem for f in emb_dir.glob("*.npy")) if emb_dir.exists() else []
        dates = dates[: len(traj)]
        x_vals = dates if len(dates) == len(traj) else list(range(len(traj)))

        thresh = _load_scoring_thresholds()
        green_t = thresh.get("green", 0.15)
        red_t = thresh.get("red", 0.30)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=traj, mode="lines+markers", name="Cosine Distance",
            marker=dict(
                color=[
                    "green" if d < green_t else "orange" if d < red_t else "red"
                    for d in traj
                ],
                size=10,
            ),
        ))
        fig.add_hline(
            y=green_t, line_dash="dash", line_color="green",
            annotation_text=f"Green ({green_t:.4f})",
        )
        fig.add_hline(
            y=red_t, line_dash="dash", line_color="red",
            annotation_text=f"Red ({red_t:.4f})",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Cosine Distance to Reference",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No temporal trajectory data for this parcel.")

    st.subheader("Chip Gallery (RGB)")
    chip_dir = DATA_DIR / "chips" / str(selected_pid)
    if chip_dir.exists():
        chip_files = sorted(chip_dir.glob("*.npz"))[:12]
        if chip_files:
            cols = st.columns(min(len(chip_files), 6))
            for i, cf in enumerate(chip_files):
                with cols[i % 6]:
                    try:
                        data = np.load(cf, allow_pickle=True)
                        pixels = data["pixels"]
                        bands = list(data["bands"]) if "bands" in data else []
                        b04_idx = bands.index("B04") if "B04" in bands else 2
                        b03_idx = bands.index("B03") if "B03" in bands else 1
                        b02_idx = bands.index("B02") if "B02" in bands else 0
                        rgb = np.stack(
                            [pixels[b04_idx], pixels[b03_idx], pixels[b02_idx]],
                            axis=-1,
                        ).astype(np.float32)
                        rgb = np.clip(rgb / 3000.0, 0, 1)
                        st.image(rgb, caption=cf.stem, use_container_width=True)
                    except Exception:
                        st.text(f"{cf.stem} (error)")
        else:
            st.info("No chips found.")
    else:
        st.info("No chips directory for this parcel.")


# ---------------------------------------------------------------------------
# Tab: Crop Profiles
# ---------------------------------------------------------------------------
def tab_crop_profiles(ctx: dict[str, Any]) -> None:
    scored = _load_scores()
    if scored is None:
        st.info("No scored parcels available.")
        return

    scored = _filter_by_selection(scored, ctx)
    monitored = scored[scored["status"] != "GRAY"]
    if monitored.empty:
        st.info("No monitored parcels for the selected region.")
        return

    crops = sorted(monitored["crop_name"].dropna().unique())
    if not crops:
        st.info("No crop labels found in monitored parcels.")
        return

    selected_crop = st.selectbox("Select Crop Type", crops)
    crop_df = monitored[monitored["crop_name"] == selected_crop]
    valid = crop_df["health_score"].dropna()

    c1, c2, c3 = st.columns(3)
    c1.metric("Parcels", len(crop_df))
    c2.metric("Avg Health Score", f"{valid.mean():.4f}" if len(valid) > 0 else "N/A")
    c3.metric("RED Alerts", int((crop_df["status"] == "RED").sum()))

    st.subheader("Health Score Distribution")
    if len(valid) > 0:
        thresh = _load_scoring_thresholds()
        green_t = thresh.get("green", 0.15)
        red_t = thresh.get("red", 0.30)
        fig = px.histogram(
            valid, nbins=30,
            labels={"value": "Health Score", "count": "Count"},
        )
        fig.add_vline(
            x=green_t, line_dash="dash", line_color="green",
            annotation_text=f"Green ({green_t:.4f})",
        )
        fig.add_vline(
            x=red_t, line_dash="dash", line_color="red",
            annotation_text=f"Red ({red_t:.4f})",
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"{selected_crop} Parcels")
    if not crop_df.empty and "geometry" in crop_df.columns:
        _render_kepler_map(crop_df, "kepler_monitor_config.json")

    red_crop = crop_df[crop_df["status"] == "RED"]
    if not red_crop.empty:
        st.subheader("RED Outliers")
        display_cols = [
            "parcel_id", "health_score", "status",
            "n_observations", "max_deviation_date",
        ]
        avail = [c for c in display_cols if c in red_crop.columns]
        st.dataframe(
            red_crop[avail].sort_values("health_score", ascending=False),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(_APP_CSS, unsafe_allow_html=True)

    ctx = sidebar()

    tab_script, tab_out = st.tabs(["Scripting", "Outputs"])

    with tab_script:
        tab_scripting(ctx)

    with tab_out:
        sub1, sub2, sub3 = st.tabs(
            ["Campaign Overview", "Parcel Detail", "Crop Profiles"],
        )
        with sub1:
            tab_campaign_overview(ctx)
        with sub2:
            tab_parcel_detail(ctx)
        with sub3:
            tab_crop_profiles(ctx)


if __name__ == "__main__":
    main()
else:
    main()
