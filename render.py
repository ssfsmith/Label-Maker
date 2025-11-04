import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless for PDF export
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# =========================
# Constants and defaults
# =========================
APP_ID = "LabelMaker"
PROFILE_DIR = Path.home() / f".{APP_ID.lower()}" / "profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# Page targets
SHEET_WIDTH_IN = 8.5
SHEET_HEIGHT_IN = 11.0

# LX900 (Primera 74806) 3x4 inch labels
LX900_W_IN = 3.0
LX900_H_IN = 4.0

# Render defaults
PLOT_DPI = 300
DEFAULT_TITLE_COLOR = "#1e6fff"
DEFAULT_TEXT_SCALE_PCT = 100
DEFAULT_MARGIN_PCT = 7
DEFAULT_PLOT_WIDTH_PCT = 52
DEFAULT_GRAPH_LINE_COLOR = "#1F77B4"

# Template IDs
TEMPLATE_GRAPH = "graph"
TEMPLATE_VALUE_A = "valueA"
TEMPLATE_VALUE_B = "valueB"
TEMPLATE_VALUE_C = "valueC"

# Output targets
TARGET_LETTER_TWO_UP = "letter_two_up"   # place 2 copies on a Letter page (top/bottom)
TARGET_LX900_3x4 = "lx900_3x4"           # exact-size 3x4in single-label pages


# =========================
# Helpers
# =========================
def safe_float(val, default):
    try:
        f = float(val)
        return f if np.isfinite(f) and f > 0 else default
    except Exception:
        return default

def extract_first_float_from_text(text: str) -> Optional[float]:
    if not isinstance(text, str):
        return None
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def fmt_opt_num(v: Any, ndig: int = 2) -> str:
    try:
        if v is None or (hasattr(pd, "isna") and pd.isna(v)) or not np.isfinite(float(v)):
            return ""
        return f"{float(v):.{ndig}f}"
    except Exception:
        return ""

def coerce_hex_color(s: str, fallback: str = DEFAULT_TITLE_COLOR) -> str:
    ss = (s or "").strip()
    if re.fullmatch(r"#([0-9a-fA-F]{6})", ss):
        return ss
    return fallback

# --- Sanitizers ---
def _clamp(v: float, lo: float, hi: float, default: Optional[float] = None) -> float:
    try:
        vf = float(v)
        if not np.isfinite(vf):
            raise ValueError
    except Exception:
        if default is not None:
            return float(default)
        return lo
    return max(lo, min(hi, vf))

def _clamp_pct(v: float, default: float = 0.0) -> float:
    return _clamp(v, 0.0, 100.0, default)

def _safe_inches(val, default, lo=0.5, hi=24.0) -> float:
    # coerce to float with default, then clamp to sane inches
    try:
        v = float(val)
        if not np.isfinite(v):
            v = float(default)
    except Exception:
        v = float(default)
    return _clamp(v, lo, hi)

def _sanitize_profile_for_draw(
    *,
    title_x_pct: Optional[float],
    title_y_pct: Optional[float],
    subtitle_y_pct: Optional[float],
    labels_x_pct: Optional[float],
    values_x_pct: Optional[float],
    label_value_gap_pct: Optional[float],
    rows_top_y_pct: Optional[float],
    line_height_pct: Optional[float],
    margin_pct: int,
    plot_width_pct: Optional[int],
    graph_margin_pct: Optional[int],
    is_graph: bool,
):
    # percents in 0..100 then normalize 0..1 where needed
    t_x = _clamp_pct(title_x_pct if title_x_pct is not None else 0.0, 0.0) / 100.0
    t_y = _clamp_pct(title_y_pct if title_y_pct is not None else 96.0, 96.0) / 100.0
    s_y = _clamp_pct(subtitle_y_pct if subtitle_y_pct is not None else 86.0, 86.0) / 100.0

    l_x = _clamp_pct(labels_x_pct if labels_x_pct is not None else 0.0, 0.0) / 100.0
    gap = _clamp_pct(label_value_gap_pct if label_value_gap_pct is not None else 10.0, 10.0) / 100.0

    if values_x_pct is not None:
        v_x = _clamp_pct(values_x_pct, min(100.0, (l_x * 100.0) + 5.0)) / 100.0
    else:
        v_x = l_x + gap

    # keep a minimum visual gap of 0.03
    if v_x <= l_x + 0.03:
        v_x = min(0.98, l_x + 0.03)
    # keep inside the box
    l_x = _clamp(l_x, 0.0, 0.98)
    v_x = _clamp(v_x, 0.0, 0.98)

    top_y = _clamp_pct(rows_top_y_pct if rows_top_y_pct is not None else 78.0, 78.0) / 100.0
    line_h = _clamp(line_height_pct if line_height_pct is not None else 7.5, 2.0, 25.0) / 100.0

    m_pct = int(_clamp(margin_pct, 0, 25, DEFAULT_MARGIN_PCT))
    g_margin = int(_clamp(graph_margin_pct if graph_margin_pct is not None else m_pct, 0, 25, m_pct))

    if is_graph:
        pw = int(_clamp(plot_width_pct if plot_width_pct is not None else DEFAULT_PLOT_WIDTH_PCT, 30, 80, DEFAULT_PLOT_WIDTH_PCT))
    else:
        pw = 0

    return (t_x, t_y, s_y, l_x, v_x, top_y, line_h, m_pct, g_margin, pw)

def read_excel_sheet_names(xlsx_path: str) -> List[str]:
    xl = pd.ExcelFile(xlsx_path)
    try:
        return xl.sheet_names
    finally:
        try:
            xl.close()
        except Exception:
            pass

def read_excel_sheet(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, sheet_name=sheet_name)


# =========================
# Template definitions
# =========================
def template_spec(template_id: str) -> Dict[str, Any]:
    if template_id == TEMPLATE_GRAPH:
        return {
            "id": TEMPLATE_GRAPH,
            "name": "Graph Label",
            "needs_logo": False,
            "fields": [
                {"key": "title", "type": "text", "label": "Title", "default": "Neb Label"},
                {"key": "x_flow_columns", "type": "multicol", "label": "Flow columns to plot (X)", "required": True},
                {"key": "include_trendline", "type": "bool", "label": "Include trendline", "default": True},
                {"key": "slot1_label", "type": "text", "label": "Label 1"},
                {"key": "slot1_col",   "type": "column", "label": "Value 1 (column)"},
                {"key": "slot2_label", "type": "text", "label": "Label 2"},
                {"key": "slot2_col",   "type": "column", "label": "Value 2 (column)"},
                {"key": "slot3_label", "type": "text", "label": "Label 3"},
                {"key": "slot3_col",   "type": "column", "label": "Value 3 (column)"},
                {"key": "slot4_label", "type": "text", "label": "Label 4"},
                {"key": "slot4_col",   "type": "column", "label": "Value 4 (column)"},
                {"key": "slot5_label", "type": "text", "label": "Label 5"},
                {"key": "slot5_col",   "type": "column", "label": "Value 5 (column)"},
                {"key": "slot6_label", "type": "text", "label": "Label 6"},
                {"key": "slot6_col",   "type": "column", "label": "Value 6 (column)"},
            ]
        }
    base_value = {
        "needs_logo": True,
        "fields": [
            {"key": "title", "type": "text", "label": "Title", "default": "Neb Label"},
            {"key": "subtitle", "type": "text", "label": "Subtitle"},
            {"key": "logo_path", "type": "file", "label": "Logo file"},
            {"key": "slot1_label", "type": "text", "label": "Label 1"},
            {"key": "slot1_col", "type": "column", "label": "Value 1 (column)"},
            {"key": "slot2_label", "type": "text", "label": "Label 2"},
            {"key": "slot2_col", "type": "column", "label": "Value 2 (column)"},
            {"key": "slot3_label", "type": "text", "label": "Label 3"},
            {"key": "slot3_col", "type": "column", "label": "Value 3 (column)"},
            {"key": "slot4_label", "type": "text", "label": "Label 4"},
            {"key": "slot4_col", "type": "column", "label": "Value 4 (column)"},
            {"key": "slot5_label", "type": "text", "label": "Label 5"},
            {"key": "slot5_col", "type": "column", "label": "Value 5 (column)"},
            {"key": "slot6_label", "type": "text", "label": "Label 6"},
            {"key": "slot6_col", "type": "column", "label": "Value 6 (column)"},
        ]
    }
    if template_id == TEMPLATE_VALUE_A:
        d = dict(base_value); d.update({"id": TEMPLATE_VALUE_A, "name": "Value Label A"}); return d
    if template_id == TEMPLATE_VALUE_B:
        d = dict(base_value); d.update({"id": TEMPLATE_VALUE_B, "name": "Value Label B"}); return d
    if template_id == TEMPLATE_VALUE_C:
        d = dict(base_value); d.update({"id": TEMPLATE_VALUE_C, "name": "Value Label C"}); return d
    return template_spec(TEMPLATE_GRAPH)


# =========================
# Profile schema and IO
# =========================
def default_profile(template_id: str = TEMPLATE_GRAPH) -> Dict[str, Any]:
    t = template_spec(template_id)
    prof = {
        "profile_name": f"{t['name']} Profile",
        "template_id": template_id,
        "title_color": DEFAULT_TITLE_COLOR,
        "text_scale_pct": DEFAULT_TEXT_SCALE_PCT,
        "margin_pct": DEFAULT_MARGIN_PCT,
        "plot_width_pct": DEFAULT_PLOT_WIDTH_PCT,  # graph-only
        "label_w_in": LX900_W_IN if template_id != TEMPLATE_GRAPH else 2.0,
        "label_h_in": LX900_H_IN if template_id != TEMPLATE_GRAPH else 1.0,
        "output_target": TARGET_LETTER_TWO_UP if template_id == TEMPLATE_GRAPH else TARGET_LX900_3x4,
        "bindings": {},
        "filename_pattern": "{serial}_{profile}.pdf",
        "copies": 1,
        "open_after": True,
    }
    for fld in t["fields"]:
        key = fld["key"]
        if "default" in fld:
            prof["bindings"][key] = fld["default"]
        else:
            prof["bindings"].setdefault(key, "")
    return prof

def profile_path(name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()) or "profile"
    return PROFILE_DIR / f"{safe}.json"

def list_profiles() -> List[str]:
    return [p.stem for p in PROFILE_DIR.glob("*.json")]

def save_profile(profile: Dict[str, Any]) -> Path:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    name = profile.get("profile_name") or "profile"
    p = profile_path(name)
    with p.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    return p

def load_profile(name: str) -> Dict[str, Any]:
    p = profile_path(name)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Plot helpers (graph template)
# =========================
def parse_flow_xy_from_columns(row: pd.Series, col_names: List[str]) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    next_idx = 1.0
    for c in col_names:
        y = row.get(c, None)
        x = extract_first_float_from_text(str(c))
        if x is None:
            x = next_idx
            next_idx += 1.0
        try:
            yn = float(y)
        except Exception:
            yn = None
        if yn is not None and np.isfinite(yn):
            xs.append(float(x))
            ys.append(float(yn))
    if len(xs) > 1:
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
    return xs, ys

def make_plot_ax(ax, xs: List[float], ys: List[float], include_trendline: bool, axis_label_fs: float, tick_fs: float,
                 line_color: str = DEFAULT_GRAPH_LINE_COLOR, axis_font_family: str = "Helvetica",
                 axis_color: str = "#666666", grid_enabled: bool = True):
    ax.clear()
    if not xs or not ys:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=max(6, tick_fs))
        ax.axis("off")
        return
    ax.plot(xs, ys, "-o", linewidth=1.2, markersize=2.8, color=coerce_hex_color(line_color, DEFAULT_GRAPH_LINE_COLOR), label="Data")
    try:
        ax.set_xlabel("Air flow rate (L/min)", fontdict={"fontsize": axis_label_fs, "family": axis_font_family, "color": axis_color})
        ax.set_ylabel("Self-aspiration rate (µL/min)", fontdict={"fontsize": axis_label_fs, "family": axis_font_family, "color": axis_color})
    except Exception:
        ax.set_xlabel("Air flow rate (L/min)", fontsize=axis_label_fs, color=axis_color)
        ax.set_ylabel("Self-aspiration rate (µL/min)", fontsize=axis_label_fs, color=axis_color)
    try:
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontsize(tick_fs)
            t.set_fontfamily(axis_font_family)
            t.set_color(axis_color)
    except Exception:
        ax.tick_params(axis="both", labelsize=tick_fs, colors=axis_color)
    ax.grid(bool(grid_enabled), linewidth=0.3, alpha=0.5)
    ax.margins(x=0.05, y=0.12)
    if include_trendline and len(xs) >= 2:
        try:
            coef = np.polyfit(xs, ys, 1)
            xfit = np.linspace(min(xs), max(xs), 100)
            yfit = np.polyval(coef, xfit)
            ax.plot(xfit, yfit, "-", color="#2b7bff", linewidth=1.0, label="Trend")
            ax.legend(loc="best", fontsize=max(6, tick_fs - 1))
        except Exception:
            pass


# =========================
# Rendering core
# =========================
def _draw_value_grid(axL, y_start: float, entries: List[Tuple[str, str]], fs_label: float, fs_val: float, vscale: float, labels_x: float, values_x: float, line_h_frac: float) -> float:
    y = y_start
    for lab, val in entries:
        if not lab and not val:
            continue
        if y < 0.10:
            break
        if lab:
            axL.text(labels_x, y, f"{lab}", fontsize=fs_label, va="top", ha="left", transform=axL.transAxes)
        if val:
            axL.text(values_x, y, f"{val}", fontsize=fs_val, va="top", ha="left", transform=axL.transAxes)
        # clamp line height to avoid runaway
        y -= max(0.02, min(0.25, line_h_frac)) * vscale
    return y

def _draw_value_grid_two_col(axL, y_start: float, entries: List[Tuple[str, str]], fs_label: float, fs_val: float, vscale: float, line_h_frac: float) -> float:
    y = y_start
    col_x = [0.0, 0.52]
    i = 0
    step = max(0.02, min(0.25, line_h_frac)) * vscale
    while i < len(entries):
        if y < 0.10:
            break
        lab, val = entries[i]
        if lab or val:
            axL.text(col_x[0], y, f"{lab}", fontsize=fs_label, va="top", ha="left", transform=axL.transAxes)
            axL.text(col_x[0] + 0.42, y, f"{val}", fontsize=fs_val, va="top", ha="left", transform=axL.transAxes)
        if i + 1 < len(entries):
            lab2, val2 = entries[i+1]
            if lab2 or val2:
                axL.text(col_x[1], y, f"{lab2}", fontsize=fs_label, va="top", ha="left", transform=axL.transAxes)
                axL.text(col_x[1] + 0.42, y, f"{val2}", fontsize=fs_val, va="top", ha="left", transform=axL.transAxes)
        y -= step
        i += 2
    return y

def _draw_logo_absolute(fig: plt.Figure, parent_rect: Tuple[float, float, float, float], logo_path: str, x_frac: float, y_frac: float, w_frac: float, h_frac: Optional[float]):
    if not logo_path or not Path(logo_path).exists():
        return
    try:
        img = plt.imread(logo_path)
    except Exception:
        return
    l, b, w, h = parent_rect
    iw = w * _clamp(w_frac, 0.02, 0.98, 0.28)
    if h_frac is None or h_frac <= 0:
        try:
            ih_img, iw_img = img.shape[0], img.shape[1]
            ratio = (ih_img / max(1.0, iw_img))
        except Exception:
            ratio = 0.35
        ih = iw * ratio
    else:
        ih = h * _clamp(h_frac, 0.02, 0.98, 0.20)
    x0 = l + _clamp(x_frac, 0.0, 0.98, 0.02) * w
    y0 = b + _clamp(y_frac, 0.0, 0.98, 0.70) * h
    ax_logo = fig.add_axes([x0, y0, max(iw, 1e-3), max(ih, 1e-3)])
    ax_logo.imshow(img)
    ax_logo.axis("off")

def _draw_logo_aligned(fig: plt.Figure, parent_rect: Tuple[float, float, float, float], align: str, logo_path: str, max_rel_w: float, max_rel_h: float):
    if not logo_path or not Path(logo_path).exists():
        return
    try:
        img = plt.imread(logo_path)
    except Exception:
        return
    l, b, w, h = parent_rect
    iw = w * _clamp(max_rel_w, 0.02, 0.98, 0.28)
    ih = h * _clamp(max_rel_h, 0.02, 0.98, 0.28)
    if align == "top-left":
        x0, y0 = l + 0.01 * w, b + h - ih - 0.01 * h
    elif align == "top-right":
        x0, y0 = l + w - iw - 0.01 * w, b + h - ih - 0.01 * h
    else:
        x0, y0 = l + (w - iw) / 2.0, b + h - ih - 0.01 * h
    ax_logo = fig.add_axes([x0, y0, max(iw, 1e-3), max(ih, 1e-3)])
    ax_logo.imshow(img)
    ax_logo.axis("off")

def _draw_label_into(
    fig: plt.Figure,
    rect_frac: Tuple[float, float, float, float],
    *,
    # Common
    title_text: str,
    title_color: str,
    text_scale_pct: int,
    margin_pct: int,
    # Title formatting/position
    title_font_family: Optional[str] = None,
    title_font_size: Optional[float] = None,
    title_x_pct: Optional[float] = None,
    title_y_pct: Optional[float] = None,
    # Subtitle formatting/position
    subtitle_text: str = "",
    subtitle_font_family: Optional[str] = None,
    subtitle_font_size: Optional[float] = None,
    subtitle_y_pct: Optional[float] = None,
    subtitle_color: Optional[str] = None,
    # Label/value formatting
    labels_font_family: Optional[str] = None,
    labels_font_size: Optional[float] = None,
    labels_color: Optional[str] = None,
    values_font_family: Optional[str] = None,
    values_font_size: Optional[float] = None,
    values_color: Optional[str] = None,
    labels_x_pct: Optional[float] = None,
    values_x_pct: Optional[float] = None,
    label_value_gap_pct: Optional[float] = None,
    rows_top_y_pct: Optional[float] = None,
    line_height_pct: Optional[float] = None,
    # Optional static text (legacy)
    static_text_enabled: bool = False,
    static_text: str = "",
    static_text_x_pct: float = 0.0,
    static_text_y_pct: float = 0.82,
    static_text_font_family: str = "Helvetica",
    static_text_font_size: float = 12.0,
    static_text_color: str = "#111111",
    # Graph specifics
    plot_width_pct: Optional[int] = None,
    include_trendline: Optional[bool] = None,
    graph_line_color: str = DEFAULT_GRAPH_LINE_COLOR,
    graph_margin_pct: Optional[int] = None,
    axis_font_family: str = "Helvetica",
    axis_color: str = "#666666",
    grid_enabled: bool = True,
    # Value specifics
    value_entries: Optional[List[Tuple[str, str]]] = None,
    logo_path: Optional[str] = None,
    # Optional absolute logo placement (value labels only)
    logo_x_pct: Optional[float] = None,
    logo_y_pct: Optional[float] = None,
    logo_w_pct: Optional[float] = None,
    logo_h_pct: Optional[float] = None,
    # legacy align option if absolute not provided
    logo_align: str = "top-left",
    value_style: str = "A",
    # Graph plot data
    xs: Optional[List[float]] = None,
    ys: Optional[List[float]] = None,
):
    # Scales (clamped)
    tscale = _clamp(text_scale_pct, 50, 200, DEFAULT_TEXT_SCALE_PCT) / 100.0
    fs_title   = float(title_font_size if title_font_size is not None else (11.0 * tscale))
    fs_body    = float(labels_font_size if labels_font_size is not None else (8.0  * tscale))
    fs_val     = float(values_font_size if values_font_size is not None else fs_body)
    fs_axis    = 7.0  * tscale
    fs_tick    = 6.0  * tscale
    fs_sub     = float(subtitle_font_size if subtitle_font_size is not None else (12.0 * tscale))

    is_graph = xs is not None

    # Sanitize all geometry inputs
    (t_x, t_y, s_y, l_x, v_x, top_y, line_h, m_pct, g_margin, pw) = _sanitize_profile_for_draw(
        title_x_pct=title_x_pct, title_y_pct=title_y_pct, subtitle_y_pct=subtitle_y_pct,
        labels_x_pct=labels_x_pct, values_x_pct=values_x_pct, label_value_gap_pct=label_value_gap_pct,
        rows_top_y_pct=rows_top_y_pct, line_height_pct=line_height_pct,
        margin_pct=margin_pct, plot_width_pct=plot_width_pct, graph_margin_pct=graph_margin_pct,
        is_graph=is_graph,
    )

    l, b, w, h = rect_frac
    # Outer border of the label
    axC = fig.add_axes([l, b, max(w, 1e-4), max(h, 1e-4)])
    axC.axis("off")
    axC.add_patch(Rectangle((0.003, 0.003), 0.994, 0.994, transform=axC.transAxes,
                            fill=False, linewidth=1, edgecolor="black"))

    # Inner layout
    lm = rm = tm = bm = m_pct / 100.0
    total_w = max(1e-6, 1 - lm - rm)
    total_h = max(1e-6, 1 - tm - bm)

    # Graph right-side width (as fraction of inner area)
    right_frac = (pw / 100.0) if is_graph else 0.0
    left_w_frac = (1.0 - right_frac) if is_graph else 1.0

    leftL = l + w * lm
    botL  = b + h * bm
    widL  = w * total_w * left_w_frac
    heiL  = h * total_h

    axL = fig.add_axes([leftL, botL, max(widL, 1e-4), max(heiL, 1e-4)])
    axL.axis("off")

    # Title
    try:
        axL.text(t_x, t_y, title_text or "",  # bindings["title"]
                 color=coerce_hex_color(title_color),
                 fontsize=fs_title,
                 fontfamily=(title_font_family or "Helvetica"),
                 fontweight="bold",
                 va="top", ha="left", transform=axL.transAxes)
    except Exception:
        axL.text(t_x, t_y, title_text or "", color=coerce_hex_color(title_color), fontsize=fs_title,
                 va="top", ha="left", transform=axL.transAxes)

    # Subtitle (left-aligned at x=0.0), using Subtitle formatting
    if (subtitle_text or "").strip():
        try:
            axL.text(0.0, s_y, subtitle_text,
                     color=coerce_hex_color(subtitle_color or "#333333", "#333333"),
                     fontsize=fs_sub,
                     fontfamily=(subtitle_font_family or "Helvetica"),
                     va="top", ha="left", transform=axL.transAxes)
        except Exception:
            axL.text(0.0, s_y, subtitle_text, color=coerce_hex_color(subtitle_color or "#333333", "#333333"),
                     fontsize=fs_sub, va="top", ha="left", transform=axL.transAxes)

    # Optional static text (legacy support)
    if static_text_enabled and (static_text or "").strip():
        try:
            axL.text(_clamp(static_text_x_pct, 0.0, 1.0, 0.0),
                     _clamp(static_text_y_pct, 0.0, 1.0, 0.82),
                     static_text,
                     color=coerce_hex_color(static_text_color, "#111111"),
                     fontsize=float(static_text_font_size or 12.0),
                     fontfamily=(static_text_font_family or "Helvetica"),
                     va="top", ha="left", transform=axL.transAxes)
        except Exception:
            axL.text(_clamp(static_text_x_pct, 0.0, 1.0, 0.0),
                     _clamp(static_text_y_pct, 0.0, 1.0, 0.82),
                     static_text, color=coerce_hex_color(static_text_color, "#111111"),
                     fontsize=float(static_text_font_size or 12.0), va="top", ha="left", transform=axL.transAxes)

    # Values block colors
    lab_col = coerce_hex_color(labels_color or "#333333", "#333333")
    val_col = coerce_hex_color(values_color or "#000000", "#000000")

    # Values
    y = top_y
    if value_entries:
        if not is_graph:
            if value_style == "A":
                y = _draw_value_grid(axL, y, value_entries, fs_body, fs_val, 1.0, l_x, v_x, line_h)
                # Logo
                if logo_path:
                    parent = (l + w*lm, b + h*bm, w*total_w, h*total_h)
                    if logo_x_pct is not None and logo_y_pct is not None and logo_w_pct is not None:
                        _draw_logo_absolute(fig, parent, logo_path,
                                            x_frac=_clamp(logo_x_pct, 0.0, 1.0, 0.02),
                                            y_frac=_clamp(logo_y_pct, 0.0, 1.0, 0.70),
                                            w_frac=_clamp(logo_w_pct, 0.02, 0.98, 0.28),
                                            h_frac=logo_h_pct if logo_h_pct else None)
                    else:
                        _draw_logo_aligned(fig, parent, "top-left", logo_path, 0.28, 0.28)
            elif value_style == "B":
                big_fs = fs_body * 1.22
                small_fs = fs_body
                first = value_entries[:3]
                rest = value_entries[3:]
                y = _draw_value_grid(axL, y, first, big_fs, big_fs, 1.0, l_x, v_x, line_h)
                y -= 0.02
                y = _draw_value_grid(axL, y, rest, small_fs, small_fs, 1.0, l_x, v_x, line_h)
                if logo_path:
                    parent = (l + w*lm, b + h*bm, w*total_w, h*total_h)
                    if logo_x_pct is not None and logo_y_pct is not None and logo_w_pct is not None:
                        _draw_logo_absolute(fig, parent, logo_path,
                                            x_frac=_clamp(logo_x_pct, 0.0, 1.0, 0.70),
                                            y_frac=_clamp(logo_y_pct, 0.0, 1.0, 0.70),
                                            w_frac=_clamp(logo_w_pct, 0.02, 0.98, 0.26),
                                            h_frac=logo_h_pct if logo_h_pct else None)
                    else:
                        _draw_logo_aligned(fig, parent, "top-right", logo_path, 0.26, 0.26)
            else:  # C
                y = _draw_value_grid_two_col(axL, y, value_entries, fs_body * 0.95, fs_val * 0.95, 1.0, line_h)
                if logo_path:
                    parent = (l + w*lm, b + h*bm, w*total_w, h*total_h)
                    if logo_x_pct is not None and logo_y_pct is not None and logo_w_pct is not None:
                        _draw_logo_absolute(fig, parent, logo_path,
                                            x_frac=_clamp(logo_x_pct, 0.0, 1.0, 0.32),
                                            y_frac=_clamp(logo_y_pct, 0.0, 1.0, 0.70),
                                            w_frac=_clamp(logo_w_pct, 0.02, 0.98, 0.36),
                                            h_frac=logo_h_pct if logo_h_pct else None)
                    else:
                        _draw_logo_aligned(fig, parent, "top-center", logo_path, 0.36, 0.22)
            # Colorize placed text
            for txt in axL.texts:
                if txt.get_text() == (title_text or ""):
                    txt.set_color(coerce_hex_color(title_color))
                else:
                    if abs(txt.get_position()[0] - v_x) < 1e-3:
                        txt.set_color(val_col)
                        try:
                            txt.set_fontfamily(values_font_family or "Helvetica")
                            txt.set_fontsize(fs_val)
                        except Exception:
                            pass
                    else:
                        # subtitle keeps its own font/color already
                        if txt.get_text() != (subtitle_text or ""):
                            txt.set_color(lab_col)
                            try:
                                txt.set_fontfamily(labels_font_family or "Helvetica")
                                txt.set_fontsize(fs_body)
                            except Exception:
                                pass
            return
        else:
            # Graph + values
            y = _draw_value_grid(axL, y, value_entries, fs_body, fs_val, 1.0, l_x, v_x, line_h)

    if not is_graph:
        return

    # graph mode: right plot area
    gap = 0.012
    leftR  = l + w * (lm + total_w * left_w_frac) + w * gap
    widR   = max(1e-4, w * (total_w * (1.0 - left_w_frac) - gap))
    botR   = botL
    heiR   = max(1e-4, heiL)

    # Apply graph inner margin
    gmf = g_margin / 100.0
    inner_left = leftR + widR * gmf
    inner_bot  = botR  + heiR * gmf
    inner_w    = max(1e-4, widR  * (1 - 2 * gmf))
    inner_h    = max(1e-4, heiR  * (1 - 2 * gmf))

    axR = fig.add_axes([inner_left, inner_bot, inner_w, inner_h])
    make_plot_ax(
        axR, xs or [], ys or [], bool(include_trendline),
        axis_label_fs=fs_body, tick_fs=fs_tick, line_color=graph_line_color,
        axis_font_family=(axis_font_family or "Helvetica"),
        axis_color=coerce_hex_color(axis_color, "#666666"),
        grid_enabled=bool(grid_enabled)
    )

def render_letter_two_up(
    out_pdf: Path,
    *,
    title_text: str,
    title_color: str,
    text_scale_pct: int,
    margin_pct: int,
    plot_width_pct: int,
    include_trendline: bool,
    xs: List[float],
    ys: List[float],
    graph_line_color: str,
    graph_margin_pct: Optional[int] = None,
    value_entries: Optional[List[Tuple[str, str]]] = None,
    # Formatting (optional)
    title_font_family: Optional[str] = None,
    title_font_size: Optional[float] = None,
    title_x_pct: Optional[float] = None,
    title_y_pct: Optional[float] = None,
    subtitle_text: str = "",
    subtitle_font_family: Optional[str] = None,
    subtitle_font_size: Optional[float] = None,
    subtitle_y_pct: Optional[float] = None,
    subtitle_color: Optional[str] = None,
    labels_font_family: Optional[str] = None,
    labels_font_size: Optional[float] = None,
    labels_color: Optional[str] = None,
    values_font_family: Optional[str] = None,
    values_font_size: Optional[float] = None,
    values_color: Optional[str] = None,
    labels_x_pct: Optional[float] = None,
    values_x_pct: Optional[float] = None,
    label_value_gap_pct: Optional[float] = None,
    rows_top_y_pct: Optional[float] = None,
    line_height_pct: Optional[float] = None,
    # Static text
    static_text_enabled: bool = False,
    static_text: str = "",
    static_text_x_pct: float = 0.0,
    static_text_y_pct: float = 0.82,
    static_text_font_family: str = "Helvetica",
    static_text_font_size: float = 12.0,
    static_text_color: str = "#111111",
    # Axis
    axis_font_family: str = "Helvetica",
    axis_color: str = "#666666",
    grid_enabled: bool = True,
):
    page_w, page_h = SHEET_WIDTH_IN, SHEET_HEIGHT_IN
    fig = plt.figure(figsize=(page_w, page_h), dpi=PLOT_DPI)
    fig.subplots_adjust(0, 0, 1, 1)

    lab_h_frac = 0.44
    lab_w_frac = 0.86
    left_frac = (1.0 - lab_w_frac) / 2.0
    bottom_bottom = 0.06
    bottom_top = 1.0 - bottom_bottom - lab_h_frac

    common_kwargs = dict(
        title_text=title_text, title_color=title_color,
        text_scale_pct=text_scale_pct, margin_pct=margin_pct,
        plot_width_pct=plot_width_pct, include_trendline=include_trendline,
        xs=xs, ys=ys,
        value_entries=value_entries,
        graph_line_color=graph_line_color, graph_margin_pct=graph_margin_pct,
        title_font_family=title_font_family, title_font_size=title_font_size,
        title_x_pct=title_x_pct, title_y_pct=title_y_pct,
        subtitle_text=subtitle_text, subtitle_font_family=subtitle_font_family, subtitle_font_size=subtitle_font_size,
        subtitle_y_pct=subtitle_y_pct, subtitle_color=subtitle_color,
        labels_font_family=labels_font_family, labels_font_size=labels_font_size, labels_color=labels_color,
        values_font_family=values_font_family, values_font_size=values_font_size, values_color=values_color,
        labels_x_pct=labels_x_pct, values_x_pct=values_x_pct, label_value_gap_pct=label_value_gap_pct,
        rows_top_y_pct=rows_top_y_pct, line_height_pct=line_height_pct,
        static_text_enabled=static_text_enabled, static_text=static_text,
        static_text_x_pct=static_text_x_pct, static_text_y_pct=static_text_y_pct,
        static_text_font_family=static_text_font_family, static_text_font_size=static_text_font_size,
        static_text_color=static_text_color,
        axis_font_family=axis_font_family, axis_color=axis_color, grid_enabled=grid_enabled,
    )

    _draw_label_into(fig, (left_frac, bottom_bottom, lab_w_frac, lab_h_frac), **common_kwargs)
    _draw_label_into(fig, (left_frac, bottom_top,    lab_w_frac, lab_h_frac), **common_kwargs)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pp:
        pp.savefig(fig, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def render_lx900_single(
    out_pdf: Path,
    *,
    label_w_in: float,
    label_h_in: float,
    title_text: str,
    title_color: str,
    text_scale_pct: int,
    margin_pct: int,
    # value label
    value_entries: Optional[List[Tuple[str, str]]] = None,
    logo_path: Optional[str] = None,
    logo_align: str = "top-left",
    value_style: str = "A",
    # graph label
    plot_width_pct: Optional[int] = None,
    include_trendline: Optional[bool] = None,
    xs: Optional[List[float]] = None,
    ys: Optional[List[float]] = None,
    # Formatting (optional)
    title_font_family: Optional[str] = None,
    title_font_size: Optional[float] = None,
    title_x_pct: Optional[float] = None,
    title_y_pct: Optional[float] = None,
    subtitle_text: str = "",
    subtitle_font_family: Optional[str] = None,
    subtitle_font_size: Optional[float] = None,
    subtitle_y_pct: Optional[float] = None,
    subtitle_color: Optional[str] = None,
    labels_font_family: Optional[str] = None,
    labels_font_size: Optional[float] = None,
    labels_color: Optional[str] = None,
    values_font_family: Optional[str] = None,
    values_font_size: Optional[float] = None,
    values_color: Optional[str] = None,
    labels_x_pct: Optional[float] = None,
    values_x_pct: Optional[float] = None,
    label_value_gap_pct: Optional[float] = None,
    rows_top_y_pct: Optional[float] = None,
    line_height_pct: Optional[float] = None,
    # Static text
    static_text_enabled: bool = False,
    static_text: str = "",
    static_text_x_pct: float = 0.0,
    static_text_y_pct: float = 0.82,
    static_text_font_family: str = "Helvetica",
    static_text_font_size: float = 12.0,
    static_text_color: str = "#111111",
    # Graph options
    graph_line_color: str = DEFAULT_GRAPH_LINE_COLOR,
    graph_margin_pct: Optional[int] = None,
    # Absolute logo positioning
    logo_x_pct: Optional[float] = None,
    logo_y_pct: Optional[float] = None,
    logo_w_pct: Optional[float] = None,
    logo_h_pct: Optional[float] = None,
    # Axis
    axis_font_family: str = "Helvetica",
    axis_color: str = "#666666",
    grid_enabled: bool = True,
):
    w_in = _safe_inches(label_w_in, LX900_W_IN, 0.5, 24.0)
    h_in = _safe_inches(label_h_in, LX900_H_IN, 0.5, 24.0)
    fig = plt.figure(figsize=(w_in, h_in), dpi=PLOT_DPI)
    fig.subplots_adjust(0, 0, 1, 1)

    _draw_label_into(
        fig, (0, 0, 1, 1),
        title_text=title_text, title_color=title_color,
        text_scale_pct=text_scale_pct, margin_pct=margin_pct,
        value_entries=value_entries, logo_path=logo_path, logo_align=logo_align,
        value_style=value_style,
        plot_width_pct=plot_width_pct, include_trendline=include_trendline,
        xs=xs, ys=ys,
        title_font_family=title_font_family, title_font_size=title_font_size,
        title_x_pct=title_x_pct, title_y_pct=title_y_pct,
        subtitle_text=subtitle_text, subtitle_font_family=subtitle_font_family, subtitle_font_size=subtitle_font_size,
        subtitle_y_pct=subtitle_y_pct, subtitle_color=subtitle_color,
        labels_font_family=labels_font_family, labels_font_size=labels_font_size, labels_color=labels_color,
        values_font_family=values_font_family, values_font_size=values_font_size, values_color=values_color,
        labels_x_pct=labels_x_pct, values_x_pct=values_x_pct, label_value_gap_pct=label_value_gap_pct,
        rows_top_y_pct=rows_top_y_pct, line_height_pct=line_height_pct,
        static_text_enabled=static_text_enabled, static_text=static_text,
        static_text_x_pct=static_text_x_pct, static_text_y_pct=static_text_y_pct,
        static_text_font_family=static_text_font_family, static_text_font_size=static_text_font_size,
        static_text_color=static_text_color,
        graph_line_color=graph_line_color, graph_margin_pct=graph_margin_pct,
        logo_x_pct=logo_x_pct, logo_y_pct=logo_y_pct, logo_w_pct=logo_w_pct, logo_h_pct=logo_h_pct,
        axis_font_family=axis_font_family, axis_color=axis_color, grid_enabled=grid_enabled,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pp:
        pp.savefig(fig, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def render_profile_to_pdf(
    profile: Dict[str, Any],
    df: pd.DataFrame,
    row: pd.Series,
    out_pdf: Path,
):
    template_id = profile.get("template_id", TEMPLATE_GRAPH)
    b = profile.get("bindings", {}) or {}

    # Use bindings["title"] here (not profile name)
    title = (b.get("title") or "").strip() or "Neb Label"
    subtitle = (b.get("subtitle") or "").strip()

    title_color = profile.get("title_color", DEFAULT_TITLE_COLOR)
    text_scale_pct = int(profile.get("text_scale_pct", DEFAULT_TEXT_SCALE_PCT))
    margin_pct = int(profile.get("margin_pct", DEFAULT_MARGIN_PCT))
    plot_width_pct = int(profile.get("plot_width_pct", DEFAULT_PLOT_WIDTH_PCT))
    target = profile.get("output_target", TARGET_LETTER_TWO_UP)
    graph_line_color = profile.get("graph_line_color", DEFAULT_GRAPH_LINE_COLOR)
    graph_margin_pct = int(profile.get("graph_margin_pct", margin_pct))

    # Formatting fields
    fmt = {
        # Title
        "title_font_family": profile.get("title_font_family"),
        "title_font_size": profile.get("title_font_size"),
        "title_x_pct": profile.get("title_x_pct"),
        "title_y_pct": profile.get("title_y_pct"),
        # Subtitle
        "subtitle_text": subtitle,
        "subtitle_font_family": profile.get("subtitle_font_family"),
        "subtitle_font_size": profile.get("subtitle_font_size"),
        "subtitle_y_pct": profile.get("subtitle_y_pct"),
        "subtitle_color": profile.get("subtitle_color"),
        # Label/Values
        "labels_font_family": profile.get("labels_font_family"),
        "labels_font_size": profile.get("labels_font_size"),
        "labels_color": profile.get("labels_color"),
        "values_font_family": profile.get("values_font_family"),
        "values_font_size": profile.get("values_font_size"),
        "values_color": profile.get("values_color"),
        "labels_x_pct": profile.get("labels_x_pct"),
        "values_x_pct": profile.get("values_x_pct"),
        "label_value_gap_pct": profile.get("label_value_gap_pct"),
        "rows_top_y_pct": profile.get("rows_top_y_pct"),
        "line_height_pct": profile.get("line_height_pct"),
        # Static text (legacy)
        "static_text_enabled": bool(profile.get("static_text_enabled", False)),
        "static_text": profile.get("static_text", "Calibration Data"),
        "static_text_x_pct": float(profile.get("static_text_x_pct", 0.0)),
        "static_text_y_pct": float(profile.get("static_text_y_pct", 0.82)),
        "static_text_font_family": profile.get("static_text_font_family", "Helvetica"),
        "static_text_font_size": float(profile.get("static_text_font_size", 12.0)),
        "static_text_color": profile.get("static_text_color", "#111111"),
        # Axis
        "axis_font_family": profile.get("axis_font_family", "Helvetica"),
        "axis_color": profile.get("axis_color", "#666666"),
        "grid_enabled": bool(profile.get("grid_enabled", True)),
    }

    def build_entries() -> List[Tuple[str, str]]:
        entries: List[Tuple[str, str]] = []
        for i in range(1, 25):
            lab = (b.get(f"slot{i}_label", "") or "").strip()
            col = (b.get(f"slot{i}_col", "") or "").strip()
            val = ""
            if col and col in df.columns:
                v = row.get(col)
                val = fmt_opt_num(v, 2) if isinstance(v, (int, float, np.floating)) else ("" if (pd.isna(v) if hasattr(pd, "isna") else False) else str(v))
            entries.append((lab, val))
        while entries and not any(entries[-1]):
            entries.pop()
        return entries

    if template_id == TEMPLATE_GRAPH:
        x_cols = b.get("x_flow_columns", [])
        if isinstance(x_cols, str):
            x_cols = [x.strip() for x in x_cols.split("|") if x.strip()]
        xs, ys = parse_flow_xy_from_columns(row, x_cols)
        include_trendline = bool(b.get("include_trendline", True))
        entries = build_entries()

        if target == TARGET_LETTER_TWO_UP:
            render_letter_two_up(
                out_pdf=out_pdf,
                title_text=title,
                title_color=title_color,
                text_scale_pct=text_scale_pct,
                margin_pct=margin_pct,
                plot_width_pct=plot_width_pct,
                include_trendline=include_trendline,
                xs=xs, ys=ys,
                value_entries=entries,
                graph_line_color=graph_line_color,
                graph_margin_pct=graph_margin_pct,
                **fmt,
            )
        else:
            w_in = _safe_inches(profile.get("label_w_in", LX900_W_IN), LX900_W_IN, 0.5, 24.0)
            h_in = _safe_inches(profile.get("label_h_in", LX900_H_IN), LX900_H_IN, 0.5, 24.0)
            render_lx900_single(
                out_pdf=out_pdf,
                label_w_in=w_in, label_h_in=h_in,
                title_text=title, title_color=title_color,
                text_scale_pct=text_scale_pct, margin_pct=margin_pct,
                plot_width_pct=plot_width_pct, include_trendline=include_trendline,
                xs=xs, ys=ys,
                value_entries=entries,
                graph_line_color=graph_line_color,
                graph_margin_pct=graph_margin_pct,
                **fmt,
            )
        return

    # Value-only labels (A/B/C)
    logo_path = b.get("logo_path", "")
    logo_x_pct = profile.get("logo_x_pct")
    logo_y_pct = profile.get("logo_y_pct")
    logo_w_pct = profile.get("logo_w_pct")
    logo_h_pct = profile.get("logo_h_pct")
    entries = build_entries()
    style = "A" if template_id == TEMPLATE_VALUE_A else ("B" if template_id == TEMPLATE_VALUE_B else "C")

    if target == TARGET_LX900_3x4:
        w_in = _safe_inches(profile.get("label_w_in", LX900_W_IN), LX900_W_IN, 0.5, 24.0)
        h_in = _safe_inches(profile.get("label_h_in", LX900_H_IN), LX900_H_IN, 0.5, 24.0)
        render_lx900_single(
            out_pdf=out_pdf,
            label_w_in=w_in, label_h_in=h_in,
            title_text=title, title_color=title_color,
            text_scale_pct=text_scale_pct, margin_pct=margin_pct,
            value_entries=entries, logo_path=logo_path, value_style=style,
            logo_x_pct=logo_x_pct, logo_y_pct=logo_y_pct, logo_w_pct=logo_w_pct, logo_h_pct=logo_h_pct,
            **fmt,
        )
    else:
        page_w, page_h = SHEET_WIDTH_IN, SHEET_HEIGHT_IN
        fig = plt.figure(figsize=(page_w, page_h), dpi=PLOT_DPI)
        fig.subplots_adjust(0, 0, 1, 1)

        lab_h_frac = 0.44
        lab_w_frac = 0.86
        left_frac = (1.0 - lab_w_frac) / 2.0
        bottom_bottom = 0.06
        bottom_top = 1.0 - bottom_bottom - lab_h_frac

        common_kwargs = dict(
            title_text=title, title_color=title_color,
            text_scale_pct=text_scale_pct, margin_pct=margin_pct,
            value_entries=entries, logo_path=logo_path, value_style=style,
            logo_x_pct=logo_x_pct, logo_y_pct=logo_y_pct, logo_w_pct=logo_w_pct, logo_h_pct=logo_h_pct,
            **fmt,
        )

        _draw_label_into(fig, (left_frac, bottom_bottom, lab_w_frac, lab_h_frac), **common_kwargs)
        _draw_label_into(fig, (left_frac, bottom_top,    lab_w_frac, lab_h_frac), **common_kwargs)

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(out_pdf) as pp:
            pp.savefig(fig, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
