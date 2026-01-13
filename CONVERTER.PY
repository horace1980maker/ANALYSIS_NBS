"""
NbS Analyzer — Streamlit Converter UI

This is the first page of the NbS Analysis project.
Users upload their raw Excel file and it gets transformed into an analysis-ready format.

Usage:
    streamlit run converter.py
"""

from __future__ import annotations

import io
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.table import Table, TableStyleInfo


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="NbS Analyzer — Data Converter",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# Custom CSS — Premium Dark Glassmorphism Theme
# =============================================================================

st.markdown("""
<style>
    /* Root variables */
    :root {
        --primary: #10b981;
        --primary-dark: #059669;
        --secondary: #6366f1;
        --accent: #f59e0b;
        --bg-dark: #0f172a;
        --bg-card: rgba(30, 41, 59, 0.7);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: rgba(148, 163, 184, 0.2);
        --glass: rgba(255, 255, 255, 0.05);
        --success: #22c55e;
        --warning: #f59e0b;
        --error: #ef4444;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-attachment: fixed;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Glass card effect */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Hero section */
    .hero {
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(99, 102, 241, 0.1));
        border-radius: 20px;
        border: 1px solid var(--border);
    }
    
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #10b981, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero p {
        color: var(--text-secondary);
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Step indicator */
    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .step {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem 1.5rem;
        background: var(--glass);
        border-radius: 12px;
        border: 1px solid var(--border);
        transition: all 0.3s ease;
    }
    
    .step.active {
        background: rgba(16, 185, 129, 0.2);
        border-color: var(--primary);
    }
    
    .step.completed {
        background: rgba(34, 197, 94, 0.2);
        border-color: var(--success);
    }
    
    .step-number {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--border);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .step.active .step-number {
        background: var(--primary);
    }
    
    .step.completed .step-number {
        background: var(--success);
    }
    
    .step-text {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    .step.active .step-text,
    .step.completed .step-text {
        color: var(--text-primary);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed var(--border);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        background: var(--glass);
    }
    
    .upload-area:hover {
        border-color: var(--primary);
        background: rgba(16, 185, 129, 0.1);
    }
    
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    
    /* Status messages */
    .status-success {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid var(--success);
        color: var(--success);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid var(--error);
        color: var(--error);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .status-info {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid var(--secondary);
        color: var(--secondary);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--secondary), #4f46e5);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: var(--glass);
        border: 2px dashed var(--border);
        border-radius: 16px;
        padding: 1rem;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary);
    }
    
    /* Table styling */
    .dataframe {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
    }
    
    .dataframe th {
        background: rgba(16, 185, 129, 0.2) !important;
        color: var(--text-primary) !important;
    }
    
    .dataframe td {
        color: var(--text-secondary) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: var(--primary);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--text-primary);
    }
    
    /* Info box */
    .info-box {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid var(--secondary);
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: var(--secondary);
        margin-bottom: 0.5rem;
    }
    
    .info-box p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions (from original converter.py)
# =============================================================================

def remove_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def clean_col(col: str) -> str:
    col = (col or "").strip()
    col = remove_accents(col)
    col = col.replace("–", "-")
    col = re.sub(r"[^0-9A-Za-z]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    col = col.lower()
    if re.match(r"^\d", col):
        col = "x" + col
    return col


def make_unique(values: List[str]) -> List[str]:
    seen = {}
    out = []
    for v in values:
        if v not in seen:
            seen[v] = 1
            out.append(v)
        else:
            seen[v] += 1
            out.append(f"{v}_{seen[v]}")
    return out


THREAT_CODE_RE = re.compile(r"\b(AC\d{2}|ANC\d{2})\b")
GAP_CODE_RE = re.compile(r"\bB?(\d+\.\d+)\b")


def extract_threat_code(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = THREAT_CODE_RE.search(text)
    return m.group(1) if m else None


def extract_gap_code(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = GAP_CODE_RE.search(text)
    if not m:
        return None
    return "B" + m.group(1)


def add_df_sheet(
    wb: openpyxl.Workbook,
    title: str,
    df: pd.DataFrame,
    *,
    table_style: str = "TableStyleMedium9",
) -> None:
    if title in wb.sheetnames:
        del wb[title]
    ws = wb.create_sheet(title)

    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)

    header_fill = PatternFill("solid", fgColor="FFB7E1CD")
    header_font = Font(bold=True)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(wrap_text=True, vertical="center")

    body_align = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = body_align

    for col_idx, col_name in enumerate(df.columns, start=1):
        max_len = len(str(col_name))
        sample = df[col_name].astype(str).head(200)
        for v in sample:
            max_len = max(max_len, len(v))
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max(10, max_len + 2), 60)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    try:
        end_row = ws.max_row
        end_col = ws.max_column
        if end_row >= 2 and end_col >= 1:
            safe = re.sub(r"[^A-Za-z0-9]", "", title)[:18]
            tbl = Table(displayName=f"{safe}{np.random.randint(0, 9999)}", ref=f"A1:{get_column_letter(end_col)}{end_row}")
            tbl.tableStyleInfo = TableStyleInfo(
                name=table_style,
                showFirstColumn=False,
                showLastColumn=False,
                showRowStripes=True,
                showColumnStripes=False,
            )
            ws.add_table(tbl)
    except Exception:
        pass


def extract_lookup_threats(wb: openpyxl.Workbook, sheet_name: str = "AN - ANC") -> pd.DataFrame:
    if sheet_name not in wb.sheetnames:
        return pd.DataFrame()
    
    ws = wb[sheet_name]
    rows: List[Dict[str, Any]] = []

    for r in range(15, 200):
        code = ws.cell(r, 2).value
        if isinstance(code, str) and re.match(r"^AC\d{2}$", code.strip()):
            rows.append({
                "threat_code": code.strip(),
                "threat_type": "climatic",
                "group": ws.cell(r, 3).value,
                "threat": ws.cell(r, 4).value,
                "examples": ws.cell(r, 5).value,
            })
        elif r > 25 and code is None:
            break

    for r in range(34, 400):
        code = ws.cell(r, 2).value
        if isinstance(code, str) and re.match(r"^ANC\d{2}$", code.strip()):
            rows.append({
                "threat_code": code.strip(),
                "threat_type": "non_climatic",
                "group": ws.cell(r, 3).value,
                "threat": ws.cell(r, 4).value,
                "examples": ws.cell(r, 5).value,
            })
        elif r > 56 and ws.cell(r, 2).value is None and ws.cell(r, 3).value is None:
            break

    return pd.DataFrame(rows)


def extract_lookup_security(wb: openpyxl.Workbook, sheet_name: str = "Seguridad") -> pd.DataFrame:
    if sheet_name not in wb.sheetnames:
        return pd.DataFrame()
    
    ws = wb[sheet_name]
    rows: List[Dict[str, Any]] = []

    for r in range(14, 30):
        var = ws.cell(r, 3).value
        if isinstance(var, str) and var.startswith("Seg_"):
            rows.append({
                "column_letter": ws.cell(r, 2).value,
                "variable_raw": var,
                "variable": clean_col(var),
                "covers": ws.cell(r, 4).value,
                "mark_when": ws.cell(r, 5).value,
                "evidence_examples": ws.cell(r, 6).value,
            })

    return pd.DataFrame(rows)


def extract_lookup_traits(wb: openpyxl.Workbook, sheet_name: str = "Rasgos Transformadores") -> pd.DataFrame:
    if sheet_name not in wb.sheetnames:
        return pd.DataFrame()
    
    ws = wb[sheet_name]
    short_defs: Dict[str, Dict[str, Any]] = {}

    for r in range(13, 19):
        trait = ws.cell(r, 2).value
        if isinstance(trait, str) and trait.startswith("T"):
            short_defs[clean_col(trait)] = {"trait_raw": trait, "short_definition": ws.cell(r, 3).value}

    long_defs: Dict[str, Any] = {}
    for r in range(26, 32):
        trait = ws.cell(r, 2).value
        if isinstance(trait, str) and trait.startswith("T"):
            long_defs[clean_col(trait)] = ws.cell(r, 3).value

    rows = []
    for trait_clean, d in short_defs.items():
        rows.append({
            "trait_raw": d["trait_raw"],
            "trait": trait_clean,
            "short_definition": d["short_definition"],
            "long_definition": long_defs.get(trait_clean),
        })
    return pd.DataFrame(rows)


def extract_lookup_gov_gaps(wb: openpyxl.Workbook, sheet_name: str = "Brechas Gobernanza") -> pd.DataFrame:
    if sheet_name not in wb.sheetnames:
        return pd.DataFrame()
    
    ws = wb[sheet_name]
    rows: List[Dict[str, Any]] = []
    current_group: Optional[str] = None

    for r in range(1, 400):
        v = ws.cell(r, 2).value
        if isinstance(v, str) and v.strip().lower().startswith("retos de"):
            current_group = v.strip()

        if isinstance(v, str) and re.match(r"^\d+\.\d+\s", v.strip()):
            code = v.strip().split(" ")[0]
            name = " ".join(v.strip().split(" ")[1:])
            rows.append({
                "gap_code": "B" + code,
                "gap_group": current_group,
                "gap_name": name,
                "gap_description": ws.cell(r, 3).value,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["gap_code"])
    return df


def transform_workbook(input_bytes: bytes, main_sheet: str) -> Tuple[bytes, Dict[str, Any]]:
    """Transform workbook and return bytes + stats."""
    
    # Load workbook
    wb = openpyxl.load_workbook(io.BytesIO(input_bytes))
    
    # Read main sheet
    df_raw = pd.read_excel(io.BytesIO(input_bytes), sheet_name=main_sheet)
    
    raw_cols = list(df_raw.columns)
    cleaned = [clean_col(c) for c in raw_cols]
    cleaned = make_unique(cleaned)
    col_map = dict(zip(raw_cols, cleaned))
    
    fact = df_raw.rename(columns=col_map).copy()
    
    # Standardize missing values
    missing_markers = {"ND", "N/D", "NA", "N.A.", "", " ", "nd", "n/d", "na"}
    for c in fact.columns:
        if fact[c].dtype == object:
            fact[c] = fact[c].apply(lambda x: np.nan if isinstance(x, str) and x.strip() in missing_markers else x)
            fact[c] = fact[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Numeric conversions
    numeric_candidates = [c for c in fact.columns if c.endswith("_n")] + ["seg_total", "t_score"]
    for c in numeric_candidates:
        if c in fact.columns:
            fact[c] = pd.to_numeric(fact[c], errors="coerce")
    
    seg_cols = [c for c in fact.columns if c.startswith("seg_") and c not in ("seg_total", "seg_justificacion")]
    trait_cols = [c for c in fact.columns if re.match(r"^t[1-6]_", c)]
    for c in seg_cols + trait_cols:
        fact[c] = pd.to_numeric(fact[c], errors="coerce").astype("Int64")
    
    # Extract lookups
    lookup_threats = extract_lookup_threats(wb)
    lookup_security = extract_lookup_security(wb)
    lookup_traits = extract_lookup_traits(wb)
    lookup_gov = extract_lookup_gov_gaps(wb)
    
    # Build maps
    threat_map = {}
    if not lookup_threats.empty:
        threat_map = lookup_threats.set_index("threat_code")[["threat_type", "group", "threat"]].to_dict(orient="index")
    
    gap_map = {}
    if not lookup_gov.empty:
        gap_map = lookup_gov.set_index("gap_code")[["gap_group", "gap_name"]].to_dict(orient="index")
    
    sec_map = {}
    if not lookup_security.empty:
        sec_map = lookup_security.set_index("variable")[["covers"]].to_dict(orient="index")
    
    trait_map = {}
    if not lookup_traits.empty:
        trait_map = lookup_traits.set_index("trait")[["short_definition"]].to_dict(orient="index")
    
    # Build tidy tables
    if "id_nbs" not in fact.columns:
        raise ValueError("Expected an 'ID_NbS' column in the main sheet")
    
    # TIDY_THREATS
    clim_cols = [c for c in fact.columns if c.startswith("amenaza_climatica_")]
    non_cols = [c for c in fact.columns if c.startswith("amenaza_no_climatica_")]
    threat_records: List[Dict[str, Any]] = []
    for col in clim_cols + non_cols:
        fallback_type = "climatic" if col in clim_cols else "non_climatic"
        for _, row in fact[["id_nbs", col]].iterrows():
            val = row[col]
            if pd.isna(val):
                continue
            code = extract_threat_code(val)
            rec = {
                "id_nbs": row["id_nbs"],
                "source_column": col,
                "threat_text_raw": val,
                "threat_code": code,
                "threat_type": fallback_type,
            }
            if code and code in threat_map:
                rec["threat_type"] = threat_map[code]["threat_type"]
                rec["threat_group"] = threat_map[code]["group"]
                rec["threat_label"] = threat_map[code]["threat"]
            else:
                rec["threat_group"] = None
                rec["threat_label"] = None
            threat_records.append(rec)
    tidy_threats = pd.DataFrame(threat_records)
    
    # TIDY_GOV_GAPS
    gap_cols = [c for c in fact.columns if c.startswith("brecha_nbs_")]
    gap_records: List[Dict[str, Any]] = []
    for col in gap_cols:
        for _, row in fact[["id_nbs", col]].iterrows():
            val = row[col]
            if pd.isna(val):
                continue
            code = extract_gap_code(val)
            rec = {"id_nbs": row["id_nbs"], "source_column": col, "gap_text_raw": val, "gap_code": code}
            if code and code in gap_map:
                rec["gap_group"] = gap_map[code]["gap_group"]
                rec["gap_name"] = gap_map[code]["gap_name"]
            else:
                rec["gap_group"] = None
                rec["gap_name"] = None
            gap_records.append(rec)
    tidy_gov = pd.DataFrame(gap_records)
    
    # TIDY_SECURITY
    dim_vars = [v for v in lookup_security["variable"].tolist() if v not in ("seg_total", "seg_justificacion")] if not lookup_security.empty else []
    dim_cols = [c for c in fact.columns if c in dim_vars]
    if dim_cols:
        tidy_security = fact[["id_nbs"] + dim_cols].melt(id_vars=["id_nbs"], var_name="dimension", value_name="value")
        tidy_security["value"] = pd.to_numeric(tidy_security["value"], errors="coerce")
        tidy_security = tidy_security.dropna(subset=["value"])
        tidy_security["definition"] = tidy_security["dimension"].map(lambda d: sec_map.get(d, {}).get("covers"))
    else:
        tidy_security = pd.DataFrame(columns=["id_nbs", "dimension", "value", "definition"])
    
    # TIDY_TRAITS
    trait_cols_clean = [c for c in fact.columns if c in lookup_traits["trait"].tolist()] if not lookup_traits.empty else []
    if trait_cols_clean:
        tidy_traits = fact[["id_nbs"] + trait_cols_clean].melt(id_vars=["id_nbs"], var_name="trait", value_name="value")
        tidy_traits["value"] = pd.to_numeric(tidy_traits["value"], errors="coerce")
        tidy_traits = tidy_traits.dropna(subset=["value"])
        tidy_traits["definition"] = tidy_traits["trait"].map(lambda t: trait_map.get(t, {}).get("short_definition"))
    else:
        tidy_traits = pd.DataFrame(columns=["id_nbs", "trait", "value", "definition"])
    
    # QA tables
    missing_summary = pd.DataFrame({
        "column": fact.columns,
        "missing_n": [int(fact[c].isna().sum()) for c in fact.columns],
        "missing_pct": [float(fact[c].isna().mean()) for c in fact.columns],
    }).sort_values("missing_pct", ascending=False)
    
    qa_seg_total = pd.DataFrame()
    if "seg_total" in fact.columns and dim_cols:
        seg_sum = fact[dim_cols].fillna(0).sum(axis=1)
        qa_seg_total = fact[["id_nbs", "seg_total"]].copy()
        qa_seg_total["seg_sum_dims"] = seg_sum
        qa_seg_total["match"] = qa_seg_total["seg_total"].fillna(-999) == qa_seg_total["seg_sum_dims"]
    
    qa_t_score = pd.DataFrame()
    if "t_score" in fact.columns and trait_cols_clean:
        t_sum = fact[trait_cols_clean].fillna(0).sum(axis=1)
        qa_t_score = fact[["id_nbs", "t_score"]].copy()
        qa_t_score["t_sum_traits"] = t_sum
        qa_t_score["match"] = qa_t_score["t_score"].fillna(-999) == qa_t_score["t_sum_traits"]
    
    column_map_df = pd.DataFrame({"raw_column": raw_cols, "clean_column": [col_map[c] for c in raw_cols]})
    
    # Create output workbook
    out_wb = openpyxl.load_workbook(io.BytesIO(input_bytes))
    
    add_df_sheet(out_wb, "COLUMN_MAP", column_map_df)
    add_df_sheet(out_wb, "FACT_NBS", fact)
    add_df_sheet(out_wb, "TIDY_THREATS", tidy_threats)
    add_df_sheet(out_wb, "TIDY_GOV_GAPS", tidy_gov)
    add_df_sheet(out_wb, "TIDY_SECURITY", tidy_security)
    add_df_sheet(out_wb, "TIDY_TRAITS", tidy_traits)
    add_df_sheet(out_wb, "LOOKUP_THREATS", lookup_threats)
    add_df_sheet(out_wb, "LOOKUP_GOV_GAPS", lookup_gov)
    add_df_sheet(out_wb, "LOOKUP_SECURITY", lookup_security)
    add_df_sheet(out_wb, "LOOKUP_TRAITS", lookup_traits)
    add_df_sheet(out_wb, "QA_MISSINGNESS", missing_summary)
    if not qa_seg_total.empty:
        add_df_sheet(out_wb, "QA_SEG_TOTAL", qa_seg_total)
    if not qa_t_score.empty:
        add_df_sheet(out_wb, "QA_T_SCORE", qa_t_score)
    
    # Reorder sheets
    desired_order = [
        main_sheet,
        "FACT_NBS",
        "TIDY_THREATS",
        "TIDY_GOV_GAPS",
        "TIDY_SECURITY",
        "TIDY_TRAITS",
        "LOOKUP_THREATS",
        "LOOKUP_GOV_GAPS",
        "LOOKUP_SECURITY",
        "LOOKUP_TRAITS",
        "QA_MISSINGNESS",
        "QA_SEG_TOTAL",
        "QA_T_SCORE",
        "COLUMN_MAP",
    ]
    existing = out_wb.sheetnames
    order = [name for name in desired_order if name in existing] + [name for name in existing if name not in desired_order]
    out_wb._sheets = [out_wb[name] for name in order]
    
    # Save to bytes
    output = io.BytesIO()
    out_wb.save(output)
    output.seek(0)
    
    # Stats
    stats = {
        "n_cases": len(fact),
        "n_columns": len(fact.columns),
        "n_threats": len(tidy_threats),
        "n_gaps": len(tidy_gov),
        "n_security_records": len(tidy_security),
        "n_traits_records": len(tidy_traits),
        "sheets_created": len([s for s in out_wb.sheetnames if s not in wb.sheetnames]),
    }
    
    return output.getvalue(), stats


# =============================================================================
# UI Components
# =============================================================================

def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>🌿 NbS Analyzer</h1>
        <p>Transform your raw Excel data into an analysis-ready format for Nature-based Solutions portfolio analysis</p>
    </div>
    """, unsafe_allow_html=True)


def render_step_indicator(current_step: int):
    steps = [
        ("1", "Upload File"),
        ("2", "Configure"),
        ("3", "Transform"),
        ("4", "Download"),
    ]
    
    html = '<div class="step-indicator">'
    for i, (num, text) in enumerate(steps, 1):
        status = "completed" if i < current_step else ("active" if i == current_step else "")
        html += f'''
        <div class="step {status}">
            <div class="step-number">{num if i >= current_step else "✓"}</div>
            <div class="step-text">{text}</div>
        </div>
        '''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_stats(stats: Dict[str, Any]):
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{stats['n_cases']}</div>
            <div class="stat-label">NbS Cases</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['n_columns']}</div>
            <div class="stat-label">Columns</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['n_threats']}</div>
            <div class="stat-label">Threat Records</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['n_gaps']}</div>
            <div class="stat-label">Gap Records</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['sheets_created']}</div>
            <div class="stat-label">New Sheets</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main App
# =============================================================================

def main():
    render_hero()
    
    # Initialize session state
    if "transformed_data" not in st.session_state:
        st.session_state.transformed_data = None
    if "transform_stats" not in st.session_state:
        st.session_state.transform_stats = None
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1
    
    # Step indicator
    render_step_indicator(st.session_state.current_step)
    
    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # File upload
        st.markdown("### 📁 Upload Your Excel File")
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["xlsx", "xls"],
            help="Upload your raw NbS analysis Excel file (e.g., Tierra Viva-Analisis NBS.xlsx)"
        )
        
        if uploaded_file:
            st.session_state.current_step = 2
            
            # Read sheet names
            try:
                xl = pd.ExcelFile(uploaded_file)
                sheets = xl.sheet_names
                
                st.markdown("### ⚙️ Configuration")
                
                # Find likely main sheet
                default_sheet = next(
                    (s for s in sheets if "general" in s.lower() and "nbs" in s.lower()),
                    sheets[0] if sheets else None
                )
                
                main_sheet = st.selectbox(
                    "Select Main Data Sheet",
                    sheets,
                    index=sheets.index(default_sheet) if default_sheet in sheets else 0,
                    help="Select the sheet containing the main NbS cases data"
                )
                
                # Info box
                st.markdown(f"""
                <div class="info-box">
                    <h4>📋 File Info</h4>
                    <p><strong>Filename:</strong> {uploaded_file.name}<br>
                    <strong>Sheets found:</strong> {len(sheets)}<br>
                    <strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Transform button
                if st.button("🔄 Transform Data", use_container_width=True):
                    st.session_state.current_step = 3
                    
                    with st.spinner("Transforming data..."):
                        try:
                            uploaded_file.seek(0)
                            input_bytes = uploaded_file.read()
                            
                            transformed_bytes, stats = transform_workbook(input_bytes, main_sheet)
                            
                            st.session_state.transformed_data = transformed_bytes
                            st.session_state.transform_stats = stats
                            st.session_state.current_step = 4
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.markdown(f"""
                            <div class="status-error">
                                <strong>❌ Transformation Error</strong><br>
                                {str(e)}
                            </div>
                            """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Show results if transformed
        if st.session_state.transformed_data:
            st.markdown("### ✅ Transformation Complete!")
            
            st.markdown("""
            <div class="status-success">
                <strong>✓ Success!</strong> Your file has been transformed into an analysis-ready format.
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.transform_stats:
                render_stats(st.session_state.transform_stats)
            
            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"NBS_analysis_ready_{timestamp}.xlsx"
            
            st.download_button(
                label="⬇️ Download Analysis-Ready File",
                data=st.session_state.transformed_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            
            # What's next section
            st.markdown("""
            <div class="info-box">
                <h4>📊 What's Next?</h4>
                <p>Your analysis-ready file contains:
                <ul style="margin-top: 0.5rem; padding-left: 1.5rem; color: var(--text-secondary);">
                    <li><strong>FACT_NBS</strong> — Clean main table with standardized columns</li>
                    <li><strong>TIDY_THREATS</strong> — Long table of threats per case</li>
                    <li><strong>TIDY_GOV_GAPS</strong> — Long table of governance gaps</li>
                    <li><strong>TIDY_SECURITY</strong> — Security dimensions per case</li>
                    <li><strong>TIDY_TRAITS</strong> — Transformative traits per case</li>
                    <li><strong>LOOKUP_*</strong> — Reference tables for codes</li>
                    <li><strong>QA_*</strong> — Data quality summaries</li>
                </ul>
                Run the NbS Analyzer CLI to generate full Storyline A analysis!
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Reset button
            if st.button("🔄 Transform Another File"):
                st.session_state.transformed_data = None
                st.session_state.transform_stats = None
                st.session_state.current_step = 1
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🌿 NbS Analyzer v1.0.0 | Storyline A (Risk-First) | CATIE Consultancy</p>
        <p>Transform → Analyze → Decide</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
