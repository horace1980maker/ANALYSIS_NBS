"""
Storyline B (Benefits/Equity-First) Pipeline Orchestrator.

Implements the complete Storyline B analysis:
- B1: Beneficiary map (who benefits)
- B2: Co-benefits profile (security dimensions)
- B3: Benefits × security cross analysis
- B4: Link co-benefits to threats
- B5: Link benefits to governance gaps
- B6: Shortlists generation
- B7: One-pager generation
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from nbs_analyzer.io import load_workbook, save_csv, save_excel
from nbs_analyzer.schema import REQUIRED_SHEETS, OPTIONAL_SHEETS
from nbs_analyzer.validate import generate_qa_summary
from nbs_analyzer.utils import (
    pick_col,
    load_column_mappings,
    ensure_deterministic_sort,
    create_output_dirs,
    safe_percentage,
    stable_round,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class StorylineBResults:
    """Container for all Storyline B outputs."""
    
    # Beneficiary analysis (B1)
    beneficiary_summary_overall: pd.DataFrame = field(default_factory=pd.DataFrame)
    beneficiary_by_country: pd.DataFrame = field(default_factory=pd.DataFrame)
    beneficiary_by_landscape: pd.DataFrame = field(default_factory=pd.DataFrame)
    beneficiary_by_org: pd.DataFrame = field(default_factory=pd.DataFrame)
    has_numeric_beneficiaries: bool = False
    
    # Security dimensions (B2)
    security_dimension_rates: pd.DataFrame = field(default_factory=pd.DataFrame)
    security_breadth_by_case: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Benefits × Security (B3)
    benefits_security_cross: pd.DataFrame = field(default_factory=pd.DataFrame)
    beneficiary_group_mentions: pd.DataFrame = field(default_factory=pd.DataFrame)
    beneficiary_group_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Threat-Security links (B4)
    threat_to_security_profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    security_to_threat_profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Gap-Security links (B5)
    gapload_by_security_archetype: pd.DataFrame = field(default_factory=pd.DataFrame)
    gaps_by_security_dimension: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Shortlists (B6)
    shortlist_B1_equity_leaders: pd.DataFrame = field(default_factory=pd.DataFrame)
    shortlist_B2_needs_enablers: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # One-pagers (B7)
    onepagers: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # QA
    qa_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Metadata
    input_file: str = ""
    output_dir: str = ""
    run_timestamp: str = ""
    n_cases: int = 0
    missing_lookups: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_beneficiary_keywords(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load beneficiary keywords configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "beneficiary_keywords.yml"
    
    if not config_path.exists():
        return {
            "beneficiary_groups": {
                "women": {"keywords": ["mujer", "mujeres", "women", "woman"]},
                "youth": {"keywords": ["joven", "jovenes", "youth", "young"]},
                "farmers": {"keywords": ["agricultor", "farmer", "campesino"]},
                "indigenous": {"keywords": ["indigena", "indigenous"]},
            },
            "proxy_settings": {"min_description_length": 10}
        }
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: Any) -> str:
    """Normalize text for keyword matching (lowercase, remove accents)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def coerce_to_binary(series: pd.Series) -> pd.Series:
    """Coerce a series to binary (0/1) values."""
    true_values = {True, "1", "SI", "Sí", "si", "sí", "yes", "Yes", "YES", "Y", "y", 1, 1.0}
    false_values = {False, "0", "NO", "No", "no", "N", "n", 0, 0.0}
    
    def convert(x):
        if pd.isna(x):
            return np.nan
        if x in true_values:
            return 1
        if x in false_values:
            return 0
        return np.nan
    
    return series.apply(convert)


def compute_percentile_thresholds(series: pd.Series, n: int) -> Dict[str, float]:
    """Compute percentile thresholds adaptively based on sample size."""
    if n >= 8:
        return {
            "high": series.quantile(0.75),
            "low": series.quantile(0.25),
        }
    else:
        sorted_vals = series.dropna().sort_values()
        k = max(1, int(np.ceil(0.25 * n)))
        return {
            "high": sorted_vals.iloc[-k] if len(sorted_vals) >= k else sorted_vals.max(),
            "low": sorted_vals.iloc[k-1] if len(sorted_vals) >= k else sorted_vals.min(),
        }


def assign_archetype(value: float, thresholds: Dict[str, float]) -> str:
    """Assign archetype based on thresholds."""
    if pd.isna(value):
        return "Unknown"
    if value >= thresholds["high"]:
        return "High"
    elif value <= thresholds["low"]:
        return "Low"
    else:
        return "Medium"


# ---------------------------------------------------------------------------
# B1: Beneficiary Map
# ---------------------------------------------------------------------------

def compute_beneficiary_summary(
    fact_nbs: pd.DataFrame,
    mappings: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Compute beneficiary summary statistics (B1).
    
    Returns dict with:
    - summary_overall: overall beneficiary stats
    - by_country, by_landscape, by_org: grouped stats
    - has_numeric: whether numeric beneficiary data exists
    """
    id_col = pick_col(fact_nbs, mappings.get("id_nbs", ["id_nbs"]))
    n_cases = fact_nbs[id_col].nunique()
    
    # Find beneficiary columns
    direct_n_col = pick_col(fact_nbs, mappings.get("benef_directos_n", []))
    indirect_n_col = pick_col(fact_nbs, mappings.get("benef_indirectos_n", []))
    direct_desc_col = pick_col(fact_nbs, mappings.get("benef_directos_desc", []))
    indirect_desc_col = pick_col(fact_nbs, mappings.get("benef_indirectos_desc", []))
    
    # Grouping columns
    country_col = pick_col(fact_nbs, mappings.get("country", []))
    landscape_col = pick_col(fact_nbs, mappings.get("landscape", []))
    org_col = pick_col(fact_nbs, mappings.get("organization", []))
    
    # Coerce numeric columns
    has_numeric = False
    if direct_n_col:
        fact_nbs["_direct_n"] = pd.to_numeric(fact_nbs[direct_n_col], errors="coerce")
        pct_valid = fact_nbs["_direct_n"].notna().sum() / len(fact_nbs)
        if pct_valid >= 0.3:
            has_numeric = True
    
    if indirect_n_col:
        fact_nbs["_indirect_n"] = pd.to_numeric(fact_nbs[indirect_n_col], errors="coerce")
    
    # Overall summary
    summary = {
        "n_cases": n_cases,
        "pct_direct_count_present": 0.0,
        "pct_indirect_count_present": 0.0,
        "pct_any_description_present": 0.0,
    }
    
    if direct_n_col:
        summary["pct_direct_count_present"] = stable_round(
            100 * fact_nbs["_direct_n"].notna().sum() / n_cases, 1
        )
    
    if indirect_n_col:
        summary["pct_indirect_count_present"] = stable_round(
            100 * fact_nbs.get("_indirect_n", pd.Series()).notna().sum() / n_cases, 1
        )
    
    # Check descriptions
    has_desc = pd.Series([False] * len(fact_nbs))
    for col in [direct_desc_col, indirect_desc_col]:
        if col:
            has_desc = has_desc | (fact_nbs[col].notna() & (fact_nbs[col].str.len() > 5))
    summary["pct_any_description_present"] = stable_round(100 * has_desc.sum() / n_cases, 1)
    
    if has_numeric:
        fact_nbs["_total_benef"] = fact_nbs.get("_direct_n", 0).fillna(0) + \
                                    fact_nbs.get("_indirect_n", 0).fillna(0)
        summary["total_direct"] = int(fact_nbs["_direct_n"].sum())
        summary["total_indirect"] = int(fact_nbs.get("_indirect_n", pd.Series([0])).sum())
        summary["total_beneficiaries"] = int(fact_nbs["_total_benef"].sum())
        summary["mean_total_per_case"] = stable_round(fact_nbs["_total_benef"].mean(), 1)
        summary["median_total_per_case"] = stable_round(fact_nbs["_total_benef"].median(), 1)
    
    overall_df = pd.DataFrame([summary])
    
    # Grouped summaries
    def compute_group_stats(group_col: Optional[str]) -> pd.DataFrame:
        if not group_col or not has_numeric:
            return pd.DataFrame()
        
        grouped = fact_nbs.groupby(group_col).agg(
            n_cases=(id_col, "nunique"),
            total_direct=("_direct_n", "sum"),
            total_indirect=("_indirect_n", lambda x: x.sum() if "_indirect_n" in fact_nbs else 0),
            total_beneficiaries=("_total_benef", "sum"),
            mean_per_case=("_total_benef", "mean"),
            median_per_case=("_total_benef", "median"),
        ).reset_index()
        
        for col in ["mean_per_case", "median_per_case"]:
            grouped[col] = grouped[col].apply(lambda x: stable_round(x, 1))
        
        return ensure_deterministic_sort(grouped, [group_col])
    
    return {
        "summary_overall": overall_df,
        "by_country": compute_group_stats(country_col),
        "by_landscape": compute_group_stats(landscape_col),
        "by_org": compute_group_stats(org_col),
        "has_numeric": has_numeric,
    }


# ---------------------------------------------------------------------------
# B2: Security Dimensions Profile
# ---------------------------------------------------------------------------

def compute_security_profile(
    tidy_security: pd.DataFrame,
    fact_nbs: pd.DataFrame,
    mappings: Dict[str, List[str]]
) -> Dict[str, pd.DataFrame]:
    """
    Compute security dimension rates and breadth per case (B2).
    """
    id_col = pick_col(tidy_security, mappings.get("id_nbs", ["id_nbs"]))
    dim_col = pick_col(tidy_security, mappings.get("dimension", ["dimension"]))
    val_col = pick_col(tidy_security, mappings.get("value", ["value"]))
    
    fact_id_col = pick_col(fact_nbs, mappings.get("id_nbs", ["id_nbs"]))
    n_cases = fact_nbs[fact_id_col].nunique()
    
    # Coerce to binary
    tidy_security["_value"] = coerce_to_binary(tidy_security[val_col])
    
    # Security dimension rates
    flagged = tidy_security[tidy_security["_value"] == 1]
    dim_rates = flagged.groupby(dim_col).agg(
        n_cases_flagged=(id_col, "nunique")
    ).reset_index()
    dim_rates["pct_cases_flagged"] = dim_rates["n_cases_flagged"].apply(
        lambda x: stable_round(100 * x / n_cases, 1)
    )
    dim_rates = ensure_deterministic_sort(dim_rates, ["pct_cases_flagged"], ascending=False)
    dim_rates.columns = ["dimension", "n_cases_flagged", "pct_cases_flagged"]
    
    # Security breadth per case (SBS)
    sbs = flagged.groupby(id_col).agg(
        SBS=("_value", "sum")
    ).reset_index()
    sbs.columns = ["id_nbs", "SBS"]
    
    # Assign archetypes
    thresholds = compute_percentile_thresholds(sbs["SBS"], len(sbs))
    sbs["security_archetype"] = sbs["SBS"].apply(lambda x: assign_archetype(x, thresholds))
    sbs = ensure_deterministic_sort(sbs, ["SBS"], ascending=False)
    
    return {
        "dimension_rates": dim_rates,
        "breadth_by_case": sbs,
        "thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
# B3: Benefits × Security Cross Analysis
# ---------------------------------------------------------------------------

def compute_benefits_security_cross(
    fact_nbs: pd.DataFrame,
    sbs_df: pd.DataFrame,
    mappings: Dict[str, List[str]],
    has_numeric: bool
) -> pd.DataFrame:
    """
    Compute cross-analysis of beneficiaries vs security breadth (B3).
    """
    if not has_numeric:
        return pd.DataFrame()
    
    id_col = pick_col(fact_nbs, mappings.get("id_nbs", ["id_nbs"]))
    
    # Merge SBS with beneficiary totals
    merged = fact_nbs[[id_col, "_total_benef"]].merge(
        sbs_df[["id_nbs", "SBS", "security_archetype"]],
        left_on=id_col,
        right_on="id_nbs",
        how="left"
    )
    
    # Beneficiary tiers
    thresholds = compute_percentile_thresholds(merged["_total_benef"], len(merged))
    merged["benef_tier"] = merged["_total_benef"].apply(lambda x: assign_archetype(x, thresholds))
    
    # Cross stats: avg SBS by beneficiary tier
    tier_stats = merged.groupby("benef_tier").agg(
        n_cases=("id_nbs", "count"),
        avg_SBS=("SBS", "mean"),
        avg_benef=("_total_benef", "mean"),
    ).reset_index()
    
    # Archetype stats: avg beneficiaries by security archetype
    arch_stats = merged.groupby("security_archetype").agg(
        n_cases=("id_nbs", "count"),
        avg_SBS=("SBS", "mean"),
        avg_benef=("_total_benef", "mean"),
    ).reset_index()
    
    # Combine into one table
    tier_stats["group_type"] = "beneficiary_tier"
    tier_stats = tier_stats.rename(columns={"benef_tier": "group"})
    arch_stats["group_type"] = "security_archetype"
    arch_stats = arch_stats.rename(columns={"security_archetype": "group"})
    
    result = pd.concat([tier_stats, arch_stats], ignore_index=True)
    for col in ["avg_SBS", "avg_benef"]:
        result[col] = result[col].apply(lambda x: stable_round(x, 2))
    
    return ensure_deterministic_sort(result, ["group_type", "group"])


def compute_beneficiary_keywords_proxy(
    fact_nbs: pd.DataFrame,
    mappings: Dict[str, List[str]],
    keywords_config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """
    Scan beneficiary descriptions for priority group keywords (B3 proxy).
    """
    id_col = pick_col(fact_nbs, mappings.get("id_nbs", ["id_nbs"]))
    
    # Find description columns
    desc_cols = []
    for key in ["benef_directos_desc", "benef_indirectos_desc", "description"]:
        col = pick_col(fact_nbs, mappings.get(key, []))
        if col:
            desc_cols.append(col)
    
    if not desc_cols:
        return {"mentions": pd.DataFrame(), "summary": pd.DataFrame()}
    
    groups = keywords_config.get("beneficiary_groups", {})
    
    # Scan each case
    records = []
    for _, row in fact_nbs.iterrows():
        nbs_id = row[id_col]
        combined_text = " ".join(
            str(row.get(col, "")) for col in desc_cols if pd.notna(row.get(col))
        )
        normalized = normalize_text(combined_text)
        
        for group_name, group_info in groups.items():
            keywords = group_info.get("keywords", [])
            found = any(normalize_text(kw) in normalized for kw in keywords)
            records.append({
                "id_nbs": nbs_id,
                "group": group_name,
                "mentioned": 1 if found else 0,
            })
    
    mentions_df = pd.DataFrame(records)
    
    # Summary: count per group
    summary = mentions_df.groupby("group").agg(
        n_cases_mentioned=("mentioned", "sum"),
        pct_cases=("mentioned", lambda x: stable_round(100 * x.sum() / len(fact_nbs), 1))
    ).reset_index()
    summary = ensure_deterministic_sort(summary, ["n_cases_mentioned"], ascending=False)
    
    # Pivot mentions for per-case view
    mentions_pivot = mentions_df.pivot(index="id_nbs", columns="group", values="mentioned").reset_index()
    mentions_pivot = ensure_deterministic_sort(mentions_pivot, ["id_nbs"])
    
    return {"mentions": mentions_pivot, "summary": summary}


# ---------------------------------------------------------------------------
# B4: Threat-Security Links
# ---------------------------------------------------------------------------

def compute_threat_security_links(
    tidy_threats: pd.DataFrame,
    tidy_security: pd.DataFrame,
    sbs_df: pd.DataFrame,
    mappings: Dict[str, List[str]]
) -> Dict[str, pd.DataFrame]:
    """
    Compute threat-to-security and security-to-threat profiles (B4).
    """
    threat_id_col = pick_col(tidy_threats, mappings.get("id_nbs", ["id_nbs"]))
    threat_code_col = pick_col(tidy_threats, mappings.get("threat_code", ["threat_code"]))
    
    sec_id_col = pick_col(tidy_security, mappings.get("id_nbs", ["id_nbs"]))
    dim_col = pick_col(tidy_security, mappings.get("dimension", ["dimension"]))
    val_col = pick_col(tidy_security, mappings.get("value", ["value"]))
    
    # Coerce security values
    tidy_security["_value"] = coerce_to_binary(tidy_security[val_col])
    flagged_sec = tidy_security[tidy_security["_value"] == 1]
    
    # Threat -> Security profile
    threat_sec_records = []
    for threat_code in tidy_threats[threat_code_col].dropna().unique():
        threat_cases = set(tidy_threats[tidy_threats[threat_code_col] == threat_code][threat_id_col])
        if not threat_cases:
            continue
        
        # Security dimensions for these cases
        sec_for_threat = flagged_sec[flagged_sec[sec_id_col].isin(threat_cases)]
        n_cases = len(threat_cases)
        
        # Dimension rates within threat
        dim_counts = sec_for_threat.groupby(dim_col)[sec_id_col].nunique()
        
        # Avg SBS
        avg_sbs = sbs_df[sbs_df["id_nbs"].isin(threat_cases)]["SBS"].mean()
        
        for dim, count in dim_counts.items():
            threat_sec_records.append({
                "threat_code": threat_code,
                "dimension": dim,
                "n_cases": count,
                "pct_of_threat_cases": stable_round(100 * count / n_cases, 1),
                "avg_SBS": stable_round(avg_sbs, 2),
            })
    
    threat_to_sec = pd.DataFrame(threat_sec_records)
    threat_to_sec = ensure_deterministic_sort(threat_to_sec, ["threat_code", "n_cases"], ascending=False)
    
    # Security -> Threat profile
    sec_threat_records = []
    for dim in flagged_sec[dim_col].dropna().unique():
        dim_cases = set(flagged_sec[flagged_sec[dim_col] == dim][sec_id_col])
        if not dim_cases:
            continue
        
        # Threats for these cases
        threats_for_dim = tidy_threats[tidy_threats[threat_id_col].isin(dim_cases)]
        threat_counts = threats_for_dim.groupby(threat_code_col)[threat_id_col].nunique()
        
        for threat, count in threat_counts.nlargest(10).items():
            sec_threat_records.append({
                "dimension": dim,
                "threat_code": threat,
                "n_cases": count,
            })
    
    sec_to_threat = pd.DataFrame(sec_threat_records)
    sec_to_threat = ensure_deterministic_sort(sec_to_threat, ["dimension", "n_cases"], ascending=False)
    
    return {
        "threat_to_security": threat_to_sec,
        "security_to_threat": sec_to_threat,
    }


# ---------------------------------------------------------------------------
# B5: Gap-Security Links
# ---------------------------------------------------------------------------

def compute_gap_security_links(
    tidy_gov_gaps: pd.DataFrame,
    sbs_df: pd.DataFrame,
    tidy_security: pd.DataFrame,
    mappings: Dict[str, List[str]]
) -> Dict[str, pd.DataFrame]:
    """
    Compute governance gap load by security archetype and gaps per dimension (B5).
    """
    gap_id_col = pick_col(tidy_gov_gaps, mappings.get("id_nbs", ["id_nbs"]))
    gap_code_col = pick_col(tidy_gov_gaps, mappings.get("gap_code", ["gap_code"]))
    
    sec_id_col = pick_col(tidy_security, mappings.get("id_nbs", ["id_nbs"]))
    dim_col = pick_col(tidy_security, mappings.get("dimension", ["dimension"]))
    val_col = pick_col(tidy_security, mappings.get("value", ["value"]))
    
    # Gap load per case
    ggl = tidy_gov_gaps.groupby(gap_id_col)[gap_code_col].nunique().reset_index()
    ggl.columns = ["id_nbs", "GGL"]
    
    # Merge with SBS archetypes
    merged = ggl.merge(sbs_df[["id_nbs", "security_archetype"]], on="id_nbs", how="left")
    
    # Stats by archetype
    arch_stats = merged.groupby("security_archetype").agg(
        n_cases=("id_nbs", "count"),
        mean_GGL=("GGL", "mean"),
        median_GGL=("GGL", "median"),
        q25_GGL=("GGL", lambda x: x.quantile(0.25)),
        q75_GGL=("GGL", lambda x: x.quantile(0.75)),
    ).reset_index()
    
    for col in ["mean_GGL", "median_GGL", "q25_GGL", "q75_GGL"]:
        arch_stats[col] = arch_stats[col].apply(lambda x: stable_round(x, 2))
    
    # Gaps by security dimension
    tidy_security["_value"] = coerce_to_binary(tidy_security[val_col])
    flagged_sec = tidy_security[tidy_security["_value"] == 1]
    
    gap_dim_records = []
    for dim in flagged_sec[dim_col].dropna().unique():
        dim_cases = set(flagged_sec[flagged_sec[dim_col] == dim][sec_id_col])
        
        gaps_for_dim = tidy_gov_gaps[tidy_gov_gaps[gap_id_col].isin(dim_cases)]
        gap_counts = gaps_for_dim.groupby(gap_code_col)[gap_id_col].nunique()
        
        for gap, count in gap_counts.nlargest(10).items():
            gap_dim_records.append({
                "dimension": dim,
                "gap_code": gap,
                "n_cases": count,
            })
    
    gaps_by_dim = pd.DataFrame(gap_dim_records)
    gaps_by_dim = ensure_deterministic_sort(gaps_by_dim, ["dimension", "n_cases"], ascending=False)
    
    return {
        "gapload_by_archetype": arch_stats,
        "gaps_by_dimension": gaps_by_dim,
        "ggl_by_case": ggl,
    }


# ---------------------------------------------------------------------------
# B6: Shortlists
# ---------------------------------------------------------------------------

def compute_shortlists_b(
    fact_nbs: pd.DataFrame,
    sbs_df: pd.DataFrame,
    ggl_df: pd.DataFrame,
    mappings: Dict[str, List[str]],
    has_numeric: bool,
    beneficiary_mentions: Optional[pd.DataFrame] = None
) -> Dict[str, pd.DataFrame]:
    """
    Compute Storyline B shortlists (B6).
    
    B1 "Equity + breadth leaders": High SBS + (High beneficiaries OR mentions priority groups)
    B2 "High benefit potential but governance-heavy": High SBS + High GGL
    """
    id_col = pick_col(fact_nbs, mappings.get("id_nbs", ["id_nbs"]))
    n_cases = len(fact_nbs)
    
    # Merge scores
    merged = fact_nbs[[id_col]].copy()
    merged = merged.merge(sbs_df[["id_nbs", "SBS", "security_archetype"]], 
                          left_on=id_col, right_on="id_nbs", how="left")
    merged = merged.merge(ggl_df[["id_nbs", "GGL"]], on="id_nbs", how="left")
    
    # SBS thresholds (top 20%)
    sbs_80 = merged["SBS"].quantile(0.80) if len(merged) >= 5 else merged["SBS"].max()
    ggl_60 = merged["GGL"].quantile(0.60) if len(merged) >= 5 else merged["GGL"].median()
    ggl_40 = merged["GGL"].quantile(0.40) if len(merged) >= 5 else merged["GGL"].median()
    
    # Add beneficiary info
    if has_numeric and "_total_benef" in fact_nbs.columns:
        merged = merged.merge(
            fact_nbs[[id_col, "_total_benef"]],
            left_on="id_nbs", right_on=id_col, how="left", suffixes=("", "_y")
        )
        benef_80 = merged["_total_benef"].quantile(0.80)
        merged["high_benef"] = merged["_total_benef"] >= benef_80
    else:
        merged["high_benef"] = False
        merged["_total_benef"] = np.nan
    
    # Check beneficiary mentions
    merged["has_mentions"] = False
    if beneficiary_mentions is not None and not beneficiary_mentions.empty:
        mention_cols = [c for c in beneficiary_mentions.columns if c != "id_nbs"]
        if mention_cols:
            mentions_sum = beneficiary_mentions.set_index("id_nbs")[mention_cols].sum(axis=1)
            merged = merged.merge(
                mentions_sum.rename("_mentions_count").reset_index(),
                on="id_nbs", how="left"
            )
            merged["has_mentions"] = merged["_mentions_count"].fillna(0) > 0
    
    # Check descriptions
    desc_cols = []
    for key in ["benef_directos_desc", "benef_indirectos_desc"]:
        col = pick_col(fact_nbs, mappings.get(key, []))
        if col:
            desc_cols.append(col)
    
    merged["has_description"] = False
    if desc_cols:
        for col in desc_cols:
            if col in fact_nbs.columns:
                merged = merged.merge(
                    fact_nbs[[id_col, col]].rename(columns={col: f"_{col}"}),
                    left_on="id_nbs", right_on=id_col, how="left", suffixes=("", "_z")
                )
                merged["has_description"] = merged["has_description"] | (
                    merged[f"_{col}"].notna() & (merged[f"_{col}"].str.len() > 10)
                )
    
    # B1: Equity + breadth leaders
    # High SBS AND (high beneficiaries OR has mentions OR has description)
    # Optional: GGL <= 60th percentile for implementability
    b1_mask = (
        (merged["SBS"] >= sbs_80) &
        (merged["high_benef"] | merged["has_mentions"] | merged["has_description"]) &
        (merged["GGL"] <= ggl_60)
    )
    
    b1 = merged[b1_mask].copy()
    b1 = ensure_deterministic_sort(b1, ["SBS", "_total_benef"], ascending=False)
    
    # B2: High benefit potential but governance-heavy
    # High SBS AND High GGL (top 40%)
    ggl_high_threshold = merged["GGL"].quantile(0.60) if len(merged) >= 5 else merged["GGL"].median()
    b2_mask = (merged["SBS"] >= sbs_80) & (merged["GGL"] >= ggl_high_threshold)
    
    b2 = merged[b2_mask].copy()
    b2 = ensure_deterministic_sort(b2, ["SBS", "GGL"], ascending=False)
    
    # Clean up output columns
    output_cols = ["id_nbs", "SBS", "security_archetype", "GGL"]
    if has_numeric:
        output_cols.append("_total_benef")
    
    b1_out = b1[[c for c in output_cols if c in b1.columns]].copy()
    b2_out = b2[[c for c in output_cols if c in b2.columns]].copy()
    
    if "_total_benef" in b1_out.columns:
        b1_out = b1_out.rename(columns={"_total_benef": "total_beneficiaries"})
    if "_total_benef" in b2_out.columns:
        b2_out = b2_out.rename(columns={"_total_benef": "total_beneficiaries"})
    
    return {
        "B1_equity_leaders": b1_out,
        "B2_needs_enablers": b2_out,
        "thresholds": {
            "sbs_80": sbs_80,
            "ggl_60": ggl_60,
            "ggl_40": ggl_40,
            "n_cases": n_cases,
        }
    }


# ---------------------------------------------------------------------------
# B7: One-Pagers
# ---------------------------------------------------------------------------

def build_onepagers_b(
    data: Dict[str, pd.DataFrame],
    results: StorylineBResults,
    mappings: Dict[str, List[str]],
    keywords_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Build one-pager data for each NbS case (B7 - Storyline B version).
    """
    fact = data.get("FACT_NBS", pd.DataFrame())
    if fact.empty:
        return pd.DataFrame()
    
    id_col = pick_col(fact, mappings.get("id_nbs", ["id_nbs"]))
    country_col = pick_col(fact, mappings.get("country", []))
    landscape_col = pick_col(fact, mappings.get("landscape", []))
    org_col = pick_col(fact, mappings.get("organization", []))
    combo_col = pick_col(fact, mappings.get("combination", []))
    desc_col = pick_col(fact, mappings.get("description", []))
    
    # Beneficiary columns
    direct_n_col = pick_col(fact, mappings.get("benef_directos_n", []))
    indirect_n_col = pick_col(fact, mappings.get("benef_indirectos_n", []))
    direct_desc_col = pick_col(fact, mappings.get("benef_directos_desc", []))
    indirect_desc_col = pick_col(fact, mappings.get("benef_indirectos_desc", []))
    
    # Get tidy data
    tidy_threats = data.get("TIDY_THREATS", pd.DataFrame())
    tidy_gov = data.get("TIDY_GOV_GAPS", pd.DataFrame())
    tidy_security = data.get("TIDY_SECURITY", pd.DataFrame())
    tidy_traits = data.get("TIDY_TRAITS", pd.DataFrame())
    
    threat_id_col = pick_col(tidy_threats, mappings.get("id_nbs", []))
    threat_code_col = pick_col(tidy_threats, mappings.get("threat_code", []))
    
    gap_id_col = pick_col(tidy_gov, mappings.get("id_nbs", []))
    gap_code_col = pick_col(tidy_gov, mappings.get("gap_code", []))
    
    sec_id_col = pick_col(tidy_security, mappings.get("id_nbs", []))
    dim_col = pick_col(tidy_security, mappings.get("dimension", []))
    sec_val_col = pick_col(tidy_security, mappings.get("value", []))
    
    trait_id_col = pick_col(tidy_traits, mappings.get("id_nbs", []))
    trait_col = pick_col(tidy_traits, mappings.get("trait", []))
    trait_val_col = pick_col(tidy_traits, mappings.get("value", []))
    
    records = []
    
    for _, fact_row in fact.iterrows():
        nbs_id = fact_row[id_col]
        record = {"id_nbs": nbs_id}
        
        # Identity
        if country_col:
            record["country"] = fact_row.get(country_col)
        if landscape_col:
            record["landscape"] = fact_row.get(landscape_col)
        if org_col:
            record["organization"] = fact_row.get(org_col)
        if combo_col:
            record["combination"] = fact_row.get(combo_col)
        if desc_col:
            record["description"] = fact_row.get(desc_col)
        
        # Beneficiaries
        if direct_n_col:
            val = pd.to_numeric(fact_row.get(direct_n_col), errors="coerce")
            record["direct_beneficiaries_n"] = val if pd.notna(val) else None
        if indirect_n_col:
            val = pd.to_numeric(fact_row.get(indirect_n_col), errors="coerce")
            record["indirect_beneficiaries_n"] = val if pd.notna(val) else None
        if direct_desc_col:
            record["direct_beneficiaries_desc"] = fact_row.get(direct_desc_col)
        if indirect_desc_col:
            record["indirect_beneficiaries_desc"] = fact_row.get(indirect_desc_col)
        
        # Keyword mentions (if proxy mode)
        if not results.beneficiary_group_mentions.empty:
            mentions = results.beneficiary_group_mentions
            if nbs_id in mentions["id_nbs"].values:
                case_mentions = mentions[mentions["id_nbs"] == nbs_id].iloc[0]
                groups = keywords_config.get("beneficiary_groups", {})
                mentioned = [g for g in groups if case_mentions.get(g, 0) == 1]
                record["keyword_groups_mentioned"] = "; ".join(mentioned)
        
        # Security dimensions
        if sec_id_col and dim_col and sec_val_col:
            case_sec = tidy_security[tidy_security[sec_id_col] == nbs_id].copy()
            case_sec["_val"] = coerce_to_binary(case_sec[sec_val_col])
            flagged = case_sec[case_sec["_val"] == 1][dim_col].dropna().tolist()
            record["security_dimensions"] = "; ".join(sorted(set(flagged)))
            record["SBS"] = len(set(flagged))
        
        # Threats
        if threat_id_col and threat_code_col:
            case_threats = tidy_threats[tidy_threats[threat_id_col] == nbs_id]
            codes = case_threats[threat_code_col].dropna().tolist()
            ac_codes = [c for c in codes if str(c).upper().startswith("AC") and not str(c).upper().startswith("ANC")]
            anc_codes = [c for c in codes if str(c).upper().startswith("ANC")]
            record["threats_AC"] = "; ".join(sorted(set(ac_codes)))
            record["threats_ANC"] = "; ".join(sorted(set(anc_codes)))
            record["threats_count"] = len(set(codes))
        
        # Governance gaps
        if gap_id_col and gap_code_col:
            case_gaps = tidy_gov[tidy_gov[gap_id_col] == nbs_id]
            gaps = case_gaps[gap_code_col].dropna().tolist()
            record["governance_gaps"] = "; ".join(sorted(set(gaps)))
            record["GGL"] = len(set(gaps))
        
        # Traits
        if trait_id_col and trait_col and trait_val_col and not tidy_traits.empty:
            case_traits = tidy_traits[tidy_traits[trait_id_col] == nbs_id].copy()
            case_traits["_val"] = coerce_to_binary(case_traits[trait_val_col])
            flagged = case_traits[case_traits["_val"] == 1][trait_col].dropna().tolist()
            record["traits"] = "; ".join(sorted(set(flagged)))
            record["TTS"] = len(set(flagged))
        
        # Shortlist status
        record["in_B1_equity_leaders"] = nbs_id in results.shortlist_B1_equity_leaders["id_nbs"].values
        record["in_B2_needs_enablers"] = nbs_id in results.shortlist_B2_needs_enablers["id_nbs"].values
        
        records.append(record)
    
    onepagers = pd.DataFrame(records)
    return ensure_deterministic_sort(onepagers, ["id_nbs"])


# ---------------------------------------------------------------------------
# Plot Generation
# ---------------------------------------------------------------------------

def generate_plots_b(
    results: StorylineBResults,
    output_dir: Path,
    fact_nbs: pd.DataFrame,
    mappings: Dict[str, List[str]]
) -> Dict[str, Path]:
    """Generate all Storyline B visualization plots."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths: Dict[str, Path] = {}
    
    # Plot configuration
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })
    
    # 1. Security dimension rates
    if not results.security_dimension_rates.empty:
        fig, ax = plt.subplots()
        data = results.security_dimension_rates.head(10)
        
        y_pos = np.arange(len(data))
        bars = ax.barh(y_pos, data["pct_cases_flagged"], align="center", color="#2196f3")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data["dimension"].tolist())
        ax.invert_yaxis()
        ax.set_xlabel("% of Cases Flagged")
        ax.set_title("Security Dimension Coverage Rates")
        
        for bar, val in zip(bars, data["pct_cases_flagged"]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                   f"{val}%", va="center", fontsize=9)
        
        plt.tight_layout()
        path = figures_dir / "security_dimension_rates.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["security_dimension_rates"] = path
        logger.info(f"Generated: {path}")
    
    # 2. Security breadth distribution
    if not results.security_breadth_by_case.empty:
        fig, ax = plt.subplots()
        arch_counts = results.security_breadth_by_case["security_archetype"].value_counts()
        
        colors = {"High": "#4caf50", "Medium": "#ff9800", "Low": "#f44336", "Unknown": "#9e9e9e"}
        bar_colors = [colors.get(a, "#9e9e9e") for a in arch_counts.index]
        
        bars = ax.bar(arch_counts.index, arch_counts.values, color=bar_colors)
        ax.set_xlabel("Security Archetype")
        ax.set_ylabel("Number of Cases")
        ax.set_title("Security Breadth Distribution (SBS Archetypes)")
        
        for bar, val in zip(bars, arch_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, 
                   str(val), ha="center", fontsize=10)
        
        plt.tight_layout()
        path = figures_dir / "security_breadth_distribution.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["security_breadth_distribution"] = path
        logger.info(f"Generated: {path}")
    
    # 3. Beneficiaries by landscape (if numeric)
    if results.has_numeric_beneficiaries and not results.beneficiary_by_landscape.empty:
        fig, ax = plt.subplots()
        data = results.beneficiary_by_landscape.sort_values("total_beneficiaries", ascending=False).head(10)
        
        y_pos = np.arange(len(data))
        landscape_col = data.columns[0]  # First column is the grouping column
        bars = ax.barh(y_pos, data["total_beneficiaries"], align="center", color="#9c27b0")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data[landscape_col].tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Total Beneficiaries")
        ax.set_title("Beneficiaries by Landscape")
        
        for bar, val in zip(bars, data["total_beneficiaries"]):
            ax.text(val + max(data["total_beneficiaries"])*0.01, 
                   bar.get_y() + bar.get_height()/2,
                   f"{int(val):,}", va="center", fontsize=9)
        
        plt.tight_layout()
        path = figures_dir / "beneficiaries_by_landscape.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["beneficiaries_by_landscape"] = path
        logger.info(f"Generated: {path}")
    
    # 4. Beneficiaries vs Security scatter (if numeric)
    if results.has_numeric_beneficiaries and not results.security_breadth_by_case.empty:
        id_col = pick_col(fact_nbs, mappings.get("id_nbs", ["id_nbs"]))
        if "_total_benef" in fact_nbs.columns:
            merged = fact_nbs[[id_col, "_total_benef"]].merge(
                results.security_breadth_by_case[["id_nbs", "SBS"]],
                left_on=id_col, right_on="id_nbs", how="inner"
            )
            
            if len(merged) >= 2:
                fig, ax = plt.subplots()
                
                ax.scatter(merged["SBS"], merged["_total_benef"], 
                          alpha=0.7, s=60, edgecolors="white", linewidth=0.5, c="#2196f3")
                
                ax.set_xlabel("Security Breadth Score (SBS)")
                ax.set_ylabel("Total Beneficiaries")
                ax.set_title("Beneficiaries vs. Security Breadth")
                
                # Add trend line if enough points
                if len(merged) >= 5:
                    z = np.polyfit(merged["SBS"], merged["_total_benef"], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(merged["SBS"].min(), merged["SBS"].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", alpha=0.5, label="Trend")
                    ax.legend()
                
                plt.tight_layout()
                path = figures_dir / "beneficiaries_vs_security.png"
                fig.savefig(path, bbox_inches="tight")
                plt.close(fig)
                plot_paths["beneficiaries_vs_security"] = path
                logger.info(f"Generated: {path}")
    
    return plot_paths


# ---------------------------------------------------------------------------
# Apply Filters (reuse from storyline_a)
# ---------------------------------------------------------------------------

def apply_filters_b(
    data: Dict[str, pd.DataFrame],
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, pd.DataFrame]:
    """Apply optional filters to FACT_NBS and propagate to tidy tables."""
    if not filters:
        return data
    
    mappings = load_column_mappings()
    fact = data.get("FACT_NBS")
    
    if fact is None or fact.empty:
        return data
    
    id_col = pick_col(fact, mappings.get("id_nbs", ["id_nbs"]))
    
    for filter_key, filter_value in filters.items():
        col_candidates = mappings.get(filter_key, [filter_key])
        actual_col = pick_col(fact, col_candidates)
        
        if actual_col:
            logger.info(f"Filtering {actual_col} = {filter_value}")
            fact = fact[fact[actual_col] == filter_value]
    
    if fact.empty:
        logger.warning("Filters resulted in empty dataset!")
        return data
    
    data["FACT_NBS"] = fact
    
    valid_ids = set(fact[id_col].dropna().unique())
    
    for sheet_name in ["TIDY_THREATS", "TIDY_GOV_GAPS", "TIDY_SECURITY", "TIDY_TRAITS"]:
        if sheet_name in data:
            df = data[sheet_name]
            tidy_id_col = pick_col(df, mappings.get("id_nbs", ["id_nbs"]))
            if tidy_id_col:
                data[sheet_name] = df[df[tidy_id_col].isin(valid_ids)]
    
    return data


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run_storyline_b(
    input_path: str | Path,
    output_dir: str | Path,
    filters: Optional[Dict[str, Any]] = None
) -> StorylineBResults:
    """
    Execute the complete Storyline B (Benefits/Equity-First) analysis pipeline.
    
    Args:
        input_path: Path to analysis-ready Excel workbook
        output_dir: Directory for outputs
        filters: Optional filters (country, landscape, org)
    
    Returns:
        StorylineBResults with all outputs
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    results = StorylineBResults(
        input_file=str(input_path),
        output_dir=str(output_dir),
        run_timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"Starting Storyline B analysis: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directories
    dirs = create_output_dirs(output_dir)
    
    # Load column mappings and keywords config
    mappings = load_column_mappings()
    keywords_config = load_beneficiary_keywords()
    
    # Step 0: Load data
    logger.info("Loading workbook...")
    
    # Make TIDY_TRAITS optional for Storyline B
    required_sheets_b = ["FACT_NBS", "TIDY_THREATS", "TIDY_GOV_GAPS", "TIDY_SECURITY"]
    optional_sheets_b = ["TIDY_TRAITS", "LOOKUP_THREATS", "LOOKUP_GOV_GAPS", 
                         "LOOKUP_SECURITY", "LOOKUP_TRAITS"]
    
    data, missing_optional = load_workbook(
        input_path,
        required_sheets=required_sheets_b,
        optional_sheets=optional_sheets_b
    )
    results.missing_lookups = missing_optional
    
    if missing_optional:
        results.warnings.append(f"Missing optional sheets: {missing_optional}")
    
    # Apply filters
    if filters:
        logger.info(f"Applying filters: {filters}")
        data = apply_filters_b(data, filters)
    
    # Get case count
    fact = data["FACT_NBS"]
    id_col = pick_col(fact, mappings.get("id_nbs", ["id_nbs"]))
    results.n_cases = fact[id_col].nunique()
    logger.info(f"Analyzing {results.n_cases} NbS cases")
    
    # Step 1: Validate data
    logger.info("Validating data...")
    results.qa_summary = generate_qa_summary(
        data, 
        dirs["tables"] / "qa_summary_storyline_b.csv"
    )
    
    # Step B1: Beneficiary map
    logger.info("B1: Analyzing beneficiary map...")
    benef_results = compute_beneficiary_summary(fact, mappings)
    results.beneficiary_summary_overall = benef_results["summary_overall"]
    results.beneficiary_by_country = benef_results["by_country"]
    results.beneficiary_by_landscape = benef_results["by_landscape"]
    results.beneficiary_by_org = benef_results["by_org"]
    results.has_numeric_beneficiaries = benef_results["has_numeric"]
    
    save_csv(results.beneficiary_summary_overall, dirs["tables"] / "beneficiary_summary_overall.csv")
    if not results.beneficiary_by_country.empty:
        save_csv(results.beneficiary_by_country, dirs["tables"] / "beneficiary_by_country.csv")
    if not results.beneficiary_by_landscape.empty:
        save_csv(results.beneficiary_by_landscape, dirs["tables"] / "beneficiary_by_landscape.csv")
    if not results.beneficiary_by_org.empty:
        save_csv(results.beneficiary_by_org, dirs["tables"] / "beneficiary_by_org.csv")
    
    if not results.has_numeric_beneficiaries:
        results.warnings.append(
            "Numeric beneficiary counts not available or insufficient (<30% non-null). "
            "Using keyword proxy mode for equity analysis."
        )
    
    # Step B2: Security dimensions profile
    logger.info("B2: Analyzing security dimensions profile...")
    sec_results = compute_security_profile(data["TIDY_SECURITY"], fact, mappings)
    results.security_dimension_rates = sec_results["dimension_rates"]
    results.security_breadth_by_case = sec_results["breadth_by_case"]
    
    save_csv(results.security_dimension_rates, dirs["tables"] / "security_dimension_rates.csv")
    save_csv(results.security_breadth_by_case, dirs["tables"] / "security_breadth_by_case.csv")
    
    # Step B3: Benefits × Security cross analysis
    logger.info("B3: Cross-analyzing benefits and security...")
    if results.has_numeric_beneficiaries:
        results.benefits_security_cross = compute_benefits_security_cross(
            fact, results.security_breadth_by_case, mappings, True
        )
        if not results.benefits_security_cross.empty:
            save_csv(results.benefits_security_cross, dirs["tables"] / "benefits_security_cross.csv")
    
    # Keyword proxy analysis (always run for additional insights)
    proxy_results = compute_beneficiary_keywords_proxy(fact, mappings, keywords_config)
    results.beneficiary_group_mentions = proxy_results["mentions"]
    results.beneficiary_group_summary = proxy_results["summary"]
    
    if not results.beneficiary_group_mentions.empty:
        save_csv(results.beneficiary_group_mentions, dirs["tables"] / "beneficiary_group_mentions_by_case.csv")
    if not results.beneficiary_group_summary.empty:
        save_csv(results.beneficiary_group_summary, dirs["tables"] / "beneficiary_group_mentions_summary.csv")
    
    # Step B4: Threat-Security links
    logger.info("B4: Linking threats to security dimensions...")
    threat_sec_results = compute_threat_security_links(
        data["TIDY_THREATS"],
        data["TIDY_SECURITY"],
        results.security_breadth_by_case,
        mappings
    )
    results.threat_to_security_profile = threat_sec_results["threat_to_security"]
    results.security_to_threat_profile = threat_sec_results["security_to_threat"]
    
    save_csv(results.threat_to_security_profile, dirs["tables"] / "threat_to_security_profile.csv")
    save_csv(results.security_to_threat_profile, dirs["tables"] / "security_to_threat_profile.csv")
    
    # Step B5: Gap-Security links
    logger.info("B5: Linking governance gaps to security...")
    gap_sec_results = compute_gap_security_links(
        data["TIDY_GOV_GAPS"],
        results.security_breadth_by_case,
        data["TIDY_SECURITY"],
        mappings
    )
    results.gapload_by_security_archetype = gap_sec_results["gapload_by_archetype"]
    results.gaps_by_security_dimension = gap_sec_results["gaps_by_dimension"]
    ggl_by_case = gap_sec_results["ggl_by_case"]
    
    save_csv(results.gapload_by_security_archetype, dirs["tables"] / "gapload_by_security_archetype.csv")
    save_csv(results.gaps_by_security_dimension, dirs["tables"] / "gaps_by_security_dimension.csv")
    
    # Step B6: Shortlists
    logger.info("B6: Generating shortlists...")
    shortlist_results = compute_shortlists_b(
        fact,
        results.security_breadth_by_case,
        ggl_by_case,
        mappings,
        results.has_numeric_beneficiaries,
        results.beneficiary_group_mentions
    )
    results.shortlist_B1_equity_leaders = shortlist_results["B1_equity_leaders"]
    results.shortlist_B2_needs_enablers = shortlist_results["B2_needs_enablers"]
    
    save_csv(results.shortlist_B1_equity_leaders, dirs["tables"] / "shortlist_B1_equity_leaders.csv")
    save_csv(results.shortlist_B2_needs_enablers, dirs["tables"] / "shortlist_B2_needs_enablers.csv")
    
    if results.n_cases < 5:
        results.warnings.append(
            f"Small dataset ({results.n_cases} cases): shortlist thresholds may be unreliable"
        )
    
    # Step B7: One-pagers
    logger.info("B7: Building one-pagers...")
    results.onepagers = build_onepagers_b(data, results, mappings, keywords_config)
    
    save_csv(results.onepagers, dirs["tables"] / "nbs_onepagers_storyline_b.csv")
    
    # Decide on Excel format based on case count
    if results.n_cases <= 30:
        # One sheet per case
        sheets = {}
        for _, row in results.onepagers.iterrows():
            sheet_name = str(row["id_nbs"])[:31]
            sheets[sheet_name] = pd.DataFrame([row])
        save_excel(sheets, dirs["tables"] / "nbs_onepagers_storyline_b.xlsx")
    else:
        # Single ONEPAGERS sheet
        save_excel(
            {"ONEPAGERS": results.onepagers},
            dirs["tables"] / "nbs_onepagers_storyline_b.xlsx"
        )
    
    # Generate plots
    logger.info("Generating visualizations...")
    plot_paths = generate_plots_b(results, output_dir, fact, mappings)
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    generate_html_report_b(results, plot_paths, dirs["reports"] / "report_storyline_b.html")
    
    logger.info(f"Storyline B analysis complete. Outputs in: {output_dir}")
    
    return results


# ---------------------------------------------------------------------------
# HTML Report Generation
# ---------------------------------------------------------------------------

def generate_html_report_b(
    results: StorylineBResults,
    plot_paths: Dict[str, Path],
    output_path: Path
) -> None:
    """Generate Storyline B HTML report."""
    from jinja2 import Environment, FileSystemLoader, BaseLoader
    import base64
    
    # Try to load from template file
    template_dir = Path(__file__).parent.parent.parent / "templates"
    template_file = template_dir / "report_storyline_b.html.j2"
    
    if template_file.exists():
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("report_storyline_b.html.j2")
    else:
        # Use inline template
        template_str = get_inline_template_b()
        env = Environment(loader=BaseLoader())
        template = env.from_string(template_str)
    
    # Encode images as base64
    def encode_image(path: Path) -> str:
        if path.exists():
            with open(path, "rb") as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        return ""
    
    # Prepare template context
    context = {
        "input_file": Path(results.input_file).name,
        "timestamp": results.run_timestamp,
        "n_cases": results.n_cases,
        "has_numeric_beneficiaries": results.has_numeric_beneficiaries,
        "warnings": results.warnings,
        
        # Beneficiary summary
        "benef_summary": results.beneficiary_summary_overall.to_dict("records")[0] 
                        if not results.beneficiary_summary_overall.empty else {},
        
        # Security rates
        "security_rates": results.security_dimension_rates.head(10).to_dict("records"),
        
        # Archetype distribution
        "archetype_counts": results.security_breadth_by_case["security_archetype"].value_counts().to_dict()
                           if not results.security_breadth_by_case.empty else {},
        
        # Benefits × Security
        "benefits_cross": results.benefits_security_cross.to_dict("records")
                         if not results.benefits_security_cross.empty else [],
        
        # Keyword proxy
        "keyword_summary": results.beneficiary_group_summary.head(10).to_dict("records")
                          if not results.beneficiary_group_summary.empty else [],
        
        # Gap load by archetype
        "gapload_archetype": results.gapload_by_security_archetype.to_dict("records")
                            if not results.gapload_by_security_archetype.empty else [],
        
        # Shortlists
        "shortlist_b1": results.shortlist_B1_equity_leaders.head(10).to_dict("records"),
        "shortlist_b2": results.shortlist_B2_needs_enablers.head(10).to_dict("records"),
        "n_b1": len(results.shortlist_B1_equity_leaders),
        "n_b2": len(results.shortlist_B2_needs_enablers),
        
        # Plots
        "plot_security_rates": encode_image(plot_paths.get("security_dimension_rates", Path())),
        "plot_security_dist": encode_image(plot_paths.get("security_breadth_distribution", Path())),
        "plot_benef_landscape": encode_image(plot_paths.get("beneficiaries_by_landscape", Path())),
        "plot_benef_security": encode_image(plot_paths.get("beneficiaries_vs_security", Path())),
    }
    
    html = template.render(**context)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    logger.info(f"Generated HTML report: {output_path}")


def get_inline_template_b() -> str:
    """Return inline HTML template for Storyline B report."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NbS Analyzer - Storyline B Report</title>
    <style>
        :root {
            --primary: #9c27b0;
            --secondary: #64748b;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header {
            background: linear-gradient(135deg, var(--primary), #7b1fa2);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
        header .meta { opacity: 0.9; font-size: 0.9rem; }
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border);
        }
        .card h2 {
            color: var(--primary);
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .stat-card {
            background: linear-gradient(135deg, #f3e5f5, #e1bee7);
            padding: 1.25rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #ce93d8;
        }
        .stat-card.success {
            background: linear-gradient(135deg, #ecfdf5, #d1fae5);
            border-color: #6ee7b7;
        }
        .stat-card.warning {
            background: linear-gradient(135deg, #fffbeb, #fef3c7);
            border-color: #fcd34d;
        }
        .stat-value { font-size: 2rem; font-weight: 700; color: var(--primary); }
        .stat-label { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem; }
        table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
        th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
        th { background: #f1f5f9; font-weight: 600; color: var(--text); }
        tr:hover { background: #f8fafc; }
        .figure { text-align: center; margin: 1.5rem 0; }
        .figure img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); }
        .warning-box {
            background: #fffbeb;
            border: 1px solid #fcd34d;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #92400e;
        }
        .info-box {
            background: #eff6ff;
            border: 1px solid #93c5fd;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #1e40af;
        }
        .two-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; }
        footer { text-align: center; padding: 2rem; color: var(--text-muted); font-size: 0.85rem; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌿 NbS Analyzer — Storyline B Report</h1>
            <div class="meta">
                <strong>Benefits/Equity-First Analysis</strong><br>
                Input: {{ input_file }}<br>
                Generated: {{ timestamp }}
            </div>
        </header>
        
        <div class="card">
            <h2>📊 Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ n_cases }}</div>
                    <div class="stat-label">Total NbS Cases</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-value">{{ n_b1 }}</div>
                    <div class="stat-label">Equity Leaders (B1)</div>
                </div>
                <div class="stat-card warning">
                    <div class="stat-value">{{ n_b2 }}</div>
                    <div class="stat-label">Needs Enablers (B2)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ security_rates|length }}</div>
                    <div class="stat-label">Security Dimensions</div>
                </div>
            </div>
            
            {% if warnings %}
            <div class="warning-box">
                <strong>⚠️ Warnings</strong>
                {% for warning in warnings %}
                <div>{{ warning }}</div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if not has_numeric_beneficiaries %}
            <div class="info-box">
                <strong>ℹ️ Proxy Mode Active</strong>
                <div>Numeric beneficiary counts not available. Using keyword-based proxy analysis.</div>
            </div>
            {% endif %}
        </div>
        
        <div class="card">
            <h2>👥 Beneficiary Summary</h2>
            {% if has_numeric_beneficiaries %}
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ "{:,}".format(benef_summary.total_beneficiaries|default(0)|int) }}</div>
                    <div class="stat-label">Total Beneficiaries</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ "{:,}".format(benef_summary.total_direct|default(0)|int) }}</div>
                    <div class="stat-label">Direct</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ "{:,}".format(benef_summary.total_indirect|default(0)|int) }}</div>
                    <div class="stat-label">Indirect</div>
                </div>
            </div>
            {% endif %}
            <table>
                <tr><td>% Cases with Direct Count</td><td><strong>{{ benef_summary.pct_direct_count_present|default(0) }}%</strong></td></tr>
                <tr><td>% Cases with Indirect Count</td><td><strong>{{ benef_summary.pct_indirect_count_present|default(0) }}%</strong></td></tr>
                <tr><td>% Cases with Any Description</td><td><strong>{{ benef_summary.pct_any_description_present|default(0) }}%</strong></td></tr>
            </table>
        </div>
        
        <div class="card">
            <h2>🔒 Security Dimension Rates</h2>
            <table>
                <thead>
                    <tr><th>Dimension</th><th>Cases</th><th>%</th></tr>
                </thead>
                <tbody>
                    {% for row in security_rates %}
                    <tr>
                        <td>{{ row.dimension }}</td>
                        <td>{{ row.n_cases_flagged }}</td>
                        <td>{{ row.pct_cases_flagged }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% if plot_security_rates %}
            <div class="figure">
                <img src="{{ plot_security_rates }}" alt="Security Dimension Rates">
            </div>
            {% endif %}
        </div>
        
        {% if plot_security_dist %}
        <div class="card">
            <h2>📊 Security Breadth Distribution</h2>
            <div class="figure">
                <img src="{{ plot_security_dist }}" alt="Security Breadth Distribution">
            </div>
        </div>
        {% endif %}
        
        {% if keyword_summary %}
        <div class="card">
            <h2>🏷️ Priority Group Mentions (Keyword Proxy)</h2>
            <table>
                <thead>
                    <tr><th>Group</th><th>Cases</th><th>%</th></tr>
                </thead>
                <tbody>
                    {% for row in keyword_summary %}
                    <tr>
                        <td>{{ row.group }}</td>
                        <td>{{ row.n_cases_mentioned }}</td>
                        <td>{{ row.pct_cases }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        <div class="card">
            <h2>✅ Shortlist B1: Equity Leaders</h2>
            <p style="color: var(--text-muted); margin-bottom: 1rem;">
                High security breadth + beneficiary evidence + manageable governance barriers.
            </p>
            <table>
                <thead>
                    <tr><th>ID</th><th>SBS</th><th>Archetype</th><th>GGL</th></tr>
                </thead>
                <tbody>
                    {% for row in shortlist_b1 %}
                    <tr>
                        <td>{{ row.id_nbs }}</td>
                        <td><strong>{{ row.SBS }}</strong></td>
                        <td>{{ row.security_archetype }}</td>
                        <td>{{ row.GGL }}</td>
                    </tr>
                    {% else %}
                    <tr><td colspan="4" style="text-align: center;">No cases meet criteria</td></tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>⚙️ Shortlist B2: Needs Enablers</h2>
            <p style="color: var(--text-muted); margin-bottom: 1rem;">
                High security breadth but significant governance challenges.
            </p>
            <table>
                <thead>
                    <tr><th>ID</th><th>SBS</th><th>Archetype</th><th>GGL</th></tr>
                </thead>
                <tbody>
                    {% for row in shortlist_b2 %}
                    <tr>
                        <td>{{ row.id_nbs }}</td>
                        <td><strong>{{ row.SBS }}</strong></td>
                        <td>{{ row.security_archetype }}</td>
                        <td>{{ row.GGL }}</td>
                    </tr>
                    {% else %}
                    <tr><td colspan="4" style="text-align: center;">No cases meet criteria</td></tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        {% if gapload_archetype %}
        <div class="card">
            <h2>🏛️ Governance Gap Load by Security Archetype</h2>
            <table>
                <thead>
                    <tr><th>Archetype</th><th>N</th><th>Mean GGL</th><th>Median</th><th>Q25</th><th>Q75</th></tr>
                </thead>
                <tbody>
                    {% for row in gapload_archetype %}
                    <tr>
                        <td>{{ row.security_archetype }}</td>
                        <td>{{ row.n_cases }}</td>
                        <td>{{ row.mean_GGL }}</td>
                        <td>{{ row.median_GGL }}</td>
                        <td>{{ row.q25_GGL }}</td>
                        <td>{{ row.q75_GGL }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        <footer>
            <p>Generated by NbS Analyzer v1.0.0 | Storyline B (Benefits/Equity-First)</p>
            <p>{{ timestamp }}</p>
        </footer>
    </div>
</body>
</html>'''
