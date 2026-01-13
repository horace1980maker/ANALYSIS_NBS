"""
Metrics computations for the NbS Analyzer.

Implements all scoring indices and analytics for Storyline A:
- TCS (Threat Coverage Score)
- SBS (Security Breadth Score)
- TTS (Transformative Traits Score)
- GGL (Governance Gap Load)
- ValueScore (combined)
- Threat frequencies and co-occurrence
- Shortlist generation
"""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from nbs_analyzer.utils import (
    pick_col,
    load_column_mappings,
    ensure_deterministic_sort,
    stable_round,
    infer_threat_type,
)

logger = logging.getLogger(__name__)


def compute_threat_frequencies(
    tidy_threats: pd.DataFrame,
    threat_type: str = "all"
) -> pd.DataFrame:
    """
    Compute frequency table for threats.
    
    Args:
        tidy_threats: TIDY_THREATS DataFrame
        threat_type: 'climatic', 'non_climatic', or 'all'
    
    Returns:
        DataFrame with columns: threat_code, count, percentage
    """
    mappings = load_column_mappings()
    id_col = pick_col(tidy_threats, mappings.get("id_nbs", ["id_nbs"]))
    code_col = pick_col(tidy_threats, mappings.get("threat_code", ["threat_code"]))
    type_col = pick_col(tidy_threats, mappings.get("threat_type", ["threat_type"]))
    label_col = pick_col(tidy_threats, ["threat_label", "threat", "amenaza_label"])
    
    if code_col is None:
        logger.warning("No threat_code column found")
        return pd.DataFrame(columns=["threat_code", "count", "percentage"])
    
    df = tidy_threats.copy()
    
    # Filter by threat type if specified
    if threat_type != "all" and type_col:
        df = df[df[type_col] == threat_type]
    elif threat_type != "all" and code_col:
        # Infer type from code prefix
        df["_inferred_type"] = df[code_col].apply(infer_threat_type)
        df = df[df["_inferred_type"] == threat_type]
    
    if df.empty:
        return pd.DataFrame(columns=["threat_code", "count", "percentage"])
    
    # Count distinct id_nbs per threat_code
    freq = df.groupby(code_col)[id_col].nunique().reset_index()
    freq.columns = ["threat_code", "count"]
    
    total_cases = tidy_threats[id_col].nunique()
    freq["percentage"] = freq["count"].apply(
        lambda x: stable_round(100.0 * x / total_cases) if total_cases > 0 else 0
    )
    
    # Add labels if available
    if label_col:
        label_map = df.drop_duplicates(subset=[code_col])[[code_col, label_col]]
        label_map.columns = ["threat_code", "threat_label"]
        freq = freq.merge(label_map, on="threat_code", how="left")
    
    # Sort by count descending, then by code for determinism
    freq = freq.sort_values(
        by=["count", "threat_code"],
        ascending=[False, True]
    ).reset_index(drop=True)
    
    return freq


def compute_cooccurrence(
    tidy_threats: pd.DataFrame,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Compute threat co-occurrence pairs.
    
    For each id_nbs, create all combinations of threat_codes and count
    across all cases.
    
    Returns:
        DataFrame with columns: pair, code_a, code_b, count
    """
    mappings = load_column_mappings()
    id_col = pick_col(tidy_threats, mappings.get("id_nbs", ["id_nbs"]))
    code_col = pick_col(tidy_threats, mappings.get("threat_code", ["threat_code"]))
    
    if id_col is None or code_col is None:
        logger.warning("Required columns not found for co-occurrence")
        return pd.DataFrame(columns=["pair", "code_a", "code_b", "count"])
    
    # Get unique sorted threat codes per id_nbs
    grouped = (
        tidy_threats
        .dropna(subset=[id_col, code_col])
        .groupby(id_col)[code_col]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )
    
    # Generate pairs
    pair_counts: Dict[Tuple[str, str], int] = {}
    
    for _, row in grouped.iterrows():
        codes = row[code_col]
        if len(codes) < 2:
            continue
        
        for pair in itertools.combinations(codes, 2):
            # Ensure lexicographic order
            ordered = tuple(sorted(pair))
            pair_counts[ordered] = pair_counts.get(ordered, 0) + 1
    
    if not pair_counts:
        return pd.DataFrame(columns=["pair", "code_a", "code_b", "count"])
    
    # Convert to DataFrame
    pairs_df = pd.DataFrame([
        {
            "code_a": k[0],
            "code_b": k[1],
            "pair": f"{k[0]}|{k[1]}",
            "count": v
        }
        for k, v in pair_counts.items()
    ])
    
    # Sort by count descending, then by pair for determinism
    pairs_df = pairs_df.sort_values(
        by=["count", "pair"],
        ascending=[False, True]
    ).reset_index(drop=True)
    
    return pairs_df.head(top_n)


def compute_tcs(
    tidy_threats: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute Threat Coverage Score (TCS) per id_nbs.
    
    TCS = number of unique threats per case
    Also computes n_AC and n_ANC separately.
    
    Returns:
        DataFrame with columns: id_nbs, TCS, n_AC, n_ANC
    """
    mappings = load_column_mappings()
    id_col = pick_col(tidy_threats, mappings.get("id_nbs", ["id_nbs"]))
    code_col = pick_col(tidy_threats, mappings.get("threat_code", ["threat_code"]))
    type_col = pick_col(tidy_threats, mappings.get("threat_type", ["threat_type"]))
    
    if id_col is None:
        logger.warning("No id_nbs column found for TCS")
        return pd.DataFrame(columns=["id_nbs", "TCS", "n_AC", "n_ANC"])
    
    df = tidy_threats.copy()
    
    # Infer type if not present
    if type_col is None and code_col:
        df["threat_type"] = df[code_col].apply(infer_threat_type)
        type_col = "threat_type"
    
    # Total unique threats per id
    tcs = df.groupby(id_col)[code_col].nunique().reset_index()
    tcs.columns = ["id_nbs", "TCS"]
    
    # Count by type
    if type_col:
        df_ac = df[df[type_col] == "climatic"]
        df_anc = df[df[type_col] == "non_climatic"]
        
        n_ac = df_ac.groupby(id_col)[code_col].nunique().reset_index()
        n_ac.columns = ["id_nbs", "n_AC"]
        
        n_anc = df_anc.groupby(id_col)[code_col].nunique().reset_index()
        n_anc.columns = ["id_nbs", "n_ANC"]
        
        tcs = tcs.merge(n_ac, on="id_nbs", how="left")
        tcs = tcs.merge(n_anc, on="id_nbs", how="left")
        tcs["n_AC"] = tcs["n_AC"].fillna(0).astype(int)
        tcs["n_ANC"] = tcs["n_ANC"].fillna(0).astype(int)
    else:
        tcs["n_AC"] = 0
        tcs["n_ANC"] = 0
    
    return ensure_deterministic_sort(tcs, ["id_nbs"])


def compute_sbs(
    tidy_security: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute Security Breadth Score (SBS) per id_nbs.
    
    SBS = sum of value==1 across security dimensions per case.
    
    Returns:
        DataFrame with columns: id_nbs, SBS
    """
    mappings = load_column_mappings()
    id_col = pick_col(tidy_security, mappings.get("id_nbs", ["id_nbs"]))
    value_col = pick_col(tidy_security, mappings.get("value", ["value"]))
    
    if id_col is None:
        logger.warning("No id_nbs column found for SBS")
        return pd.DataFrame(columns=["id_nbs", "SBS"])
    
    if value_col is None:
        logger.warning("No value column found for SBS")
        return pd.DataFrame(columns=["id_nbs", "SBS"])
    
    df = tidy_security.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    
    # Sum of value==1 per id
    sbs = df[df[value_col] == 1].groupby(id_col).size().reset_index()
    sbs.columns = ["id_nbs", "SBS"]
    
    return ensure_deterministic_sort(sbs, ["id_nbs"])


def compute_tts(
    tidy_traits: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute Transformative Traits Score (TTS) per id_nbs.
    
    TTS = sum of value==1 across traits per case.
    
    Returns:
        DataFrame with columns: id_nbs, TTS
    """
    mappings = load_column_mappings()
    id_col = pick_col(tidy_traits, mappings.get("id_nbs", ["id_nbs"]))
    value_col = pick_col(tidy_traits, mappings.get("value", ["value"]))
    
    if id_col is None:
        logger.warning("No id_nbs column found for TTS")
        return pd.DataFrame(columns=["id_nbs", "TTS"])
    
    if value_col is None:
        logger.warning("No value column found for TTS")
        return pd.DataFrame(columns=["id_nbs", "TTS"])
    
    df = tidy_traits.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    
    # Sum of value==1 per id
    tts = df[df[value_col] == 1].groupby(id_col).size().reset_index()
    tts.columns = ["id_nbs", "TTS"]
    
    return ensure_deterministic_sort(tts, ["id_nbs"])


def compute_ggl(
    tidy_gov_gaps: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute Governance Gap Load (GGL) per id_nbs.
    
    GGL = number of unique governance gaps per case.
    
    Returns:
        DataFrame with columns: id_nbs, GGL
    """
    mappings = load_column_mappings()
    id_col = pick_col(tidy_gov_gaps, mappings.get("id_nbs", ["id_nbs"]))
    gap_col = pick_col(tidy_gov_gaps, mappings.get("gap_code", ["gap_code"]))
    
    if id_col is None:
        logger.warning("No id_nbs column found for GGL")
        return pd.DataFrame(columns=["id_nbs", "GGL"])
    
    if gap_col is None:
        # Try to count rows if no gap_code column
        ggl = tidy_gov_gaps.groupby(id_col).size().reset_index()
    else:
        ggl = tidy_gov_gaps.groupby(id_col)[gap_col].nunique().reset_index()
    
    ggl.columns = ["id_nbs", "GGL"]
    
    return ensure_deterministic_sort(ggl, ["id_nbs"])


def compute_value_score(
    tcs_df: pd.DataFrame,
    sbs_df: pd.DataFrame,
    tts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute ValueScore (VS) = TCS + SBS + TTS.
    
    Returns:
        DataFrame with columns: id_nbs, TCS, SBS, TTS, VS
    """
    # Merge all scores
    result = tcs_df.copy()
    result = result.merge(sbs_df, on="id_nbs", how="outer")
    result = result.merge(tts_df, on="id_nbs", how="outer")
    
    # Fill NaN with 0 for score computation
    for col in ["TCS", "n_AC", "n_ANC", "SBS", "TTS"]:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)
    
    result["VS"] = result["TCS"] + result["SBS"] + result["TTS"]
    
    return ensure_deterministic_sort(result, ["id_nbs"])


def compute_all_scores(
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compute all scores from tidy tables and merge into single DataFrame.
    
    Returns:
        DataFrame with id_nbs and all score columns
    """
    # Get all unique ids from FACT_NBS
    mappings = load_column_mappings()
    fact = data.get("FACT_NBS", pd.DataFrame())
    id_col = pick_col(fact, mappings.get("id_nbs", ["id_nbs"]))
    
    if id_col is None or fact.empty:
        logger.error("FACT_NBS not found or empty")
        return pd.DataFrame()
    
    all_ids = fact[[id_col]].drop_duplicates().rename(columns={id_col: "id_nbs"})
    
    # Compute individual scores
    tcs_df = compute_tcs(data.get("TIDY_THREATS", pd.DataFrame()))
    sbs_df = compute_sbs(data.get("TIDY_SECURITY", pd.DataFrame()))
    tts_df = compute_tts(data.get("TIDY_TRAITS", pd.DataFrame()))
    ggl_df = compute_ggl(data.get("TIDY_GOV_GAPS", pd.DataFrame()))
    
    # Merge all
    result = all_ids.copy()
    result = result.merge(tcs_df, on="id_nbs", how="left")
    result = result.merge(sbs_df, on="id_nbs", how="left")
    result = result.merge(tts_df, on="id_nbs", how="left")
    result = result.merge(ggl_df, on="id_nbs", how="left")
    
    # Fill NaN with 0
    score_cols = ["TCS", "n_AC", "n_ANC", "SBS", "TTS", "GGL"]
    for col in score_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)
    
    # Compute ValueScore
    result["VS"] = result["TCS"] + result["SBS"] + result["TTS"]
    
    return ensure_deterministic_sort(result, ["id_nbs"])


def compute_shortlists(
    scores_df: pd.DataFrame,
    high_value_percentile: float = 80,
    low_friction_percentile: float = 40,
    high_friction_percentile: float = 60
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Compute shortlists based on rank-based thresholds.
    
    Args:
        scores_df: DataFrame with VS and GGL columns
        high_value_percentile: Percentile for high value (default 80th)
        low_friction_percentile: Percentile for low friction (default 40th)
        high_friction_percentile: Percentile for high friction (default 60th)
    
    Returns:
        Tuple of (low_friction_shortlist, needs_enablers_shortlist, thresholds_dict)
    """
    n = len(scores_df)
    
    if n == 0:
        empty = pd.DataFrame(columns=["id_nbs", "VS", "GGL"])
        return empty, empty, {}
    
    # Calculate thresholds
    vs_threshold = scores_df["VS"].quantile(high_value_percentile / 100)
    ggl_low_threshold = scores_df["GGL"].quantile(low_friction_percentile / 100)
    ggl_high_threshold = scores_df["GGL"].quantile(high_friction_percentile / 100)
    
    # For small N, ensure at least some cases pass
    min_high_value = int(max(3, math.ceil(0.2 * n)))
    
    # Get high value cases
    high_value = scores_df[scores_df["VS"] >= vs_threshold].copy()
    
    # If threshold too strict, take top N by VS
    if len(high_value) < min_high_value:
        high_value = scores_df.nlargest(int(min_high_value), "VS")
    
    # Split into low friction and needs enablers
    low_friction = high_value[high_value["GGL"] <= ggl_low_threshold].copy()
    needs_enablers = high_value[high_value["GGL"] >= ggl_high_threshold].copy()
    
    # Sort for determinism
    low_friction = ensure_deterministic_sort(low_friction, ["VS", "id_nbs"], ascending=False)
    needs_enablers = ensure_deterministic_sort(needs_enablers, ["GGL", "id_nbs"], ascending=False)
    
    thresholds = {
        "n_cases": n,
        "vs_threshold": stable_round(vs_threshold),
        "ggl_low_threshold": stable_round(ggl_low_threshold),
        "ggl_high_threshold": stable_round(ggl_high_threshold),
        "n_high_value": len(high_value),
        "n_low_friction": len(low_friction),
        "n_needs_enablers": len(needs_enablers),
    }
    
    logger.info(f"Shortlist thresholds: {thresholds}")
    
    return low_friction, needs_enablers, thresholds


def compute_top_gov_gaps(
    tidy_gov_gaps: pd.DataFrame,
    fact_nbs: Optional[pd.DataFrame] = None,
    by_landscape: bool = False
) -> pd.DataFrame:
    """
    Compute most common governance gaps.
    
    Args:
        tidy_gov_gaps: TIDY_GOV_GAPS DataFrame
        fact_nbs: FACT_NBS DataFrame (for landscape grouping)
        by_landscape: If True, group by landscape
    
    Returns:
        DataFrame with gap frequencies
    """
    mappings = load_column_mappings()
    id_col = pick_col(tidy_gov_gaps, mappings.get("id_nbs", ["id_nbs"]))
    gap_col = pick_col(tidy_gov_gaps, mappings.get("gap_code", ["gap_code"]))
    gap_name_col = pick_col(tidy_gov_gaps, mappings.get("gap_name", ["gap_name"]))
    
    if gap_col is None:
        logger.warning("No gap_code column found")
        return pd.DataFrame(columns=["gap_code", "count"])
    
    df = tidy_gov_gaps.copy()
    
    if by_landscape and fact_nbs is not None:
        landscape_col = pick_col(fact_nbs, mappings.get("landscape", ["landscape", "paisaje"]))
        
        if landscape_col:
            # Join landscape from FACT_NBS
            fact_id_col = pick_col(fact_nbs, mappings.get("id_nbs", ["id_nbs"]))
            landscape_map = fact_nbs[[fact_id_col, landscape_col]].drop_duplicates()
            landscape_map.columns = ["id_nbs", "landscape"]
            
            df = df.merge(landscape_map, left_on=id_col, right_on="id_nbs", how="left")
            
            # Group by landscape and gap
            result = df.groupby(["landscape", gap_col])[id_col].nunique().reset_index()
            result.columns = ["landscape", "gap_code", "count"]
            
            if gap_name_col:
                name_map = tidy_gov_gaps[[gap_col, gap_name_col]].drop_duplicates()
                name_map.columns = ["gap_code", "gap_name"]
                result = result.merge(name_map, on="gap_code", how="left")
            
            return result.sort_values(["landscape", "count"], ascending=[True, False]).reset_index(drop=True)
    
    # Overall frequency
    result = df.groupby(gap_col)[id_col].nunique().reset_index()
    result.columns = ["gap_code", "count"]
    
    if gap_name_col:
        name_map = tidy_gov_gaps[[gap_col, gap_name_col]].drop_duplicates()
        name_map.columns = ["gap_code", "gap_name"]
        result = result.merge(name_map, on="gap_code", how="left")
    
    total_cases = df[id_col].nunique()
    result["percentage"] = result["count"].apply(
        lambda x: stable_round(100.0 * x / total_cases) if total_cases > 0 else 0
    )
    
    return result.sort_values("count", ascending=False).reset_index(drop=True)


def load_enabling_packages(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load enabling packages configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "enabling_packages.yml"
    
    if not config_path.exists():
        logger.warning(f"Enabling packages config not found: {config_path}")
        return {
            "packages": {},
            "default_package": "Uncategorized"
        }
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def map_gaps_to_packages(
    gap_codes: List[str],
    packages_config: Optional[Dict[str, Any]] = None
) -> Dict[str, List[str]]:
    """Map gap codes to enabling packages."""
    if packages_config is None:
        packages_config = load_enabling_packages()
    
    packages = packages_config.get("packages", {})
    default = packages_config.get("default_package", "Uncategorized")
    
    # Build reverse mapping: gap_code -> package_name
    gap_to_package = {}
    for pkg_name, pkg_data in packages.items():
        if isinstance(pkg_data, dict):
            codes = pkg_data.get("gap_codes", [])
        else:
            codes = pkg_data  # Direct list
        
        for code in codes:
            gap_to_package[code] = pkg_name
    
    # Group input gaps by package
    result: Dict[str, List[str]] = {}
    for gap in gap_codes:
        pkg = gap_to_package.get(gap, default)
        if pkg not in result:
            result[pkg] = []
        result[pkg].append(gap)
    
    return result


def compute_nbs_threat_matching(
    tidy_threats: pd.DataFrame,
    scores_df: pd.DataFrame,
    fact_nbs: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    For each threat_code, list NbS cases and average scores.
    
    Returns:
        DataFrame with threat_code, n_cases, cases_list, avg_SBS, avg_TTS, avg_GGL
    """
    mappings = load_column_mappings()
    id_col = pick_col(tidy_threats, mappings.get("id_nbs", ["id_nbs"]))
    code_col = pick_col(tidy_threats, mappings.get("threat_code", ["threat_code"]))
    
    if id_col is None or code_col is None:
        return pd.DataFrame()
    
    # Get cases per threat
    grouped = tidy_threats.groupby(code_col)[id_col].apply(list).reset_index()
    grouped.columns = ["threat_code", "cases"]
    grouped["n_cases"] = grouped["cases"].apply(len)
    
    # Compute average scores per threat
    def get_avg_scores(cases):
        case_scores = scores_df[scores_df["id_nbs"].isin(cases)]
        if case_scores.empty:
            return pd.Series({"avg_SBS": np.nan, "avg_TTS": np.nan, "avg_GGL": np.nan})
        return pd.Series({
            "avg_SBS": stable_round(case_scores["SBS"].mean()),
            "avg_TTS": stable_round(case_scores["TTS"].mean()),
            "avg_GGL": stable_round(case_scores["GGL"].mean()),
        })
    
    avg_scores = grouped["cases"].apply(get_avg_scores)
    result = pd.concat([grouped, avg_scores], axis=1)
    
    # Join case descriptions if available
    if fact_nbs is not None:
        fact_id_col = pick_col(fact_nbs, mappings.get("id_nbs", ["id_nbs"]))
        desc_col = pick_col(fact_nbs, mappings.get("combination", ["combinacion", "combination"]))
        
        if fact_id_col and desc_col:
            desc_map = fact_nbs.set_index(fact_id_col)[desc_col].to_dict()
            result["cases_descriptions"] = result["cases"].apply(
                lambda cs: [f"{c}: {desc_map.get(c, '')}" for c in cs]
            )
    
    return result.sort_values("n_cases", ascending=False).reset_index(drop=True)
