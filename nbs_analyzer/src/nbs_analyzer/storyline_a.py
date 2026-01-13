"""
Storyline A (Risk-First) Pipeline Orchestrator.

Implements the complete Storyline A analysis:
- A1: Threat landscape analysis
- A2: NbS-Threat matching
- A3: Index computations
- A4: Shortlist generation
- A5: Enabling packages
- A6: One-pager generation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbs_analyzer.io import load_workbook, save_csv, save_excel
from nbs_analyzer.schema import REQUIRED_SHEETS, OPTIONAL_SHEETS
from nbs_analyzer.validate import run_all_validations, generate_qa_summary
from nbs_analyzer.metrics import (
    compute_threat_frequencies,
    compute_cooccurrence,
    compute_all_scores,
    compute_shortlists,
    compute_top_gov_gaps,
    compute_nbs_threat_matching,
    map_gaps_to_packages,
    load_enabling_packages,
)
from nbs_analyzer.utils import (
    pick_col,
    load_column_mappings,
    ensure_deterministic_sort,
    create_output_dirs,
)
from nbs_analyzer.report import generate_html_report

logger = logging.getLogger(__name__)


@dataclass
class StorylineAResults:
    """Container for all Storyline A outputs."""
    
    # Threat analysis
    threat_freq_ac: pd.DataFrame = field(default_factory=pd.DataFrame)
    threat_freq_anc: pd.DataFrame = field(default_factory=pd.DataFrame)
    threat_pairs: pd.DataFrame = field(default_factory=pd.DataFrame)
    threat_matching: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Scores
    nbs_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Shortlists
    shortlist_low_friction: pd.DataFrame = field(default_factory=pd.DataFrame)
    shortlist_needs_enablers: pd.DataFrame = field(default_factory=pd.DataFrame)
    shortlist_thresholds: Dict[str, Any] = field(default_factory=dict)
    
    # Governance
    top_gov_gaps_overall: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_gov_gaps_by_landscape: pd.DataFrame = field(default_factory=pd.DataFrame)
    enabling_packages: Dict[str, Any] = field(default_factory=dict)
    
    # One-pagers
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


def apply_filters(
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
    
    # Apply each filter
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
    
    # Filter tidy tables to matching IDs
    valid_ids = set(fact[id_col].dropna().unique())
    
    for sheet_name in ["TIDY_THREATS", "TIDY_GOV_GAPS", "TIDY_SECURITY", "TIDY_TRAITS"]:
        if sheet_name in data:
            df = data[sheet_name]
            tidy_id_col = pick_col(df, mappings.get("id_nbs", ["id_nbs"]))
            if tidy_id_col:
                data[sheet_name] = df[df[tidy_id_col].isin(valid_ids)]
    
    return data


def generate_plots(
    results: StorylineAResults,
    output_dir: Path
) -> Dict[str, Path]:
    """Generate all visualization plots."""
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
    
    # Top AC threats
    if not results.threat_freq_ac.empty:
        fig, ax = plt.subplots()
        data = results.threat_freq_ac.head(10)
        
        y_pos = np.arange(len(data))
        bars = ax.barh(y_pos, data["count"], align="center")
        ax.set_yticks(y_pos)
        
        labels = data["threat_code"].tolist()
        if "threat_label" in data.columns:
            labels = [f"{c}: {l[:30]}" if pd.notna(l) else c 
                     for c, l in zip(data["threat_code"], data["threat_label"])]
        
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Number of NbS Cases")
        ax.set_title("Top 10 Climatic Threats (AC)")
        
        # Add value labels
        for bar, val in zip(bars, data["count"]):
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                   str(val), va="center", fontsize=9)
        
        plt.tight_layout()
        path = figures_dir / "top_ac.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["top_ac"] = path
        logger.info(f"Generated: {path}")
    
    # Top ANC threats
    if not results.threat_freq_anc.empty:
        fig, ax = plt.subplots()
        data = results.threat_freq_anc.head(10)
        
        y_pos = np.arange(len(data))
        bars = ax.barh(y_pos, data["count"], align="center", color="coral")
        ax.set_yticks(y_pos)
        
        labels = data["threat_code"].tolist()
        if "threat_label" in data.columns:
            labels = [f"{c}: {l[:30]}" if pd.notna(l) else c 
                     for c, l in zip(data["threat_code"], data["threat_label"])]
        
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Number of NbS Cases")
        ax.set_title("Top 10 Non-Climatic Threats (ANC)")
        
        for bar, val in zip(bars, data["count"]):
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                   str(val), va="center", fontsize=9)
        
        plt.tight_layout()
        path = figures_dir / "top_anc.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["top_anc"] = path
        logger.info(f"Generated: {path}")
    
    # Value vs Friction scatter
    if not results.nbs_scores.empty and len(results.nbs_scores) >= 2:
        fig, ax = plt.subplots()
        
        x = results.nbs_scores["GGL"]
        y = results.nbs_scores["VS"]
        
        ax.scatter(x, y, alpha=0.7, s=60, edgecolors="white", linewidth=0.5)
        
        ax.set_xlabel("Governance Gap Load (GGL) — Friction")
        ax.set_ylabel("Value Score (VS)")
        ax.set_title("Value vs. Friction Analysis")
        
        # Add quadrant lines at median
        if len(x) > 1:
            ax.axvline(x.median(), color="gray", linestyle="--", alpha=0.5)
            ax.axhline(y.median(), color="gray", linestyle="--", alpha=0.5)
        
        # Annotate shortlist cases if few
        if len(results.shortlist_low_friction) <= 5:
            for _, row in results.shortlist_low_friction.iterrows():
                case_data = results.nbs_scores[results.nbs_scores["id_nbs"] == row["id_nbs"]]
                if not case_data.empty:
                    ax.annotate(
                        row["id_nbs"],
                        (case_data["GGL"].values[0], case_data["VS"].values[0]),
                        fontsize=8,
                        alpha=0.8
                    )
        
        plt.tight_layout()
        path = figures_dir / "value_vs_friction.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plot_paths["value_vs_friction"] = path
        logger.info(f"Generated: {path}")
    
    return plot_paths


def build_onepagers(
    data: Dict[str, pd.DataFrame],
    results: StorylineAResults
) -> pd.DataFrame:
    """Build one-pager data for each NbS case."""
    mappings = load_column_mappings()
    fact = data.get("FACT_NBS", pd.DataFrame())
    
    if fact.empty:
        return pd.DataFrame()
    
    id_col = pick_col(fact, mappings.get("id_nbs", ["id_nbs"]))
    
    # Identity fields
    country_col = pick_col(fact, mappings.get("country", ["country", "pais"]))
    landscape_col = pick_col(fact, mappings.get("landscape", ["landscape", "paisaje"]))
    org_col = pick_col(fact, mappings.get("organization", ["organization", "org"]))
    combo_col = pick_col(fact, mappings.get("combination", ["combinacion", "combination"]))
    desc_col = pick_col(fact, mappings.get("description", ["descripcion", "description"]))
    
    # Get tidy data
    tidy_threats = data.get("TIDY_THREATS", pd.DataFrame())
    tidy_gov = data.get("TIDY_GOV_GAPS", pd.DataFrame())
    tidy_security = data.get("TIDY_SECURITY", pd.DataFrame())
    tidy_traits = data.get("TIDY_TRAITS", pd.DataFrame())
    
    threat_id_col = pick_col(tidy_threats, mappings.get("id_nbs", ["id_nbs"]))
    threat_code_col = pick_col(tidy_threats, mappings.get("threat_code", ["threat_code"]))
    threat_label_col = pick_col(tidy_threats, ["threat_label", "threat"])
    
    gap_id_col = pick_col(tidy_gov, mappings.get("id_nbs", ["id_nbs"]))
    gap_code_col = pick_col(tidy_gov, mappings.get("gap_code", ["gap_code"]))
    gap_name_col = pick_col(tidy_gov, mappings.get("gap_name", ["gap_name"]))
    
    sec_id_col = pick_col(tidy_security, mappings.get("id_nbs", ["id_nbs"]))
    sec_dim_col = pick_col(tidy_security, mappings.get("dimension", ["dimension"]))
    sec_val_col = pick_col(tidy_security, mappings.get("value", ["value"]))
    
    trait_id_col = pick_col(tidy_traits, mappings.get("id_nbs", ["id_nbs"]))
    trait_col = pick_col(tidy_traits, mappings.get("trait", ["trait"]))
    trait_val_col = pick_col(tidy_traits, mappings.get("value", ["value"]))
    
    packages_config = load_enabling_packages()
    
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
        
        # Threats
        if threat_id_col and threat_code_col:
            case_threats = tidy_threats[tidy_threats[threat_id_col] == nbs_id]
            codes = case_threats[threat_code_col].dropna().tolist()
            record["threats_codes"] = "; ".join(sorted(set(codes)))
            record["threats_count"] = len(set(codes))
            
            if threat_label_col:
                labels = case_threats[threat_label_col].dropna().tolist()
                record["threats_labels"] = "; ".join(sorted(set(labels)))
        
        # Governance gaps
        if gap_id_col and gap_code_col:
            case_gaps = tidy_gov[tidy_gov[gap_id_col] == nbs_id]
            gap_codes = case_gaps[gap_code_col].dropna().tolist()
            record["gaps_codes"] = "; ".join(sorted(set(gap_codes)))
            record["gaps_count"] = len(set(gap_codes))
            
            # Map to packages
            packages = map_gaps_to_packages(gap_codes, packages_config)
            record["enabling_packages"] = "; ".join(sorted(packages.keys()))
            
            if gap_name_col:
                names = case_gaps[gap_name_col].dropna().tolist()
                record["gaps_names"] = "; ".join(sorted(set(names)))
        
        # Security dimensions
        if sec_id_col and sec_dim_col and sec_val_col:
            case_sec = tidy_security[tidy_security[sec_id_col] == nbs_id]
            case_sec[sec_val_col] = pd.to_numeric(case_sec[sec_val_col], errors="coerce")
            flagged = case_sec[case_sec[sec_val_col] == 1][sec_dim_col].dropna().tolist()
            record["security_dimensions"] = "; ".join(sorted(set(flagged)))
            record["security_count"] = len(set(flagged))
        
        # Traits
        if trait_id_col and trait_col and trait_val_col:
            case_traits = tidy_traits[tidy_traits[trait_id_col] == nbs_id]
            case_traits[trait_val_col] = pd.to_numeric(case_traits[trait_val_col], errors="coerce")
            flagged = case_traits[case_traits[trait_val_col] == 1][trait_col].dropna().tolist()
            record["transformative_traits"] = "; ".join(sorted(set(flagged)))
            record["traits_count"] = len(set(flagged))
        
        # Scores
        scores_row = results.nbs_scores[results.nbs_scores["id_nbs"] == nbs_id]
        if not scores_row.empty:
            for col in ["TCS", "n_AC", "n_ANC", "SBS", "TTS", "GGL", "VS"]:
                if col in scores_row.columns:
                    record[col] = scores_row[col].values[0]
        
        # Shortlist status
        record["in_shortlist_low_friction"] = nbs_id in results.shortlist_low_friction["id_nbs"].values
        record["in_shortlist_needs_enablers"] = nbs_id in results.shortlist_needs_enablers["id_nbs"].values
        
        records.append(record)
    
    onepagers = pd.DataFrame(records)
    return ensure_deterministic_sort(onepagers, ["id_nbs"])


def run_storyline_a(
    input_path: str | Path,
    output_dir: str | Path,
    filters: Optional[Dict[str, Any]] = None
) -> StorylineAResults:
    """
    Execute the complete Storyline A (Risk-First) analysis pipeline.
    
    Args:
        input_path: Path to analysis-ready Excel workbook
        output_dir: Directory for outputs
        filters: Optional filters (country, landscape, org)
    
    Returns:
        StorylineAResults with all outputs
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    results = StorylineAResults(
        input_file=str(input_path),
        output_dir=str(output_dir),
        run_timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"Starting Storyline A analysis: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directories
    dirs = create_output_dirs(output_dir)
    
    # Step 0: Load data
    logger.info("Loading workbook...")
    data, missing_optional = load_workbook(
        input_path,
        required_sheets=REQUIRED_SHEETS,
        optional_sheets=OPTIONAL_SHEETS
    )
    results.missing_lookups = missing_optional
    
    if missing_optional:
        results.warnings.append(f"Missing optional sheets: {missing_optional}")
    
    # Apply filters
    if filters:
        logger.info(f"Applying filters: {filters}")
        data = apply_filters(data, filters)
    
    # Get case count
    mappings = load_column_mappings()
    fact = data["FACT_NBS"]
    id_col = pick_col(fact, mappings.get("id_nbs", ["id_nbs"]))
    results.n_cases = fact[id_col].nunique()
    logger.info(f"Analyzing {results.n_cases} NbS cases")
    
    # Step 1: Validate data
    logger.info("Validating data...")
    results.qa_summary = generate_qa_summary(
        data, 
        dirs["tables"] / "qa_summary.csv"
    )
    
    # Step A1: Threat landscape
    logger.info("A1: Analyzing threat landscape...")
    results.threat_freq_ac = compute_threat_frequencies(
        data["TIDY_THREATS"], 
        threat_type="climatic"
    )
    results.threat_freq_anc = compute_threat_frequencies(
        data["TIDY_THREATS"], 
        threat_type="non_climatic"
    )
    results.threat_pairs = compute_cooccurrence(data["TIDY_THREATS"], top_n=50)
    
    save_csv(results.threat_freq_ac, dirs["tables"] / "threat_frequencies_ac.csv")
    save_csv(results.threat_freq_anc, dirs["tables"] / "threat_frequencies_anc.csv")
    save_csv(results.threat_pairs, dirs["tables"] / "threat_pairs.csv")
    
    # Step A3: Compute scores
    logger.info("A3: Computing indices...")
    results.nbs_scores = compute_all_scores(data)
    save_csv(results.nbs_scores, dirs["tables"] / "nbs_scores.csv")
    
    # Step A2: NbS-Threat matching
    logger.info("A2: Matching NbS to threats...")
    results.threat_matching = compute_nbs_threat_matching(
        data["TIDY_THREATS"],
        results.nbs_scores,
        fact
    )
    save_csv(results.threat_matching, dirs["tables"] / "nbs_threat_matching.csv")
    
    # Step A4: Shortlists
    logger.info("A4: Generating shortlists...")
    (
        results.shortlist_low_friction,
        results.shortlist_needs_enablers,
        results.shortlist_thresholds
    ) = compute_shortlists(results.nbs_scores)
    
    save_csv(results.shortlist_low_friction, dirs["tables"] / "shortlist_low_friction.csv")
    save_csv(results.shortlist_needs_enablers, dirs["tables"] / "shortlist_needs_enablers.csv")
    
    # Check for small N warnings
    if results.n_cases < 5:
        results.warnings.append(
            f"Small dataset ({results.n_cases} cases): shortlist thresholds may be unreliable"
        )
    
    # Step A5: Governance analysis
    logger.info("A5: Analyzing governance gaps...")
    results.top_gov_gaps_overall = compute_top_gov_gaps(data["TIDY_GOV_GAPS"])
    save_csv(results.top_gov_gaps_overall, dirs["tables"] / "top_gov_gaps_overall.csv")
    
    # By landscape if available
    landscape_col = pick_col(fact, mappings.get("landscape", ["landscape", "paisaje"]))
    if landscape_col:
        results.top_gov_gaps_by_landscape = compute_top_gov_gaps(
            data["TIDY_GOV_GAPS"],
            fact,
            by_landscape=True
        )
        save_csv(
            results.top_gov_gaps_by_landscape, 
            dirs["tables"] / "top_gov_gaps_by_landscape.csv"
        )
    
    # Load enabling packages
    results.enabling_packages = load_enabling_packages()
    
    # Step A6: One-pagers
    logger.info("A6: Building one-pagers...")
    results.onepagers = build_onepagers(data, results)
    
    save_csv(results.onepagers, dirs["tables"] / "nbs_onepagers.csv")
    save_excel(
        {"OnePagers": results.onepagers},
        dirs["tables"] / "nbs_onepagers.xlsx"
    )
    
    # Generate plots
    logger.info("Generating visualizations...")
    plot_paths = generate_plots(results, output_dir)
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    generate_html_report(
        results,
        plot_paths,
        dirs["reports"] / "report_storyline_a.html"
    )
    
    logger.info(f"Storyline A analysis complete. Outputs in: {output_dir}")
    
    return results
