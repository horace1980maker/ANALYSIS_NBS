"""
Storyline C (Transformation-First) Pipeline Orchestrator.

Implements Journey C:
- C1: Transformative traits profile
- C2: Transformation patterns (signatures + co-occurrence)
- C3: Link to co-benefits (security)
- C4: Link to threats (coverage + lift)
- C5: Governance enabling conditions
- C6: Transformation-first shortlists
- C7: One-pager generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbs_analyzer.io import load_workbook, save_csv, save_excel
from nbs_analyzer.schema import REQUIRED_SHEETS, OPTIONAL_SHEETS
from nbs_analyzer.validate import generate_qa_summary
from nbs_analyzer.metrics import (
    compute_tcs,
    compute_sbs,
    compute_tts,
    compute_ggl,
    load_enabling_packages,
    map_gaps_to_packages,
)
from nbs_analyzer.utils import (
    pick_col,
    load_column_mappings,
    ensure_deterministic_sort,
    create_output_dirs,
    stable_round,
)
from nbs_analyzer.report import generate_html_report

logger = logging.getLogger(__name__)

@dataclass
class StorylineCResults:
    """Container for all Storyline C outputs."""
    # Traits
    trait_rates: pd.DataFrame = field(default_factory=pd.DataFrame)
    tts_by_case: pd.DataFrame = field(default_factory=pd.DataFrame)
    trait_signatures: pd.DataFrame = field(default_factory=pd.DataFrame)
    trait_pairs: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Links
    sbs_by_tts_group: pd.DataFrame = field(default_factory=pd.DataFrame)
    security_lift: pd.DataFrame = field(default_factory=pd.DataFrame)
    threat_coverage_by_tts_group: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_threats_high_tts: pd.DataFrame = field(default_factory=pd.DataFrame)
    threat_lift: pd.DataFrame = field(default_factory=pd.DataFrame)
    ggl_by_tts_group: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_gaps_high_tts: pd.DataFrame = field(default_factory=pd.DataFrame)
    gap_lift: pd.DataFrame = field(default_factory=pd.DataFrame)
    packages_high_tts: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Shortlists
    shortlist_c1: pd.DataFrame = field(default_factory=pd.DataFrame)
    shortlist_c2: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Outputs
    onepagers: pd.DataFrame = field(default_factory=pd.DataFrame)
    qa_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Metadata
    input_file: str = ""
    output_dir: str = ""
    run_timestamp: str = ""
    n_cases: int = 0
    warnings: List[str] = field(default_factory=list)
    archetype_counts: Dict[str, int] = field(default_factory=dict)

def define_tts_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    """Assign High/Med/Low archetypes based on TTS."""
    n = len(df)
    if n == 0:
        df["archetype"] = None
        return df

    if n >= 8:
        q25 = df["TTS"].quantile(0.25)
        q75 = df["TTS"].quantile(0.75)
        def get_arch(tts):
            if tts >= q75: return "High"
            if tts <= q25: return "Low"
            return "Medium"
    else:
        # For small N: top ceil(0.25*N), bottom ceil(0.25*N)
        k = int(np.ceil(0.25 * n))
        # Handle cases where k=0
        k = max(1, k)
        
        # Sort for ranking
        sorted_df = df.sort_values(by=["TTS", "id_nbs"], ascending=[False, True])
        high_ids = sorted_df.head(k)["id_nbs"].tolist()
        low_ids = sorted_df.tail(k)["id_nbs"].tolist()
        
        def get_arch_small(row):
            if row["id_nbs"] in high_ids: return "High"
            if row["id_nbs"] in low_ids: return "Low"
            return "Medium"
        
        df["archetype"] = df.apply(get_arch_small, axis=1)
        return df

    df["archetype"] = df["TTS"].apply(get_arch)
    return df

def compute_lift(df_tidy: pd.DataFrame, df_scores: pd.DataFrame, group_col: str, item_col: str, val_col: str = None) -> pd.DataFrame:
    """Compute lift (pct_high - pct_low) for binary flags or presence."""
    mappings = load_column_mappings()
    id_col = "id_nbs" # Standardized
    
    # Ensure id_nbs in df_tidy
    if id_col not in df_tidy.columns:
        tidy_id = pick_col(df_tidy, mappings.get("id_nbs", ["id_nbs"]))
        if tidy_id: df_tidy = df_tidy.rename(columns={tidy_id: id_col})
    
    # Map item_col if needed
    actual_item_col = pick_col(df_tidy, mappings.get(item_col, [item_col])) or item_col
    
    high_cases = df_scores[df_scores["archetype"] == "High"][id_col].unique()
    low_cases = df_scores[df_scores["archetype"] == "Low"][id_col].unique()
    
    n_high = len(high_cases)
    n_low = len(low_cases)
    
    if n_high == 0 or n_low == 0:
        return pd.DataFrame(columns=[item_col, "pct_high", "pct_low", "lift"])

    # If val_col is provided, we check value == 1. If not, we check presence in the table.
    if val_col:
        actual_val_col = pick_col(df_tidy, mappings.get(val_col, [val_col])) or val_col
        df_val = df_tidy[df_tidy[actual_val_col].astype(float) == 1]
    else:
        df_val = df_tidy

    items = df_val[actual_item_col].dropna().unique()
    records = []
    
    for item in items:
        cases_with_item = df_val[df_val[actual_item_col] == item][id_col].unique()
        
        c_high = len(set(cases_with_item) & set(high_cases))
        c_low = len(set(cases_with_item) & set(low_cases))
        
        pct_high = 100.0 * c_high / n_high
        pct_low = 100.0 * c_low / n_low
        lift = pct_high - pct_low
        
        records.append({
            item_col: item,
            "pct_high": stable_round(pct_high),
            "pct_low": stable_round(pct_low),
            "lift": stable_round(lift)
        })
        
    lift_df = pd.DataFrame(records)
    if not lift_df.empty:
        lift_df = lift_df.sort_values(by=["lift", item_col], ascending=[False, True]).reset_index(drop=True)
    return lift_df

def generate_plots(results: StorylineCResults, output_dir: Path) -> Dict[str, Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 150})
    plot_paths = {}

    # trait_rates.png
    if not results.trait_rates.empty:
        fig, ax = plt.subplots()
        data = results.trait_rates.sort_values("percentage", ascending=True)
        ax.barh(data["trait"], data["percentage"], color="skyblue")
        ax.set_title("Transformative Trait Frequencies (%)")
        ax.set_xlabel("Percentage of cases")
        plt.tight_layout()
        path = figures_dir / "trait_rates.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths["trait_rates"] = path

    # tts_distribution.png
    if not results.tts_by_case.empty:
        fig, ax = plt.subplots()
        counts = results.tts_by_case["archetype"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        ax.bar(counts.index, counts.values, color=["forestgreen", "gold", "salmon"])
        ax.set_title("TTS Archetype Distribution")
        ax.set_ylabel("Number of cases")
        plt.tight_layout()
        path = figures_dir / "tts_distribution.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths["tts_distribution"] = path

    # security_lift.png
    if not results.security_lift.empty:
        fig, ax = plt.subplots()
        data = results.security_lift.head(15).sort_values("lift", ascending=True)
        ax.barh(data["dimension"], data["lift"], color="teal")
        ax.set_title("Security Dimension 'Lift' (High vs Low TTS)")
        ax.set_xlabel("Lift (pct_high - pct_low)")
        plt.tight_layout()
        path = figures_dir / "security_lift.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths["security_lift"] = path

    # threat_lift.png
    if not results.threat_lift.empty:
        fig, ax = plt.subplots()
        # Take top 15 by absolute lift
        data = results.threat_lift.assign(abs_lift=results.threat_lift["lift"].abs()).sort_values("abs_lift", ascending=False).head(15)
        data = data.sort_values("lift", ascending=True)
        ax.barh(data["threat_code"], data["lift"], color="coral")
        ax.set_title("Threat 'Lift' (High vs Low TTS)")
        ax.set_xlabel("Lift (pct_high - pct_low)")
        plt.tight_layout()
        path = figures_dir / "threat_lift.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths["threat_lift"] = path

    # gap_lift.png
    if not results.gap_lift.empty:
        fig, ax = plt.subplots()
        data = results.gap_lift.assign(abs_lift=results.gap_lift["lift"].abs()).sort_values("abs_lift", ascending=False).head(15)
        data = data.sort_values("lift", ascending=True)
        ax.barh(data["gap_code"], data["lift"], color="purple")
        ax.set_title("Governance Gap 'Lift' (High vs Low TTS)")
        ax.set_xlabel("Lift (pct_high - pct_low)")
        plt.tight_layout()
        path = figures_dir / "gap_lift.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths["gap_lift"] = path

    # value_map_transformation.png (GGL vs TTS bubble size=SBS)
    if not results.tts_by_case.empty:
        fig, ax = plt.subplots()
        df = results.tts_by_case
        sc = ax.scatter(df["GGL"], df["TTS"], s=df["SBS"]*50, alpha=0.6, c=df["TTS"], cmap="viridis")
        ax.set_xlabel("Governance Gap Load (GGL)")
        ax.set_ylabel("Transformative Traits Score (TTS)")
        ax.set_title("Transformation Value Map (Bubble size = SBS)")
        plt.colorbar(sc, label="TTS")
        plt.tight_layout()
        path = figures_dir / "value_map_transformation.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths["value_map_transformation"] = path

    return plot_paths

def build_onepagers(data: Dict[str, pd.DataFrame], results: StorylineCResults) -> pd.DataFrame:
    mappings = load_column_mappings()
    fact = data.get("FACT_NBS", pd.DataFrame())
    if fact.empty: return pd.DataFrame()
    
    id_col = pick_col(fact, mappings.get("id_nbs", ["id_nbs"]))
    country_col = pick_col(fact, mappings.get("country", ["country", "pais"]))
    landscape_col = pick_col(fact, mappings.get("landscape", ["landscape", "paisaje"]))
    org_col = pick_col(fact, mappings.get("organization", ["organization", "org"]))
    combo_col = pick_col(fact, mappings.get("combination", ["combinacion", "combination"]))
    desc_col = pick_col(fact, mappings.get("description", ["descripcion", "description"]))
    
    tidy_traits = data.get("TIDY_TRAITS", pd.DataFrame())
    tidy_sec = data.get("TIDY_SECURITY", pd.DataFrame())
    tidy_threats = data.get("TIDY_THREATS", pd.DataFrame())
    tidy_gov = data.get("TIDY_GOV_GAPS", pd.DataFrame())
    
    trait_col = pick_col(tidy_traits, ["trait", "rasgo", "t_code"])
    trait_val = pick_col(tidy_traits, ["value", "flag", "bin"])
    sec_dim = pick_col(tidy_sec, ["dimension", "dim"])
    sec_val = pick_col(tidy_sec, ["value", "flag", "bin"])
    threat_code = pick_col(tidy_threats, ["threat_code", "code"])
    gap_code = pick_col(tidy_gov, ["gap_code", "code"])
    
    packages_config = load_enabling_packages()
    
    records = []
    for _, row in fact.iterrows():
        nid = row[id_col]
        rec = {"id_nbs": nid}
        if country_col: rec["country"] = row.get(country_col)
        if landscape_col: rec["landscape"] = row.get(landscape_col)
        if org_col: rec["organization"] = row.get(org_col)
        if combo_col: rec["combination"] = row.get(combo_col)
        if desc_col: rec["description"] = row.get(desc_col)
        
        # Transformation
        c_traits = tidy_traits[tidy_traits[id_col] == nid]
        flagged_traits = sorted(c_traits[c_traits[trait_val].astype(float) == 1][trait_col].dropna().unique())
        rec["traits_flagged"] = ", ".join(flagged_traits)
        rec["signature"] = "+".join(flagged_traits) if flagged_traits else "None"
        
        # Scores from results.tts_by_case
        case_scores = results.tts_by_case[results.tts_by_case["id_nbs"] == nid]
        if not case_scores.empty:
            for col in ["TTS", "SBS", "TCS", "n_AC", "n_ANC", "GGL", "archetype"]:
                if col in case_scores.columns: rec[col] = case_scores[col].values[0]
        
        # Co-benefits
        c_sec = tidy_sec[tidy_sec[id_col] == nid]
        rec["security_dims"] = ", ".join(sorted(c_sec[c_sec[sec_val].astype(float) == 1][sec_dim].dropna().unique()))
        
        # Threats
        c_threats = tidy_threats[tidy_threats[id_col] == nid]
        rec["threats_ac"] = ", ".join(sorted([t for t in c_threats[threat_code].dropna().unique() if t.startswith("AC")]))
        rec["threats_anc"] = ", ".join(sorted([t for t in c_threats[threat_code].dropna().unique() if t.startswith("ANC")]))
        
        # Gov
        c_gov = tidy_gov[tidy_gov[id_col] == nid]
        gaps = c_gov[gap_code].dropna().unique()
        rec["gaps"] = ", ".join(sorted(gaps))
        pkgs = map_gaps_to_packages(gaps, packages_config)
        rec["enabling_packages"] = ", ".join(sorted(pkgs.keys()))
        
        records.append(rec)
        
    return pd.DataFrame(records)

def run_storyline_c(
    input_path: str | Path,
    output_dir: str | Path,
    filters: Optional[Dict[str, Any]] = None
) -> StorylineCResults:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    results = StorylineCResults(
        input_file=str(input_path),
        output_dir=str(output_dir),
        run_timestamp=datetime.now().isoformat()
    )
    
    logger.info("C0: Loading workbook...")
    data, missing_lookups = load_workbook(input_path, REQUIRED_SHEETS, OPTIONAL_SHEETS)
    
    # Simple filtering (clone logic from Storyline A if needed, but keeping it brief)
    fact = data["FACT_NBS"]
    mappings = load_column_mappings()
    id_col = pick_col(fact, mappings.get("id_nbs", ["id_nbs"]))
    
    if filters:
        for k, v in filters.items():
            col = pick_col(fact, mappings.get(k, [k]))
            if col: fact = fact[fact[col] == v]
        valid_ids = fact[id_col].unique()
        data["FACT_NBS"] = fact
        for sheet in ["TIDY_TRAITS", "TIDY_GOV_GAPS", "TIDY_SECURITY", "TIDY_THREATS"]:
            if sheet in data:
                tid = pick_col(data[sheet], mappings.get("id_nbs", ["id_nbs"]))
                data[sheet] = data[sheet][data[sheet][tid].isin(valid_ids)]

    results.n_cases = fact[id_col].nunique()
    dirs = create_output_dirs(output_dir)
    
    # Step QA
    results.qa_summary = generate_qa_summary(data, dirs["tables"] / "qa_summary_storyline_c.csv")

    # C1: Transformative traits
    logger.info("C1: Trait analysis")
    tidy_traits = data["TIDY_TRAITS"]
    trait_col = pick_col(tidy_traits, ["trait", "rasgo", "t_code"])
    trait_val = pick_col(tidy_traits, ["value", "flag", "bin"])
    
    # trait_rates
    counts = tidy_traits[tidy_traits[trait_val].astype(float) == 1].groupby(trait_col)[id_col].nunique().reset_index()
    counts.columns = ["trait", "n_cases"]
    counts["percentage"] = (100.0 * counts["n_cases"] / results.n_cases).apply(stable_round)
    results.trait_rates = counts.sort_values("n_cases", ascending=False).reset_index(drop=True)
    save_csv(results.trait_rates, dirs["tables"] / "trait_rates.csv")
    
    # tts_by_case
    tts_df = compute_tts(tidy_traits)
    tts_df = define_tts_archetypes(tts_df)
    results.archetype_counts = tts_df["archetype"].value_counts().to_dict()
    
    # Add other scores for link analysis
    sbs_df = compute_sbs(data["TIDY_SECURITY"])
    tcs_df = compute_tcs(data["TIDY_THREATS"])
    ggl_df = compute_ggl(data["TIDY_GOV_GAPS"])
    
    all_scores = tts_df.merge(sbs_df, on="id_nbs", how="left").merge(tcs_df, on="id_nbs", how="left").merge(ggl_df, on="id_nbs", how="left")
    for c in ["SBS", "TCS", "n_AC", "n_ANC", "GGL"]: all_scores[c] = all_scores[c].fillna(0).astype(int)
    results.tts_by_case = all_scores
    save_csv(results.tts_by_case, dirs["tables"] / "tts_by_case.csv")

    # C2: Patterns
    logger.info("C2: Patterns")
    sigs = tidy_traits[tidy_traits[trait_val].astype(float) == 1].groupby(id_col)[trait_col].apply(lambda x: "+".join(sorted(x))).reset_index()
    sigs.columns = [id_col, "signature"]
    results.trait_signatures = sigs["signature"].value_counts().reset_index()
    results.trait_signatures.columns = ["signature", "n_cases"]
    save_csv(results.trait_signatures, dirs["tables"] / "trait_signatures.csv")
    
    # Co-occurrence
    pair_counts = {}
    for traits_list in tidy_traits[tidy_traits[trait_val].astype(float) == 1].groupby(id_col)[trait_col].apply(lambda x: sorted(set(x))):
        if len(traits_list) >= 2:
            for p in itertools.combinations(traits_list, 2):
                pair_counts[p] = pair_counts.get(p, 0) + 1
    results.trait_pairs = pd.DataFrame([{"trait_a": k[0], "trait_b": k[1], "n_cases": v} for k, v in pair_counts.items()]).sort_values("n_cases", ascending=False).reset_index(drop=True)
    save_csv(results.trait_pairs, dirs["tables"] / "trait_pairs.csv")

    # C3: Security Link
    logger.info("C3: Security Lift")
    results.sbs_by_tts_group = all_scores.groupby("archetype")["SBS"].agg(["mean", "median", "count"]).reset_index()
    save_csv(results.sbs_by_tts_group, dirs["tables"] / "sbs_by_tts_group.csv")
    results.security_lift = compute_lift(data["TIDY_SECURITY"], all_scores, "archetype", "dimension", "value")
    save_csv(results.security_lift, dirs["tables"] / "security_lift_high_vs_low_tts.csv")

    # C4: Threat Link
    logger.info("C4: Threat Lift")
    results.threat_coverage_by_tts_group = all_scores.groupby("archetype")[["TCS", "n_AC", "n_ANC"]].mean().reset_index()
    save_csv(results.threat_coverage_by_tts_group, dirs["tables"] / "threat_coverage_by_tts_group.csv")
    
    high_ids = all_scores[all_scores["archetype"] == "High"]["id_nbs"]
    high_threats = data["TIDY_THREATS"][data["TIDY_THREATS"][id_col].isin(high_ids)]
    hc = pick_col(high_threats, ["threat_code", "code"])
    results.top_threats_high_tts = high_threats.groupby(hc)[id_col].nunique().sort_values(ascending=False).reset_index()
    results.top_threats_high_tts.columns = ["threat_code", "n_cases"]
    save_csv(results.top_threats_high_tts, dirs["tables"] / "top_threats_high_tts.csv")
    
    results.threat_lift = compute_lift(data["TIDY_THREATS"], all_scores, "archetype", "threat_code")
    save_csv(results.threat_lift, dirs["tables"] / "threat_lift_high_vs_low_tts.csv")

    # C5: Gov Gaps
    logger.info("C5: Gov Lift")
    results.ggl_by_tts_group = all_scores.groupby("archetype")["GGL"].agg(["mean", "median", "count"]).reset_index()
    save_csv(results.ggl_by_tts_group, dirs["tables"] / "ggl_by_tts_group.csv")
    
    high_gaps = data["TIDY_GOV_GAPS"][data["TIDY_GOV_GAPS"][id_col].isin(high_ids)]
    gc = pick_col(high_gaps, ["gap_code", "code"])
    results.top_gaps_high_tts = high_gaps.groupby(gc)[id_col].nunique().sort_values(ascending=False).reset_index()
    results.top_gaps_high_tts.columns = ["gap_code", "n_cases"]
    save_csv(results.top_gaps_high_tts, dirs["tables"] / "top_gaps_high_tts.csv")
    
    results.gap_lift = compute_lift(data["TIDY_GOV_GAPS"], all_scores, "archetype", "gap_code")
    save_csv(results.gap_lift, dirs["tables"] / "gap_lift_high_vs_low_tts.csv")
    
    # Packages
    pkg_cfg = load_enabling_packages()
    pkg_rows = []
    for h_id in high_ids:
        case_gaps = data["TIDY_GOV_GAPS"][data["TIDY_GOV_GAPS"][id_col] == h_id][gc].dropna().unique()
        pkgs = map_gaps_to_packages(case_gaps, pkg_cfg)
        for p in pkgs: pkg_rows.append({"id_nbs": h_id, "package": p})
    pkg_df = pd.DataFrame(pkg_rows)
    if not pkg_df.empty:
        pkg_counts = pkg_df.groupby("package")["id_nbs"].nunique().reset_index()
        pkg_counts.columns = ["package", "n_cases"]
        pkg_counts["percentage"] = (100.0 * pkg_counts["n_cases"] / len(high_ids)).apply(stable_round)
        results.packages_high_tts = pkg_counts.sort_values("n_cases", ascending=False).reset_index(drop=True)
    save_csv(results.packages_high_tts, dirs["tables"] / "packages_high_tts.csv")

    # C6: Shortlists
    logger.info("C6: Shortlists")
    # C1 Transformative leaders: High TTS (top 20%)
    k1 = max(3, int(np.ceil(0.2 * results.n_cases)))
    results.shortlist_c1 = all_scores.sort_values(["TTS", "SBS", "TCS", "id_nbs"], ascending=[False, False, False, True]).head(k1)
    
    # C2 Needs enablers: High TTS AND High GGL (top 40% GGL)
    high_tts = all_scores[all_scores["archetype"] == "High"]
    ggl_q = all_scores["GGL"].quantile(0.6) # top 40%
    results.shortlist_c2 = high_tts[high_tts["GGL"] >= ggl_q].sort_values(["TTS", "SBS", "TCS", "id_nbs"], ascending=[False, False, False, True])
    
    save_csv(results.shortlist_c1, dirs["tables"] / "shortlist_C1_transformative_leaders.csv")
    save_csv(results.shortlist_c2, dirs["tables"] / "shortlist_C2_needs_enablers.csv")

    # C7: One-pagers
    logger.info("C7: One-pagers")
    results.onepagers = build_onepagers(data, results)
    save_csv(results.onepagers, dirs["tables"] / "nbs_onepagers_storyline_C.csv")
    
    onepager_sheets = {"SUMMARY": results.onepagers}
    if results.n_cases <= 30:
        for nid in sorted(results.onepagers["id_nbs"].unique()):
            onepager_sheets[str(nid)] = results.onepagers[results.onepagers["id_nbs"] == nid].T.reset_index()
    save_excel(onepager_sheets, dirs["tables"] / "nbs_onepagers_storyline_C.xlsx")

    # Plots
    logger.info("Generating plots...")
    plot_paths = generate_plots(results, output_dir)
    
    # Report
    logger.info("Generating report...")
    generate_html_report(results, plot_paths, dirs["reports"] / "report_storyline_c.html", template_name="report_storyline_c.html.j2")
    
    return results
