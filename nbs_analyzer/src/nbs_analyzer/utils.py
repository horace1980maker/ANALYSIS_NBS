"""
Utility functions for the NbS Analyzer.

Provides column matching, normalization, and deterministic operations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_column_mappings(config_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load column name mappings from YAML config."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "column_mappings.yml"
    
    if not config_path.exists():
        # Return minimal defaults if config not found
        return {
            "id_nbs": ["id_nbs", "ID_NbS", "id", "nbs_id"],
            "threat_code": ["threat_code", "codigo_amenaza", "amenaza", "threat"],
            "threat_type": ["threat_type", "tipo_amenaza", "type"],
            "gap_code": ["gap_code", "codigo_brecha", "brecha", "gap"],
            "value": ["value", "flag", "seleccion", "selected"],
            "dimension": ["dimension", "security_dimension", "dim", "variable"],
            "trait": ["trait", "rasgo", "transformative_trait"],
        }
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = False,
    context: str = ""
) -> Optional[str]:
    """
    Find the first column in df matching any candidate (case-insensitive).
    
    Args:
        df: DataFrame to search
        candidates: List of possible column names
        required: If True, raise ValueError when not found
        context: Context string for error messages
    
    Returns:
        Matched column name or None if not found and not required
    
    Raises:
        ValueError: If required=True and no match found
    """
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    
    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        if candidate_lower in df_cols_lower:
            return df_cols_lower[candidate_lower]
    
    if required:
        raise ValueError(
            f"Required column not found{f' for {context}' if context else ''}. "
            f"Tried: {candidates}. Available: {list(df.columns)}"
        )
    
    return None


def normalize_missing(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Replace common missing value markers with NaN.
    
    Handles: ND, N/D, NA, N.A., empty strings, whitespace-only strings
    """
    if not inplace:
        df = df.copy()
    
    missing_markers = {"ND", "N/D", "NA", "N.A.", "N/A", "", " ", "nd", "n/d", "na", "n.a.", "n/a"}
    
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: np.nan if isinstance(x, str) and x.strip() in missing_markers else x
            )
            # Also strip whitespace from remaining strings
            df[col] = df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
    
    return df


def normalize_columns(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """Trim whitespace from column names."""
    if not inplace:
        df = df.copy()
    
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_threat_type(code: Any) -> Optional[str]:
    """
    Infer threat type from code prefix.
    
    Returns:
        'climatic' for AC codes, 'non_climatic' for ANC codes, None otherwise
    """
    if not isinstance(code, str):
        return None
    
    code = code.strip().upper()
    if code.startswith("AC") and not code.startswith("ANC"):
        return "climatic"
    elif code.startswith("ANC"):
        return "non_climatic"
    
    return None


def stable_round(value: float, decimals: int = 3) -> float:
    """Round a value with consistent behavior for edge cases."""
    if pd.isna(value):
        return np.nan
    return round(float(value), decimals)


def ensure_deterministic_sort(
    df: pd.DataFrame,
    sort_cols: List[str],
    ascending: bool = True
) -> pd.DataFrame:
    """
    Sort DataFrame deterministically.
    
    Always uses stable sort and handles ties by adding index as secondary key.
    """
    # Filter to existing columns
    valid_cols = [c for c in sort_cols if c in df.columns]
    
    if not valid_cols:
        return df.reset_index(drop=True)
    
    return df.sort_values(
        by=valid_cols,
        ascending=ascending,
        kind="stable",
        na_position="last"
    ).reset_index(drop=True)


def extract_threat_code(text: Any) -> Optional[str]:
    """Extract threat code (AC## or ANC##) from text."""
    if not isinstance(text, str):
        return None
    
    pattern = re.compile(r"\b(ANC\d{2}|AC\d{2})\b", re.IGNORECASE)
    match = pattern.search(text)
    
    if match:
        return match.group(1).upper()
    
    return None


def extract_gap_code(text: Any) -> Optional[str]:
    """Extract gap code (B#.#) from text."""
    if not isinstance(text, str):
        return None
    
    # Match patterns like "1.4", "B1.4", "2.3"
    pattern = re.compile(r"\b[Bb]?(\d+\.\d+)\b")
    match = pattern.search(text)
    
    if match:
        code = match.group(1)
        # Normalize to B prefix
        return f"B{code}"
    
    return None


def safe_percentage(value: float, total: float, decimals: int = 1) -> float:
    """Calculate percentage safely, returning 0 for division by zero."""
    if total == 0 or pd.isna(total):
        return 0.0
    return round(100.0 * value / total, decimals)


def create_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create output directory structure."""
    dirs = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "reports": output_dir / "reports",
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return dirs
