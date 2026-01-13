"""
Excel I/O functions for robust sheet and data handling.

Provides functions to load workbooks, find sheets case-insensitively,
normalize data, and save outputs in multiple formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from nbs_analyzer.utils import normalize_columns, normalize_missing

logger = logging.getLogger(__name__)


def find_sheet(
    sheet_names: List[str],
    candidates: List[str],
    required: bool = False,
    context: str = ""
) -> Optional[str]:
    """
    Find a sheet name case-insensitively with whitespace trimming.
    
    Args:
        sheet_names: List of available sheet names in workbook
        candidates: List of possible sheet names to match
        required: If True, raise ValueError when not found
        context: Context string for error messages
    
    Returns:
        Matched sheet name or None if not found and not required
    """
    normalized = {s.strip().lower(): s for s in sheet_names}
    
    for candidate in candidates:
        candidate_lower = candidate.strip().lower()
        if candidate_lower in normalized:
            return normalized[candidate_lower]
    
    if required:
        raise ValueError(
            f"Required sheet not found{f' for {context}' if context else ''}. "
            f"Tried: {candidates}. Available: {sheet_names}"
        )
    
    return None


def load_workbook(
    input_path: Path,
    required_sheets: Optional[List[str]] = None,
    optional_sheets: Optional[List[str]] = None
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Load an Excel workbook with robust sheet matching.
    
    Args:
        input_path: Path to Excel file
        required_sheets: List of sheet names that must exist
        optional_sheets: List of sheet names to load if present
    
    Returns:
        Tuple of (dict of sheet name -> DataFrame, list of missing optional sheets)
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required sheets are missing
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading workbook: {input_path}")
    
    # Get all sheet names
    xl = pd.ExcelFile(input_path, engine="openpyxl")
    available_sheets = xl.sheet_names
    logger.info(f"Available sheets: {available_sheets}")
    
    required_sheets = required_sheets or []
    optional_sheets = optional_sheets or []
    
    data: Dict[str, pd.DataFrame] = {}
    missing_optional: List[str] = []
    
    # Load required sheets
    for sheet_key in required_sheets:
        sheet_name = find_sheet(
            available_sheets,
            [sheet_key, sheet_key.lower(), sheet_key.upper()],
            required=True,
            context=sheet_key
        )
        logger.info(f"Loading required sheet: {sheet_name}")
        df = pd.read_excel(xl, sheet_name=sheet_name, engine="openpyxl")
        df = normalize_columns(df)
        df = normalize_missing(df)
        data[sheet_key] = df
    
    # Load optional sheets
    for sheet_key in optional_sheets:
        sheet_name = find_sheet(
            available_sheets,
            [sheet_key, sheet_key.lower(), sheet_key.upper()],
            required=False
        )
        if sheet_name:
            logger.info(f"Loading optional sheet: {sheet_name}")
            df = pd.read_excel(xl, sheet_name=sheet_name, engine="openpyxl")
            df = normalize_columns(df)
            df = normalize_missing(df)
            data[sheet_key] = df
        else:
            logger.warning(f"Optional sheet not found: {sheet_key}")
            missing_optional.append(sheet_key)
    
    xl.close()
    
    return data, missing_optional


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save DataFrame to CSV with consistent encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    logger.info(f"Saved CSV: {path} ({len(df)} rows)")


def save_excel(
    dfs: Dict[str, pd.DataFrame],
    path: Path,
    index: bool = False
) -> None:
    """Save multiple DataFrames to Excel with each as a sheet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            # Excel sheet names limited to 31 chars
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=index)
    
    logger.info(f"Saved Excel: {path} ({len(dfs)} sheets)")


def save_outputs(
    tables: Dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str = ""
) -> Dict[str, Path]:
    """
    Save all tables as both CSV and combined Excel.
    
    Returns dict of table name -> CSV path
    """
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths: Dict[str, Path] = {}
    
    # Save individual CSVs
    for name, df in tables.items():
        csv_path = tables_dir / f"{prefix}{name}.csv"
        save_csv(df, csv_path)
        saved_paths[name] = csv_path
    
    # Save combined Excel
    if tables:
        xlsx_path = tables_dir / f"{prefix}all_tables.xlsx"
        save_excel(tables, xlsx_path)
        saved_paths["_combined_xlsx"] = xlsx_path
    
    return saved_paths
