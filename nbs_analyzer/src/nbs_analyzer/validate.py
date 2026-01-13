"""
Data validation and QA checks for the NbS Analyzer.

Provides functions to validate data quality and generate QA reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import numpy as np

from nbs_analyzer.utils import pick_col, load_column_mappings
from nbs_analyzer.schema import SCHEMAS, get_all_valid_ids

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    
    check_name: str
    passed: bool
    message: str
    severity: str = "error"  # "error", "warning", "info"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAReport:
    """Complete QA report for a dataset."""
    
    validations: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        """Check if all error-level validations passed."""
        return all(v.passed for v in self.validations if v.severity == "error")
    
    @property
    def error_count(self) -> int:
        return sum(1 for v in self.validations if not v.passed and v.severity == "error")
    
    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.validations if not v.passed and v.severity == "warning")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert validations to DataFrame."""
        return pd.DataFrame([
            {
                "check": v.check_name,
                "passed": v.passed,
                "severity": v.severity,
                "message": v.message,
            }
            for v in self.validations
        ])


def validate_id_uniqueness(
    df: pd.DataFrame,
    sheet_name: str,
    id_col: str = "id_nbs"
) -> ValidationResult:
    """Validate that id_nbs values are unique in the sheet."""
    mappings = load_column_mappings()
    actual_col = pick_col(df, mappings.get("id_nbs", [id_col]))
    
    if actual_col is None:
        return ValidationResult(
            check_name=f"{sheet_name}_id_column_exists",
            passed=False,
            message=f"ID column not found in {sheet_name}",
            severity="error"
        )
    
    duplicates = df[actual_col].dropna()
    duplicate_values = duplicates[duplicates.duplicated()].unique().tolist()
    
    if duplicate_values:
        return ValidationResult(
            check_name=f"{sheet_name}_id_uniqueness",
            passed=False,
            message=f"Found {len(duplicate_values)} duplicate IDs in {sheet_name}",
            severity="error",
            details={"duplicate_ids": duplicate_values[:10]}  # Limit to first 10
        )
    
    return ValidationResult(
        check_name=f"{sheet_name}_id_uniqueness",
        passed=True,
        message=f"All {len(duplicates)} IDs in {sheet_name} are unique",
        severity="info"
    )


def validate_foreign_keys(
    tidy_df: pd.DataFrame,
    fact_df: pd.DataFrame,
    tidy_name: str,
    id_col: str = "id_nbs"
) -> ValidationResult:
    """Validate that all IDs in tidy table exist in FACT_NBS."""
    mappings = load_column_mappings()
    id_candidates = mappings.get("id_nbs", [id_col])
    
    tidy_id_col = pick_col(tidy_df, id_candidates)
    fact_id_col = pick_col(fact_df, id_candidates)
    
    if tidy_id_col is None or fact_id_col is None:
        return ValidationResult(
            check_name=f"{tidy_name}_foreign_key",
            passed=False,
            message=f"ID column not found in {tidy_name} or FACT_NBS",
            severity="error"
        )
    
    valid_ids = set(fact_df[fact_id_col].dropna().unique())
    tidy_ids = set(tidy_df[tidy_id_col].dropna().unique())
    
    orphans = tidy_ids - valid_ids
    
    if orphans:
        return ValidationResult(
            check_name=f"{tidy_name}_foreign_key",
            passed=False,
            message=f"Found {len(orphans)} orphan IDs in {tidy_name} not in FACT_NBS",
            severity="warning",
            details={"orphan_ids": list(orphans)[:10]}
        )
    
    return ValidationResult(
        check_name=f"{tidy_name}_foreign_key",
        passed=True,
        message=f"All {len(tidy_ids)} IDs in {tidy_name} exist in FACT_NBS",
        severity="info"
    )


def validate_binary_columns(
    df: pd.DataFrame,
    sheet_name: str,
    value_col: str = "value"
) -> ValidationResult:
    """Validate that value columns contain only 0, 1, or NaN."""
    mappings = load_column_mappings()
    actual_col = pick_col(df, mappings.get("value", [value_col]))
    
    if actual_col is None:
        return ValidationResult(
            check_name=f"{sheet_name}_binary_values",
            passed=True,
            message=f"No value column found in {sheet_name} (skipping check)",
            severity="info"
        )
    
    values = df[actual_col].dropna()
    
    # Convert to numeric
    numeric_values = pd.to_numeric(values, errors="coerce")
    valid_values = {0, 1, 0.0, 1.0}
    
    invalid_mask = ~numeric_values.isin(valid_values) & numeric_values.notna()
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        invalid_examples = numeric_values[invalid_mask].head(5).tolist()
        return ValidationResult(
            check_name=f"{sheet_name}_binary_values",
            passed=False,
            message=f"Found {invalid_count} non-binary values in {sheet_name}.{actual_col}",
            severity="warning",
            details={"invalid_examples": invalid_examples}
        )
    
    return ValidationResult(
        check_name=f"{sheet_name}_binary_values",
        passed=True,
        message=f"All {len(values)} values in {sheet_name}.{actual_col} are binary (0/1)",
        severity="info"
    )


def validate_required_columns(
    df: pd.DataFrame,
    sheet_name: str,
    required_cols: List[str]
) -> ValidationResult:
    """Validate that required columns exist."""
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        return ValidationResult(
            check_name=f"{sheet_name}_required_columns",
            passed=False,
            message=f"Missing required columns in {sheet_name}: {missing}",
            severity="error",
            details={"missing_columns": missing}
        )
    
    return ValidationResult(
        check_name=f"{sheet_name}_required_columns",
        passed=True,
        message=f"All required columns present in {sheet_name}",
        severity="info"
    )


def validate_minimum_rows(
    df: pd.DataFrame,
    sheet_name: str,
    min_rows: int = 1
) -> ValidationResult:
    """Validate that sheet has minimum required rows."""
    row_count = len(df)
    
    if row_count < min_rows:
        return ValidationResult(
            check_name=f"{sheet_name}_minimum_rows",
            passed=False,
            message=f"{sheet_name} has {row_count} rows, minimum {min_rows} required",
            severity="error"
        )
    
    return ValidationResult(
        check_name=f"{sheet_name}_minimum_rows",
        passed=True,
        message=f"{sheet_name} has {row_count} rows",
        severity="info"
    )


def run_all_validations(data: Dict[str, pd.DataFrame]) -> QAReport:
    """Run all validation checks on the loaded data."""
    report = QAReport()
    
    # Validate FACT_NBS
    if "FACT_NBS" in data:
        fact = data["FACT_NBS"]
        report.validations.append(validate_id_uniqueness(fact, "FACT_NBS"))
        report.validations.append(validate_minimum_rows(fact, "FACT_NBS", min_rows=1))
    else:
        report.validations.append(ValidationResult(
            check_name="FACT_NBS_exists",
            passed=False,
            message="FACT_NBS sheet not found",
            severity="error"
        ))
        return report  # Can't continue without FACT_NBS
    
    # Validate tidy tables
    tidy_tables = ["TIDY_THREATS", "TIDY_GOV_GAPS", "TIDY_SECURITY", "TIDY_TRAITS"]
    
    for tidy_name in tidy_tables:
        if tidy_name in data:
            tidy_df = data[tidy_name]
            report.validations.append(validate_minimum_rows(tidy_df, tidy_name))
            report.validations.append(validate_foreign_keys(tidy_df, fact, tidy_name))
            
            if tidy_name in ("TIDY_SECURITY", "TIDY_TRAITS"):
                report.validations.append(validate_binary_columns(tidy_df, tidy_name))
        else:
            report.validations.append(ValidationResult(
                check_name=f"{tidy_name}_exists",
                passed=False,
                message=f"{tidy_name} sheet not found",
                severity="error"
            ))
    
    # Build summary
    report.summary = {
        "total_checks": len(report.validations),
        "passed": sum(1 for v in report.validations if v.passed),
        "failed": sum(1 for v in report.validations if not v.passed),
        "errors": report.error_count,
        "warnings": report.warning_count,
    }
    
    return report


def generate_qa_summary(
    data: Dict[str, pd.DataFrame],
    output_path: Path
) -> pd.DataFrame:
    """Generate and save QA summary report."""
    report = run_all_validations(data)
    qa_df = report.to_dataframe()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    logger.info(f"QA Report: {report.summary}")
    
    if not report.passed:
        logger.warning(f"QA validation has {report.error_count} errors")
    
    return qa_df
