"""
Schema definitions for the NbS Analyzer.

Defines required sheets, columns, and expected data types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# Required sheets that must be present
REQUIRED_SHEETS = [
    "FACT_NBS",
    "TIDY_THREATS",
    "TIDY_GOV_GAPS",
    "TIDY_SECURITY",
    "TIDY_TRAITS",
]

# Optional sheets (will use if present)
OPTIONAL_SHEETS = [
    "LOOKUP_THREATS",
    "LOOKUP_GOV_GAPS",
    "LOOKUP_SECURITY",
    "LOOKUP_TRAITS",
]


@dataclass
class SheetSchema:
    """Schema definition for a single sheet."""
    
    name: str
    required_columns: List[str] = field(default_factory=list)
    optional_columns: List[str] = field(default_factory=list)
    primary_key: Optional[str] = None
    foreign_key: Optional[str] = None  # Column that references another table


# Schema definitions for each sheet
SCHEMAS: Dict[str, SheetSchema] = {
    "FACT_NBS": SheetSchema(
        name="FACT_NBS",
        required_columns=["id_nbs"],
        optional_columns=[
            "country", "pais",
            "landscape", "paisaje",
            "organization", "org", "organizacion",
            "combinacion", "combination",
            "descripcion", "description",
            "seg_total",
            "t_score",
        ],
        primary_key="id_nbs",
    ),
    
    "TIDY_THREATS": SheetSchema(
        name="TIDY_THREATS",
        required_columns=["id_nbs"],
        optional_columns=[
            "threat_code", "codigo_amenaza", "amenaza",
            "threat_type", "tipo_amenaza",
            "threat_label", "amenaza_label",
            "threat_group",
        ],
        foreign_key="id_nbs",
    ),
    
    "TIDY_GOV_GAPS": SheetSchema(
        name="TIDY_GOV_GAPS",
        required_columns=["id_nbs"],
        optional_columns=[
            "gap_code", "codigo_brecha", "brecha",
            "gap_name", "nombre_brecha",
            "gap_group", "grupo_brecha",
        ],
        foreign_key="id_nbs",
    ),
    
    "TIDY_SECURITY": SheetSchema(
        name="TIDY_SECURITY",
        required_columns=["id_nbs"],
        optional_columns=[
            "dimension", "security_dimension",
            "value", "flag",
            "definition",
        ],
        foreign_key="id_nbs",
    ),
    
    "TIDY_TRAITS": SheetSchema(
        name="TIDY_TRAITS",
        required_columns=["id_nbs"],
        optional_columns=[
            "trait", "rasgo",
            "value", "flag",
            "definition",
        ],
        foreign_key="id_nbs",
    ),
    
    "LOOKUP_THREATS": SheetSchema(
        name="LOOKUP_THREATS",
        required_columns=[],
        optional_columns=[
            "threat_code",
            "threat_type", "tipo_amenaza",
            "threat", "amenaza",
            "group", "grupo",
            "examples", "ejemplos",
        ],
    ),
    
    "LOOKUP_GOV_GAPS": SheetSchema(
        name="LOOKUP_GOV_GAPS",
        required_columns=[],
        optional_columns=[
            "gap_code", "codigo_brecha",
            "gap_name", "nombre_brecha",
            "gap_group", "grupo_brecha",
            "gap_description", "descripcion_brecha",
        ],
    ),
    
    "LOOKUP_SECURITY": SheetSchema(
        name="LOOKUP_SECURITY",
        required_columns=[],
        optional_columns=[
            "variable",
            "covers",
            "mark_when",
            "evidence_examples",
        ],
    ),
    
    "LOOKUP_TRAITS": SheetSchema(
        name="LOOKUP_TRAITS",
        required_columns=[],
        optional_columns=[
            "trait", "trait_raw",
            "short_definition",
            "long_definition",
        ],
    ),
}


def get_required_columns(sheet_name: str) -> List[str]:
    """Get required columns for a sheet."""
    if sheet_name not in SCHEMAS:
        return []
    return SCHEMAS[sheet_name].required_columns


def get_all_valid_ids(fact_nbs: 'pd.DataFrame', id_col: str = "id_nbs") -> Set:
    """Extract all valid IDs from FACT_NBS."""
    import pandas as pd
    
    if id_col not in fact_nbs.columns:
        return set()
    
    return set(fact_nbs[id_col].dropna().unique())
