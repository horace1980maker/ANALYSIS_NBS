"""
Smoke tests for Storyline B (Benefits/Equity-First) analysis.

Tests basic functionality with minimal datasets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


def create_minimal_workbook_b(path: Path) -> None:
    """Create a minimal test workbook for Storyline B with 2 cases."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # FACT_NBS with beneficiary data
        fact = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS002"],
            "country": ["Honduras", "Honduras"],
            "landscape": ["Zona Alta", "Zona Baja"],
            "combinacion": ["Agroforestry", "Reforestation"],
            "benef_directos_n": [150, 200],
            "benef_indirectos_n": [500, 800],
            "benef_directos_desc": ["Mujeres agricultoras de la zona", "Jóvenes y campesinos"],
        })
        fact.to_excel(writer, sheet_name="FACT_NBS", index=False)
        
        # TIDY_THREATS
        threats = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS001", "NBS002"],
            "threat_code": ["AC01", "ANC05", "AC02"],
            "threat_type": ["climatic", "non_climatic", "climatic"],
            "threat_label": ["Drought", "Deforestation", "Flooding"],
        })
        threats.to_excel(writer, sheet_name="TIDY_THREATS", index=False)
        
        # TIDY_GOV_GAPS
        gaps = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS002", "NBS002"],
            "gap_code": ["B1.1", "B2.1", "B2.2"],
            "gap_name": ["Land tenure", "Coordination", "Multi-stakeholder"],
        })
        gaps.to_excel(writer, sheet_name="TIDY_GOV_GAPS", index=False)
        
        # TIDY_SECURITY
        security = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS001", "NBS001", "NBS002", "NBS002"],
            "dimension": ["seg_water", "seg_food", "seg_energy", "seg_water", "seg_climate"],
            "value": [1, 1, 1, 1, 1],
        })
        security.to_excel(writer, sheet_name="TIDY_SECURITY", index=False)
        
        # TIDY_TRAITS (optional)
        traits = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS002", "NBS002"],
            "trait": ["t1_systemic", "t2_anticipatory", "t3_flexible"],
            "value": [1, 1, 1],
        })
        traits.to_excel(writer, sheet_name="TIDY_TRAITS", index=False)


def create_workbook_no_numeric_benef(path: Path) -> None:
    """Create a workbook without numeric beneficiary data (proxy mode)."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # FACT_NBS with descriptions only (no numeric counts)
        fact = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS002"],
            "country": ["Honduras", "Guatemala"],
            "landscape": ["Zona Alta", "Zona Baja"],
            "benef_directos_desc": ["Mujeres indígenas de la comunidad", "Agricultores y jóvenes"],
            "benef_indirectos_desc": ["Comunidades rurales", "Familias campesinas"],
        })
        fact.to_excel(writer, sheet_name="FACT_NBS", index=False)
        
        # TIDY_THREATS
        threats = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS002"],
            "threat_code": ["AC01", "AC02"],
        })
        threats.to_excel(writer, sheet_name="TIDY_THREATS", index=False)
        
        # TIDY_GOV_GAPS
        gaps = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS002"],
            "gap_code": ["B1.1", "B2.1"],
        })
        gaps.to_excel(writer, sheet_name="TIDY_GOV_GAPS", index=False)
        
        # TIDY_SECURITY
        security = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS001", "NBS002"],
            "dimension": ["seg_water", "seg_food", "seg_water"],
            "value": [1, 1, 1],
        })
        security.to_excel(writer, sheet_name="TIDY_SECURITY", index=False)


class TestStorylineBMinimal:
    """Tests with a minimal 2-case dataset."""
    
    @pytest.fixture
    def minimal_workbook(self, tmp_path: Path) -> Path:
        """Create minimal workbook fixture with numeric beneficiaries."""
        wb_path = tmp_path / "minimal_b.xlsx"
        create_minimal_workbook_b(wb_path)
        return wb_path
    
    @pytest.fixture
    def proxy_workbook(self, tmp_path: Path) -> Path:
        """Create workbook for proxy mode (no numeric beneficiaries)."""
        wb_path = tmp_path / "proxy_mode.xlsx"
        create_workbook_no_numeric_benef(wb_path)
        return wb_path
    
    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create output directory fixture."""
        out = tmp_path / "outputs"
        out.mkdir()
        return out
    
    def test_full_pipeline_with_numeric(self, minimal_workbook: Path, output_dir: Path) -> None:
        """Test full Storyline B pipeline with numeric beneficiary data."""
        from nbs_analyzer.storyline_b import run_storyline_b
        
        results = run_storyline_b(
            input_path=minimal_workbook,
            output_dir=output_dir,
        )
        
        assert results.n_cases == 2
        assert results.has_numeric_beneficiaries is True
        assert not results.security_dimension_rates.empty
        assert not results.security_breadth_by_case.empty
        assert not results.beneficiary_summary_overall.empty
        assert not results.onepagers.empty
        
        # Check output files exist
        assert (output_dir / "tables" / "security_dimension_rates.csv").exists()
        assert (output_dir / "tables" / "security_breadth_by_case.csv").exists()
        assert (output_dir / "tables" / "beneficiary_summary_overall.csv").exists()
        assert (output_dir / "tables" / "nbs_onepagers_storyline_b.xlsx").exists()
        assert (output_dir / "reports" / "report_storyline_b.html").exists()
    
    def test_proxy_mode(self, proxy_workbook: Path, output_dir: Path) -> None:
        """Test Storyline B pipeline falls back to proxy mode when no numeric data."""
        from nbs_analyzer.storyline_b import run_storyline_b
        
        results = run_storyline_b(
            input_path=proxy_workbook,
            output_dir=output_dir,
        )
        
        assert results.n_cases == 2
        assert results.has_numeric_beneficiaries is False
        assert not results.beneficiary_group_summary.empty
        
        # Check warning was added
        assert any("proxy" in w.lower() for w in results.warnings)
    
    def test_security_profile(self, minimal_workbook: Path) -> None:
        """Test security dimension profile computation."""
        from nbs_analyzer.io import load_workbook
        from nbs_analyzer.storyline_b import compute_security_profile
        from nbs_analyzer.utils import load_column_mappings
        
        data, _ = load_workbook(
            minimal_workbook,
            required_sheets=["FACT_NBS", "TIDY_SECURITY"]
        )
        mappings = load_column_mappings()
        
        results = compute_security_profile(
            data["TIDY_SECURITY"],
            data["FACT_NBS"],
            mappings
        )
        
        assert not results["dimension_rates"].empty
        assert not results["breadth_by_case"].empty
        assert "SBS" in results["breadth_by_case"].columns
        assert "security_archetype" in results["breadth_by_case"].columns
    
    def test_beneficiary_keywords_proxy(self, minimal_workbook: Path) -> None:
        """Test beneficiary keyword proxy scan."""
        from nbs_analyzer.io import load_workbook
        from nbs_analyzer.storyline_b import (
            compute_beneficiary_keywords_proxy,
            load_beneficiary_keywords,
        )
        from nbs_analyzer.utils import load_column_mappings
        
        data, _ = load_workbook(
            minimal_workbook,
            required_sheets=["FACT_NBS"]
        )
        mappings = load_column_mappings()
        keywords_config = load_beneficiary_keywords()
        
        results = compute_beneficiary_keywords_proxy(
            data["FACT_NBS"],
            mappings,
            keywords_config
        )
        
        assert not results["mentions"].empty
        assert not results["summary"].empty
        
        # Check that "women" and "youth" groups are detected
        # (descriptions contain "mujeres" and "jóvenes")
        summary = results["summary"]
        women_row = summary[summary["group"] == "women"]
        if not women_row.empty:
            assert women_row["n_cases_mentioned"].values[0] >= 1
    
    def test_determinism(self, minimal_workbook: Path, output_dir: Path) -> None:
        """Test that same input produces identical output."""
        from nbs_analyzer.storyline_b import run_storyline_b
        
        # Run twice
        out1 = output_dir / "run1"
        out2 = output_dir / "run2"
        
        run_storyline_b(minimal_workbook, out1)
        run_storyline_b(minimal_workbook, out2)
        
        # Compare CSV outputs
        for csv_name in ["security_dimension_rates.csv", "security_breadth_by_case.csv"]:
            df1 = pd.read_csv(out1 / "tables" / csv_name)
            df2 = pd.read_csv(out2 / "tables" / csv_name)
            
            pd.testing.assert_frame_equal(df1, df2)
    
    def test_shortlists_b(self, minimal_workbook: Path) -> None:
        """Test Storyline B shortlist generation."""
        from nbs_analyzer.io import load_workbook
        from nbs_analyzer.storyline_b import (
            compute_security_profile,
            compute_gap_security_links,
            compute_shortlists_b,
            compute_beneficiary_summary,
        )
        from nbs_analyzer.utils import load_column_mappings
        
        data, _ = load_workbook(
            minimal_workbook,
            required_sheets=["FACT_NBS", "TIDY_SECURITY", "TIDY_GOV_GAPS"]
        )
        mappings = load_column_mappings()
        
        # Compute required inputs
        benef = compute_beneficiary_summary(data["FACT_NBS"], mappings)
        sec = compute_security_profile(data["TIDY_SECURITY"], data["FACT_NBS"], mappings)
        gap = compute_gap_security_links(
            data["TIDY_GOV_GAPS"], 
            sec["breadth_by_case"],
            data["TIDY_SECURITY"],
            mappings
        )
        
        # Compute shortlists
        shortlists = compute_shortlists_b(
            data["FACT_NBS"],
            sec["breadth_by_case"],
            gap["ggl_by_case"],
            mappings,
            benef["has_numeric"]
        )
        
        # Should have both shortlist types
        assert "B1_equity_leaders" in shortlists
        assert "B2_needs_enablers" in shortlists
        assert "thresholds" in shortlists


class TestStorylineBHelpers:
    """Tests for Storyline B helper functions."""
    
    def test_coerce_to_binary(self) -> None:
        """Test binary coercion."""
        from nbs_analyzer.storyline_b import coerce_to_binary
        import numpy as np
        
        series = pd.Series([1, 0, "SI", "NO", "yes", "no", None, "invalid"])
        result = coerce_to_binary(series)
        
        assert result.iloc[0] == 1
        assert result.iloc[1] == 0
        assert result.iloc[2] == 1
        assert result.iloc[3] == 0
        assert result.iloc[4] == 1
        assert result.iloc[5] == 0
        assert pd.isna(result.iloc[6])
        assert pd.isna(result.iloc[7])
    
    def test_normalize_text(self) -> None:
        """Test text normalization for keyword matching."""
        from nbs_analyzer.storyline_b import normalize_text
        
        assert normalize_text("Mujeres") == "mujeres"
        assert normalize_text("indígenas") == "indigenas"
        assert normalize_text("Jóvenes") == "jovenes"
        assert normalize_text("") == ""
        assert normalize_text(None) == ""
    
    def test_assign_archetype(self) -> None:
        """Test archetype assignment."""
        from nbs_analyzer.storyline_b import assign_archetype
        import numpy as np
        
        thresholds = {"high": 5.0, "low": 2.0}
        
        assert assign_archetype(6.0, thresholds) == "High"
        assert assign_archetype(5.0, thresholds) == "High"
        assert assign_archetype(3.0, thresholds) == "Medium"
        assert assign_archetype(2.0, thresholds) == "Low"
        assert assign_archetype(1.0, thresholds) == "Low"
        assert assign_archetype(np.nan, thresholds) == "Unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
