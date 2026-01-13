"""
Smoke tests for Storyline C (Transformation-First) analysis.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


def create_minimal_workbook_c(path: Path) -> None:
    """Create a minimal test workbook for Storyline C with 5 cases."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # FACT_NBS
        fact = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS002", "NBS003", "NBS004", "NBS005"],
            "country": ["Honduras", "Honduras", "Costa Rica", "Costa Rica", "Panama"],
            "landscape": ["L1", "L1", "L2", "L2", "L3"],
            "organization": ["OrgA", "OrgB", "OrgA", "OrgC", "OrgB"],
        })
        fact.to_excel(writer, sheet_name="FACT_NBS", index=False)
        
        # TIDY_THREATS
        threats = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS001", "NBS002", "NBS003", "NBS004", "NBS005"],
            "threat_code": ["AC01", "ANC05", "AC01", "AC02", "ANC05", "AC03"],
        })
        threats.to_excel(writer, sheet_name="TIDY_THREATS", index=False)
        
        # TIDY_GOV_GAPS
        gaps = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS002", "NBS002", "NBS003", "NBS004", "NBS005"],
            "gap_code": ["B1.1", "B1.1", "B2.1", "B1.1", "B2.1", "B3.1"],
        })
        gaps.to_excel(writer, sheet_name="TIDY_GOV_GAPS", index=False)
        
        # TIDY_SECURITY
        security = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS001", "NBS002", "NBS003", "NBS004", "NBS005"],
            "dimension": ["water", "food", "water", "climate", "food", "water"],
            "value": [1, 1, 1, 1, 1, 1],
        })
        security.to_excel(writer, sheet_name="TIDY_SECURITY", index=False)
        
        # TIDY_TRAITS
        traits = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS001", "NBS001", "NBS002", "NBS003", "NBS004", "NBS005"],
            "trait": ["systemic", "anticipatory", "flexible", "systemic", "inclusive", "flexible", "systemic"],
            "value": [1, 1, 1, 1, 1, 1, 0],
        })
        # Note: NBS005 has systemic=0, others have varied traits
        traits.to_excel(writer, sheet_name="TIDY_TRAITS", index=False)


class TestStorylineCMinimal:
    """Tests with a minimal 5-case dataset."""
    
    @pytest.fixture
    def workbook(self, tmp_path: Path) -> Path:
        """Create minimal workbook fixture."""
        wb_path = tmp_path / "minimal_c.xlsx"
        create_minimal_workbook_c(wb_path)
        return wb_path
    
    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create output directory fixture."""
        out = tmp_path / "outputs"
        out.mkdir()
        return out
    
    def test_full_pipeline(self, workbook: Path, output_dir: Path) -> None:
        """Test full Storyline C pipeline."""
        from nbs_analyzer.storyline_c import run_storyline_c
        
        results = run_storyline_c(
            input_path=workbook,
            output_dir=output_dir,
        )
        
        assert results.n_cases == 5
        assert not results.trait_rates.empty
        assert not results.tts_by_case.empty
        assert not results.security_lift.empty
        assert not results.threat_lift.empty
        assert not results.gap_lift.empty
        assert not results.shortlist_c1.empty
        # shortlist_c2 might be empty depending on GGL thresholds for 5 cases, but we check if it runs
        
        # Check output files exist
        assert (output_dir / "tables" / "trait_rates.csv").exists()
        assert (output_dir / "tables" / "tts_by_case.csv").exists()
        assert (output_dir / "tables" / "security_lift_high_vs_low_tts.csv").exists()
        assert (output_dir / "tables" / "nbs_onepagers_storyline_C.xlsx").exists()
        assert (output_dir / "reports" / "report_storyline_c.html").exists()
        
        # Check figures
        assert (output_dir / "figures" / "trait_rates.png").exists()
        assert (output_dir / "figures" / "tts_distribution.png").exists()
        assert (output_dir / "figures" / "security_lift.png").exists()

    def test_determinism(self, workbook: Path, output_dir: Path) -> None:
        """Test that same input produces identical output."""
        from nbs_analyzer.storyline_c import run_storyline_c
        
        # Run twice
        out1 = output_dir / "run1"
        out2 = output_dir / "run2"
        
        run_storyline_c(workbook, out1)
        run_storyline_c(workbook, out2)
        
        # Compare CSV outputs
        for csv_name in ["trait_rates.csv", "tts_by_case.csv", "security_lift_high_vs_low_tts.csv"]:
            df1 = pd.read_csv(out1 / "tables" / csv_name)
            df2 = pd.read_csv(out2 / "tables" / csv_name)
            
            pd.testing.assert_frame_equal(df1, df2)

    def test_with_filters(self, workbook: Path, output_dir: Path) -> None:
        """Test pipeline with filters."""
        from nbs_analyzer.storyline_c import run_storyline_c
        
        results = run_storyline_c(
            input_path=workbook,
            output_dir=output_dir,
            filters={"country": "Honduras"}
        )
        
        assert results.n_cases == 2 # Only NBS001 and NBS002
        assert all(results.tts_by_case["id_nbs"].isin(["NBS001", "NBS002"]))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
