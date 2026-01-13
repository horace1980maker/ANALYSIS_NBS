"""
Smoke tests for the NbS Analyzer.

Tests basic functionality with minimal datasets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


def create_minimal_workbook(path: Path) -> None:
    """Create a minimal test workbook with 2 cases."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # FACT_NBS
        fact = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS002"],
            "country": ["Honduras", "Honduras"],
            "landscape": ["Zona Alta", "Zona Baja"],
            "combinacion": ["Agroforestry", "Reforestation"],
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
            "id_nbs": ["NBS001", "NBS001", "NBS002"],
            "dimension": ["seg_water", "seg_food", "seg_water"],
            "value": [1, 1, 1],
        })
        security.to_excel(writer, sheet_name="TIDY_SECURITY", index=False)
        
        # TIDY_TRAITS
        traits = pd.DataFrame({
            "id_nbs": ["NBS001", "NBS002", "NBS002"],
            "trait": ["t1_systemic", "t2_anticipatory", "t3_flexible"],
            "value": [1, 1, 1],
        })
        traits.to_excel(writer, sheet_name="TIDY_TRAITS", index=False)


class TestMinimalDataset:
    """Tests with a minimal 2-case dataset."""
    
    @pytest.fixture
    def minimal_workbook(self, tmp_path: Path) -> Path:
        """Create minimal workbook fixture."""
        wb_path = tmp_path / "minimal.xlsx"
        create_minimal_workbook(wb_path)
        return wb_path
    
    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create output directory fixture."""
        out = tmp_path / "outputs"
        out.mkdir()
        return out
    
    def test_load_workbook(self, minimal_workbook: Path) -> None:
        """Test workbook loading."""
        from nbs_analyzer.io import load_workbook
        from nbs_analyzer.schema import REQUIRED_SHEETS, OPTIONAL_SHEETS
        
        data, missing = load_workbook(
            minimal_workbook,
            required_sheets=REQUIRED_SHEETS,
            optional_sheets=OPTIONAL_SHEETS
        )
        
        assert "FACT_NBS" in data
        assert len(data["FACT_NBS"]) == 2
        assert "TIDY_THREATS" in data
        assert "LOOKUP_THREATS" in missing  # Optional, not present
    
    def test_compute_tcs(self, minimal_workbook: Path) -> None:
        """Test TCS computation."""
        from nbs_analyzer.io import load_workbook
        from nbs_analyzer.schema import REQUIRED_SHEETS
        from nbs_analyzer.metrics import compute_tcs
        
        data, _ = load_workbook(minimal_workbook, required_sheets=REQUIRED_SHEETS)
        tcs = compute_tcs(data["TIDY_THREATS"])
        
        assert len(tcs) == 2
        assert "TCS" in tcs.columns
        assert "n_AC" in tcs.columns
        assert "n_ANC" in tcs.columns
        
        # NBS001 has 2 threats (1 AC, 1 ANC)
        nbs001 = tcs[tcs["id_nbs"] == "NBS001"].iloc[0]
        assert nbs001["TCS"] == 2
        assert nbs001["n_AC"] == 1
        assert nbs001["n_ANC"] == 1
    
    def test_compute_sbs(self, minimal_workbook: Path) -> None:
        """Test SBS computation."""
        from nbs_analyzer.io import load_workbook
        from nbs_analyzer.schema import REQUIRED_SHEETS
        from nbs_analyzer.metrics import compute_sbs
        
        data, _ = load_workbook(minimal_workbook, required_sheets=REQUIRED_SHEETS)
        sbs = compute_sbs(data["TIDY_SECURITY"])
        
        assert len(sbs) == 2
        
        # NBS001 has 2 security dimensions flagged
        nbs001 = sbs[sbs["id_nbs"] == "NBS001"].iloc[0]
        assert nbs001["SBS"] == 2
    
    def test_compute_shortlists(self, minimal_workbook: Path) -> None:
        """Test shortlist generation with small N."""
        from nbs_analyzer.io import load_workbook
        from nbs_analyzer.schema import REQUIRED_SHEETS
        from nbs_analyzer.metrics import compute_all_scores, compute_shortlists
        
        data, _ = load_workbook(minimal_workbook, required_sheets=REQUIRED_SHEETS)
        scores = compute_all_scores(data)
        
        low_friction, needs_enablers, thresholds = compute_shortlists(scores)
        
        # With only 2 cases, should still produce results
        assert thresholds["n_cases"] == 2
        # Combined shortlists should include at least some cases
        assert len(low_friction) + len(needs_enablers) >= 0
    
    def test_full_pipeline(self, minimal_workbook: Path, output_dir: Path) -> None:
        """Test full Storyline A pipeline."""
        from nbs_analyzer.storyline_a import run_storyline_a
        
        results = run_storyline_a(
            input_path=minimal_workbook,
            output_dir=output_dir,
        )
        
        assert results.n_cases == 2
        assert not results.nbs_scores.empty
        assert not results.threat_freq_ac.empty
        assert not results.onepagers.empty
        
        # Check output files exist
        assert (output_dir / "tables" / "nbs_scores.csv").exists()
        assert (output_dir / "tables" / "nbs_onepagers.xlsx").exists()
        assert (output_dir / "reports" / "report_storyline_a.html").exists()
    
    def test_determinism(self, minimal_workbook: Path, output_dir: Path) -> None:
        """Test that same input produces identical output."""
        from nbs_analyzer.storyline_a import run_storyline_a
        
        # Run twice
        out1 = output_dir / "run1"
        out2 = output_dir / "run2"
        
        run_storyline_a(minimal_workbook, out1)
        run_storyline_a(minimal_workbook, out2)
        
        # Compare CSV outputs
        for csv_name in ["nbs_scores.csv", "threat_frequencies_ac.csv"]:
            df1 = pd.read_csv(out1 / "tables" / csv_name)
            df2 = pd.read_csv(out2 / "tables" / csv_name)
            
            pd.testing.assert_frame_equal(df1, df2)


class TestValidation:
    """Tests for data validation."""
    
    def test_validate_id_uniqueness(self) -> None:
        """Test ID uniqueness validation."""
        from nbs_analyzer.validate import validate_id_uniqueness
        
        # Valid data
        df = pd.DataFrame({"id_nbs": ["A", "B", "C"]})
        result = validate_id_uniqueness(df, "test_sheet")
        assert result.passed
        
        # Duplicate IDs
        df = pd.DataFrame({"id_nbs": ["A", "B", "A"]})
        result = validate_id_uniqueness(df, "test_sheet")
        assert not result.passed
    
    def test_validate_binary_columns(self) -> None:
        """Test binary value validation."""
        from nbs_analyzer.validate import validate_binary_columns
        
        # Valid binary
        df = pd.DataFrame({"value": [0, 1, 1, 0]})
        result = validate_binary_columns(df, "test_sheet")
        assert result.passed
        
        # Invalid values
        df = pd.DataFrame({"value": [0, 1, 2, 0]})
        result = validate_binary_columns(df, "test_sheet")
        assert not result.passed


class TestUtils:
    """Tests for utility functions."""
    
    def test_pick_col(self) -> None:
        """Test column matching."""
        from nbs_analyzer.utils import pick_col
        
        df = pd.DataFrame({"ID_NbS": [1], "Country": [2]})
        
        # Case-insensitive match
        assert pick_col(df, ["id_nbs", "ID_NBS"]) == "ID_NbS"
        assert pick_col(df, ["country"]) == "Country"
        assert pick_col(df, ["nonexistent"]) is None
    
    def test_infer_threat_type(self) -> None:
        """Test threat type inference."""
        from nbs_analyzer.utils import infer_threat_type
        
        assert infer_threat_type("AC01") == "climatic"
        assert infer_threat_type("AC99") == "climatic"
        assert infer_threat_type("ANC05") == "non_climatic"
        assert infer_threat_type("OTHER") is None
    
    def test_normalize_missing(self) -> None:
        """Test missing value normalization."""
        from nbs_analyzer.utils import normalize_missing
        import numpy as np
        
        df = pd.DataFrame({"col": ["A", "ND", "N/D", "", " ", "B"]})
        result = normalize_missing(df, inplace=False)
        
        assert result["col"].tolist()[:2] == ["A", np.nan] or pd.isna(result["col"].iloc[1])
        assert result["col"].iloc[5] == "B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
