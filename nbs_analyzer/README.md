# NbS Analyzer — Storylines A & B

A deterministic, production-ready analyzer for Nature-based Solutions (NbS) portfolios. This tool processes analysis-ready Excel workbooks and generates comprehensive analytics, one-pagers, and reports following two complementary methodologies:

- **Storyline A (Risk-First)**: Starts from threats → NbS → governance gaps
- **Storyline B (Benefits/Equity-First)**: Starts from beneficiaries → co-benefits (security) → threats & gaps
- **Storyline C (Transformation-First)**: Starts from transformative traits → co-benefits → threats & governance

## Features

### Common Features
- **Deterministic**: Same input always produces identical outputs
- **Robust**: Handles missing data, small datasets (even 2 rows), and varied column names
- **Comprehensive QA**: Validates data quality and reports issues

### Storyline A (Risk-First)
- Threat landscape analysis (AC/ANC frequencies, co-occurrence)
- NbS scoring (TCS, SBS, TTS, GGL, ValueScore)
- Shortlist generation (low friction, needs enablers)
- Governance packages mapping
- One-pagers and HTML report

### Storyline C (Transformation-First)
- Transformative traits portfolio analytics (rates, signatures, co-occurrence)
- TTS (Transformative Traits Score) archetypes (High/Med/Low)
- Link transformation to co-benefits (security lift)
- Link transformation to threat coverage (threat lift)
- Governance enabling conditions for leaders
- Transformation-focused shortlists (C1: leaders, C2: needs enablers)
- One-pagers and HTML report

## Installation

```bash
cd nbs_analyzer
pip install -e .
```

## Usage

### CLI — Storyline A

```bash
# Full Storyline A analysis
nbs-analyze storyline-a --input TierraViva_NBS_analysis_ready.xlsx --outdir ./outputs/storyline_a

# With filters (optional)
nbs-analyze storyline-a --input data.xlsx --outdir ./outputs --filters '{"country": "Honduras"}'
```

### CLI — Storyline B

```bash
# Full Storyline B analysis
nbs-analyze storyline-b --input TierraViva_NBS_analysis_ready.xlsx --outdir ./outputs/storyline_b

# With filters (optional)
nbs-analyze storyline-b --input data.xlsx --outdir ./outputs --filters '{"landscape": "Zona Alta"}'
```

### CLI — Storyline C

```bash
# Full Storyline C analysis
nbs-analyze storyline-c --input TierraViva_NBS_analysis_ready.xlsx --outdir ./outputs/storyline_c

# With filters (optional)
nbs-analyze storyline-c --input data.xlsx --outdir ./outputs --filters '{"organization": "CATIE"}'
```

### Python API

```python
# Storyline A
from nbs_analyzer.storyline_a import run_storyline_a

results_a = run_storyline_a(
    input_path="TierraViva_NBS_analysis_ready.xlsx",
    output_dir="./outputs/storyline_a",
    filters={"landscape": "Zona Alta"}
)

# Storyline B
from nbs_analyzer.storyline_b import run_storyline_b

results_b = run_storyline_b(
    input_path="TierraViva_NBS_analysis_ready.xlsx",
    output_dir="./outputs/storyline_b",
    filters={"country": "Honduras"}
)

# Storyline C
from nbs_analyzer.storyline_c import run_storyline_c

results_c = run_storyline_c(
    input_path="TierraViva_NBS_analysis_ready.xlsx",
    output_dir="./outputs/storyline_c"
)
```

## Input Requirements

The input Excel workbook must contain these sheets:

| Sheet | Required (A) | Required (B) | Description |
|-------|:------------:|:------------:|-------------|
| `FACT_NBS` | ✅ | ✅ | Main table with one row per NbS case |
| `TIDY_THREATS` | ✅ | ✅ | Long table of threats per case |
| `TIDY_GOV_GAPS` | ✅ | ✅ | Long table of governance gaps per case |
| `TIDY_SECURITY` | ✅ | ✅ | Long table of security dimensions per case |
| `TIDY_TRAITS` | ✅ | Optional | Long table of transformative traits per case |
| `LOOKUP_THREATS` | Optional | Optional | Threat code definitions |
| `LOOKUP_GOV_GAPS` | Optional | Optional | Gap code definitions |
| `LOOKUP_SECURITY` | Optional | Optional | Security dimension definitions |
| `LOOKUP_TRAITS` | Optional | Optional | Trait definitions |

### Key Columns

- `id_nbs`: Unique identifier for each NbS case (required in all tables)
- `threat_code`: Threat identifier (e.g., AC01, ANC05)
- `gap_code`: Governance gap identifier (e.g., B1.1, B2.3)
- `dimension`: Security dimension identifier
- `value`: Binary flag (0/1) for security and traits

### Beneficiary Columns (Storyline B)

| Field | Column Candidates |
|-------|------------------|
| Direct count | `benef_directos_n`, `beneficiarios_directos_n`, `direct_beneficiaries` |
| Indirect count | `benef_indirectos_n`, `beneficiarios_indirectos_n`, `indirect_beneficiaries` |
| Direct description | `benef_directos_desc`, `beneficiarios_directos_desc` |
| Indirect description | `benef_indirectos_desc`, `beneficiarios_indirectos_desc` |

> **Note**: If numeric beneficiary counts are missing or <30% valid, Storyline B automatically switches to "proxy mode" using keyword scanning of descriptions.

## Outputs

### Storyline A Outputs

| File | Description |
|------|-------------|
| `threat_frequencies_ac.csv` | Climatic threat frequency table |
| `threat_frequencies_anc.csv` | Non-climatic threat frequency table |
| `threat_pairs.csv` | Top 50 threat co-occurrence pairs |
| `nbs_scores.csv` | All computed scores per case |
| `shortlist_low_friction.csv` | High value + low friction cases |
| `shortlist_needs_enablers.csv` | High value + high friction cases |
| `nbs_onepagers.xlsx` | One-pager data for each case |
| `report_storyline_a.html` | Complete HTML report |

### Storyline C Outputs

| File | Description |
|------|-------------|
| `trait_rates.csv` | Transformative trait frequency table |
| `tts_by_case.csv` | TTS score and archetype per case |
| `trait_signatures.csv` | Archetypal trait combinations |
| `security_lift_high_vs_low_tts.csv` | Security gains in transformative cases |
| `threat_lift_high_vs_low_tts.csv` | Threat coverage lift for leaders |
| `gap_lift_high_vs_low_tts.csv` | Governance gap lift for leaders |
| `shortlist_C1_transformative_leaders.csv` | Top-tier transformative cases |
| `nbs_onepagers_storyline_C.xlsx` | One-pager data (Storyline C) |
| `report_storyline_c.html` | Complete HTML report |

## Scoring Methodology

### Storyline A Scores

| Score | Definition |
|-------|------------|
| **TCS** | Threat Coverage Score = unique threats addressed |
| **n_AC** | Count of climatic threats (AC codes) |
| **n_ANC** | Count of non-climatic threats (ANC codes) |
| **SBS** | Security Breadth Score = flagged security dimensions |
| **TTS** | Transformative Traits Score = flagged traits |
| **GGL** | Governance Gap Load = governance barriers count |
| **VS** | Value Score = TCS + SBS + TTS |

### Storyline B Scores

| Score | Definition |
|-------|------------|
| **SBS** | Security Breadth Score = count of flagged security dimensions |
| **GGL** | Governance Gap Load = count of unique gaps |
| **Security Archetype** | High/Medium/Low based on SBS percentiles |
| **Beneficiary Tier** | High/Medium/Low based on total beneficiaries |

## Shortlist Criteria

### Storyline A
- **Low Friction**: VS ≥ 80th percentile AND GGL ≤ 40th percentile
- **Needs Enablers**: VS ≥ 80th percentile AND GGL ≥ 60th percentile

### Storyline C
- **C1 Leaders**: TTS ≥ 80th percentile AND GGL ≤ 60th percentile
- **C2 Needs Enablers**: High TTS AND Top 40% GGL

## Configuration

### Enabling Packages (Storyline A)

Edit `config/enabling_packages.yml` to customize governance gap groupings.

### Beneficiary Keywords (Storyline B)

Edit `config/beneficiary_keywords.yml` to customize priority group detection:

```yaml
beneficiary_groups:
  women:
    keywords:
      - "mujer"
      - "mujeres"
      - "women"
  youth:
    keywords:
      - "joven"
      - "jovenes"
      - "youth"
  # ... more groups
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run Storyline B tests only
pytest tests/test_storyline_b_smoke.py -v
```

## License

MIT License
