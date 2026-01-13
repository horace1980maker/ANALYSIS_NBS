"""
NbS Analyzer — Page 2: Analyzer

Run Storyline A, B, or C analysis on analysis-ready data.
Currently only Storyline A (Risk-First) is implemented.
"""

from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

# Add nbs_analyzer to path
NBS_ANALYZER_PATH = Path(__file__).parent.parent / "nbs_analyzer" / "src"
if str(NBS_ANALYZER_PATH) not in sys.path:
    sys.path.insert(0, str(NBS_ANALYZER_PATH))

try:
    from nbs_analyzer.storyline_a import run_storyline_a
    from nbs_analyzer.storyline_b import run_storyline_b
    from nbs_analyzer.storyline_c import run_storyline_c
    from nbs_analyzer.io import load_workbook
    from nbs_analyzer.schema import REQUIRED_SHEETS, OPTIONAL_SHEETS
    from nbs_analyzer.validate import run_all_validations
    NBS_AVAILABLE = True
except ImportError as e:
    NBS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="NbS Analyzer — Analyzer",
    page_icon="📊",
    layout="wide",
)

# Shared CSS
st.markdown("""
<style>
    :root {
        --primary: #10b981;
        --primary-dark: #059669;
        --secondary: #6366f1;
        --bg-card: rgba(30, 41, 59, 0.7);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: rgba(148, 163, 184, 0.2);
        --glass: rgba(255, 255, 255, 0.05);
        --success: #22c55e;
        --warning: #f59e0b;
        --error: #ef4444;
    }
    
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); }
    #MainMenu, footer { visibility: hidden; }
    
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .hero {
        text-align: center;
        padding: 2rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(99, 102, 241, 0.1));
        border-radius: 20px;
        border: 1px solid var(--border);
    }
    
    .hero h1 {
        font-size: 2rem;
        background: linear-gradient(135deg, #10b981, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero p { color: var(--text-secondary); }
    
    .storyline-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 2px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .storyline-card:hover { border-color: var(--primary); transform: translateY(-4px); }
    .storyline-card.selected { border-color: var(--primary); background: rgba(16, 185, 129, 0.15); }
    .storyline-card.disabled { opacity: 0.5; cursor: not-allowed; }
    .storyline-card.disabled:hover { transform: none; border-color: var(--border); }
    
    .storyline-card h4 { color: var(--text-primary); margin-bottom: 0.5rem; }
    .storyline-card p { color: var(--text-secondary); font-size: 0.85rem; margin: 0; }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .badge-available { background: rgba(34, 197, 94, 0.2); color: var(--success); }
    .badge-coming { background: rgba(245, 158, 11, 0.2); color: var(--warning); }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value { font-size: 1.5rem; font-weight: 700; color: var(--primary); }
    .stat-label { font-size: 0.75rem; color: var(--text-secondary); }
    
    .status-success {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid var(--success);
        color: var(--success);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid var(--error);
        color: var(--error);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        border: 1px solid var(--warning);
        color: var(--warning);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .status-info {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid var(--secondary);
        color: #a5b4fc;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid var(--secondary);
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    
    .info-box h4 { color: #a5b4fc; margin-bottom: 0.5rem; }
    .info-box p, .info-box li { color: var(--text-secondary); font-size: 0.9rem; }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--secondary), #4f46e5);
        color: white;
        border: none;
        border-radius: 12px;
        width: 100%;
    }
    
    .results-section {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .results-section h4 { color: var(--text-primary); margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions
# =============================================================================

def create_results_zip(output_dir: Path) -> bytes:
    """Create a ZIP file from all outputs."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ["tables", "figures", "reports"]:
            folder_path = output_dir / folder
            if folder_path.exists():
                for file_path in folder_path.iterdir():
                    if file_path.is_file():
                        arcname = f"{folder}/{file_path.name}"
                        zf.write(file_path, arcname)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()



ABBREVIATIONS = {
    "TCS": "TCS (Threat Coverage Score)",
    "SBS": "SBS (Security Breadth Score)",
    "TTS": "TTS (Transformative Traits Score)",
    "GGL": "GGL (Governance Gap Load)",
    "VS": "VS (ValueScore)",
    "n_AC": "n_AC (# Climatic Threats)",
    "n_ANC": "n_ANC (# Non-Climatic Threats)",
    "id_nbs": "ID (NbS Identifier)",
    "n_cases": "# Cases",
    "threat_code": "Code",
    "threat_label": "Threat Name",
    "gap_code": "Gap Code",
    "gap_name": "Gap Name",
    "dimension": "Security Dimension",
    "trait": "Trait",
    "count": "# Cases",
    "percentage": "% Cases",
}


def rename_with_abbreviations(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to include full names for better readability."""
    return df.rename(columns=ABBREVIATIONS)


def render_stats(stats: Dict[str, Any]):
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-value">{stats.get('n_cases', 0)}</div><div class="stat-label">NbS Cases</div></div>
        <div class="stat-card"><div class="stat-value">{stats.get('n_ac', 0)}</div><div class="stat-label">Climatic Threats (AC)</div></div>
        <div class="stat-card"><div class="stat-value">{stats.get('n_anc', 0)}</div><div class="stat-label">Non-Climatic Threats (ANC)</div></div>
        <div class="stat-card"><div class="stat-value">{stats.get('n_low_friction', 0)}</div><div class="stat-label">Low Friction</div></div>
        <div class="stat-card"><div class="stat-value">{stats.get('n_enablers', 0)}</div><div class="stat-label">Needs Enablers</div></div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main UI
# =============================================================================

st.markdown("""
<div class="hero">
    <h1>📊 NbS Analyzer</h1>
    <p>Run comprehensive analysis on your analysis-ready data</p>
</div>
""", unsafe_allow_html=True)

# Check if nbs_analyzer is available
if not NBS_AVAILABLE:
    st.markdown(f"""
    <div class="status-warning">
        <strong>⚠️ NbS Analyzer module not loaded</strong><br>
        Please install the analyzer package first:<br>
        <code>cd nbs_analyzer && pip install -e .</code><br>
        Error: {IMPORT_ERROR}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Initialize session state
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "output_zip" not in st.session_state:
    st.session_state.output_zip = None
if "html_report" not in st.session_state:
    st.session_state.html_report = None
if "selected_storyline" not in st.session_state:
    st.session_state.selected_storyline = "A"
if "show_report" not in st.session_state:
    st.session_state.show_report = False

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Select Storyline")
    
    # Storyline selection
    storyline = st.radio(
        "Choose analysis approach:",
        ["A — Risk-First", "B — Opportunity-First", "C — Transformation-First"],
        index=0,
        help="Select the analysis storyline to run"
    )
    
    st.session_state.selected_storyline = storyline[0]
    
    if storyline.startswith("A"):
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Storyline A: Risk-First</h4>
            <p>Analyzes threats, computes NbS scores, and generates shortlists based on value and friction.</p>
            <ul style="padding-left: 1.5rem; margin-top: 0.5rem;">
                <li>Threat landscape analysis</li>
                <li>TCS, SBS, TTS, GGL, ValueScore</li>
                <li>Value vs. friction shortlists</li>
                <li>Governance enabling packages</li>
                <li>One-pagers per NbS</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif storyline.startswith("B"):
        st.markdown("""
        <div class="info-box" style="border-left-color: #9c27b0;">
            <h4 style="color: #ce93d8;">🌱 Storyline B: Benefits-First</h4>
            <p>Analyzes beneficiaries and co-benefits (security dimensions), linking to threats and governance gaps.</p>
            <ul style="padding-left: 1.5rem; margin-top: 0.5rem;">
                <li>Beneficiary analytics (direct/indirect)</li>
                <li>Security dimension profile (SBS)</li>
                <li>Priority group keyword detection</li>
                <li>Equity-focused shortlists</li>
                <li>One-pagers per NbS</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif storyline.startswith("C"):
        st.markdown("""
        <div class="info-box" style="border-left-color: #f59e0b;">
            <h4 style="color: #fbbf24;">✨ Storyline C: Transformation-First</h4>
            <p>Analyzes transformative traits, signatures, and computes "lift" for security and threats.</p>
            <ul style="padding-left: 1.5rem; margin-top: 0.5rem;">
                <li>Trait portfolio analytics (rates, signatures)</li>
                <li>TTS (Transformative Traits Score) archetypes</li>
                <li>Security & Threat Lift analysis</li>
                <li>Strategic shortlists (C1 Leaders, C2 Enablers)</li>
                <li>One-pagers per NbS</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # File upload
    st.markdown("### 📤 Upload Analysis-Ready File")
    st.markdown("""
    <p style="color: var(--text-secondary); font-size: 0.9rem;">
        Upload a file converted using the Converter page, or any Excel with the required sheets:
        FACT_NBS, TIDY_THREATS, TIDY_GOV_GAPS, TIDY_SECURITY, TIDY_TRAITS
    </p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload analysis-ready Excel",
        type=["xlsx"],
        help="Must contain required sheets from conversion"
    )
    
    if uploaded_file and st.session_state.selected_storyline in ["A", "B", "C"]:
        # Validate file
        try:
            xl = pd.ExcelFile(uploaded_file)
            sheets = xl.sheet_names
            
            # Storyline B has slightly different requirements (TIDY_TRAITS is optional)
            if st.session_state.selected_storyline == "B":
                required_sheets_b = ["FACT_NBS", "TIDY_THREATS", "TIDY_GOV_GAPS", "TIDY_SECURITY"]
                required_found = [s for s in required_sheets_b if s in sheets]
                required_missing = [s for s in required_sheets_b if s not in sheets]
            else:
                required_found = [s for s in REQUIRED_SHEETS if s in sheets]
                required_missing = [s for s in REQUIRED_SHEETS if s not in sheets]
            optional_found = [s for s in OPTIONAL_SHEETS if s in sheets]
            
            if required_missing:
                st.markdown(f"""
                <div class="status-error">
                    <strong>❌ Missing Required Sheets</strong><br>
                    {', '.join(required_missing)}<br>
                    <em>Please use the Converter page first to prepare your data.</em>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-success">
                    <strong>✓ File Valid</strong> — All required sheets found ({len(required_found)} sheets)
                </div>
                """, unsafe_allow_html=True)
                
                if optional_found:
                    st.markdown(f"<p style='color: #94a3b8; font-size: 0.85rem;'>Optional sheets: {', '.join(optional_found)}</p>", unsafe_allow_html=True)
                
                # Run analysis button
                st.markdown("### ⚡ Run Analysis")
                
                storyline_label = st.session_state.selected_storyline
                button_style = "primary"
                
                if st.button(f"🚀 Run Storyline {storyline_label} Analysis", use_container_width=True, type=button_style):
                    with st.spinner(f"Running Storyline {storyline_label} analysis... This may take a moment."):
                        try:
                            # Save uploaded file to temp
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_input:
                                uploaded_file.seek(0)
                                tmp_input.write(uploaded_file.read())
                                tmp_input_path = Path(tmp_input.name)
                            
                            # Create temp output dir
                            with tempfile.TemporaryDirectory() as tmp_output:
                                output_path = Path(tmp_output)
                                
                                if st.session_state.selected_storyline == "A":
                                    # Run Storyline A
                                    results = run_storyline_a(
                                        input_path=tmp_input_path,
                                        output_dir=output_path,
                                    )
                                    
                                    # Store results in session
                                    st.session_state.analysis_results = {
                                        "storyline": "A",
                                        "n_cases": results.n_cases,
                                        "n_ac": len(results.threat_freq_ac),
                                        "n_anc": len(results.threat_freq_anc),
                                        "n_low_friction": len(results.shortlist_low_friction),
                                        "n_enablers": len(results.shortlist_needs_enablers),
                                        "warnings": results.warnings,
                                        "scores": results.nbs_scores,
                                        "shortlist_low": results.shortlist_low_friction,
                                        "shortlist_high": results.shortlist_needs_enablers,
                                        "threats_ac": results.threat_freq_ac,
                                        "threats_anc": results.threat_freq_anc,
                                        "gaps": results.top_gov_gaps_overall,
                                    }
                                    
                                    html_path = output_path / "reports" / "report_storyline_a.html"
                                elif st.session_state.selected_storyline == "B":
                                    # Run Storyline B
                                    results = run_storyline_b(
                                        input_path=tmp_input_path,
                                        output_dir=output_path,
                                    )
                                    
                                    # Store results in session
                                    st.session_state.analysis_results = {
                                        "storyline": "B",
                                        "n_cases": results.n_cases,
                                        "has_numeric": results.has_numeric_beneficiaries,
                                        "n_b1": len(results.shortlist_B1_equity_leaders),
                                        "n_b2": len(results.shortlist_B2_needs_enablers),
                                        "n_dimensions": len(results.security_dimension_rates),
                                        "warnings": results.warnings,
                                        "security_rates": results.security_dimension_rates,
                                        "security_breadth": results.security_breadth_by_case,
                                        "shortlist_b1": results.shortlist_B1_equity_leaders,
                                        "shortlist_b2": results.shortlist_B2_needs_enablers,
                                        "benef_summary": results.beneficiary_summary_overall,
                                        "keyword_summary": results.beneficiary_group_summary,
                                    }
                                    
                                    html_path = output_path / "reports" / "report_storyline_b.html"
                                elif st.session_state.selected_storyline == "C":
                                    # Run Storyline C
                                    results = run_storyline_c(
                                        input_path=tmp_input_path,
                                        output_dir=output_path,
                                    )
                                    
                                    # Store results in session
                                    st.session_state.analysis_results = {
                                        "storyline": "C",
                                        "n_cases": results.n_cases,
                                        "trait_rates": results.trait_rates,
                                        "tts_by_case": results.tts_by_case,
                                        "security_lift": results.security_lift,
                                        "threat_lift": results.threat_lift,
                                        "gap_lift": results.gap_lift,
                                        "shortlist_c1": results.shortlist_c1,
                                        "shortlist_c2": results.shortlist_c2,
                                        "warnings": results.warnings,
                                    }
                                    
                                    html_path = output_path / "reports" / "report_storyline_c.html"
                                
                                # Create zip of outputs
                                st.session_state.output_zip = create_results_zip(output_path)
                                
                                # Store HTML report content
                                if html_path.exists():
                                    st.session_state.html_report = html_path.read_text(encoding="utf-8")
                            
                            # Clean up input
                            tmp_input_path.unlink()
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.markdown(f"""
                            <div class="status-error">
                                <strong>❌ Analysis Error</strong><br>
                                {str(e)}
                            </div>
                            """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    
    st.markdown('</div>', unsafe_allow_html=True)

# Results section
if st.session_state.analysis_results:
    st.markdown("---")
    
    results = st.session_state.analysis_results
    storyline = results.get("storyline", "A")
    
    if storyline == "A":
        st.markdown("## 📈 Storyline A Results (Risk-First)")
        render_stats(results)
    elif storyline == "B":
        st.markdown("## 📈 Storyline B Results (Benefits-First)")
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value">{results.get('n_cases', 0)}</div><div class="stat-label">NbS Cases</div></div>
            <div class="stat-card"><div class="stat-value">{results.get('n_dimensions', 0)}</div><div class="stat-label">Security Dimensions</div></div>
            <div class="stat-card"><div class="stat-value">{results.get('n_b1', 0)}</div><div class="stat-label">Equity Leaders</div></div>
            <div class="stat-card"><div class="stat-value">{results.get('n_b2', 0)}</div><div class="stat-label">Needs Enablers</div></div>
            <div class="stat-card"><div class="stat-value">{'✓' if results.get('has_numeric') else 'Proxy'}</div><div class="stat-label">Beneficiary Data</div></div>
        </div>
        """, unsafe_allow_html=True)
    elif storyline == "C":
        st.markdown("## 📈 Storyline C Results (Transformation-First)")
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value">{results.get('n_cases', 0)}</div><div class="stat-label">NbS Cases</div></div>
            <div class="stat-card"><div class="stat-value">{len(results.get('trait_rates', []))}</div><div class="stat-label">Active Traits</div></div>
            <div class="stat-card"><div class="stat-value">{len(results.get('shortlist_c1', []))}</div><div class="stat-label">Transformative Leaders</div></div>
            <div class="stat-card"><div class="stat-value">{len(results.get('shortlist_c2', []))}</div><div class="stat-label">Needs Enablers</div></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Warnings
    if results.get("warnings"):
        for warning in results["warnings"]:
            st.markdown(f"""
            <div class="status-warning">⚠️ {warning}</div>
            """, unsafe_allow_html=True)
    
    # Download and View buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"NBS_Storyline{storyline}_Results_{timestamp}.zip"
        st.download_button(
            "⬇️ Download All Results (ZIP)",
            st.session_state.output_zip,
            zip_name,
            "application/zip",
            use_container_width=True
        )
    with col2:
        if st.session_state.html_report:
            if st.button("📄 View HTML Report", use_container_width=True):
                st.session_state.show_report = not st.session_state.show_report
    with col3:
        if st.session_state.html_report:
            html_filename = f"report_storyline_{storyline.lower()}.html"
            st.download_button(
                "📥 Download Report (HTML)",
                st.session_state.html_report,
                html_filename,
                "text/html",
                use_container_width=True
            )
    
    # Display HTML report inline if toggled
    if st.session_state.show_report and st.session_state.html_report:
        st.markdown("---")
        st.markdown("### 📄 HTML Report Preview")
        
        # Add "Open in New Tab" button using a robust components.html approach
        import base64
        html_b64 = base64.b64encode(st.session_state.html_report.encode("utf-8")).decode("utf-8")
        
        # Determine colors based on storyline
        if storyline == "A":
            primary_color, dark_color = "#10b981", "#059669"
            shadow_rgba = "rgba(16, 185, 129, 0.3)"
        elif storyline == "B":
            primary_color, dark_color = "#9c27b0", "#7b1fa2"
            shadow_rgba = "rgba(156, 39, 176, 0.3)"
        else:  # Storyline C
            primary_color, dark_color = "#f59e0b", "#d97706"
            shadow_rgba = "rgba(245, 158, 11, 0.3)"

        import streamlit.components.v1 as components
        
        button_html = f'''
        <script>
        function openReportInNewTab() {{
            try {{
                var b64 = "{html_b64}";
                var binaryString = window.atob(b64);
                var len = binaryString.length;
                var bytes = new Uint8Array(len);
                for (var i = 0; i < len; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}
                var blob = new Blob([bytes], {{type: "text/html"}});
                var url = URL.createObjectURL(blob);
                window.open(url, "_blank");
            }} catch (e) {{
                console.error("Error opening report:", e);
                alert("Failed to open report in new tab. Error: " + e.message);
            }}
        }}
        </script>
        <div style="display: flex; justify-content: flex-start; margin-bottom: 1rem;">
            <button onclick="openReportInNewTab()" style="
                background: linear-gradient(135deg, {primary_color}, {dark_color});
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.75rem 2rem;
                font-weight: 600;
                font-size: 1rem;
                cursor: pointer;
                box-shadow: 0 4px 15px {shadow_rgba};
                font-family: 'Source Sans Pro', sans-serif;
                transition: transform 0.2s ease;
            " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                🔗 Open Report in New Tab
            </button>
        </div>
        '''
        components.html(button_html, height=70)
        components.html(st.session_state.html_report, height=800, scrolling=True)
    
    # Results tabs - different for A vs B
    if storyline == "A":
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Scores", "✅ Shortlists", "🌡️ Threats", "🏛️ Governance"])
        
        with tab1:
            st.markdown("### NbS Scores")
            if not results["scores"].empty:
                st.dataframe(rename_with_abbreviations(results["scores"]), use_container_width=True, height=400)
            else:
                st.info("No scores computed.")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Low Friction Shortlist")
                st.markdown("<p style='color: #94a3b8;'>High value + low governance barriers</p>", unsafe_allow_html=True)
                if not results["shortlist_low"].empty:
                    st.dataframe(rename_with_abbreviations(results["shortlist_low"]), use_container_width=True)
                else:
                    st.info("No cases meet low friction criteria.")
            
            with col2:
                st.markdown("### ⚙️ Needs Enablers Shortlist")
                st.markdown("<p style='color: #94a3b8;'>High value + high governance barriers</p>", unsafe_allow_html=True)
                if not results["shortlist_high"].empty:
                    st.dataframe(rename_with_abbreviations(results["shortlist_high"]), use_container_width=True)
                else:
                    st.info("No cases meet enablers criteria.")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌡️ Climatic Threats (AC)")
                if not results["threats_ac"].empty:
                    st.dataframe(rename_with_abbreviations(results["threats_ac"].head(15)), use_container_width=True)
                else:
                    st.info("No AC threats found.")
            
            with col2:
                st.markdown("### 🔥 Non-Climatic Threats (ANC)")
                if not results["threats_anc"].empty:
                    st.dataframe(rename_with_abbreviations(results["threats_anc"].head(15)), use_container_width=True)
                else:
                    st.info("No ANC threats found.")
        
        with tab4:
            st.markdown("### 🏛️ Top Governance Gaps")
            if not results["gaps"].empty:
                st.dataframe(rename_with_abbreviations(results["gaps"].head(20)), use_container_width=True)
            else:
                st.info("No governance gaps found.")
    elif storyline == "B":
        tab1, tab2, tab3, tab4 = st.tabs(["🔒 Security", "✅ Shortlists", "👥 Beneficiaries", "🏷️ Keywords"])
        
        with tab1:
            st.markdown("### Security Dimension Rates")
            if not results["security_rates"].empty:
                st.dataframe(rename_with_abbreviations(results["security_rates"]), use_container_width=True, height=400)
            else:
                st.info("No security dimension data.")
            
            st.markdown("### Security Breadth by Case")
            if not results["security_breadth"].empty:
                st.dataframe(rename_with_abbreviations(results["security_breadth"].head(20)), use_container_width=True)
            else:
                st.info("No security breadth data.")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ B1: Equity Leaders")
                st.markdown("<p style='color: #94a3b8;'>High SBS + beneficiary evidence + low GGL</p>", unsafe_allow_html=True)
                if not results["shortlist_b1"].empty:
                    st.dataframe(rename_with_abbreviations(results["shortlist_b1"]), use_container_width=True)
                else:
                    st.info("No cases meet equity leaders criteria.")
            
            with col2:
                st.markdown("### ⚙️ B2: Needs Enablers")
                st.markdown("<p style='color: #94a3b8;'>High SBS + high governance barriers</p>", unsafe_allow_html=True)
                if not results["shortlist_b2"].empty:
                    st.dataframe(rename_with_abbreviations(results["shortlist_b2"]), use_container_width=True)
                else:
                    st.info("No cases meet enablers criteria.")
        
        with tab3:
            st.markdown("### 👥 Beneficiary Summary")
            if not results["benef_summary"].empty:
                st.dataframe(rename_with_abbreviations(results["benef_summary"]), use_container_width=True)
            else:
                st.info("No beneficiary summary data.")
        
        with tab4:
            st.markdown("### 🏷️ Priority Group Keyword Mentions")
            st.markdown("<p style='color: #94a3b8;'>Groups detected from beneficiary descriptions</p>", unsafe_allow_html=True)
            if not results["keyword_summary"].empty:
                st.dataframe(rename_with_abbreviations(results["keyword_summary"]), use_container_width=True)
            else:
                st.info("No keyword mentions detected.")
    
    elif storyline == "C":
        tab1, tab2, tab3, tab4 = st.tabs(["✨ Traits", "📊 TTS Scores", "🚀 Lift Analysis", "✅ Shortlists"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Transformative Trait Rates")
                if not results["trait_rates"].empty:
                    st.dataframe(rename_with_abbreviations(results["trait_rates"]), use_container_width=True)
            with col2:
                st.markdown("### Trait Portfolio Description")
                st.info("See the full HTML report for detailed trait co-occurrence analysis and signatures.")
        
        with tab2:
            st.markdown("### Case Performance (TTS & SBS)")
            if not results["tts_by_case"].empty:
                st.dataframe(rename_with_abbreviations(results["tts_by_case"]), use_container_width=True)
            else:
                st.info("No scores available.")
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Security Dimensions Lift")
                if not results["security_lift"].empty:
                    st.dataframe(rename_with_abbreviations(results["security_lift"]), use_container_width=True)
            with col2:
                st.markdown("### Threat Coverage Lift")
                if not results["threat_lift"].empty:
                    st.dataframe(rename_with_abbreviations(results["threat_lift"]), use_container_width=True)
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ C1: Transformative Leaders")
                if not results["shortlist_c1"].empty:
                    st.dataframe(rename_with_abbreviations(results["shortlist_c1"]), use_container_width=True)
            with col2:
                st.markdown("### ⚙️ C2: Needs Enablers")
                if not results["shortlist_c2"].empty:
                    st.dataframe(rename_with_abbreviations(results["shortlist_c2"]), use_container_width=True)
    
    # Run another analysis
    st.markdown("---")
    if st.button("🔄 Run Another Analysis"):
        st.session_state.analysis_results = None
        st.session_state.output_zip = None
        st.session_state.html_report = None
        st.session_state.show_report = False
        st.rerun()

