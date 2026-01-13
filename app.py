"""
NbS Analyzer — Multi-Page Streamlit Application

Main entry point that sets up the navigation and shared styling.

Usage:
    streamlit run app.py
"""

import streamlit as st

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="NbS Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Shared CSS — Premium Dark Glassmorphism Theme
# =============================================================================

st.markdown("""
<style>
    /* Root variables */
    :root {
        --primary: #10b981;
        --primary-dark: #059669;
        --secondary: #6366f1;
        --accent: #f59e0b;
        --bg-dark: #0f172a;
        --bg-card: rgba(30, 41, 59, 0.7);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: rgba(148, 163, 184, 0.2);
        --glass: rgba(255, 255, 255, 0.05);
        --success: #22c55e;
        --warning: #f59e0b;
        --error: #ef4444;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-attachment: fixed;
    }
    
    /* Hide default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* Glass card effect */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Navigation cards */
    .nav-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-card:hover {
        border-color: var(--primary);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(16, 185, 129, 0.2);
    }
    
    .nav-card h3 {
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .nav-card p {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    .nav-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Hero section */
    .hero {
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(99, 102, 241, 0.1));
        border-radius: 20px;
        border: 1px solid var(--border);
    }
    
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #10b981, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero p {
        color: var(--text-secondary);
        font-size: 1.1rem;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    
    /* Status messages */
    .status-success {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid var(--success);
        color: var(--success);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid var(--error);
        color: var(--error);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .status-info {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid var(--secondary);
        color: #a5b4fc;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        border: 1px solid var(--warning);
        color: var(--warning);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--secondary), #4f46e5);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* Radio buttons as cards */
    .stRadio > div {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .stRadio > div > label {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stRadio > div > label:hover {
        border-color: var(--primary);
    }
    
    /* Info box */
    .info-box {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid var(--secondary);
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #a5b4fc;
        margin-bottom: 0.5rem;
    }
    
    .info-box p, .info-box li {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border);
    }
    
    /* Storyline cards */
    .storyline-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 2px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .storyline-card.available {
        cursor: pointer;
    }
    
    .storyline-card.available:hover {
        border-color: var(--primary);
        transform: translateX(8px);
    }
    
    .storyline-card.disabled {
        opacity: 0.5;
    }
    
    .storyline-card h4 {
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .storyline-card p {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin: 0;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .badge-available {
        background: rgba(34, 197, 94, 0.2);
        color: var(--success);
    }
    
    .badge-coming {
        background: rgba(245, 158, 11, 0.2);
        color: var(--warning);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar Navigation
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #10b981; margin: 0;">🌿 NbS Analyzer</h2>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">
            Nature-based Solutions<br>Portfolio Analysis Suite
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📚 Documentation")
    st.markdown("""
    - [README](./nbs_analyzer/README.md)
    - [Methodology](#methodology)
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; font-size: 0.8rem; color: #64748b;">
        v1.0.0 | CATIE Consultancy
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Main Home Page Content
# =============================================================================

st.markdown("""
<div class="hero">
    <h1>🌿 NbS Analyzer Suite</h1>
    <p>A comprehensive toolkit for analyzing Nature-based Solutions portfolios. 
    Transform your data, run risk-first analysis, and generate actionable insights.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🚀 Quick Start")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">📁</div>
        <h3>Step 1: Data Converter</h3>
        <p>Transform your raw Excel file into an analysis-ready format with standardized tables and lookups.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Go to Converter →", key="nav_converter", use_container_width=True):
        st.switch_page("pages/1_📁_Converter.py")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">📊</div>
        <h3>Step 2: Analyzer</h3>
        <p>Run Storyline A, B, or C analysis on your prepared data to generate scores, shortlists, and reports.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Go to Analyzer →", key="nav_analyzer", use_container_width=True):
        st.switch_page("pages/2_📊_Analyzer.py")

# Workflow overview
st.markdown("---")
st.markdown("### 📋 Workflow Overview")

st.markdown("""
<div class="info-box">
    <h4>Two-Step Process</h4>
    <ol style="padding-left: 1.5rem; margin-top: 0.5rem;">
        <li><strong>Convert</strong> — Upload your raw Excel file and transform it into the standardized analysis-ready format</li>
        <li><strong>Analyze</strong> — Choose a storyline (A, B, or C) and generate comprehensive analytics, shortlists, and reports</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Storylines preview
st.markdown("### 🎯 Available Storylines")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="storyline-card available">
        <h4>Storyline A <span class="badge badge-available">Available</span></h4>
        <p><strong>Risk-First Analysis</strong></p>
        <p>Analyze threat landscape, compute NbS scores (TCS, SBS, TTS, GGL), generate value vs. friction shortlists.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="storyline-card available">
        <h4>Storyline B <span class="badge badge-available">Available</span></h4>
        <p><strong>Benefits-First Analysis</strong></p>
        <p>Analyze beneficiaries, co-benefits (security dimensions), and equity-focused shortlists.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="storyline-card available">
        <h4>Storyline C <span class="badge badge-available">Available</span></h4>
        <p><strong>Transformation-First Analysis</strong></p>
        <p>Analyze transformative traits, signatures, and co-benefit/threat lifts for strategic portfolio planning.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>🌿 NbS Analyzer Suite v1.0.0 | Built for CATIE Consultancy</p>
    <p>Transform → Analyze → Decide</p>
</div>
""", unsafe_allow_html=True)
