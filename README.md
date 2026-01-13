# NbS Analyzer Suite 🌿

A comprehensive web application for transforming and analyzing Nature-based Solutions (NbS) portfolios. Built for **CATIE**, this suite provides a deterministic, two-step workflow to move from raw data to actionable strategic insights.

## components

### 📁 1. Data Converter
Transforms raw survey/project Excel files into a standardized, "analysis-ready" format.
- Cleans and standardizes column names.
- Extracts threat codes (AC/ANC) and governance gap codes.
- Creates "tidy" (long-format) tables for scalable analysis.
- Generates exhaustive QA reports on data missingness.

### 📊 2. Analyzer Dashboard
Runs multi-perspective analysis based on three distinct "Storylines":
- **Storyline A (Risk-First)**: Focuses on threat landscape, governance barriers, and value-vs-friction shortlists.
- **Storyline B (Benefits/Equity-First)**: Focuses on beneficiaries, security dimensions (SBS), and equity-focused leaders.
- **Storyline C (Transformation-First)**: Focuses on transformative traits, archetypes (TTS), and strategic "lifts" in co-benefits and threat coverage.

### ⚙️ 3. NbS Analyzer Core
A standalone Python package (`nbs_analyzer`) containing the heavy-lifting logic, reporting engines, and deterministic scoring algorithms.

## Quick Start (Local)

### 1. Install Dependencies
```bash
# Install core analyzer
pip install -e ./nbs_analyzer

# Install app requirements
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

## Deployment (Docker / Coolify)

This suite is production-ready for deployment on **Hetzner** via **Coolify**.

### Using Docker
```bash
docker build -t nbs-analyzer .
docker run -p 8501:8501 nbs-analyzer
```

### Using Coolify
1. Point Coolify to your repository.
2. It will automatically detect the `docker-compose.yml`.
3. Set your custom domain, and you're live on port 8501.

## Project Structure

```text
.
├── app.py                # Main hub / Landing page
├── pages/                # Streamlit page definitions
│   ├── 1_📁_Converter.py # Data transformation logic
│   └── 2_📊_Analyzer.py  # Analysis dashboard logic
├── nbs_analyzer/         # Core logic package
│   ├── src/              # Orchestrators and metrics
│   ├── templates/        # Jinja2 HTML report templates
│   └── tests/            # Automated smoke tests
├── Dockerfile            # Production container definition
├── docker-compose.yml    # Coolify deployment config
└── requirements.txt      # Python dependencies
```

## Documentation
- For detailed methodology and CLI usage, see the [nbs_analyzer README](nbs_analyzer/README.md).

## License
MIT License - Built for CATIE Consultancy.
