"""
HTML Report Generation for the NbS Analyzer.

Uses Jinja2 templates to generate self-contained HTML reports.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, BaseLoader

if TYPE_CHECKING:
    from nbs_analyzer.storyline_a import StorylineAResults

logger = logging.getLogger(__name__)


# Inline template for when template file is not found
INLINE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NbS Analyzer - Storyline A Report</title>
    <style>
        :root {
            --primary: #2563eb;
            --secondary: #64748b;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            background: linear-gradient(135deg, var(--primary), #1d4ed8);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        header .meta {
            opacity: 0.9;
            font-size: 0.9rem;
        }
        
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border);
        }
        
        .card h2 {
            color: var(--primary);
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            padding: 1.25rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #bae6fd;
        }
        
        .stat-card.success {
            background: linear-gradient(135deg, #ecfdf5, #d1fae5);
            border-color: #6ee7b7;
        }
        
        .stat-card.warning {
            background: linear-gradient(135deg, #fffbeb, #fef3c7);
            border-color: #fcd34d;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }
        
        .stat-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            background: #f1f5f9;
            font-weight: 600;
            color: var(--text);
        }
        
        tr:hover {
            background: #f8fafc;
        }
        
        .figure {
            text-align: center;
            margin: 1.5rem 0;
        }
        
        .figure img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .figure-caption {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }
        
        .warning-box {
            background: #fffbeb;
            border: 1px solid #fcd34d;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #92400e;
        }
        
        .warning-box strong {
            display: block;
            margin-bottom: 0.25rem;
        }
        
        .download-links {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        
        .download-link {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1rem;
            background: var(--primary);
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-size: 0.85rem;
            transition: background 0.2s;
        }
        
        .download-link:hover {
            background: #1d4ed8;
        }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        
        .two-col {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 1rem;
            }
            
            .two-col {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌿 NbS Analyzer — Storyline A Report</h1>
            <div class="meta">
                <strong>Risk-First Analysis</strong><br>
                Input: {{ input_file }}<br>
                Generated: {{ timestamp }}
            </div>
        </header>
        
        <div class="card">
            <h2>📊 Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ n_cases }}</div>
                    <div class="stat-label">Total NbS Cases</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-value">{{ n_low_friction }}</div>
                    <div class="stat-label">Low Friction (Ready)</div>
                </div>
                <div class="stat-card warning">
                    <div class="stat-value">{{ n_needs_enablers }}</div>
                    <div class="stat-label">Needs Enablers</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ n_threats_ac }}</div>
                    <div class="stat-label">Distinct AC Threats</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ n_threats_anc }}</div>
                    <div class="stat-label">Distinct ANC Threats</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ n_gaps }}</div>
                    <div class="stat-label">Distinct Gov Gaps</div>
                </div>
            </div>
            
            {% if warnings %}
            <div class="warning-box">
                <strong>⚠️ Warnings</strong>
                {% for warning in warnings %}
                <div>{{ warning }}</div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
        
        <div class="two-col">
            <div class="card">
                <h2>🌡️ Top Climatic Threats (AC)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Code</th>
                            <th>Threat</th>
                            <th>Cases</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in top_ac %}
                        <tr>
                            <td>{{ row.threat_code }}</td>
                            <td>{{ row.threat_label or '-' }}</td>
                            <td>{{ row.count }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>🔥 Top Non-Climatic Threats (ANC)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Code</th>
                            <th>Threat</th>
                            <th>Cases</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in top_anc %}
                        <tr>
                            <td>{{ row.threat_code }}</td>
                            <td>{{ row.threat_label or '-' }}</td>
                            <td>{{ row.count }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        {% if plot_top_ac %}
        <div class="card">
            <h2>📈 Visualizations</h2>
            <div class="two-col">
                <div class="figure">
                    <img src="{{ plot_top_ac }}" alt="Top AC Threats">
                    <div class="figure-caption">Top 10 Climatic Threats</div>
                </div>
                {% if plot_top_anc %}
                <div class="figure">
                    <img src="{{ plot_top_anc }}" alt="Top ANC Threats">
                    <div class="figure-caption">Top 10 Non-Climatic Threats</div>
                </div>
                {% endif %}
            </div>
            
            {% if plot_scatter %}
            <div class="figure">
                <img src="{{ plot_scatter }}" alt="Value vs Friction">
                <div class="figure-caption">Value Score vs. Governance Gap Load (Friction)</div>
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <div class="card">
            <h2>✅ Shortlist: Low Friction Cases</h2>
            <p style="color: var(--text-muted); margin-bottom: 1rem;">
                High value cases with low governance barriers — ready for implementation.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>VS</th>
                        <th>TCS</th>
                        <th>SBS</th>
                        <th>TTS</th>
                        <th>GGL</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in shortlist_low_friction %}
                    <tr>
                        <td>{{ row.id_nbs }}</td>
                        <td><strong>{{ row.VS }}</strong></td>
                        <td>{{ row.TCS }}</td>
                        <td>{{ row.SBS }}</td>
                        <td>{{ row.TTS }}</td>
                        <td>{{ row.GGL }}</td>
                    </tr>
                    {% endfor %}
                    {% if not shortlist_low_friction %}
                    <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No cases meet criteria</td></tr>
                    {% endif %}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>⚙️ Shortlist: Needs Enabling Packages</h2>
            <p style="color: var(--text-muted); margin-bottom: 1rem;">
                High value cases requiring governance interventions before implementation.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>VS</th>
                        <th>TCS</th>
                        <th>SBS</th>
                        <th>TTS</th>
                        <th>GGL</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in shortlist_needs_enablers %}
                    <tr>
                        <td>{{ row.id_nbs }}</td>
                        <td><strong>{{ row.VS }}</strong></td>
                        <td>{{ row.TCS }}</td>
                        <td>{{ row.SBS }}</td>
                        <td>{{ row.TTS }}</td>
                        <td>{{ row.GGL }}</td>
                    </tr>
                    {% endfor %}
                    {% if not shortlist_needs_enablers %}
                    <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No cases meet criteria</td></tr>
                    {% endif %}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>🏛️ Top Governance Gaps</h2>
            <table>
                <thead>
                    <tr>
                        <th>Code</th>
                        <th>Gap</th>
                        <th>Cases</th>
                        <th>%</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in top_gaps %}
                    <tr>
                        <td>{{ row.gap_code }}</td>
                        <td>{{ row.gap_name or '-' }}</td>
                        <td>{{ row.count }}</td>
                        <td>{{ row.percentage }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>📁 Output Files</h2>
            <div class="download-links">
                <a href="tables/nbs_scores.csv" class="download-link">📊 NbS Scores (CSV)</a>
                <a href="tables/shortlist_low_friction.csv" class="download-link">✅ Low Friction Shortlist</a>
                <a href="tables/shortlist_needs_enablers.csv" class="download-link">⚙️ Needs Enablers Shortlist</a>
                <a href="tables/threat_frequencies_ac.csv" class="download-link">🌡️ AC Threats</a>
                <a href="tables/threat_frequencies_anc.csv" class="download-link">🔥 ANC Threats</a>
                <a href="tables/threat_pairs.csv" class="download-link">🔗 Threat Pairs</a>
                <a href="tables/top_gov_gaps_overall.csv" class="download-link">🏛️ Gov Gaps</a>
                <a href="tables/nbs_onepagers.xlsx" class="download-link">📋 One-Pagers (Excel)</a>
                <a href="tables/qa_summary.csv" class="download-link">🔍 QA Summary</a>
            </div>
        </div>
        
        <div class="card">
            <h2>📐 Methodology</h2>
            <table>
                <thead>
                    <tr><th>Score</th><th>Definition</th></tr>
                </thead>
                <tbody>
                    <tr><td><strong>TCS</strong></td><td>Threat Coverage Score = unique threats addressed</td></tr>
                    <tr><td><strong>n_AC</strong></td><td>Count of climatic threats (AC codes)</td></tr>
                    <tr><td><strong>n_ANC</strong></td><td>Count of non-climatic threats (ANC codes)</td></tr>
                    <tr><td><strong>SBS</strong></td><td>Security Breadth Score = flagged security dimensions</td></tr>
                    <tr><td><strong>TTS</strong></td><td>Transformative Traits Score = flagged traits</td></tr>
                    <tr><td><strong>GGL</strong></td><td>Governance Gap Load = governance barriers count</td></tr>
                    <tr><td><strong>VS</strong></td><td>Value Score = TCS + SBS + TTS</td></tr>
                </tbody>
            </table>
            
            <h3 style="margin-top: 1.5rem; margin-bottom: 0.5rem;">Shortlist Thresholds</h3>
            <ul style="padding-left: 1.5rem; color: var(--text-muted);">
                <li><strong>High Value:</strong> VS ≥ 80th percentile ({{ thresholds.vs_threshold }})</li>
                <li><strong>Low Friction:</strong> GGL ≤ 40th percentile ({{ thresholds.ggl_low_threshold }})</li>
                <li><strong>High Friction:</strong> GGL ≥ 60th percentile ({{ thresholds.ggl_high_threshold }})</li>
            </ul>
        </div>
        
        <footer>
            <p>Generated by NbS Analyzer v1.0.0 | Storyline A (Risk-First)</p>
            <p>{{ timestamp }}</p>
        </footer>
    </div>
</body>
</html>
"""


def encode_image_base64(path: Path) -> str:
    """Encode an image file as base64 data URI."""
    if not path.exists():
        return ""
    
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }.get(suffix, "image/png")
    
    return f"data:{mime};base64,{data}"


def generate_html_report(
    results: Any,
    plot_paths: Dict[str, Path],
    output_path: Path,
    template_name: str = "report_storyline_a.html.j2",
    embed_images: bool = True
) -> None:
    """
    Generate HTML report from analysis results.
    
    Args:
        results: Result dataclass (StorylineAResults, etc)
        plot_paths: Dict of plot name -> Path
        output_path: Output HTML file path
        template_name: Name of the Jinja2 template to use
        embed_images: If True, embed images as base64
    """
    # Try to load template file
    template_dir = Path(__file__).parent.parent.parent / "templates"
    template_file = template_dir / template_name
    
    if template_file.exists():
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template(template_name)
    else:
        # Fallback to inline or error if template_name is not default
        if template_name == "report_storyline_a.html.j2":
            env = Environment(loader=BaseLoader())
            template = env.from_string(INLINE_TEMPLATE)
        else:
            logger.error(f"Template {template_name} not found in {template_dir}")
            return

    # Prepare context based on result type
    context = {
        "input_file": Path(results.input_file).name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": results.n_cases,
        "warnings": getattr(results, "warnings", []),
    }

    # Embed or link images
    images = {}
    for name, path in plot_paths.items():
        if embed_images:
            images[f"plot_{name}"] = encode_image_base64(path)
        else:
            images[f"plot_{name}"] = f"../figures/{path.name}"
    context.update(images)

    # Dispatch context preparation by result class
    class_name = results.__class__.__name__
    
    if class_name == "StorylineAResults":
        context.update({
            "n_low_friction": len(results.shortlist_low_friction),
            "n_needs_enablers": len(results.shortlist_needs_enablers),
            "n_threats_ac": len(results.threat_freq_ac),
            "n_threats_anc": len(results.threat_freq_anc),
            "n_gaps": len(results.top_gov_gaps_overall),
            "top_ac": results.threat_freq_ac.head(10).to_dict("records"),
            "top_anc": results.threat_freq_anc.head(10).to_dict("records"),
            "top_gaps": results.top_gov_gaps_overall.head(10).to_dict("records"),
            "shortlist_low_friction": results.shortlist_low_friction.head(10).to_dict("records"),
            "shortlist_needs_enablers": results.shortlist_needs_enablers.head(10).to_dict("records"),
            "thresholds": results.shortlist_thresholds,
        })
    elif class_name == "StorylineCResults":
        context.update({
            "trait_rates": results.trait_rates.to_dict("records"),
            "trait_signatures": results.trait_signatures.head(10).to_dict("records"),
            "trait_pairs": results.trait_pairs.head(10).to_dict("records"),
            "archetype_counts": results.archetype_counts,
            "security_lift": results.security_lift.head(15).to_dict("records"),
            "threat_lift": results.threat_lift.head(15).to_dict("records"),
            "gap_lift": results.gap_lift.head(15).to_dict("records"),
            "packages_high_tts": results.packages_high_tts.head(10).to_dict("records"),
            "shortlist_c1": results.shortlist_c1.to_dict("records"),
            "shortlist_c2": results.shortlist_c2.to_dict("records"),
            "sbs_by_tts": results.sbs_by_tts_group.to_dict("records"),
            "threat_by_tts": results.threat_coverage_by_tts_group.to_dict("records"),
        })
    # Add B if needed, but B handles it inline so far. I'll stick to C for now.

    # Render
    html = template.render(**context)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    logger.info(f"Generated HTML report: {output_path}")
