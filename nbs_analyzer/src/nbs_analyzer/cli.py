"""
CLI entrypoint for the NbS Analyzer.

Provides command-line interface using Typer.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from nbs_analyzer import __version__
from nbs_analyzer.storyline_a import run_storyline_a
from nbs_analyzer.storyline_b import run_storyline_b
from nbs_analyzer.storyline_c import run_storyline_c

# Initialize CLI app
app = typer.Typer(
    name="nbs-analyze",
    help="NbS Analyzer — Deterministic analysis for Nature-based Solutions portfolios",
    add_completion=False,
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]NbS Analyzer[/bold blue] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    ),
) -> None:
    """NbS Analyzer — Risk-First Analysis for Nature-based Solutions."""
    pass


@app.command("storyline-a")
def storyline_a_cmd(
    input_file: Path = typer.Option(
        ..., "--input", "-i",
        help="Path to analysis-ready Excel workbook",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        "./outputs", "--outdir", "-o",
        help="Output directory for results",
    ),
    filters: Optional[str] = typer.Option(
        None, "--filters", "-f",
        help='JSON string of filters, e.g. \'{"country": "Honduras"}\'',
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-V",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Run Storyline A (Risk-First) analysis.
    
    Processes an analysis-ready Excel workbook and generates:
    - Threat frequency tables and co-occurrence analysis
    - NbS scores (TCS, SBS, TTS, GGL, ValueScore)
    - Shortlists (low friction and needs enablers)
    - Governance gap analysis with enabling packages
    - One-pagers for each NbS case
    - HTML report with visualizations
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Parse filters
    parsed_filters = None
    if filters:
        try:
            parsed_filters = json.loads(filters)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing filters JSON:[/red] {e}")
            raise typer.Exit(1)
    
    # Header
    console.print(Panel.fit(
        "[bold blue]🌿 NbS Analyzer — Storyline A[/bold blue]\n"
        "[dim]Risk-First Analysis Pipeline[/dim]",
        border_style="blue"
    ))
    
    console.print(f"\n[bold]Input:[/bold] {input_file}")
    console.print(f"[bold]Output:[/bold] {output_dir}")
    if parsed_filters:
        console.print(f"[bold]Filters:[/bold] {parsed_filters}")
    console.print()
    
    try:
        # Run analysis
        with console.status("[bold green]Running Storyline A analysis..."):
            results = run_storyline_a(
                input_path=input_file,
                output_dir=output_dir,
                filters=parsed_filters,
            )
        
        # Summary table
        table = Table(title="Analysis Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        
        table.add_row("Total NbS Cases", str(results.n_cases))
        table.add_row("Distinct AC Threats", str(len(results.threat_freq_ac)))
        table.add_row("Distinct ANC Threats", str(len(results.threat_freq_anc)))
        table.add_row("Low Friction Shortlist", str(len(results.shortlist_low_friction)))
        table.add_row("Needs Enablers Shortlist", str(len(results.shortlist_needs_enablers)))
        table.add_row("Governance Gaps", str(len(results.top_gov_gaps_overall)))
        
        console.print()
        console.print(table)
        
        # Warnings
        if results.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in results.warnings:
                console.print(f"  ⚠️  {warning}")
        
        # Output files
        console.print("\n[bold green]✓ Analysis complete![/bold green]")
        console.print(f"\n[bold]Output files in:[/bold] {output_dir}")
        console.print("  📊 tables/ — CSV and Excel files")
        console.print("  📈 figures/ — PNG visualizations")
        console.print("  📄 reports/ — HTML report")
        
    except FileNotFoundError as e:
        console.print(f"[red]File not found:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("storyline-b")
def storyline_b_cmd(
    input_file: Path = typer.Option(
        ..., "--input", "-i",
        help="Path to analysis-ready Excel workbook",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        "./outputs", "--outdir", "-o",
        help="Output directory for results",
    ),
    filters: Optional[str] = typer.Option(
        None, "--filters", "-f",
        help='JSON string of filters, e.g. \'{"country": "Honduras"}\'',
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-V",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Run Storyline B (Benefits/Equity-First) analysis.
    
    Processes an analysis-ready Excel workbook and generates:
    - Beneficiary analytics (who benefits)
    - Security dimension coverage (co-benefits profile)
    - Benefits × Security cross analysis
    - Threat-to-security and security-to-threat profiles
    - Governance gap analysis linked to security
    - Shortlists (equity leaders and needs enablers)
    - One-pagers for each NbS case
    - HTML report with visualizations
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Parse filters
    parsed_filters = None
    if filters:
        try:
            parsed_filters = json.loads(filters)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing filters JSON:[/red] {e}")
            raise typer.Exit(1)
    
    # Header
    console.print(Panel.fit(
        "[bold magenta]🌿 NbS Analyzer — Storyline B[/bold magenta]\n"
        "[dim]Benefits/Equity-First Analysis Pipeline[/dim]",
        border_style="magenta"
    ))
    
    console.print(f"\n[bold]Input:[/bold] {input_file}")
    console.print(f"[bold]Output:[/bold] {output_dir}")
    if parsed_filters:
        console.print(f"[bold]Filters:[/bold] {parsed_filters}")
    console.print()
    
    try:
        # Run analysis
        with console.status("[bold green]Running Storyline B analysis..."):
            results = run_storyline_b(
                input_path=input_file,
                output_dir=output_dir,
                filters=parsed_filters,
            )
        
        # Summary table
        table = Table(title="Analysis Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        
        table.add_row("Total NbS Cases", str(results.n_cases))
        table.add_row("Numeric Beneficiaries", "Yes" if results.has_numeric_beneficiaries else "No (proxy mode)")
        table.add_row("Security Dimensions", str(len(results.security_dimension_rates)))
        table.add_row("Equity Leaders (B1)", str(len(results.shortlist_B1_equity_leaders)))
        table.add_row("Needs Enablers (B2)", str(len(results.shortlist_B2_needs_enablers)))
        
        console.print()
        console.print(table)
        
        # Warnings
        if results.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in results.warnings:
                console.print(f"  ⚠️  {warning}")
        
        # Output files
        console.print("\n[bold green]✓ Analysis complete![/bold green]")
        console.print(f"\n[bold]Output files in:[/bold] {output_dir}")
        console.print("  📊 tables/ — CSV and Excel files")
        console.print("  📈 figures/ — PNG visualizations")
        console.print("  📄 reports/ — HTML report")
        
    except FileNotFoundError as e:
        console.print(f"[red]File not found:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("storyline-c")
def storyline_c_cmd(
    input_file: Path = typer.Option(
        ..., "--input", "-i",
        help="Path to analysis-ready Excel workbook",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        "./outputs", "--outdir", "-o",
        help="Output directory for results",
    ),
    filters: Optional[str] = typer.Option(
        None, "--filters", "-f",
        help='JSON string of filters, e.g. \'{"country": "Honduras"}\'',
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-V",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Run Storyline C (Transformation-First) analysis.
    
    Processes an analysis-ready Excel workbook and generates:
    - Transformative traits portfolio analytics
    - Links to co-benefits, threats, and governance
    - Lifts (High vs Low TTS) for security, threats, and gaps
    - Shortlists (leaders and enabler-needing)
    - One-pagers focused on transformation
    - HTML report with visualizations
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Parse filters
    parsed_filters = None
    if filters:
        try:
            parsed_filters = json.loads(filters)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing filters JSON:[/red] {e}")
            raise typer.Exit(1)
    
    # Header
    console.print(Panel.fit(
        "[bold purple]🌿 NbS Analyzer — Storyline C[/bold purple]\n"
        "[dim]Transformation-First Analysis Pipeline[/dim]",
        border_style="purple"
    ))
    
    console.print(f"\n[bold]Input:[/bold] {input_file}")
    console.print(f"[bold]Output:[/bold] {output_dir}")
    if parsed_filters:
        console.print(f"[bold]Filters:[/bold] {parsed_filters}")
    console.print()
    
    try:
        # Run analysis
        with console.status("[bold green]Running Storyline C analysis..."):
            results = run_storyline_c(
                input_path=input_file,
                output_dir=output_dir,
                filters=parsed_filters,
            )
        
        # Summary table
        table = Table(title="Analysis Summary", show_header=True, header_style="bold purple")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        
        table.add_row("Total NbS Cases", str(results.n_cases))
        table.add_row("Traits Flagged", str(len(results.trait_rates)))
        table.add_row("High TTS Cases", str(results.archetype_counts.get("High", 0)))
        table.add_row("Leaders Shortlist (C1)", str(len(results.shortlist_c1)))
        table.add_row("Needs Enablers (C2)", str(len(results.shortlist_c2)))
        
        console.print()
        console.print(table)
        
        # Warnings
        if results.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in results.warnings:
                console.print(f"  ⚠️  {warning}")
        
        # Output files
        console.print("\n[bold green]✓ Analysis complete![/bold green]")
        console.print(f"\n[bold]Output files in:[/bold] {output_dir}")
        console.print("  📊 tables/ — CSV and Excel files")
        console.print("  📈 figures/ — PNG visualizations")
        console.print("  📄 reports/ — HTML report")
        
    except FileNotFoundError as e:
        console.print(f"[red]File not found:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("validate")
def validate_cmd(
    input_file: Path = typer.Option(
        ..., "--input", "-i",
        help="Path to Excel workbook to validate",
        exists=True,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-V",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Validate an Excel workbook without running full analysis.
    
    Checks for required sheets, columns, data quality issues.
    """
    setup_logging(verbose)
    
    from nbs_analyzer.io import load_workbook
    from nbs_analyzer.schema import REQUIRED_SHEETS, OPTIONAL_SHEETS
    from nbs_analyzer.validate import run_all_validations
    
    console.print(Panel.fit(
        "[bold blue]🔍 NbS Analyzer — Validation[/bold blue]",
        border_style="blue"
    ))
    
    console.print(f"\n[bold]Input:[/bold] {input_file}\n")
    
    try:
        # Load data
        with console.status("[bold green]Loading workbook..."):
            data, missing_optional = load_workbook(
                input_file,
                required_sheets=REQUIRED_SHEETS,
                optional_sheets=OPTIONAL_SHEETS
            )
        
        console.print(f"[green]✓[/green] Loaded {len(data)} sheets")
        
        if missing_optional:
            console.print(f"[yellow]⚠[/yellow] Missing optional sheets: {missing_optional}")
        
        # Run validations
        with console.status("[bold green]Running validations..."):
            report = run_all_validations(data)
        
        # Results table
        table = Table(title="Validation Results", show_header=True, header_style="bold cyan")
        table.add_column("Check", style="dim")
        table.add_column("Status")
        table.add_column("Message")
        
        for v in report.validations:
            status = "[green]✓ PASS[/green]" if v.passed else (
                "[red]✗ FAIL[/red]" if v.severity == "error" else "[yellow]⚠ WARN[/yellow]"
            )
            table.add_row(v.check_name, status, v.message)
        
        console.print()
        console.print(table)
        
        # Summary
        console.print(f"\n[bold]Summary:[/bold] {report.summary}")
        
        if report.passed:
            console.print("\n[bold green]✓ Validation passed![/bold green]")
        else:
            console.print(f"\n[bold red]✗ Validation failed with {report.error_count} errors[/bold red]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
