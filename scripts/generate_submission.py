#!/usr/bin/env python3
"""
Generate Submission Package

Combines all submission materials into a single comprehensive markdown file
for reviewer submission. Run after major changes to regenerate.

Usage:
    python scripts/generate_submission.py
    python scripts/generate_submission.py --output custom_name.md
"""

import argparse
import base64
from datetime import datetime
from pathlib import Path
import re


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def read_file(path: Path) -> str:
    """Read file contents, return empty string if not found."""
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError:
        print(f"  Warning: {path} not found")
        return ""


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 data URI."""
    try:
        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        suffix = image_path.suffix.lower()
        mime_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml'
        }.get(suffix, 'image/png')
        return f"data:{mime_type};base64,{data}"
    except FileNotFoundError:
        print(f"  Warning: Image {image_path} not found")
        return ""


def extract_code_section(content: str, start_line: int = 1, end_line: int = None) -> str:
    """Extract specific lines from code content."""
    lines = content.split('\n')
    if end_line is None:
        end_line = len(lines)
    return '\n'.join(lines[start_line-1:end_line])


def clean_markdown_for_embedding(content: str, header_level_offset: int = 0) -> str:
    """Clean markdown content for embedding, adjusting header levels."""
    if header_level_offset > 0:
        # Increase header levels (e.g., # -> ##)
        def increase_header(match):
            hashes = match.group(1)
            text = match.group(2)
            new_hashes = '#' * (len(hashes) + header_level_offset)
            return f"{new_hashes} {text}"
        content = re.sub(r'^(#{1,6})\s+(.+)$', increase_header, content, flags=re.MULTILINE)
    return content


def generate_submission_package(output_path: Path = None):
    """Generate the complete submission package."""
    root = get_project_root()

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = root / f"Paper/Submission_Package_{timestamp}.md"

    print(f"Generating submission package: {output_path}")

    sections = []

    # ==========================================================================
    # HEADER
    # ==========================================================================
    sections.append(f"""# Physics-Guided GNN for Power Systems: Submission Package

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Model Card](#model-card)
3. [Results](#results)
4. [Figures](#figures)
5. [Technical Architecture](#technical-architecture)
6. [Feature Audit](#feature-audit)
7. [Key Implementation Code](#key-implementation-code)
8. [Reproducibility](#reproducibility)

---
""")

    # ==========================================================================
    # EXECUTIVE SUMMARY (from Progress Report intro)
    # ==========================================================================
    print("  Adding Executive Summary...")
    progress_report = read_file(root / "Paper/Progress_Report.md")

    sections.append("""## Executive Summary

### Primary Research Claim

> "A grid-specific self-supervised, physics-consistent GNN encoder improves PF/Line Flow learning (especially low-label / OOD), and transfers to cascading-failure prediction and explanation."

### Key Results Summary

| Task | Metric | SSL Improvement | Validation |
|------|--------|-----------------|------------|
| Cascade Prediction (IEEE-24) | F1 Score | +14.2% at 10% labels | 3-seed |
| Cascade Prediction (IEEE-118) | F1 Score | +61 points at 10% labels | 5-seed |
| Power Flow | MAE | +29.1% at 10% labels | 5-seed |
| Line Flow Prediction | MAE | +26.4% at 10% labels | 5-seed |

### Validated Claims

- **SSL Transfer**: Self-supervised pretraining significantly improves downstream task performance
- **Low-Label Regime**: Benefits are most pronounced with limited labeled data (10-20%)
- **Training Stability**: SSL reduces variance by ~5x at low label percentages
- **OOD Robustness**: SSL improves out-of-distribution generalization (+22% at 1.3x load)
- **Grid Scalability**: Results replicate across IEEE 24-bus and IEEE 118-bus systems

---
""")

    # ==========================================================================
    # MODEL CARD
    # ==========================================================================
    print("  Adding Model Card...")
    model_card = read_file(root / "MODEL_CARD.md")
    if model_card:
        # Remove the first header line if it exists (we'll add our own)
        model_card = re.sub(r'^#[^#].*\n', '', model_card, count=1)
        model_card = clean_markdown_for_embedding(model_card, header_level_offset=1)
        sections.append(f"""## Model Card

{model_card}

---
""")

    # ==========================================================================
    # RESULTS
    # ==========================================================================
    print("  Adding Results...")
    results = read_file(root / "Paper/Results.md")
    if results:
        results = re.sub(r'^#[^#].*\n', '', results, count=1)
        results = clean_markdown_for_embedding(results, header_level_offset=1)
        sections.append(f"""## Results

{results}

---
""")

    # ==========================================================================
    # FIGURES (embedded as base64)
    # ==========================================================================
    print("  Adding Figures...")
    figures_dir = root / "analysis/figures"

    figure_descriptions = {
        'cascade_ssl_comparison.png': 'IEEE-24 Cascade: SSL vs Scratch Training Comparison',
        'cascade_improvement_curve.png': 'IEEE-24 Cascade: Label Efficiency Improvement Curve',
        'cascade_118_ssl_comparison.png': 'IEEE-118 Cascade: SSL vs Scratch Training Comparison',
        'cascade_118_improvement_curve.png': 'IEEE-118 Cascade: Label Efficiency Improvement Curve (Absolute ΔF1)',
        'pf_ssl_comparison.png': 'Power Flow: SSL vs Scratch Training Comparison',
        'pf_improvement_curve.png': 'Power Flow: Label Efficiency Improvement Curve',
        'lineflow_ssl_comparison.png': 'Line Flow: SSL vs Scratch Training Comparison',
        'lineflow_improvement_curve.png': 'Line Flow: Label Efficiency Improvement Curve',
        'grid_scalability_comparison.png': 'Cross-Grid Scalability: IEEE-24 vs IEEE-118',
        'multi_task_comparison.png': 'Multi-Task Performance Summary',
    }

    figures_section = "## Figures\n\n"

    for fig_name, description in figure_descriptions.items():
        fig_path = figures_dir / fig_name
        if fig_path.exists():
            base64_data = image_to_base64(fig_path)
            if base64_data:
                figures_section += f"### {description}\n\n"
                figures_section += f"![{description}]({base64_data})\n\n"
                print(f"    Embedded: {fig_name}")
        else:
            print(f"    Missing: {fig_name}")

    # Add LaTeX tables
    figures_section += "### LaTeX Tables\n\n"
    for tex_file in sorted(figures_dir.glob("*.tex")):
        tex_content = read_file(tex_file)
        if tex_content:
            figures_section += f"**{tex_file.name}**:\n```latex\n{tex_content}\n```\n\n"

    sections.append(figures_section + "\n---\n")

    # ==========================================================================
    # TECHNICAL ARCHITECTURE
    # ==========================================================================
    print("  Adding Technical Architecture...")
    architecture = read_file(root / "Paper/ModelArchitecture.md")
    if architecture:
        architecture = re.sub(r'^#[^#].*\n', '', architecture, count=1)
        architecture = clean_markdown_for_embedding(architecture, header_level_offset=1)
        sections.append(f"""## Technical Architecture

{architecture}

---
""")

    # ==========================================================================
    # FEATURE AUDIT
    # ==========================================================================
    print("  Adding Feature Audit...")
    feature_audit = read_file(root / "Paper/Feature_Audit.md")
    if feature_audit:
        feature_audit = re.sub(r'^#[^#].*\n', '', feature_audit, count=1)
        feature_audit = clean_markdown_for_embedding(feature_audit, header_level_offset=1)
        sections.append(f"""## Feature Audit

{feature_audit}

---
""")

    # ==========================================================================
    # KEY IMPLEMENTATION CODE
    # ==========================================================================
    print("  Adding Key Implementation Code...")

    code_section = """## Key Implementation Code

### PhysicsGuidedConv Layer

The core physics-informed message passing layer:

```python
"""

    # Read PhysicsGuidedConv
    encoder_path = root / "src/models/encoder.py"
    encoder_content = read_file(encoder_path)
    if encoder_content:
        # Extract PhysicsGuidedConv class
        match = re.search(r'(class PhysicsGuidedConv\(.*?\n(?:.*?\n)*?)(class |\Z)', encoder_content, re.MULTILINE)
        if match:
            conv_code = match.group(1).rstrip()
            code_section += conv_code

    code_section += """
```

### SSL Pretraining Module

```python
"""

    # Read SSL module (located in src/models/ssl.py)
    ssl_path = root / "src/models/ssl.py"
    ssl_content = read_file(ssl_path)
    if ssl_content:
        # Include the full SSL module (it's not too long)
        code_section += ssl_content.strip()

    code_section += """
```

### Physics Metrics (Thermal Violation)

```python
"""

    # Read physics metrics
    metrics_path = root / "src/metrics/physics.py"
    metrics_content = read_file(metrics_path)
    if metrics_content:
        # Extract thermal_violation_rate function
        match = re.search(r'(def thermal_violation_rate\(.*?\n(?:    .*?\n)*)', metrics_content)
        if match:
            metrics_code = match.group(1).rstrip()
            code_section += metrics_code

    code_section += """
```

### Training Script (SSL Loading with Hard Failure)

```python
"""

    # Read train_pf_opf.py SSL loading section
    train_path = root / "scripts/train_pf_opf.py"
    train_content = read_file(train_path)
    if train_content:
        # Extract SSL loading section
        match = re.search(r'(# Load pretrained encoder.*?print\(f"  Loaded pretrained encoder.*?\))', train_content, re.DOTALL)
        if match:
            train_code = match.group(1)
            code_section += train_code

    code_section += """
```

---
"""

    sections.append(code_section)

    # ==========================================================================
    # REPRODUCIBILITY
    # ==========================================================================
    print("  Adding Reproducibility...")

    # Read configs
    base_config = read_file(root / "configs/base.yaml")
    splits_config = read_file(root / "configs/splits.yaml")

    sections.append(f"""## Reproducibility

### Configuration Files

**configs/base.yaml**:
```yaml
{base_config}
```

**configs/splits.yaml**:
```yaml
{splits_config}
```

### One-Command Reproduction

```bash
# Run complete analysis pipeline
python analysis/run_all.py

# Individual tasks
python scripts/train_cascade.py --grid ieee24 --label_fractions 0.1 0.2 0.5 1.0
python scripts/train_pf_opf.py --task pf --seeds 42 123 456 789 1337
python scripts/trivial_baselines.py --grid ieee24
```

### Environment

```
Python 3.10+
PyTorch 2.0+
PyTorch Geometric 2.4+
CUDA 11.8+ (for GPU training)
```

See `requirements.txt` for complete dependencies.

### Seeds Used

| Component | Seed(s) |
|-----------|---------|
| Data Splits | 42 |
| Model Init (single-seed) | 42 |
| Multi-seed Validation | 42, 123, 456, 789, 1337 |

---

## Appendix: Output Folder Structure

```
outputs/
├── multiseed_ieee118_20251214_084423/    # IEEE-118 cascade (5-seed)
├── multiseed_ieee24_20251213_235213/     # IEEE-24 cascade (5-seed)
├── pf_multiseed_ieee24_20251214_193216/  # Power Flow (5-seed)
├── opf_multiseed_ieee24_20251214_193216/ # Line Flow (5-seed)
├── comparison_ieee118_20251214_005058/   # IEEE-118 comparison
├── comparison_ieee24_20251213_192316/    # IEEE-24 comparison
├── trivial_baselines_ieee24_20251214_*/  # Baseline results
└── trivial_baselines_ieee118_20251214_*/ # Baseline results
```

---

*End of Submission Package*
""")

    # ==========================================================================
    # WRITE OUTPUT
    # ==========================================================================
    output_content = '\n'.join(sections)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(output_content, encoding='utf-8')

    # Calculate size
    size_mb = len(output_content.encode('utf-8')) / (1024 * 1024)

    print(f"\nSubmission package generated successfully!")
    print(f"  Output: {output_path}")
    print(f"  Size: {size_mb:.2f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate submission package")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: Paper/Submission_Package_<timestamp>.md)')
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    generate_submission_package(output_path)


if __name__ == "__main__":
    main()
