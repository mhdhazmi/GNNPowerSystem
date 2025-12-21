#!/usr/bin/env python3
"""
Extract claims from LaTeX paper and generate audit report.
Usage: python extract_claims.py path/to/paper.tex
"""

import re
import json
import sys
from pathlib import Path

# Patterns indicating claims
CLAIM_PATTERNS = [
    (r'\+?\d+\.?\d*%', 'performance'),  # Percentage improvements
    (r'outperform', 'performance'),
    (r'achiev\w+\s+\d', 'performance'),
    (r'improv\w+', 'performance'),
    (r'reduc\w+.*\d', 'performance'),
    (r'first', 'novelty'),
    (r'novel', 'novelty'),
    (r'state-of-the-art', 'performance'),
    (r'SOTA', 'performance'),
    (r'robust', 'generalization'),
    (r'generaliz', 'generalization'),
    (r'transfer', 'generalization'),
    (r'efficient', 'efficiency'),
    (r'lightweight', 'efficiency'),
    (r'\d+[xX]\s+(smaller|faster|fewer)', 'efficiency'),
]

SUPERLATIVES = ['first', 'novel', 'unique', 'breakthrough', 'revolutionary', 'best']


def extract_claims(tex_content: str) -> list:
    """Extract sentences containing claim patterns."""
    claims = []
    
    # Split into sentences (rough approximation)
    sentences = re.split(r'(?<=[.!?])\s+', tex_content)
    
    for i, sent in enumerate(sentences):
        # Clean LaTeX commands
        clean = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', sent)
        clean = re.sub(r'\\[a-zA-Z]+', '', clean)
        clean = re.sub(r'[{}]', '', clean)
        
        for pattern, claim_type in CLAIM_PATTERNS:
            if re.search(pattern, clean, re.IGNORECASE):
                # Check for evidence pointers
                has_table_ref = bool(re.search(r'Table\s*\\?ref|Tab\.\s*\d', sent))
                has_figure_ref = bool(re.search(r'Fig\w*\s*\\?ref|Fig\.\s*\d', sent))
                has_citation = bool(re.search(r'\\cite\{', sent))
                
                evidence_present = has_table_ref or has_figure_ref
                
                # Risk assessment
                risk = 'LOW'
                if any(sup in clean.lower() for sup in SUPERLATIVES):
                    risk = 'HIGH'
                elif not evidence_present:
                    risk = 'MED'
                
                claims.append({
                    'id': f'C{len(claims)+1:02d}',
                    'claim': clean.strip()[:200],
                    'sentence_index': i,
                    'claim_type': claim_type,
                    'evidence_present': evidence_present,
                    'has_table_ref': has_table_ref,
                    'has_figure_ref': has_figure_ref,
                    'has_citation': has_citation,
                    'risk': risk,
                })
                break  # Only count each sentence once
    
    return claims


def generate_report(claims: list) -> str:
    """Generate markdown audit report."""
    high_risk = [c for c in claims if c['risk'] == 'HIGH']
    med_risk = [c for c in claims if c['risk'] == 'MED']
    
    report = "# Claim Audit Report\n\n"
    report += f"**Total claims found**: {len(claims)}\n"
    report += f"**High risk**: {len(high_risk)}\n"
    report += f"**Medium risk**: {len(med_risk)}\n\n"
    
    if high_risk:
        report += "## High-Risk Claims (Require Rewrite)\n\n"
        for c in high_risk:
            report += f"### {c['id']}: {c['claim_type']}\n"
            report += f"> {c['claim']}\n\n"
            report += f"- Evidence: {'Yes' if c['evidence_present'] else 'MISSING'}\n"
            report += f"- Suggested action: Soften language, add evidence reference\n\n"
    
    if med_risk:
        report += "## Medium-Risk Claims (Need Evidence)\n\n"
        for c in med_risk:
            report += f"- **{c['id']}**: {c['claim'][:100]}...\n"
    
    return report


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_claims.py path/to/paper.tex")
        sys.exit(1)
    
    tex_path = Path(sys.argv[1])
    if not tex_path.exists():
        print(f"File not found: {tex_path}")
        sys.exit(1)
    
    content = tex_path.read_text(encoding='utf-8', errors='ignore')
    claims = extract_claims(content)
    
    # Output JSON
    json_out = tex_path.with_suffix('.claims.json')
    with open(json_out, 'w') as f:
        json.dump(claims, f, indent=2)
    print(f"Claims JSON: {json_out}")
    
    # Output report
    report = generate_report(claims)
    report_out = tex_path.with_suffix('.claims.md')
    with open(report_out, 'w') as f:
        f.write(report)
    print(f"Audit report: {report_out}")
