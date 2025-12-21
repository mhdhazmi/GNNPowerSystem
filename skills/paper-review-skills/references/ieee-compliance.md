# IEEE PES GM Compliance Checklist

## Formatting Requirements

| Requirement | Specification | Check Method |
|-------------|---------------|--------------|
| Page limit | 8 pages (conference), 10 (journal) | Count PDF pages |
| Columns | Two-column format | Visual check |
| Margins | Top: 0.75", Bottom: 1", L/R: 0.625" | Template adherence |
| Font | Times New Roman, 10pt body | LaTeX: IEEEtran.cls |
| Title | 24pt, centered | Template |
| Abstract limit | 200 words max | Word count |
| Keywords | 3-7 index terms | Count |

## Citation Requirements

| Issue | Symptom | Fix |
|-------|---------|-----|
| Broken reference | [?] in text | Check .bib key matches \cite{} |
| Missing citation | Claim without [X] | Add appropriate reference |
| Self-citation excess | >30% self-cite | Diversify references |
| Old references | Most >5 years old | Add recent 2022-2025 work |

## Figure/Table Requirements

| Requirement | Specification |
|-------------|---------------|
| Resolution | 300 DPI minimum |
| Format | Vector (PDF/EPS) preferred |
| Caption | Below figures, above tables |
| Numbering | Sequential (Fig. 1, Fig. 2...) |
| In-text reference | Every figure/table must be cited |
| Font in figures | Readable at column width |

## Abstract Checklist

Must include:
- [ ] Problem statement (1-2 sentences)
- [ ] Method summary (2-3 sentences)
- [ ] Key quantitative results (2-3 sentences)
- [ ] Impact statement (1 sentence)

Must NOT include:
- References to figures/tables
- Citations
- Acronyms without definition (except IEEE, etc.)

## Common LaTeX Issues

```latex
% Broken citation
\cite{wrongKey}  % Check citations.bib for exact key

% Missing package
\usepackage{algorithm}  % Ensure in preamble

% Overfull hbox
Check for long unbreakable words/URLs

% Figure placement
\begin{figure}[!t]  % Use [!t] or [!b] for top/bottom
```

## Pre-Submission Checklist

1. [ ] PDF compiles without errors
2. [ ] No [?] broken references
3. [ ] All figures/tables referenced in text
4. [ ] Abstract ≤200 words with quantitative results
5. [ ] Author names and affiliations complete
6. [ ] Acknowledgments (if any) included
7. [ ] References formatted consistently
8. [ ] Page count within limit
