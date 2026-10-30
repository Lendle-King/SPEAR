# SPEAR manuscript (initial draft)

**Working title:** *SPEAR: Learning What Evidence to Seek for Parameter-Level Self-Improvement in LLM Agents*

**Draft date:** 2026-10-30  
**Status:** Method formulation and prespecified experiment design, **not an empirical paper yet**.

## Contents

- \`main.tex\`: two-column, portable LaTeX entrypoint, abstract and bibliography plumbing.
- \`sections/01_introduction.tex\`: motivation, research questions, contributions and scope.
- \`sections/02_related.tex\`: comparisons to SEAL, SIA, SkillSmith, KV-Skill, VOI routing, and update learning.
- \`sections/03_method.tex\`: PI-VOI objective, intervention generator, paired verification, recursive learning, algorithm and diagram.
- \`sections/04_evaluation.tex\`: proposed RQs, environments, cost-matched baselines, metrics, falsification gates.
- \`sections/05_discussion.tex\`: limitations, threats to validity, conclusion.
- \`references.bib\`: verified primary references, checked 2026-10-30.
- \`EXPERIMENTS.md\`: implementation-oriented pre-registration and audit schema.
- \`RELATED_WORK.md\`: related-work comparison and acronym collision warning.

## Build

Requires a standard TeX Live installation with \`latexmk\`, \`pdflatex\`, and common packages (\`natbib\`, \`algorithm2e\`, \`tikz\`, \`geometry\`).

\`\`\`bash
cd paper
make
# PDF: main.pdf
\`\`\`

Equivalent manual build:

\`\`\`bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
\`\`\`

If a venue template is selected, adapt the \`main.tex\` preamble and keep the scientific section files.

## Research integrity and edit boundaries

1. No experiment has been run or imported into this manuscript. It deliberately reports **zero SPEAR result numbers**.
2. The PI-VOI expression is a proposed decision-specific instantiation of classical VOI, not claimed as a new information-theoretic identity.
3. Parameter/skill generators and recursive self-editing already exist in related work; the proposed novelty is an **experimentally falsifiable coupling** between evidence source choice and durable parameter modification.
4. Before any public release, resolve the existing acronym conflict documented in \`RELATED_WORK.md\` and replace anonymous authorship placeholders with confirmed author information.
5. Only add a Results section after linking raw runs, checker validation, budgets, and the analysis protocol.
