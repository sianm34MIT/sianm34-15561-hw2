# 15.561 Homework 3: Northwind analysis

Start by reading `PROGRESS.md`. It has the assignment requirements, the current state of the webpage, the findings, data decisions and next steps.

## Ground rules
- All analysis code is Python. Every number on the page must come from the scripts in `src/`.
- After any change, run `python src/run_all.py`, then `python audit/audit.py`. The audit must end with `FAILURES: none`.
- Edit the page in `site/template.html`, never in `docs/index.html` (that file is generated).
- Docstrings: use the `python-docstrings` skill (in `.claude/skills/`). Sian invokes it herself so she can describe it in her reflection. It must change docstrings only.
- The reflection report is Sian's own writing. Do not draft or edit it.
- GitHub Pages serves the site from the `docs/` folder on the `main` branch.
