---
name: github-pr-workflow
description: Git branching, commit hygiene, pull request checklist, and pre-merge verification workflow for AuditLab. Use when preparing branches, commits, reviews, or documentation updates.
---

# GitHub & PR Engineering Workflow

## When to Use
- Creating a feature, test, fix, or documentation branch.
- Preparing commits and Pull Requests for review.
- Updating `PROJECT_STATUS.md`, `DECISIONS.md`, or `IMPLEMENTATION_PLAN.md` after a milestone.

## Branch & Commit Conventions
- **Branch naming**:
  - `feat/<short-topic>` — new environment, trainer, storage, or Web UI capability.
  - `fix/<short-topic>` — bug fixes or security hardening.
  - `test/<short-topic>` — new test suites or regression coverage.
  - `docs/<short-topic>` — architecture, ADRs, skills, or README updates.
- **Commit style**: concise imperative subject line (`feat: add stdlib RL trainer and SQLite store`, `docs: add architecture and ADRs`). Keep commits scoped to one logical change.
- **Tracked vs Ignored Artifacts**:
  - Tracked: `audit_rl/`, `configs/`, `data/train/`, `data/dev/`, `trajectories/trajectories.jsonl`, `tests/`, `.agents/skills/`, root docs.
  - Ignored: `__pycache__/`, `*.pyc`, `.venv/`, `*.egg-info/`, `.DS_Store`.

## Workflow
1. Check current branch and working tree cleanliness:
   ```bash
   git status -s && git branch -a
   ```
2. Run full local verification before committing or opening a PR:
   ```bash
   python3 -m unittest discover -s tests -v
   python3 -m audit_rl verify
   ```
3. Run security & branding hygiene tests:
   ```bash
   python3 -m unittest tests/test_branding_and_security.py -v
   ```
4. Update `PROJECT_STATUS.md` if branch status, test counts, or milestone deliverables changed.

## PR Verification Checklist
- [ ] `python3 -m unittest discover -s tests -v` passes 100%.
- [ ] `python3 -m audit_rl verify` exits 0 with 100.0/100.0 score.
- [ ] Zero external runtime dependencies added (`pyproject.toml` `dependencies = []`).
- [ ] Docs (`README.md`, `ARCHITECTURE.md`, `PROJECT_STATUS.md`) match actual code and measured metrics.
- [ ] No secrets, private eval answers, or temporary scratch files staged.
