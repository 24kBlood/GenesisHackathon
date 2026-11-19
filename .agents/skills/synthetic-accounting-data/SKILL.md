---
name: synthetic-accounting-data
description: Procedures for designing, generating, and validating causal synthetic accounting books, pre-injection double-entry invariants, distractors, error types, and permutation invariance in AuditLab. Use when modifying audit_rl/schemas.py, audit_rl/generator.py, or configs/.
---

# Synthetic Accounting Data Engineering

## When to Use
- Updating `EpisodeConfig`, `GeneratedEpisode`, or file serialization in `audit_rl/schemas.py`.
- Adding or adjusting accounting error types, interacting scenarios, or clean distractors in `audit_rl/generator.py`.
- Creating or editing dataset split configs (`configs/train_configs.json`, `dev_configs.json`, `val_configs.json`, `eval_configs.json`).

## Data & Accounting Rules
1. **Strict Determinism**: Use only a local `random.Random(seed)` instance; never touch global `random` or wall-clock `time`.
2. **Exact Value Formats**:
   - Timestamps: ISO-8601 with explicit timezone offset (e.g., `2026-01-15T14:30:00+05:00`).
   - Money: integer minor units (`*_minor: int`, never `float` or `bool`), governed by `CURRENCY_EXPONENTS` (`USD/EUR/GBP/KZT/CHF: 2`, `JPY: 0`).
   - FX rates: exact decimal strings (e.g., `"1.084500"`), evaluated with `decimal.Decimal` and `policy["rounding_rule"]`.
   - Missing values: explicit `null` in JSON/JSONL, empty string `""` in CSV parsed back to `None`.
3. **Pre-Injection Invariants**: Before injecting errors, verify `sum(debit_minor) == sum(credit_minor)` in `base_currency` and full invoice/payment reconciliation; record checks in `provenance["pre_injection_checks"]`.
4. **Clean Distractors**: Clean and erroneous books must include realistic valid patterns (split payments, refunds/credit notes, legitimate recurring identical amounts with distinct `service_date`, valid multi-currency conversions, valid period-boundary timestamps) that produce zero truth issues.
5. **Verifiable Ground Truth**: Every entry in `truth["issues"]` must contain `issue_id`, `type`, `affected_ids`, `expected_amount_minor`, `currency`, `tolerance_minor`, `acceptable_evidence_sets` (referencing only existing open IDs), and `description`.
6. **Split Hygiene**: Keep seeds strictly disjoint across `train`, `dev`, `val`, and `eval`.

## Workflow
1. Edit `audit_rl/schemas.py` or `audit_rl/generator.py` preserving `SCHEMA_VERSION` compatibility.
2. Regenerate open datasets if schemas or train/dev configs changed:
   ```bash
   python3 -m audit_rl generate-datasets
   ```
3. Run generator, invariant, and permutation tests:
   ```bash
   python3 -m unittest tests/test_generator_and_invariants.py tests/test_reproducibility_and_permutation.py -v
   ```
4. Verify all 20 eval episodes pass `verify_episode_integrity` and 20 paired `permute_and_relabel_episode` checks via:
   ```bash
   python3 -m audit_rl verify
   ```

## Verification Checklist
- [ ] Ledger row counts stay within 200–2000 and document counts within 20–200.
- [ ] Clean episodes have 0 truth issues while containing all 5 distractor categories.
- [ ] Every truth issue is independently recalculable from open files alone.
- [ ] `permute_and_relabel_episode` remaps all IDs consistently and passes `verify_episode_integrity`.
- [ ] `train`/`dev`/`val`/`eval` config seeds have zero overlap.
