# `audit-rl` — FinTech RL-среда для дообучения и оценки моделей финансового аудита

**Genesis Hackathon — FinTech Track: «RL-среды для дообучения моделей финансового аудита»**

`audit-rl` — это детерминированная, верифицируемая RL-среда (Gymnasium-совместимый JSON-контракт) на **чистом Python 3.11+ stdlib** (ноль внешних runtime-зависимостей, 100% offline, CPU-only). Среда генерирует реалистичные синтетические бухгалтерские книги компаний с предпроверенными балансовыми инвариантами, внедряет причинно связанные финансовые расхождения и оценивает агентов в изолированном процессе по открытому 1-to-1 матчер-протоколу.

---

## Ключевые особенности

- **100% pure Python 3.11+ stdlib & Offline-Ready**: никаких сторонних библиотек, без сети и без GPU; работает в контейнере с лимитом 2 CPU / 4 GB RAM.
- **Строгая изоляция оценивания**: `AuditEnv(mode="eval")` скрывает награду и эталон из `observation`/`info`, а `SubprocessAuditEnv` + `audit_rl/eval_worker.py` выносит оценивание, `truth.json`, `provenance.json` и `seed` в отдельный процесс (JSON-lines IPC).
- **Причинная генерация + бухгалтерские инварианты**: чистая книга проходит проверку `sum(debit_minor) == sum(credit_minor)` до внедрения ошибок; включает реалистичные дистракторы (split payments, возвраты, регулярные платежи с одинаковыми суммами, мультивалютные проводки, пограничные timestamp).
- **4 типа ошибок + 2 взаимодействующих сценария**:
  - `duplicate_invoice`
  - `payment_allocation_mismatch`
  - `fx_conversion_error`
  - `period_cutoff_error`
  - Взаимодействующие: `fx_period_partial_payment` и `duplicate_invoice_fx_mismatch`.
- **Узкий инструментальный интерфейс**: `list_records` (лимит 50/страницу), `get_record`, `get_document`, безопасный AST `Decimal` `calculator`, `submit` (бюджет 100 шагов).
- **Открытый 1-to-1 двудольный матчер и устойчивая награда**: `R = clip(2 * F1 - 1.0 - 0.002 * inspection_steps, -1.0, 1.0)`, устойчивая к flag-everything спаму, дубликатам и ложным обвинениям на чистых книгах.
- **Инвариантность к перестановкам и переименованию ID**: 20 парных проверок `permute_and_relabel_episode`.

---

## Структура репозитория

```text
.
├── audit_rl/
│   ├── __init__.py          # Публичный API пакета
│   ├── __main__.py          # Entrypoint `python3 -m audit_rl`
│   ├── schemas.py           # Константы, EpisodeConfig, валидация submit, сериализация
│   ├── generator.py         # Причинный генератор, verify_episode_integrity, permute_and_relabel_episode
│   ├── calculator.py        # Безопасный AST Decimal калькулятор (без eval)
│   ├── matcher.py           # Нормализация, 1-to-1 двудольный матчер (Kuhn), терминальная награда
│   ├── env.py               # AuditEnv (train/eval) и SubprocessAuditEnv (IPC-изоляция)
│   ├── eval_worker.py       # Изолированный дочерний процесс оценщика
│   ├── policies.py          # 4 бейзлайн-политики + запись траекторий
│   └── cli.py               # Команды verify, generate-datasets, run-baselines, demo
├── configs/
│   ├── train_configs.json   # 10 открытых train-эпизодов (seeds 1001..1010)
│   ├── dev_configs.json     # 10 открытых dev-эпизодов (seeds 2001..2010)
│   └── eval_configs.json    # 20 held-out eval-эпизодов (seeds 9001..9020)
├── data/
│   ├── train/               # Сгенерированные train-эпизоды
│   └── dev/                 # Сгенерированные dev-эпизоды
├── trajectories/
│   └── trajectories.jsonl   # Записанные траектории без утечек truth/provenance/seed
├── tests/                   # Полный набор unittest-тестов (pure stdlib)
├── DESIGN.md                # Технический дизайн-документ
├── DEMO.md                  # Пошаговый сценарий демонстрации и разбор эпизода
├── Dockerfile               # Offline CPU-only образ на python:3.11-slim
├── pyproject.toml           # PEP 621 метаданные и консольный скрипт `audit-rl`
├── requirements.txt         # Зафиксированное отсутствие внешних runtime-зависимостей
└── LICENSE                  # MIT License
```

---

## Быстрый старт (Quickstart)

### 1. Полная проверка одной командой (Single-Command Verification)

```bash
python3 -m audit_rl verify
```

Команда автоматически выполняет:
1. Полный набор unit- и integration-тестов (`python3 -m unittest discover -s tests -v`).
2. Независимую проверку бухгалтерской и доказательной целостности (`verify_episode_integrity`) на 20 eval-эпизодах.
3. 20 парных проверок инвариантности к перестановкам строк и переименованию ID (`permute_and_relabel_episode`).
4. Сканирование `trajectories/trajectories.jsonl` на отсутствие утечек `truth`, `provenance`, `seed` и наличие обязательных полей.
5. Прогон всех 4 бейзлайн-политик через изолированный `SubprocessAuditEnv(mode="eval")` на 20 eval-эпизодах с выводом сводной таблицы.

### 2. Отдельные CLI-команды

```bash
# Сгенерировать открытые датасеты data/train/, data/dev/ и trajectories/trajectories.jsonl
python3 -m audit_rl generate-datasets

# Запустить все 4 бейзлайн-политики на 20 held-out eval-эпизодах (в изолированном SubprocessAuditEnv)
python3 -m audit_rl run-baselines

# Интерактивный текстовый разбор одного эпизода (reset -> tools -> calculator -> submit -> оценка)
python3 -m audit_rl demo

# Запуск только тестов
python3 -m unittest discover -s tests -v
```

### 3. Запуск в Docker (100% offline, 2 CPU / 4 GB RAM, без GPU)

```bash
# Сборка образа
docker build -t audit-rl:latest .

# Запуск полной верификации без доступа к сети
docker run --rm --network=none --cpus=2 --memory=4g audit-rl:latest

# Запуск демонстрации эпизода в контейнере
docker run --rm --network=none --cpus=2 --memory=4g audit-rl:latest python -m audit_rl demo
```

---

## Формат датасетов и траекторий

### Открытые данные эпизода (`data/train/<episode_id>/`, `data/dev/<episode_id>/`)
- `manifest.json` — параметры периода (`period_start`, `period_end`, `timezone_offset`), `base_currency`, `currencies`, `schema_version`.
- `policy.json` — учетная политика (`fx_date_policy`, `rounding_rule`).
- `ledger.csv` — главная книга (`row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id`).
- `invoices.jsonl` — инвойсы с построчной детализацией `line_items` и `service_date`.
- `payments.jsonl` — платежи с распределением `invoice_refs` (или явным `null`).
- `fx.csv` — таблица валютных курсов (`date,source_currency,target_currency,rate` как десятичная строка).

### Траектории (`trajectories/trajectories.jsonl`)
Каждая строка — JSON-объект одного шага эпизода:
```json
{
  "episode_id": "ep_...",
  "policy": "SimpleReconciliationPolicy",
  "step": 1,
  "action": {"action": "list_records", "table": "invoices", "cursor": "0", "limit": 50},
  "observation": {"records": [...], "next_cursor": null, "budget_remaining": 99},
  "reward": 0.0,
  "terminated": false,
  "truncated": false
}
```
Траектории проходят автоматический leak-scan: в них гарантированно отсутствуют `truth`, `provenance`, `seed` и локальные пути файловой системы.

---

## Результаты верификации и бейзлайнов на 20 Held-Out Eval-эпизодах

Полный прогон `python3 -m audit_rl verify` завершается со статусом **Overall PASS (2.74 s)**:
- **Unit & Integration Tests** (`python3 -m unittest discover -s tests -v`): **61 / 61 OK (1.22 s)**
- **Eval Integrity Checks**: **20 / 20 PASS** (баланс до инъекции + независимый пересчет каждого расхождения из открытых данных)
- **Permutation & ID-Remapping Invariance**: **20 / 20 PASS** (перестановка строк и переименование всех ID дают идентичный `Oracle F1 = 1.0`, `Return = 1.0`)
- **Trajectory Leak Scan**: **PASS** (`trajectories/trajectories.jsonl`: 30 прогонов, 341 шаг, **0 утечек** `truth`/`provenance`/`seed`)

Оценка 4 бейзлайнов в изолированном `SubprocessAuditEnv(mode="eval")` по конфигу [configs/eval_configs.json](configs/eval_configs.json) (20 эпизодов: 5 чистых, 5 со взаимодействующими ошибками, 10 с одиночными/множественными ошибками):

| Policy | Mean Precision | Mean Recall | Mean F1 | Mean Return | Mean Steps | Mean Runtime (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `OraclePolicy` (`oracle`, evaluator-only ceiling) | `1.0000` | `1.0000` | `1.0000` | `+1.0000` | `1.00` | `0.14` |
| `SimpleReconciliationPolicy` (`simple_reconciliation`) | `1.0000` | `1.0000` | `1.0000` | `+0.9676` | `17.20` | `4.98` |
| `RandomExplorerPolicy` (`random_explorer`) | `0.1000` | `0.1000` | `0.1000` | `-0.8019` | `12.60` | `1.78` |
| `FlagEverythingAttackPolicy` (`flag_everything_attack`) | `0.0017` | `0.0250` | `0.0031` | `-0.9940` | `4.00` | `1.71` |

### Краткий анализ провалов и успехов (Measured Failure Analysis)
- **`FlagEverythingAttackPolicy` (`F1 = 0.0031`, `Return = -0.9940`)**: массовая подача до 30 гипотетических находок на эпизод уничтожает `Precision` (`0.0017`), не удовлетворяет строгим проверкам `affected_ids`, точной суммы в `minor` и подмножества доказательств `acceptable_evidence_sets`, а на 5 чистых книгах получает штраф `-1.0`. Это подтверждает устойчивость награды к specification gaming.
- **`RandomExplorerPolicy` (`F1 = 0.1000`, `Return = -0.8019`)**: случайное исследование набирает `F1 = 1.0` только тогда, когда случайно отправляет пустой отчет на чистой книге; на всех ошибочных книгах (и при случайных ложных обвинениях) не находит корректных пар доказательств и расхождений.
- **`SimpleReconciliationPolicy` (`F1 = 1.0000`, `Return = +0.9676`, `17.20 steps`)**: последовательно пагинирует открытые таблицы через узкий инструментальный интерфейс, проверяет 4 класса расхождений и формирует минимальные валидные `evidence_ids`, не попадаясь на чистые дистракторы. Разница с `OraclePolicy` (`+0.9676` против `+1.0000`) в точности отражает штраф экономии шагов (`0.002 * 16.2 ≈ 0.0324` за ~16.2 шагов инспекции).
- **`OraclePolicy` (`F1 = 1.0000`, `Return = +1.0000`, `1.00 step`)**: подтверждает достижимость верхней границы награды при 0 шагах инспекции и корректность 1-to-1 матчера.

> **Честные ограничения:** синтетический скоуп (4 класса расхождений и 2 взаимодействующих сценария), не заменяет профессиональный аудит, `SimpleReconciliationPolicy` — конечный rule-based бейзлайн под известную схему таблиц, реальные финансовые данные и PII не используются. Подробнее — в [DESIGN.md](DESIGN.md).

---

## Соответствие критериям оценки (Rubric Mapping)

| Критерий | Реализация в `audit-rl` |
|---|---|
| **Контракт RL-среды и изоляция** | `AuditEnv` (`train`/`eval`) + межпроцессная изоляция `SubprocessAuditEnv` / `eval_worker.py`; 5 узких инструментов, пагинация ≤ 50, бюджет 100 шагов, безопасный AST Decimal калькулятор (`tests/test_env_contract_and_isolation.py`). |
| **Процедурный генератор и бухгалтерские инварианты** | Детерминированный генератор (`random.Random(seed)`), проверка баланса до инъекции ошибок, 4 типа ошибок, 2 взаимодействующих сценария, 5 классов чистых дистракторов, `verify_episode_integrity` (`tests/test_generator_and_invariants.py`). |
| **Открытый матчер и устойчивость награды** | 1-to-1 двудольное паросочетание (Kuhn), проверка доказательств по подмножеству, допуски, штрафы за спам/дубликаты/невалидную схему, корректное молчание на чистых книгах (`tests/test_matcher_and_reward.py`). |
| **Воспроизводимость и инвариантность** | Побитовая воспроизводимость `(seed, config, actions)` и 20 парных проверок перестановки строк и переименования всех ID (`tests/test_reproducibility_and_permutation.py`). |
| **Бейзлайны, датасеты и траектории** | 4 политики (`Oracle`, `SimpleReconciliation`, `RandomExplorer`, `FlagEverythingAttack`), версионированные конфиги `train`/`dev`/`eval`, leak-free `trajectories/trajectories.jsonl` (`tests/test_policies_and_trajectories.py`). |
| **Упаковка, документация и воспроизводимый запуск** | Zero-deps Python 3.11+ stdlib, offline `Dockerfile`, единая команда `python3 -m audit_rl verify`, [DESIGN.md](DESIGN.md) и [DEMO.md](DEMO.md). |

---

## Документация и лицензия

- Подробная архитектура, математика награды и ограничения: [DESIGN.md](DESIGN.md)
- Пошаговая демонстрация и разбор эпизода: [DEMO.md](DEMO.md)
- Лицензия: [LICENSE](LICENSE) (MIT, 2026).
