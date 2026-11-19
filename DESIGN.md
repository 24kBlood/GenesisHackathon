# DESIGN.md — Техническая архитектура AuditLab (`audit-rl`)

**AuditLab** — *An autonomous financial audit simulation and AI training environment.*

## 1. Обзор архитектуры и строгая изоляция оценивания (Architecture & Out-of-Process Isolation)

`audit-rl` — детерминированная, верифицируемая RL-среда для обучения и честной оценки агентов финансового аудита, реализованная на **чистом Python 3.11+ stdlib** (нулевые внешние runtime-зависимости, 100% offline, 2 CPU / 4 GB RAM, без GPU).

```text
┌─────────────────────────────────┐      JSON-lines (stdin/stdout)      ┌────────────────────────────────────┐
│        Agent Process            │ ──────────────────────────────────▶ │   Evaluator Process (eval_worker)  │
│  SubprocessAuditEnv (Gymnasium) │                                     │                                    │
│  - reset(seed, config)          │ ◀────────────────────────────────── │  - AuditEnv(mode="eval")           │
│  - step(action)                 │   sanitized (obs, 0.0, term, trunc) │  - generator.py -> open + private  │
│  (NO truth / provenance / seed) │                                     │  - truth.json / provenance.json    │
└─────────────────────────────────┘                                     │  - matcher.py -> P, R, F1, reward  │
                                                                        └────────────────────────────────────┘
```

### Принципы изоляции
1. **Внутрипроцессное разделение (`AuditEnv`)**:
   - `mode="train"`: возвращает терминальную награду в `step()` при `terminated=True` или `truncated=True`.
   - `mode="eval"`: в `step()` **всегда** возвращает `reward = 0.0` и скрывает диагностику/эталон из `observation` и `info`. Отчет оценивания доступен исключительно оценщику через `get_evaluator_report()` после завершения эпизода.
   - При каждом `reset(seed, config)` состояние предыдущего эпизода полностью уничтожается.
   - В `observation` и `info` никогда не попадают `truth`, `provenance`, `seed`, пути файловой системы и внутренние атрибуты среды.
2. **Межпроцессная изоляция (`SubprocessAuditEnv` + `audit_rl/eval_worker.py`)**:
   - `SubprocessAuditEnv` запускает отдельный процесс `python3 -m audit_rl.eval_worker`, взаимодействующий по протоколу JSON-lines через `stdin`/`stdout`.
   - Эталонная разметка (`truth.json`), метаданные генерации (`provenance.json`) и `seed` **физически не попадают в адресное пространство процесса агента**.
   - Поддерживает контекстный менеджер (`with SubprocessAuditEnv(...) as env:`) и гарантированное закрытие рабочего процесса через `close()`.
   - **Честное уточнение модели угроз**: межпроцессная/API-изоляция защищает от утечек состояния Python-политик, но не является OS-level kernel sandbox (seccomp/VM) для запуска недоверенных нативных бинарников.

---

## 2. Причинная генерация истории компании и инварианты (Causal Company History & Accounting Invariants)

Генератор (`audit_rl/generator.py`) использует исключительно локальный `random.Random(seed)` (без глобального RNG и системного времени), что гарантирует побитовую воспроизводимость при одинаковых `(seed, config)`.

### Этап 1: Чистая бухгалтерская история (Pre-Injection Clean Ledger)
- Создаются компании/контрагенты, счета (инвойсы), платежи, курсы валют и двойные проводки в главной книге (`ledger.csv`) в диапазоне **200–2000 строк книги** и **20–200 документов** (инвойсы + платежи).
- Перед внедрением ошибок проверяются строгие бухгалтерские инварианты (результаты фиксируются в `provenance["pre_injection_checks"]`):
  - Сумма дебетов равна сумме кредитов в базовой валюте (`sum(debit_minor) == sum(credit_minor)`).
  - Каждый чистый инвойс и платеж полностью согласованы с проводками главной книги.

### Реалистичные чистые отвлекающие случаи (Clean Distractors)
Чтобы исключить ложные срабатывания примитивных эвристик, чистая история (включая `clean=True` эпизоды) содержит:
- **Split payments** — оплата одного инвойса 2–3 платежами, сумма которых точно равна `total_minor`.
- **Refunds / credit notes** — корректно отраженные возвраты и сторнирующие проводки.
- **Legitimate recurring identical amounts** — регулярные одинаковые суммы (например, аренда/SaaS) в разные месяцы с различными `service_date`.
- **Valid multi-currency transactions** — корректная конвертация по курсу из `fx.csv` и правилу округления из `policy.json`.
- **Valid boundary timestamps** — операции вблизи границы периода (например, `23:30` локального времени внутри отчетного периода), не нарушающие отсечение периода.

### Этап 2: Причинное внедрение 4 типов ошибок (4 Canonical Error Types)
Каждый эпизод содержит от 0 до 12 независимо проверяемых ошибок:
1. **`duplicate_invoice`** — повторный инвойс (другие `doc_id`/`invoice_id`, тот же контрагент, те же `service_date`/`line_items`/`total_minor`, дата выставления в пределах 0–2 дней) и дублирующие проводки в главной книге.
2. **`payment_allocation_mismatch`** — платеж с `invoice_refs: null` либо с суммой `allocated_minor != amount_minor` (или расхождение между документом платежа и проводкой Cash/AR).
3. **`fx_conversion_error`** — проводка по валютному инвойсу проведена по курсу неверной даты или смещенному курсу, нарушая `policy.json` (`fx_date_policy`), что создает верифицируемое расхождение в `base_currency` (`tolerance_minor = 1`).
4. **`period_cutoff_error`** — инвойс с `service_date` вне `[period_start, period_end]` (в часовом поясе `manifest.json`) ошибочно включен в текущий период главной книги (или наоборот).

### 2 взаимодействующих сценария (Interacting Multi-Document Scenarios)
1. **`fx_period_partial_payment`** — валютный инвойс вблизи границы периода с частичным платежом, пересекающим границу периода и проведенным по неверному FX-курсу; порождает 2 связанных ground-truth расхождения, разделяющих общие документы.
2. **`duplicate_invoice_fx_mismatch`** — дубликат валютного инвойса, проведенный по ошибочному FX-курсу; порождает 2 связанных расхождения (`duplicate_invoice` + `fx_conversion_error`), на которых ломается наивная дедупликация по сумме в базовой валюте.

### Независимая проверка целостности (`verify_episode_integrity`)
Для каждого сгенерированного эпизода функция `verify_episode_integrity(episode)` проверяет:
- выполнение бухгалтерского баланса до внедрения ошибок;
- соблюдение границ числа строк (200–2000) и документов (20–200);
- существование каждого `evidence_id` из `truth["issues"]` в открытых записях эпизода;
- независимый пересчет расхождения каждой ошибки напрямую из открытых данных.

---

## 3. Схемы открытых и приватных файлов и форматы значений (Schemas & Value Formats)

Версия схемы: `SCHEMA_VERSION = "1.0.0"`.

### Открытые файлы эпизода (доступны среде и агенту через инструменты)
| Файл | Формат | Поля / назначение |
|---|---|---|
| `manifest.json` | JSON | `episode_id`, `schema_version`, `period_start`, `period_end`, `timezone_offset`, `base_currency`, `currencies`, метаданные таблиц |
| `policy.json` | JSON | Правила учета: `fx_date_policy` (`invoice_issue_date` / `payment_date` / `posting_date`), `rounding_rule` (`ROUND_HALF_EVEN` / `ROUND_HALF_UP`), `period_cutoff_rule` |
| `ledger.csv` | CSV | `row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id` (пустая строка соответствует `None`) |
| `invoices.jsonl` | JSONL | `doc_id,invoice_id,entity_id,issued_at,due_at,currency,total_minor,line_items` (`description,quantity,unit_price_minor,amount_minor,service_date`) |
| `payments.jsonl` | JSONL | `doc_id,payment_id,timestamp,currency,amount_minor,invoice_refs` (список `{"invoice_id": str, "allocated_minor": int}` или `null`) |
| `fx.csv` | CSV | `date,source_currency,target_currency,rate` (курс как десятичная строка, ключ `"{date}:{source_currency}:{target_currency}"`) |

### Приватные файлы (доступны только оценщику)
- `truth.json` — список канонических расхождений (`issue_id`, `type`, `affected_ids`, `expected_amount_minor`, `currency`, `tolerance_minor`, `acceptable_evidence_sets`, `description`).
- `provenance.json` — параметры генерации, результаты `pre_injection_checks` и причинная трассировка внедренных ошибок.

### Строгие форматы значений
- **Временные метки**: ISO-8601 с явным смещением часового пояса (например, `2026-01-15T14:30:00+05:00`).
- **Денежные суммы**: целые числа в минимальных единицах валюты (`*_minor: int`, запрещены `float` и `bool`). Поддерживаемые экспоненты `CURRENCY_EXPONENTS`: `USD: 2, EUR: 2, GBP: 2, KZT: 2, CHF: 2, JPY: 0`.
- **Валютные курсы**: точные десятичные строки (например, `"1.084500"`), обрабатываемые через `decimal.Decimal`.
- **Отсутствующие значения**: явный `null` в JSON/JSONL и пустая строка `""` в CSV (десериализуемая обратно в `None`).

---

## 4. Узкий инструментальный интерфейс и безопасный калькулятор (Narrow Tool Interface & Safe AST Calculator)

Агент взаимодействует со средой по JSON-совместимому контракту Gymnasium (`reset` / `step`) через ровно 5 действий с бюджетом `DEFAULT_MAX_STEPS = 100` (и таймаутом `60s`):

| Действие | Параметры | Поведение и ограничения |
|---|---|---|
| `list_records` | `table`, `cursor`, `limit` | Пагинация таблиц (`ledger`, `invoices`, `payments`, `fx`). `1 <= limit <= 50` (`MAX_PAGE_LIMIT`), стабильные детерминированные курсоры (`"0"`, `"50"`, ...). |
| `get_record` | `table`, `id` | Получение одной записи по ключу (`row_id`, `invoice_id`, `payment_id`, `"{date}:{src}:{dst}"`). |
| `get_document` | `doc_id` | Получение первичного документа (инвойса или платежа) по `doc_id`. |
| `calculator` | `expression` | Безопасные десятичные вычисления через AST `Decimal`. |
| `submit` | `findings` | Финальная отправка отчета (`terminated=True`, `truncated=False`). |

### Безопасный AST Decimal калькулятор (`audit_rl/calculator.py`)
- Использует **исключительно** `ast.parse(expression, mode="eval")` (никакого `eval()`/`exec()`).
- Преобразует числовые литералы через `Decimal(str(val))`.
- Разрешены только операции `+`, `-`, `*`, `/`, `//`, `%`, унарные `+`/`-` и функции `abs`, `min`, `max`, `round` (округление `ROUND_HALF_EVEN`).
- Жестко отклоняются: возведение в степень (`**`), побитовые операции, сравнения, генераторы, `lambda`, обращения к атрибутам/индексам, любые имена вне белого списка, строки и выражения длиннее 256 символов.

---

## 5. Открытый 1-to-1 матчер и терминальная награда (Bipartite Matcher & Terminal Reward)

Логика оценивания (`audit_rl/matcher.py`) полностью открыта и воспроизводима.

### Валидация структуры `submit` (`validate_findings_payload`)
- совпадение `episode_id`, не более `MAX_FINDINGS = 30` находок;
- у каждой находки: валидный `type` из `ERROR_TYPES`, непустой `affected_ids: list[str]`, целый `claimed_amount_minor: int` (строго не `bool`), `currency: str`, `evidence_ids: list[str]` длиной до `10`, `explanation: str` до `1000` символов (в Evidence-First политиках структурировано как `Observed: ... Calculated: ... Rule: ...`).
- при нарушении схемы `valid_submission = False` и награда жестко равна `-1.0`.

### Детерминированное 1-to-1 сопоставление и награда
- Максимальное двудольное паросочетание (алгоритм Куна) в детерминированном отсортированном порядке `(issue_id, finding_index)`.
- Повторные находки засчитываются как `FP`.
- На чистых книгах (`len(truth_issues) == 0`): пустой отчет дает `F1 = 1.0`, ложные обвинения дают `F1 = 0.0`.
- Терминальная награда:

$$
R = \begin{cases}
-1.0, & \text{если } \neg \text{valid\_submission} \\
\operatorname{clip}\big(2 \cdot F_1 - 1.0 - 0.002 \cdot \text{inspection\_steps},\ -1.0,\ 1.0\big), & \text{иначе}
\end{cases}
$$

---

## 6. Инвариантность, обучение и хранение экспериментов

- **Перестановки и переименование ID**: `permute_and_relabel_episode(episode, perm_seed)` перемешивает порядок записей и согласованно переименовывает все ID в открытых таблицах, `truth.json` и `provenance.json`. На 20 парных эпизодах подтверждено сохранение `verify_episode_integrity` и `Oracle F1 = 1.0`.
- **Чистый stdlib RL-тренер (`audit_rl/trainer.py`)**: `LearnableAuditPolicy` и `run_training_experiment` оптимизируют параметры политики строго на `train`-сплите по терминальной награде и измеряют прирост на невидимых held-out эпизодах.
- **SQLite-хранилище (`audit_rl/storage.py`)**: `PlatformStore` сохраняет эксперименты, бенчмарки и версии политик в `data/platform.db` с параметризованными SQL-запросами.

---

## 7. Измеренные результаты верификации и анализ бейзлайнов

Полная проверка `python3 -m audit_rl verify` проходит за **2.74 s** (**Overall PASS**):
- **Core Unit/Integration тесты**: **61 / 61 OK** (`1.22 s`) + расширенные наборы `test_training_and_storage.py` и `test_branding_and_security.py`.
- **Целостность eval-эпизодов**: **20 / 20 PASS** (`verify_episode_integrity`).
- **Перестановки и переименование ID**: **20 / 20 PASS** (`permute_and_relabel_episode`).
- **Сканирование траекторий**: **PASS** (`trajectories/trajectories.jsonl`, 30 прогонов, 341 шаг, 0 утечек `truth`/`provenance`/`seed`).

Результаты на 20 held-out eval-эпизодах (`SubprocessAuditEnv`, `mode="eval"`):

| Policy | Precision | Recall | F1 | Return | Steps | Avg Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `oracle` | `1.0000` | `1.0000` | `1.0000` | `+1.0000` | `1.00` | `0.14 ms` |
| `simple_reconciliation` | `1.0000` | `1.0000` | `1.0000` | `+0.9676` | `17.20` | `4.98 ms` |
| `random_explorer` | `0.1000` | `0.1000` | `0.1000` | `-0.8019` | `12.60` | `1.78 ms` |
| `flag_everything_attack` | `0.0017` | `0.0250` | `0.0031` | `-0.9940` | `4.00` | `1.71 ms` |

---

## 8. Честные ограничения (Honest Limitations)

1. **Синтетическая область**: моделируется ограниченный набор бухгалтерских операций (инвойсы, платежи, FX, главная книга); среда не охватывает все стандарты МСФО/GAAP.
2. **Не является профессиональным аудиторским инструментом**: предназначена исключительно для исследований и обучения/бенчмаркинга RL- и LLM-агентов.
3. **Уровень изоляции**: `SubprocessAuditEnv` обеспечивает изоляцию процессов и API в Python, но не заменяет OS-level kernel sandbox (seccomp/VM) для враждебных бинарников.
4. **Отсутствие реальных финансовых данных и PII**: все компании, контрагенты, счета и транзакции полностью вымышлены.
