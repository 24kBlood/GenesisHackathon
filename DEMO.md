# DEMO.md — Пошаговая демонстрация `audit-rl` и разбор эпизода

Этот документ описывает быстрый сценарий живой демонстрации среды (3–5 минут) и приводит разбор типичного эпизода финансового аудита от `reset()` до финальной оценки.

---

## Сценарий демонстрации (Step-by-Step Demo Script)

### Шаг 1. Проверка нулевых зависимостей и единая верификация
Запускаем полную верификацию одной командой (работает полностью offline):

```bash
python3 -m audit_rl verify
```

Что показать жюри в выводе (измеренный прогон — **Overall PASS за 2.74 s**):
- **61 / 61** unit- и integration-тестов (`python3 -m unittest discover -s tests -v`) проходят за **1.22 s**;
- **20 / 20** held-out eval-эпизодов проходят `verify_episode_integrity` (бухгалтерский баланс до инъекции + независимый пересчет каждого расхождения из открытых файлов);
- **20 / 20** парных перестановок и переименований ID (`permute_and_relabel_episode`) дают идентичный `F1 = 1.0` и `reward = 1.0` у `OraclePolicy`;
- сканер утечек подтверждает **0 утечек** `truth`/`provenance`/`seed` в `trajectories/trajectories.jsonl` (30 прогонов, 341 шаг);
- сравнение 4 бейзлайнов в изолированном `SubprocessAuditEnv(mode="eval")`:
  - `oracle`: `P = 1.0000`, `R = 1.0000`, `F1 = 1.0000`, `Return = +1.0000` (`1.00 step`, `0.14 ms`) — верхняя граница;
  - `simple_reconciliation`: `P = 1.0000`, `R = 1.0000`, `F1 = 1.0000`, `Return = +0.9676` (`17.20 steps`, `4.98 ms`) — честная сверка через узкие инструменты;
  - `random_explorer`: `P = 0.1000`, `R = 0.1000`, `F1 = 0.1000`, `Return = -0.8019` (`12.60 steps`, `1.78 ms`) — случайно угадывает только отдельные чистые книги;
  - `flag_everything_attack`: `P = 0.0017`, `R = 0.0250`, `F1 = 0.0031`, `Return = -0.9940` (`4.00 steps`, `1.71 ms`) — спам-атака жестко наказывается функцией награды.

### Шаг 2. Живой прогон одного эпизода
Запускаем читаемый walkthrough одного эпизода:

```bash
python3 -m audit_rl demo
```

Вывод пошагово показывает:
1. Начальное наблюдение `reset()` (`manifest`, `policy`, `budget_remaining`, `available_actions` — без `truth` и `seed`).
2. Инспекцию таблиц через `list_records` с пагинацией (`limit <= 50`).
3. Чтение первичных документов через `get_document` и записей через `get_record`.
4. Точную проверку валютной конвертации через AST `Decimal` `calculator`.
5. Финальный `submit` и расчет метрик (`TP`, `FP`, `FN`, `Precision`, `Recall`, `F1`, `reward`).

### Шаг 3. Проверка offline Docker-контейнера
```bash
docker build -t audit-rl:latest .
docker run --rm --network=none --cpus=2 --memory=4g audit-rl:latest
```
Контейнер выполняет полный `verify` без сети, на CPU и в пределах 4 GB RAM.

---

## Разбор примерного эпизода (Sample Episode Walkthrough)

Ниже приведен пример взаимодействия агента с `SubprocessAuditEnv(mode="eval")` на эпизоде с ошибкой валютной конвертации (`fx_conversion_error`) и нераспределенным платежом (`payment_allocation_mismatch`).

### 1. Инициализация (`reset`)
```python
from audit_rl.env import SubprocessAuditEnv

with SubprocessAuditEnv(mode="eval", max_steps=100) as env:
    obs, info = env.reset(seed=9003, config=eval_config)
```

Агент получает только открытый контекст:
```json
{
  "task": "Audit the company financial records for the reporting period and submit all verifiable discrepancies.",
  "manifest": {
    "episode_id": "ep_9003_...",
    "schema_version": "1.0.0",
    "base_currency": "USD",
    "currencies": ["USD", "EUR", "GBP", "KZT", "JPY"],
    "timezone_offset": "+05:00",
    "period_start": "2026-01-01T00:00:00+05:00",
    "period_end": "2026-03-31T23:59:59+05:00"
  },
  "policy": {
    "fx_date_policy": "invoice_issue_date",
    "rounding_rule": "ROUND_HALF_EVEN"
  },
  "budget_remaining": 100,
  "step": 0,
  "available_actions": ["list_records", "get_record", "get_document", "calculator", "submit"]
}
```

### 2. Пагинация таблиц (`list_records`)
Агент запрашивает инвойсы и проводки страницами не более 50 записей:
```json
{"action": "list_records", "table": "invoices", "cursor": "0", "limit": 50}
```
```json
{"action": "list_records", "table": "ledger", "cursor": "0", "limit": 50}
```

### 3. Проверка первичного документа и курса FX (`get_document` + `get_record`)
Агент замечает валютный инвойс в `EUR`, получает документ и официальный курс на дату выставления (`invoice_issue_date`):
```json
{"action": "get_document", "doc_id": "DOC-INV-014"}
```
```json
{"action": "get_record", "table": "fx", "id": "2026-02-10:EUR:USD"}
```
Ответ среды возвращает курс `"1.084500"`.

### 4. Проверка суммы через безопасный AST Decimal калькулятор (`calculator`)
Агент вычисляет ожидаемую сумму в `USD` (в центах, с округлением `ROUND_HALF_EVEN`) и сравнивает с проведенной суммой в `ledger.csv`:
```json
{"action": "calculator", "expression": "abs(124500 - round(112000 * 1.084500))"}
```
Результат калькулятора: `"3036"` (расхождение 30.36 USD — превышает `tolerance_minor = 1`).

> Если агент попытается передать небезопасное выражение (например, `__import__('os')` или `2 ** 10000`), калькулятор вернет контролируемую ошибку без выполнения кода и спишет 1 шаг бюджета.

### 5. Отправка отчета (`submit`)
Агент завершает эпизод, прикладывая конкретные доказательства (`evidence_ids`):
```json
{
  "action": "submit",
  "findings": {
    "episode_id": "ep_9003_...",
    "findings": [
      {
        "type": "fx_conversion_error",
        "affected_ids": ["INV-014"],
        "claimed_amount_minor": 3036,
        "currency": "USD",
        "evidence_ids": ["DOC-INV-014", "LED-0087", "2026-02-10:EUR:USD"],
        "explanation": "Invoice INV-014 (1120.00 EUR) posted as 1245.00 USD instead of 1214.64 USD at 2026-02-10 EUR/USD rate 1.084500."
      }
    ]
  }
}
```

В `mode="eval"` метод `step()` возвращает `reward = 0.0, terminated = True, truncated = False` (агент не видит эталон).

### 6. Получение отчета оценщиком (`get_evaluator_report`)
После завершения эпизода оценщик запрашивает финальный отчет:
```python
report = env.get_evaluator_report()
```
Пример отчета:
```json
{
  "valid_submission": true,
  "tp": 1,
  "fp": 0,
  "fn": 0,
  "precision": 1.0,
  "recall": 1.0,
  "f1": 1.0,
  "inspection_steps": 5,
  "reward": 0.99,
  "terminated": true,
  "truncated": false
}
```
Здесь `reward = 2 * 1.0 - 1.0 - 0.002 * 5 = 0.99`.
