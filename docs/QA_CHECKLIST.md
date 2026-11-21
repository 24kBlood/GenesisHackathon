# Локальная матрица Genesis FinTech

Матрица команды покрывает контракт, изоляцию, сопоставление и награду на синтетических данных. Официальный закрытый прогон оценивает жюри.

Из корня репозитория на Python 3.11 с установленным `tests/requirements-schema.lock`:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v -k test_C
PYTHONPATH=src python -m unittest discover -s tests -v -k test_E
```

В PowerShell задайте `$env:PYTHONPATH = 'src'` перед теми же командами без префикса `PYTHONPATH=src`. ID входит в имя теста, поэтому для повтора отдельной проверки задайте `-k test_C01` или `-k test_E01`. Один ID обозначает один тест. Вложенные варианты приводятся в диагностике unittest.

## Контракт и изоляция

| ID | Имя теста | Ожидаемое и фактически проверенное поведение | Результат |
| --- | --- | --- | --- |
| C01 | `test_C01_reset_shape` | Пара, наблюдение проходит схему, бюджет 100. | Прошёл |
| C02 | `test_C02_step_shape` | Пять значений, логические флаги, сериализация JSON. | Прошёл |
| C03 | `test_C03_reset_clears_budget_terminal_and_cursors` | После reset итог пуст, бюджет 100, старый курсор отклоняется. | Прошёл |
| C04 | `test_C04_same_public_inputs_replay` | Публичные переходы совпадают при повторе входов и часов. | Прошёл |
| C05 | `test_C05_maximum_page_size` | Из 51 записи при лимите 50 выдаётся ровно 50 и курсор. | Прошёл |
| C06 | `test_C06_pages_cover_rows_in_stable_order` | Повтор первой страницы стабилен. Страницы по 50, 50 и 23 восстанавливают таблицу в точности. | Прошёл |
| C07 | `test_C07_empty_table_invalid_limits_and_nontransferable_cursors` | Чужой, неизвестный и устаревший курсоры отклоняются с ограниченным ответом. Неверные запросы расходуют шаг. | Прошёл |
| C08 | `test_C08_table_and_limit_validation` | Неизвестная таблица и лимит 51 дают `invalid_argument`. | Прошёл |
| C09 | `test_C09_index_covers_each_table_docs_and_evidence_catalog` | Известный ID даёт одну запись. Неизвестный ID расходует шаг и даёт `not_found`. | Прошёл |
| C10 | `test_C10_only_public_documents_are_addressable` | Политика и счёт доступны по ID. Неопубликованный документ недоступен. | Прошёл |
| C11 | `test_C11_calculator_public_decimal_and_grammar` | Два выражения дают строку `0.3`. Ввод вне арифметической грамматики отклонён. | Прошёл |
| C12 | `test_C12_unrecognized_action_and_wrong_types_are_bounded` | Неизвестный тип действия и неверный тип аргумента дают фиксированные коды. | Прошёл |
| C13 | `test_C13_final_step_accepts_submit` | После 99 шагов остаётся один. Submit на шаге 100 принимается. | Прошёл |
| C14 | `test_C14_budget_exhaustion_scores_empty_and_ignores_late_submit` | Сотое исследование приводит к усечению и пустой подаче. | Прошёл |
| C15 | `test_C15_exact_deadline_precedes_action_and_does_not_debit` | Управляемые часы достигают дедлайна. Действие не выполняется и шаг не списывается. | Прошёл |
| C16 | `test_C16_late_requests_preserve_the_completed_episode` | После завершения следующий вызов сохраняет финальные наблюдение, бюджет и итог. У координатора обмен прекращается после финального кадра. | Прошёл |
| C17 | `test_C17_transport_accepts_valid_frame_and_bounds_failures` | UTF-8 подача в допустимом размере принимается. Переполнение, неполный кадр, EOF и молчание обрабатываются усечением. | Прошёл |
| C18 | `test_C18_real_container_configuration_and_lifecycle` | Реальный контейнер запускается с отдельным UID, без сети и монтирований, принимает JSONL и удаляется. | Пропущен локально, тест требует Docker |
| C19 | `test_C19_identical_closed_wire_bytes_different_private_scores` | Байты ответов совпадают для разных эталонов. Политика получает 0, закрытая оценка различается. | Прошёл |
| C20 | `test_C20_private_synthetic_metadata_stays_out_of_public_projections` | В наблюдениях отсутствуют закрытые поля и внутренний путь из синтетических меток. C19 дополнительно проверяет отсутствие stderr. | Прошёл |

Для C18 предназначен Linux workflow `Policy isolation`. Модульная проверка дополнительно подтверждает аргументы раннера. Реальный запуск исходного теста из ENV-10 прошёл в [GitHub Actions](https://github.com/24kBlood/GenesisHackathon/actions/runs/44755852402). Для повторного контейнерного прогона используйте команды из [описания изоляции](ISOLATION_AND_LEAKAGE.md).

## Оценка

| ID | Имя теста | Ожидаемое и фактически проверенное поведение | Результат |
| --- | --- | --- | --- |
| E01 | `test_E01_complete_finding_matches_exactly_once` | Полный точный вывод даёт 1 TP и награду 1 при нуле исследовательских шагов. | Прошёл |
| E02 | `test_E02_empty_anomalous_submission_misses_the_finding` | Пустой ответ на ошибочной книге даёт 1 FN и меньшую награду. | Прошёл |
| E03 | `test_E03_empty_clean_book_metrics` | Пустой ответ на чистой книге даёт P=R=0, F1=1. | Прошёл |
| E04 | `test_E04_false_clean_book_finding_loses_to_empty` | Ложный вывод на чистой книге даёт 1 FP, F1=0 и меньшую награду. | Прошёл |
| E05 | `test_E05_wrong_error_type_is_unmatched` | Неверный тип даёт 1 FP и 1 FN. | Прошёл |
| E06 | `test_E06_inexact_affected_id_sets_are_unmatched` | Угаданный, лишний и пропущенный затронутый ID ломают совпадение. | Прошёл |
| E07 | `test_E07_id_permutation_preserves_canonical_match` | Перестановка ID сохраняет совпадение и диагностический результат. | Прошёл |
| E08 | `test_E08_wrong_currency_is_unmatched_at_same_amount` | Совпадение суммы при другой валюте даёт 1 FP и 1 FN. | Прошёл |
| E09 | `test_E09_declared_tolerance_includes_both_boundaries` | При эталоне 100 и допуске 2 суммы 98 и 102 приняты. | Прошёл |
| E10 | `test_E10_boundary_excess_and_zero_tolerance_are_rejected` | Суммы 97 и 103 при допуске 2 отклонены. Нулевой и опущенный допуски требуют точности. | Прошёл |
| E11 | `test_E11_alternative_evidence_sets_work_individually` | Каждый полный альтернативный набор даёт 1 TP. | Прошёл |
| E12 | `test_E12_incomplete_known_evidence_is_unmatched` | Частичный набор известных доказательств даёт 0 TP. | Прошёл |
| E13 | `test_E13_unknown_evidence_invalidates_an_otherwise_complete_set` | Неизвестный ID дисквалифицирует утверждение с остальными полными доказательствами. | Прошёл |
| E14 | `test_E14_duplicate_and_paraphrase_add_false_positives` | Исходный вывод, дубль и перефразировка дают 1 TP и 2 FP. | Прошёл |
| E15 | `test_E15_partial_evidence_is_not_merged_across_duplicates` | Два неполных набора дают 0 TP, 2 FP, 1 FN. | Прошёл |
| E16 | `test_E16_competing_candidates_get_deterministic_maximum_matching` | Конкурирующие пары дают 2 TP с тем же сопоставлением при перестановках. | Прошёл |
| E17 | `test_E17_mark_all_submits_no_more_than_thirty_valid_false_claims` | Политика на чистой книге даёт 30 FP. P=R=F1=0, возврат −1, всего 2 шага. | Прошёл |
| E18 | `test_E18_structural_limits_and_inclusive_controls` | 31 вывод, 11 доказательств и 1 001 символ дают −1. Границы 30, 10 и 1 000 допустимы. | Прошёл |
| E19 | `test_E19_equal_results_differ_by_one_inspection_cost` | При одинаковых метриках один дополнительный шаг уменьшает награду на 0,002. | Прошёл |
| E20 | `test_E20_step_exhaustion_scores_empty_at_actual_step_count` | Усечение после 100 исследований оценивает пустой ответ. Награда 0,8 на чистой книге и −1 при ошибке. | Прошёл |

Локальный прогон Python 3.11.16 дал 19/20 проверок контракта с одним пропуском Docker и 20/20 проверок оценки. В полной проверке окружения C18 учитывается после фактического контейнерного прогона.
