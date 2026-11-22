# Genesis FinTech · проверка для жюри

Проверенная версия реализации — [5f5d681](https://github.com/24kBlood/GenesisHackathon/commit/5f5d6816ade3122f789c3b76170dcb1ba0a2604e), включая [исправления независимого аудита](https://github.com/24kBlood/GenesisHackathon/pull/60). По этому коммиту можно повторить результаты ниже. Официальная оценка проводится отдельно на закрытых эпизодах организаторов.

## Запустить проверку

Если установлены Git и uv, выполните команду из любой папки в PowerShell или Bash. Для первоначальной загрузки пакета и зависимостей нужен интернет.

```sh
uvx --python 3.11 --from 'genesis-fintech-env[verify] @ git+https://github.com/24kBlood/GenesisHackathon.git@5f5d6816ade3122f789c3b76170dcb1ba0a2604e' genesis verify --all
```

Для воспроизведения офлайн-прогона соберите образ из клона на указанном SHA или перенесите заранее собранный образ. Запуск проходит без сети, с 2 CPU и 4 ГиБ RAM. [Сборка, перенос и точная команда Docker](../../docs/OFFLINE_RUN.md) · [Журнал офлайн-проверки](https://github.com/24kBlood/GenesisHackathon/actions/runs/44991344209)

Для проверки отдельных артефактов используйте Python 3.11 и [зависимости из lock-файла](../../tests/requirements-schema.lock). Команды ниже выполняются из корня клона. В PowerShell сначала задайте `$env:PYTHONPATH = 'src'`, затем запустите команды без префикса `PYTHONPATH=src`.

```sh
PYTHONPATH=src python -m genesis.cli demo
PYTHONPATH=src python -m genesis.generation.datasets check --dataset data
PYTHONPATH=src python -m unittest discover -s tests -p test_trajectories.py -v
PYTHONPATH=src python -m genesis.evaluation.reproducibility benchmark --profiles benchmarks/local_eval_profiles.json
```

## Как сверить критерии ТЗ

| Критерий §10 | Что проверять | Подтверждено на открытых данных |
| --- | --- | --- |
| Контракт и изоляция · 25 | [C01–C20](../../docs/QA_CHECKLIST.md#контракт-и-изоляция), [среда и транспорт](../../src/genesis/environment/) | 19 проверок контракта и C18 с реальным контейнером. Контейнер агента проверен в Linux CI, включая прогон встроенной политики. [Изоляция](https://github.com/24kBlood/GenesisHackathon/actions/runs/44991344043) |
| Данные и эталон · 25 | [Генератор](../../src/genesis/generation/), [train и dev](../../data/), [сверка доказательств](../../src/genesis/evaluation/public_proofs.py) | 10 train и 10 dev побайтно воспроизводятся. Для всех 20 книг тесты сверяют публичные доказательства и ссылки. Четыре книги попадают в рекомендованный ТЗ диапазон размеров. [Тесты данных и схем](https://github.com/24kBlood/GenesisHackathon/actions/runs/44991344097) |
| Надёжность награды · 25 | [Матрица E01–E20](../../tests/test_evaluation_matrix.py), [формула и сопоставление](../../docs/REWARD_AND_MATCHING.md) | Локальная матрица прошла 20 из 20. Прогон проверяет чистые случаи, ложные выводы, повторы и стоимость исследования. |
| Воспроизводимость и обобщение · 15 | [Профили](../../benchmarks/local_eval_profiles.json), [проверки повторов и замен ID](../../tests/test_reproducibility_and_permutation.py) | 20 из 20 локальных повторов и 20 из 20 пар с перестановкой записей и заменой ID. В парных проверках участвуют Oracle и простая сверка. |
| Инженерия и объяснение · 10 | [Запуск](../../README.md#быстрый-старт), [записка о дизайне](../../docs/DESIGN_NOTE.md), [ограничения](../../docs/LIMITATIONS.md) | 187 локальных тестов, два дополнительных теста с реальным Docker в Linux CI. Проверочный образ на этом SHA занимал 50 834 881 байт по метрике Docker Engine. При повреждённом диагностическом JSON команда завершилась кодом 1. [Журнал](https://github.com/24kBlood/GenesisHackathon/actions/runs/44991344209) |

Числа в первом столбце — максимумы из ТЗ. Приведённые прогоны показывают результаты на открытых примерах, оценки закрытого набора здесь нет.

## Артефакты §09

| Материал | Где смотреть |
| --- | --- |
| Код и зависимости | [Генератор](../../src/genesis/generation/) · [Среда](../../src/genesis/environment/) · [Lock-файл](../../tests/requirements-schema.lock) · [MIT](../../LICENSE) · [NOTICE](../../NOTICE) |
| Контейнер и команда проверки | [Dockerfile](../../Dockerfile) · [Автономный запуск](../../docs/OFFLINE_RUN.md) |
| Сопоставление, награда и схемы | [Оценщик](../../src/genesis/evaluation/) · [JSON Schema](../../schemas/) |
| Публичные эпизоды | [train](../../data/train/) · [dev](../../data/dev/) · [Контрольные суммы](../../data/checksums.json) |
| Тесты | [Каталог тестов](../../tests/) · [Матрица проверок](../../docs/QA_CHECKLIST.md) |
| Четыре базовых адаптера | [Политики](../../docs/BASELINE_POLICIES.md) · [Локальные профили](../../benchmarks/README.md) |
| Записанная траектория | [JSONL](../../trajectories/trajectories.jsonl) · [Формат, SHA-256 и replay](../../docs/TRAJECTORIES.md) |
| Записка о дизайне | [Устройство среды](../../docs/DESIGN_NOTE.md) |
| Демо | [Команда запуска](../../README.md#быстрый-старт). В коротком демо простая сверка находит одну FX-ошибку за 5 шагов, `TP=1 FP=0 FN=0`, `F1=1.0`, награда `0.992`. |

Записанная траектория относится к другому эпизоду из train. В ней 11 шагов, 8 верных выводов и награда `0.980`. Средние метрики четырёх политик получены в 80 локальных запусках на 20 компактных профилях. Все эти профили используют KZT, Asia/Almaty и курс на дату платежа. [Методика и цифры](../../docs/DESIGN_NOTE.md#локальные-измерения)

## Внешние материалы и подача

По положению капитан подаёт до **22 ноября 2026 года, 23:59 по времени Астаны** единый документ Google Drive со ссылками на код, видео до трёх минут и презентацию. Зрительские ссылки на видео, презентацию и документ, а также подтверждение подачи в репозитории пока отсутствуют. [Демо и презентация](https://github.com/24kBlood/GenesisHackathon/issues/26) · [Финальная подача](https://github.com/24kBlood/GenesisHackathon/issues/27)
