# GenesisHackathon · FinTech

Симулятор финансового аудита на вымышленных данных. Политика читает открытые счета, платежи, проводки и курсы, формулирует вывод и прикладывает ссылки на доказательства. Оценщик сверяет вывод с эталоном и рассчитывает награду. Среда подходит для проверки политик и записи траекторий RL.

## Быстрый старт

Требуется Python 3.11. Выполняйте команды из корня репозитория. Установка тестовых зависимостей может потребовать доступа к реестру либо заранее подготовленных wheel-файлов. Все зависимости проверок зафиксированы в [тестовом lock-файле](tests/requirements-schema.lock).

Linux:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --no-deps -r tests/requirements-schema.lock
python -m pip check
PYTHONPATH=src python -m genesis.cli verify --all
```

Windows PowerShell. Если Python Launcher отсутствует, замените первую команду на `python -m venv .venv` с установленным Python 3.11.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --no-deps -r tests/requirements-schema.lock
.venv\Scripts\python.exe -m pip check
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m genesis.cli verify --all
```

Код выхода 0 означает успешную проверку обязательных групп, код 1 — сбой. При успехе в сводке есть `PASS replay-and-permutation: replay 20/20, pairs 20/20` и `PASS policy-diagnostics: 80/80 local runs`. Пропущенный `docker-policy-smoke` отмечается отдельно. Чтобы сделать его обязательным на доверенном хосте с подготовленным образом, используйте `--require-docker`. [Подготовка smoke-образа](docs/OFFLINE_RUN.md#системная-изоляция-политики)

## Проверка в контейнере

Сборка может обращаться к Docker Hub и PyPI. Подготовленный образ выполняет проверку без сети. Пределы контейнера — 2 CPU и 4 ГиБ памяти.

```sh
docker build --tag genesis-verify:local .
docker run --rm --pull never --network none --cpus 2 --memory 4g --memory-swap 4g \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 64 \
  genesis-verify:local python -m genesis.cli verify --all
```

Перенос образа на офлайн-машину и запуск проверок изоляции внешней политики описаны отдельно. [Автономная проверка](docs/OFFLINE_RUN.md)

В Linux CI на коммите `63a01ad` Docker Engine сообщил размер образа **50 734 338 байт**. Прогон с отключённой сетью прошёл 20 повторов, 20 пар с заменой ID и 80 запусков политик. Контрольный пример с повреждённым JSON завершился кодом 1. [Лог Offline verification](https://github.com/24kBlood/GenesisHackathon/actions/runs/44880471216/job/136774466587)

## Диагностика политик

Для локального сравнения четырёх базовых политик:

```sh
PYTHONPATH=src python -m genesis.evaluation.reproducibility benchmark --profiles benchmarks/local_eval_profiles.json
```

Команда печатает JSON с числом эпизодов, средними P, R, F1, возвратом, шагами и отдельно измеренным временем. Условия, цифры и наблюдаемые сбои описаны в [записке о дизайне](docs/DESIGN_NOTE.md#локальные-измерения).

## Данные и траектории

- [10 train](data/train/) и [10 dev](data/dev/) содержат открытые артефакты синтетических эпизодов. [Индексы конфигураций](data/configs/), [SHA-256](data/checksums.json) и [команды регенерации](data/README.md) включены в репозиторий. Эталон и provenance для этих наборов воспроизводятся в доверенном контуре из открытых seed. Официальные закрытые seed жюри хранятся отдельно.
- [Записанная траектория](trajectories/trajectories.jsonl) получена запуском простой сверки на открытом FX-эпизоде. [Формат и replay](docs/TRAJECTORIES.md) описывают публичные поля журнала и границу закрытой диагностики. В `tests/fixtures/contract/` лежит отдельный рукотворный пример формата.
- [20 профилей локальной диагностики](benchmarks/local_eval_profiles.json) используются для сравнения политик и парных проверок. Результаты официальной проверки жюри определяются на закрытых эпизодах.

## Как устроен проект

| Где | Что внутри |
| --- | --- |
| [`src/genesis/generation/`](src/genesis/generation/) | История компании, открытые документы, инъекции ошибок и причинные изменения. |
| [`src/genesis/environment/`](src/genesis/environment/) | `reset`, `step`, публичные инструменты, бюджет и изоляция процесса политики. |
| [`src/genesis/evaluation/`](src/genesis/evaluation/) | Сопоставление, метрики, награда, координатор прогона и доверенный Oracle. |
| [`src/genesis/policies/`](src/genesis/policies/) | Случайный исследователь, простая сверка и контрольная политика «пометить всё». |
| [`src/genesis/trajectories/`](src/genesis/trajectories/) и [`src/genesis/cli/`](src/genesis/cli/) | Запись публичных наблюдений, валидация траекторий и команда проверки. |
| [`schemas/`](schemas/) и [`tests/`](tests/) | Контракты данных, проверки среды, оценщика и воспроизводимости. |

Начните с [записки о дизайне](docs/DESIGN_NOTE.md), [контракта данных](docs/DATA_CONTRACT.md) и [ограничений](docs/LIMITATIONS.md). Для работы команды используйте [CONTRIBUTING.md](CONTRIBUTING.md) и [план в GitHub](https://github.com/24kBlood/GenesisHackathon/issues/33).

## Атрибуция и сдача

Исходный код проекта распространяется по [лицензии MIT](LICENSE). Использованные зависимости и AI-инструменты перечислены в [NOTICE](NOTICE).

Дедлайн по положению — **22 ноября 2026, 23:59 по времени Астаны**. Капитан подаёт ссылку на единый документ Google Drive с доступными жюри ссылками на исходники, видеодемонстрацию до трёх минут и презентацию. Также требуется комплект артефактов из ТЗ FinTech. [Чек-лист сдачи](https://github.com/24kBlood/GenesisHackathon/issues/27) · [Сайт Genesis](https://genesis-exe.com/)
