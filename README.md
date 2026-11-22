# GenesisHackathon · FinTech

Симулятор финансового аудита на вымышленных данных. Политика читает открытые счета, платежи, проводки и курсы, формулирует вывод и прикладывает ссылки на доказательства. Оценщик сверяет вывод с эталоном и рассчитывает награду. Среда подходит для проверки политик и записи траекторий RL.

## Быстрый старт

Требуется Python 3.11+. Выполняйте команды из корня репозитория. Установка зависимостей может потребовать доступа к реестру либо заранее подготовленных wheel-файлов. Все зависимости проверок зафиксированы в [тестовом lock-файле](tests/requirements-schema.lock); метаданные пакета и консольный скрипт `genesis` описаны в [pyproject.toml](pyproject.toml).

Linux / macOS:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --no-deps -r tests/requirements-schema.lock
python -m pip check
# Вариант 1: запуск без установки пакета
PYTHONPATH=src python -m genesis.cli verify --all
# Вариант 2: editable-установка пакета (добавляет команду `genesis`)
python -m pip install --no-deps --no-build-isolation -e .
genesis verify --all
```

Windows PowerShell. Если Python Launcher отсутствует, замените первую команду на `python -m venv .venv` с установленным Python 3.11+.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --no-deps -r tests/requirements-schema.lock
.venv\Scripts\python.exe -m pip check
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m genesis.cli verify --all
```

Код выхода 0 означает успешную проверку обязательных групп, код 1 — сбой. При успехе в сводке есть `PASS replay-and-permutation: replay 20/20, pairs 20/20` и `PASS policy-diagnostics: 80/80 local runs`. Флаг `--all` включён по умолчанию. Пропущенный `docker-policy-smoke` отмечается отдельно; чтобы сделать его обязательным на доверенном хосте с подготовленным образом, используйте `--require-docker`. [Подготовка образов изоляции](docs/OFFLINE_RUN.md#системная-изоляция-политики)

Обычная установка через `python -m pip install ".[verify]"` включает данные, схемы и тесты. После установки `genesis verify --all` и `genesis run-policy` можно запускать из любой папки.

## Демо работающей среды

Чтобы выполнить один синтетический эпизод от старта до оценки с пошаговым выводом действий, наблюдений, выводов и метрик:

```sh
PYTHONPATH=src python -m genesis.cli demo
PYTHONPATH=src python -m genesis.cli demo --policy oracle --seed 20260304
```

Поддерживаются политики `simple_reconciliation` (по умолчанию), `random`, `mark_all` и доверенный `oracle`, а также параметр `--seed` (по умолчанию `20260304`). При установке пакета через `pyproject.toml` можно вызывать `genesis demo`.

## Проверка в контейнере

Сборка может обращаться к Docker Hub и PyPI (для полностью автономной сборки из локального каталога wheel-файлов поддерживается `--build-arg PIP_FIND_LINKS=...`). Подготовленный образ выполняет проверку без сети в пределах 2 CPU и 4 ГиБ памяти.

```sh
docker build --tag genesis-verify:local .
docker run --rm --pull never --network none --cpus 2 --memory 4g --memory-swap 4g \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 64 \
  genesis-verify:local python -m genesis.cli verify --all
```

Для запуска встроенных базовых политик в изолированном контейнере без доступа к оценщику предусмотрен отдельный [Dockerfile.policy](Dockerfile.policy):

```sh
docker build -f Dockerfile.policy --tag genesis-policy:local .
```

Перенос образов на офлайн-машину и запуск проверок изоляции внешней политики описаны отдельно. [Автономная проверка](docs/OFFLINE_RUN.md)

В Linux CI (`offline-verify.yml`) автоматически проверяются размер образа (лимит <= 1 000 000 000 байт), отключение сети (`--network none`), 2 CPU, 4 ГиБ RAM, read-only rootfs, UID/GID `65532:65532`, прохождение всех групп проверки и код выхода 1 на контрольном повреждённом JSON. Отдельный workflow `policy-isolation.yml` собирает `Dockerfile.policy` и smoke-образ и запускает `python -m genesis.cli verify --all --require-docker`.

## Диагностика политик

Для локального сравнения четырёх базовых политик:

```sh
PYTHONPATH=src python -m genesis.evaluation.reproducibility benchmark --profiles benchmarks/local_eval_profiles.json
```

Команда печатает JSON с числом эпизодов, средними P, R, F1, возвратом, шагами и отдельно измеренным временем. Условия, цифры и наблюдаемые сбои описаны в [записке о дизайне](docs/DESIGN_NOTE.md#локальные-измерения).

## Данные и траектории

Для показа полного закрытого цикла сверки в контейнере выполните [закрытый прогон базовой политики (`run-policy`)](docs/BASELINE_POLICIES.md#запуск-и-контейнер-политики): он сохраняет публичный журнал и печатает итоговые метрики после удаления контейнера.

- [10 train](data/train/) и [10 dev](data/dev/) (`v1.1.0`) содержат открытые артефакты синтетических эпизодов (от компактных 16-строчных до масштабированных книг с 220+ проводками и 100+ документами, 5 валют и 4 часовых пояса IANA). [Индексы конфигураций](data/configs/), [SHA-256](data/checksums.json) и [команды регенерации](data/README.md) включены в репозиторий. Эталон и provenance для этих наборов воспроизводятся в доверенном контуре из открытых seed. Официальные закрытые seed жюри хранятся отдельно.
- [Записанная траектория](trajectories/trajectories.jsonl) получена реальным запуском `simple_reconciliation` на масштабированном 60-обязательном эпизоде `data/train` (`episode-ce7be9493ab47ff2dd4555c4`, `seed=20260110`, 11 шагов, постраничное чтение с курсорами, все 4 типа ошибок и оба сценария взаимодействия). [Формат и replay](docs/TRAJECTORIES.md) описывают публичные поля журнала и границу закрытой диагностики. В `tests/fixtures/contract/` лежит отдельный рукотворный пример формата.
- [20 профилей локальной диагностики](benchmarks/local_eval_profiles.json) используются для сравнения политик и парных проверок (включая проверку инвариантности к перестановке ID для Oracle и `simple_reconciliation`). Результаты официальной проверки жюри определяются на закрытых эпизодах.

## Как устроен проект

| Где | Что внутри |
| --- | --- |
| [`src/genesis/generation/`](src/genesis/generation/) | История компании, открытые документы, инъекции ошибок и причинные изменения. |
| [`src/genesis/environment/`](src/genesis/environment/) | `reset`, `step`, публичные инструменты, бюджет и изоляция процесса политики. |
| [`src/genesis/evaluation/`](src/genesis/evaluation/) | Сопоставление, метрики, награда, координатор прогона и доверенный Oracle. |
| [`src/genesis/policies/`](src/genesis/policies/) | Случайный исследователь, базовая сверка по четырём типам ошибок и контрольная политика «пометить всё». |
| [`src/genesis/trajectories/`](src/genesis/trajectories/) и [`src/genesis/cli/`](src/genesis/cli/) | Запись публичных наблюдений, валидация траекторий, команды `verify`, `demo` и `run-policy`. |
| [`schemas/`](schemas/), [`tests/`](tests/), [`pyproject.toml`](pyproject.toml), [`Dockerfile.policy`](Dockerfile.policy) | Контракты данных, проверки среды/оценщика/воспроизводимости, метаданные пакета и образ воркера политики. |

Начните с [записки о дизайне](docs/DESIGN_NOTE.md), [контракта данных](docs/DATA_CONTRACT.md) и [ограничений](docs/LIMITATIONS.md). Для работы команды используйте [CONTRIBUTING.md](CONTRIBUTING.md) и [план в GitHub](https://github.com/24kBlood/GenesisHackathon/issues/33).

## Атрибуция и сдача

Исходный код проекта распространяется по [лицензии MIT](LICENSE). Использованные зависимости и AI-инструменты перечислены в [NOTICE](NOTICE).

Дедлайн по положению — **22 ноября 2026, 23:59 по времени Астаны**. Капитан подаёт ссылку на единый документ Google Drive с доступными жюри ссылками на исходники, видеодемонстрацию до трёх минут и презентацию. Также требуется комплект артефактов из ТЗ FinTech. [Чек-лист сдачи](https://github.com/24kBlood/GenesisHackathon/issues/27) · [Сайт Genesis](https://genesis-exe.com/)
