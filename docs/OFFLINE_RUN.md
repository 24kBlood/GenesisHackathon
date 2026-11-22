# Автономная проверка Genesis

## Один вызов

Локально используйте Python 3.11+ и зафиксированные зависимости из `tests/requirements-schema.lock`. Установка может обращаться к реестру; после установки проверка выполняется полностью по локальным файлам. Метаданные пакета и консольный скрипт `genesis` описаны в [`pyproject.toml`](../pyproject.toml).

```sh
python -m pip install --no-deps -r tests/requirements-schema.lock
PYTHONPATH=src python -m genesis.cli verify --all
# Или после editable-установки:
python -m pip install --no-deps --no-build-isolation -e .
genesis verify --all
```

В PowerShell задайте `$env:PYTHONPATH = 'src'` и запустите `python -m genesis.cli verify --all`. Флаг `--all` включён по умолчанию. Команда выводит статус, количество и длительность по группам, а при сбое сообщает имя первой проверки и класс ошибки (при отсутствии пакета из lock-файла — явную подсказку `ModuleNotFoundError`, а при расхождении открытого профиля — безопасный путь различия). Закрытые значения и эталон из тестов в журнал команды не попадают.

Проверяются схемы, наборы train/dev, матрицы C01–C20 и E01–E20, траектории, переходы и награда, 20 повторов и 20 пар с перестановкой ID (для Oracle и `simple_reconciliation`). Отдельно выводятся средние P, R, F1, возврат, шаги и время четырёх политик на 20 открытых диагностических профилях (результаты кэшируются внутри одного прогона без дублирования вычислений). Это диагностика команды на синтетических данных; закрытый прогон проводит жюри. [Матрица QA](QA_CHECKLIST.md).

Для наглядного пошагового запуска одного синтетического эпизода доступна команда `demo`:

```sh
PYTHONPATH=src python -m genesis.cli demo --policy simple_reconciliation --seed 20260304
```

Поддерживаются политики `simple_reconciliation`, `random`, `mark_all` и `oracle`.

## Сборка и перенос

`Dockerfile` и `Dockerfile.policy` фиксируют официальный базовый образ Python 3.11 по digest (`sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535`). При сборке сначала скачиваются зафиксированные wheel-файлы (либо используется локальный каталог через `--build-arg PIP_FIND_LINKS=...` для полностью офлайн-сборки). Финальный образ проверки содержит зависимости, исходники, `LICENSE`, `NOTICE`, `pyproject.toml` и локальные тестовые артефакты. Для переноса уже собранного образа можно использовать `docker save` и `docker load` без скачивания при запуске.

```sh
docker build --tag genesis-verify:local .
docker run --rm --pull never --network none --cpus 2 --memory 4g --memory-swap 4g \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 64 \
  genesis-verify:local python -m genesis.cli verify --all
```

Образ запускается как UID/GID 65532. Исходники и входы доступны для чтения. Тесты записывают временные данные в ограниченный `/tmp`. В CI проверяются параметры фактического контейнера и поле `.Size` образа Docker Engine в байтах с пределом 1 000 000 000. Длительности измеряются на машине запуска. Контрольный прогон подставляет нарушенный JSON диагностического профиля и проверяет код 1.

## Системная изоляция политики

Контейнер полной проверки имеет предел 2 CPU и 4 ГБ RAM. Контейнеры внешних политик проверяет отдельный workflow `Policy isolation` на доверенном хосте. Координатор запускает работника с настраиваемыми ограниченными ресурсами (по умолчанию 512 МиБ RAM, 1 CPU, 32 PID, `--no-healthcheck`, `--ulimit nofile=256:256`, `--cgroupns private`) и обменивается JSONL. Дедлайн по умолчанию составляет 60 секунд, бюджет — 100 действий. Раннер не предоставляет политике сокет Docker, монтирования, секретные переменные или сеть. [Описание изоляции](ISOLATION_AND_LEAKAGE.md).

Для контейнеризации встроенных публичных политик (`simple_reconciliation`, `random`, `mark_all`) предусмотрен [`Dockerfile.policy`](../Dockerfile.policy), включающий только публичное подмножество модулей (`genesis/domain`, `genesis/policies`, `genesis/environment/jsonl.py`), `tzdata==2025.2`, `LICENSE` и `NOTICE` под `USER 65532:65532`:

```sh
docker build -f Dockerfile.policy --tag genesis-policy:local .
```

Внутри образа проверки `docker-policy-smoke` помечается `SKIP`. Если на доверенном хосте заранее собран smoke-образ и задана переменная `GENESIS_POLICY_SMOKE_IMAGE`, используйте `python -m genesis.cli verify --all --require-docker`, чтобы эти проверки стали обязательными в том же запуске. В CI workflow [`Policy isolation`](../.github/workflows/policy-isolation.yml) собирает оба образа (`Dockerfile.policy` и smoke-образ) и запускает `verify --all --require-docker`. [Offline verification workflow](../.github/workflows/offline-verify.yml).
