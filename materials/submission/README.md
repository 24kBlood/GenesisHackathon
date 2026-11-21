# Финальный аудит Genesis FinTech

**Статус на 21 ноября 2026 года: технический кандидат проверен, комплект сдачи неполный.** Код, данные и результаты ниже относятся к коммиту `47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca` из [Draft PR #49](https://github.com/24kBlood/GenesisHackathon/pull/49). На момент аудита `main` указывал на `955c4f1937f24ed0a830ce150b1623a2bc2ed581`. Для сдачи нужно утвердить версию и повторить проверку после интеграции.

Проверенный бэкенд вошёл в [`main` коммитом `6fc8360`](https://github.com/24kBlood/GenesisHackathon/commit/6fc8360d126795785c420e5ee1215133252680b5). На этом коммите повторно прошли локальная проверка и replay. Правообладатели согласовали MIT для исходного кода. [Лицензия](../../LICENSE) · [Атрибуция](../../NOTICE)

По переданному положению дедлайн — **22 ноября 2026 года, 23:59 по времени Астаны**. Капитан подаёт через личный кабинет ссылку на единый документ Google Drive с исходниками и инструкцией, видео продолжительностью до трёх минут и презентацией. Задача #26 пропущена по решению команды. Документ на Drive, проверка всей цепочки ссылок в роли жюри и отправка через кабинет остаются незавершёнными. [Задача финального аудита](https://github.com/24kBlood/GenesisHackathon/issues/27) · [Официальный сайт](https://genesis-exe.com/)

## Ссылки для комплекта

| Материал | Ссылка и состояние доступа |
| --- | --- |
| Код и инструкция | Публичный репозиторий. Для ссылок на дерево и README кандидата получен HTTP 200 без авторизации. [Код кандидата](https://github.com/24kBlood/GenesisHackathon/tree/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca) · [Инструкция запуска](https://github.com/24kBlood/GenesisHackathon/blob/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/README.md) |
| Видео до 3:00 | Зрительская ссылка и проверка воспроизведения отсутствуют. |
| Презентация | Зрительская ссылка и проверка просмотра отсутствуют. |
| Единый документ и отправка | Документ Google Drive не создан. Подача капитаном и её подтверждение отсутствуют. |

## Девять пунктов технической сдачи

| № | Требование §09 | Результат на проверенном SHA |
| --- | --- | --- |
| 1 | Исходники генератора и среды, лицензия, зафиксированные зависимости | **Проверено.** Исходники и lock присутствовали на проверенном SHA. MIT добавлена после первоначального аудита. [Генератор](../../src/genesis/generation) · [Среда](../../src/genesis/environment) · [Lock](../../tests/requirements-schema.lock) · [LICENSE](../../LICENSE) · [NOTICE](../../NOTICE) |
| 2 | README с контейнерным запуском и единой командой проверки | **Проверено.** Команда завершилась кодом 0 локально и в контейнере CI без сети. [Инструкция офлайн-запуска](https://github.com/24kBlood/GenesisHackathon/blob/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/docs/OFFLINE_RUN.md) |
| 3 | Общие matcher и награда, схемы и политики | **Проверено.** Код и схемы доступны. Отдельная матрица оценки прошла 20 из 20. [Оценщик](https://github.com/24kBlood/GenesisHackathon/tree/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/src/genesis/evaluation) · [Схемы](https://github.com/24kBlood/GenesisHackathon/tree/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/schemas) |
| 4 | 10 открытых train и 10 dev эпизодов | **Проверено.** В каждом каталоге ровно 10 эпизодов. Регрессионная проверка сверяет воспроизводимые открытые файлы и контрольные суммы. [Данные](https://github.com/24kBlood/GenesisHackathon/tree/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/data) |
| 5 | Тесты среды, matcher и награды | **Проверено.** Локально 145 тестов прошли, два Docker-smoke теста запущены отдельным Linux job. [Тесты](https://github.com/24kBlood/GenesisHackathon/tree/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/tests) · [Изоляция политики в CI](https://github.com/24kBlood/GenesisHackathon/actions/runs/44880612847/job/136774873988) |
| 6 | Четыре базовых адаптера | **Проверено.** Случайный исследователь, простая сверка, «пометить всё» и доверенный Oracle завершили 80 из 80 локальных запусков на общих 20 профилях. Это измерения команды. [Политики](https://github.com/24kBlood/GenesisHackathon/blob/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/docs/BASELINE_POLICIES.md) · [Полный отчёт](https://github.com/24kBlood/GenesisHackathon/tree/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/benchmarks) |
| 7 | Записанная траектория без закрытого эталона в наблюдениях | **Проверено.** В `trajectories.jsonl` пять принятых шагов реального запуска простой сверки. Тесты проверяют схему, точный replay и отсутствие закрытых полей в публичной проекции. [Траектория](https://github.com/24kBlood/GenesisHackathon/blob/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/trajectories/trajectories.jsonl) · [Формат и replay](https://github.com/24kBlood/GenesisHackathon/blob/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/docs/TRAJECTORIES.md) |
| 8 | Краткая записка о дизайне | **Проверено.** Описаны происхождение ошибок, открытый протокол, изоляция, формула, результаты четырёх политик и ограничения. [Записка о дизайне](https://github.com/24kBlood/GenesisHackathon/blob/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/docs/DESIGN_NOTE.md) · [Ограничения](https://github.com/24kBlood/GenesisHackathon/blob/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/docs/LIMITATIONS.md) |
| 9 | Демо работающей среды | **Открыто.** Локальный replay FX-эпизода подтверждён. Видеодемонстрация и презентация пропущены вместе с #26, зрительских ссылок нет. |

## Воспроизведение проверок

Выполните из корня клона проекта с Python 3.11. Подготовка зависимостей может использовать сеть. Проверка использует локальные файлы.

```sh
git fetch origin draft/25-docs-and-design-note
git checkout --detach 47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca
python -m pip install --no-deps -r tests/requirements-schema.lock
PYTHONPATH=src python -m genesis.cli verify --all
PYTHONPATH=src python -m unittest discover -s tests -p test_trajectories.py -v
```

В PowerShell задайте `$env:PYTHONPATH = 'src'` и запустите обе команды Python без префикса `PYTHONPATH=src`.

Команды сборки и проверки на Linux приведены ниже. Для сборки может потребоваться сеть. При проверке сеть отключена.

```sh
docker build --tag genesis-verify:local .
docker run --rm --pull never --network none --cpus 2 --memory 4g --memory-swap 4g \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 64 \
  genesis-verify:local python -m genesis.cli verify --all
```

- **Локально, CPython 3.11.16:** `verify --all` вернул 0. Прошли 145 обязательных тестов, 20 из 20 повторов, 20 из 20 пар с перестановкой ID, 80 из 80 запусков политик. Два Docker-smoke теста в локальном общем запуске отмечены как пропущенные.
- **Изоляция на Linux:** отдельный CI job прошёл 11 из 11 тестов, включая оба Docker-smoke. [Policy isolation](https://github.com/24kBlood/GenesisHackathon/actions/runs/44880612847/job/136774873988)
- **Офлайн на Linux:** Docker Engine сообщил размер образа **50 734 207 байт**. Проверка подтвердила сеть `none`, 2 CPU, 4 ГиБ памяти, корень только для чтения и UID/GID `65532:65532`. Основной запуск завершился кодом 0. Контрольный повреждённый JSON дал `JSONDecodeError` и код 1. [Offline verification](https://github.com/24kBlood/GenesisHackathon/actions/runs/44880612895/job/136774873984)
- **Траектория:** отдельный запуск прошёл 7 из 7 тестов. FX-эпизод `ep-cf50625d` завершился на пятом шаге с терминальной наградой `0.992`. SHA-256 файла `978246df1473500ad3e1208cc434767611dbfb152af568bb2ad9b40794526c4d`.

На Draft PR с проверенным head прошли также [Data contract](https://github.com/24kBlood/GenesisHackathon/actions/runs/44880612840/job/136774873849) и [Bootstrap CI](https://github.com/24kBlood/GenesisHackathon/actions/runs/44880612980/job/136774874121). Эти прогоны измеряют состояние локального синтетического решения. Результаты закрытой оценки жюри здесь отсутствуют.

Размер открытых эпизодов ниже рекомендованного в ТЗ. В train от 16 до 41 проводки и от 7 до 17 документов на эпизод. В dev от 16 до 36 проводок и от 7 до 15 документов. [Состав данных](https://github.com/24kBlood/GenesisHackathon/blob/47c8fb37e4fcb8fefebc5c40b9ff28e9b6f6e1ca/data/README.md)

## Перед подачей

1. Зафиксировать итоговый SHA комплекта и повторить офлайн-проверку и replay на этой версии.
2. При подготовке полного конкурсного комплекта записать видео, подготовить презентацию, собрать все ссылки в одном документе Google Drive и проверить открытие каждого материала с правами жюри.
3. Капитан отправляет ссылку через кабинет Genesis и сохраняет отметку времени и подтверждение. При сбое сохраняет скриншот, время и описание для обращения к организаторам.
