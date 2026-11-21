FROM docker.io/library/python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS dependencies

COPY tests/requirements-schema.lock /build/requirements.lock
RUN python -m pip download --only-binary=:all: --no-deps --dest /wheels -r /build/requirements.lock \
    && python -m pip install --no-index --find-links=/wheels --no-deps \
       --target=/offline-deps -r /build/requirements.lock

FROM docker.io/library/python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

WORKDIR /app
COPY --from=dependencies /offline-deps /usr/local/lib/python3.11/site-packages/
RUN python -m pip check
COPY src/ src/
COPY schemas/ schemas/
COPY tests/ tests/
COPY data/ data/
COPY benchmarks/ benchmarks/
COPY trajectories/ trajectories/
COPY LICENSE NOTICE ./

ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 65532:65532
CMD ["python", "-m", "genesis.cli", "verify", "--all"]
