# Offline-capable, CPU-only container (runs comfortably within 2 CPU / 4 GB RAM, no GPU).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy repository contents (code, versioned configs, datasets, trajectories, tests, docs)
COPY . /app

# Install editable package offline if setuptools is present in the base image;
# PYTHONPATH=/app guarantees `python -m audit_rl` works 100% offline regardless.
RUN python -m pip install --no-deps --no-build-isolation -e . 2>/dev/null || true

# Default single-command verification entrypoint
CMD ["python", "-m", "audit_rl", "verify"]
