FROM python:3.11-slim

COPY worker.py /policy/worker.py
ENTRYPOINT ["python", "-I", "-B", "-u", "/policy/worker.py"]
