FROM ghcr.io/astral-sh/uv:0.12.5 AS uv
FROM python:3.14.7-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOME=/home/app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY demo ./demo
# Retain the source layout: sdk.ROOT resolves /app for the approved fixtures.
RUN uv sync --locked --no-dev && useradd --uid 10001 --create-home app
USER 10001:10001
EXPOSE 8021
CMD ["uvicorn", "hdfc_mistral_mvp.api:app", "--host", "0.0.0.0", "--port", "8021", "--workers", "1", "--no-access-log", "--timeout-graceful-shutdown", "60"]
