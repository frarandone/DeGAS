
FROM astral/uv:python3.12-bookworm-slim AS builder

WORKDIR /build

COPY pydegas /build/pydegas
COPY degas-api /build/degas-api

WORKDIR /build/degas-api

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv sync --frozen --no-dev

FROM astral/uv:python3.12-bookworm-slim AS runtime

WORKDIR /app

COPY --from=builder /build/degas-api /app
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["degas-api"]
