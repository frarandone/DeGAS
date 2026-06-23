
FROM node:22-slim AS frontend-builder

WORKDIR /build/degas-ui
COPY libs/degas-ui/package.json libs/degas-ui/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY libs/degas-ui/ ./
RUN npm run build

FROM astral/uv:python3.12-bookworm-slim AS builder

WORKDIR /build

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# Required: cache mount is a different filesystem; hardlinks don't work across
# filesystems so uv must copy. Without this flag uv silently falls back to a
# slow path and the cache provides no real benefit.
ENV UV_LINK_MODE=copy

# pydegas is a local path dep — uv must build it, so it needs full source.
COPY libs/pydegas /build/pydegas

WORKDIR /build/degas-api

RUN uv venv /opt/venv

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=libs/degas-api/pyproject.toml,target=/build/degas-api/pyproject.toml \
    --mount=type=bind,source=libs/degas-api/uv.lock,target=/build/degas-api/uv.lock \
    uv sync --frozen --no-dev --no-install-project

# now copy the real source — changes here only re-run the fast final sync
COPY libs/degas-api /build/degas-api

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM astral/uv:python3.12-bookworm-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=frontend-builder /build/degas-ui/dist /app/static

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEGAS_API_STATIC_DIR=/app/static

RUN useradd --system --no-create-home --uid 1001 appuser
USER appuser

EXPOSE 8000

ENTRYPOINT ["/opt/venv/bin/uvicorn", "degas_api.server:create_app", "--factory"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
