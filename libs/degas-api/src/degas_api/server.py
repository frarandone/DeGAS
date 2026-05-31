import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette import status

from loguru import logger

from degas_api import __version__, middlewares
from degas_api.settings import app_settings
from degas_api.log_config import configure_logging
from degas_api.routers import api_router
from degas_api.lifespan import create_lifespan


type AppLifespan = Callable[[FastAPI], AbstractAsyncContextManager[Any]]


def create_app(
    routers: list[tuple[APIRouter, dict]] = [(api_router, {})],
    lifespan: AppLifespan = create_lifespan(
        host=app_settings.host, port=app_settings.port
    ),
    docs_url: str = app_settings.docs_url,
) -> FastAPI:
    """Create a FastAPI app with the given routers."""
    from degas_api.docs.api import (
        FASTAPI_DESCRIPTION,
        FASTAPI_SUMMARY,
        FASTAPI_TITLE,
    )

    configure_logging(level=app_settings.logging.level)

    app = FastAPI(
        docs_url=docs_url,
        version=__version__,
        title=FASTAPI_TITLE,
        description=FASTAPI_DESCRIPTION,
        summary=FASTAPI_SUMMARY,
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or request.headers.get(
            "DeGAS-Request-Id"
        )
        logger.opt(exception=True).error(
            "Unhandled exception (request_id={})", request_id
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "request_id": request_id},
            media_type="application/json; charset=utf-8",
        )

    app.middleware("http")(middlewares.logging_context_middleware())
    app.middleware("http")(middlewares.add_request_id_header)
    app.middleware("http")(middlewares.api_version_middleware(__version__))
    app.middleware("http")(middlewares.add_request_duration_header)
    app.middleware("http")(middlewares.add_security_headers)

    for router, options in routers:
        app.include_router(router, **options)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, object]:
        limiter = getattr(app.state, "orl_limiter", None)
        max_concurrent = app_settings.optimization_limits.max_concurrent_runs
        slots_free = (
            await limiter.slots_free() if limiter is not None else max_concurrent
        )
        return {
            "unix": time.time(),
            "slots_free": slots_free,
            "max_concurrent": max_concurrent,
        }

    @app.get("/", tags=["root"])
    async def redirect_to_docs_root():
        logger.debug("Redirecting to docs")
        return RedirectResponse(url=docs_url)

    return app
