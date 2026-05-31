from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from degas_api.settings import app_settings
from loguru import logger

import redis.asyncio as redis_asyncio
from redis.exceptions import ConnectionError as RedisConnectionError


def create_lifespan(host: str, port: int):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from degas_api.docs.api import startup_message
        from degas_api.cache.process import OptimizationRateLimiter

        torch.set_default_dtype(torch.float64)
        logger.info(startup_message(host=host, port=port))

        orl = app_settings.optimization_rate_limiter
        limits = app_settings.optimization_limits
        app.state.orl_redis = None
        app.state.orl_limiter = None
        if orl.enabled:
            try:
                logger.info(
                    "initializing redis client host={host} port={port} db={db}",
                    host=orl.host,
                    port=orl.port,
                    db=orl.db,
                )
                app.state.orl_redis = redis_asyncio.Redis.from_url(
                    f"redis://{orl.host}:{orl.port}/{orl.db}",
                    password=orl.password,
                )

                pong = await app.state.orl_redis.ping()
                if pong:
                    logger.info(
                        "pinged redis instance host={host} port={port} db={db}",
                        host=orl.host,
                        port=orl.port,
                        db=orl.db,
                    )

                # Reset stale concurrency counter from previous (possibly crashed) run.
                await app.state.orl_redis.set("active_runs:optimization", 0)

                app.state.orl_limiter = OptimizationRateLimiter(
                    redis=app.state.orl_redis,
                    max_concurrent_runs=limits.max_concurrent_runs,
                    rate_limit_requests=limits.rate_limit_requests,
                    rate_limit_window_seconds=limits.rate_limit_window_seconds,
                )
                logger.info(
                    "optimization rate limiter initialized: "
                    "max_concurrent={max_concurrent} "
                    "rate_limit={rate}/{window}s "
                    "max_steps={max_steps} min_kmax={min_kmax}",
                    max_concurrent=limits.max_concurrent_runs,
                    rate=limits.rate_limit_requests,
                    window=limits.rate_limit_window_seconds,
                    max_steps=limits.max_steps,
                    min_kmax=limits.min_kmax,
                )
            except RedisConnectionError as e:
                logger.exception(
                    "could not connect to the redis of optimization rate limiter."
                )
                raise e

        try:
            yield
        finally:
            logger.info("shutting down the DeGAS API server...")

            if app.state.orl_redis:
                await app.state.orl_redis.close()

                logger.info(
                    "redis session closed with host {host}/{port}/{db}",
                    host=orl.host,
                    port=orl.port,
                    db=orl.db,
                )

            from degas_api.log_config import cleanup_logging

            cleanup_logging()

    return lifespan
