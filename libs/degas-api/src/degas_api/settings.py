from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv(dotenv_path=find_dotenv(usecwd=True))


class LoggingSettings(BaseSettings):
    level: str = "INFO"


class RedisRateLimiterSettings(BaseSettings):
    enabled: bool = True
    host: str = "localhost"
    port: str = "6379"
    db: int = 0
    password: str | None = None
    username: str | None = None
    ssl: bool = False


class OptimizationLimitsSettings(BaseSettings):
    max_concurrent_runs: int = 5
    max_steps: int = 500
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60
    min_kmax: int = 15
    max_run_seconds: int = 120  # wall-clock timeout per optimization run
    max_trajectory_rows: int = 500  # rows in uploaded trajectory CSV
    max_trajectory_cols: int = 200  # columns (time steps) per trajectory


class AppSettings(BaseSettings):
    """main application settings with hierarchical configuration."""

    host: str
    port: int
    workers: int

    docs_url: str = "/docs"

    logging: LoggingSettings = LoggingSettings()
    optimization_rate_limiter: RedisRateLimiterSettings = RedisRateLimiterSettings()
    optimization_limits: OptimizationLimitsSettings = OptimizationLimitsSettings()

    # prefix for environment variables and enable nested delimiter
    model_config = SettingsConfigDict(
        env_prefix="degas_api_",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
    )


app_settings = AppSettings()
