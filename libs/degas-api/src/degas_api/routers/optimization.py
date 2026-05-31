from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Literal, Union, get_args

import torch
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from pydegas.cfg.builder import from_text
from pydegas.cfg.smoother import smooth
from pydegas.exceptions import PyDeGASError
from pydegas.optimize.losses import LOSS_PARAM_SCHEMA, LOSS_REGISTRY, get_loss
from pydegas.optimize.runner import OPTIMIZER_REGISTRY, OptimizationRun, StepResult
from pydegas.parse.preprocessor import compile_to_soga_text

from degas_api.cache.process import OptimizationRateLimiter
from degas_api.settings import app_settings


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/optimization",
    tags=["optimization"],
)


def _sse_event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class LossValidateRequest(BaseModel):
    source: str


class LossParamInfo(BaseModel):
    name: str
    type: str | None


class LossValidateResponse(BaseModel):
    errors: list[str]
    params: list[LossParamInfo]


class OptimizationRequest(BaseModel):
    program: str
    program_language: Literal["soga", "soga_highlevel"] = "soga"
    compile_seed: int | None = None
    smooth_eps: float | None = None

    loss_function: str | None = Field(
        default=None,
        description="Name from /optimization/loss-functions. Ignored when loss_source is set.",
    )
    loss_kwargs: dict[str, Any] = Field(default_factory=dict)

    loss_source: str | None = Field(
        default=None,
        description="DeGASLoss DSL source. When set, loss_function is ignored.",
    )
    loss_bindings: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON bindings for non-dist parameters in the custom loss DSL.",
    )

    @model_validator(mode="after")
    def check_loss_provided(self) -> "OptimizationRequest":
        if self.loss_source is None and self.loss_function is None:
            raise ValueError("Either loss_function or loss_source must be provided.")
        return self

    optimizer: str = Field(..., description="Name from /optimization/optimizers")
    initial_params: dict[str, float]
    n_steps: int = 100
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)

    Kmax: int | None = Field(
        default=20,
        description="Maximum number of Gaussian components per merge. None = no pruning.",
    )
    pruning: Literal["classic", "ranking", "kmeans"] = "classic"

    tolerance: float | None = Field(
        default=1e-8,
        ge=0,
        description="Loss-stability threshold for convergence detection. None disables it.",
    )
    patience: int = Field(
        default=30,
        ge=1,
        description="Number of consecutive stable steps required before flagging converged.",
    )

    return_dist_summary: bool = True


class LossParamDef(BaseModel):
    name: str
    type: str
    required: bool
    default: float | int | None = None


class LossFunctionInfo(BaseModel):
    name: str
    params: list[LossParamDef]


class DistSummary(BaseModel):
    var_list: list[str]
    mean: list[float]


class StepOut(BaseModel):
    step: int
    loss: float
    params: dict[str, float]
    elapsed_ms: float = 0.0
    dist: DistSummary | None = None


class OptimizationResponse(BaseModel):
    steps: list[StepOut]
    converged: bool
    final_params: dict[str, float]


def _step_to_out(step: StepResult, *, include_dist: bool) -> StepOut:
    dist_summary: DistSummary | None = None
    if include_dist:
        mean = step.dist.gm.mean().tolist()
        dist_summary = DistSummary(
            var_list=step.dist.var_list, mean=[float(x) for x in mean]
        )
    return StepOut(
        step=step.step,
        loss=step.loss,
        params=step.params,
        elapsed_ms=step.elapsed_ms,
        dist=dist_summary,
    )


async def check_optimization_limits(
    http_request: Request,
    body: OptimizationRequest,
) -> OptimizationRequest:
    """Caller must release the acquired concurrency slot in a finally block."""
    limits = app_settings.optimization_limits

    if body.n_steps > limits.max_steps:
        raise HTTPException(
            status_code=422,
            detail=f"n_steps={body.n_steps} exceeds maximum allowed value of {limits.max_steps}.",
        )
    if body.Kmax is not None and body.Kmax < limits.min_kmax:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Kmax={body.Kmax} is below the minimum of {limits.min_kmax}. "
                "Set Kmax=null to disable pruning entirely."
            ),
        )

    limiter: OptimizationRateLimiter | None = getattr(
        http_request.app.state, "orl_limiter", None
    )
    if limiter is None:
        return body

    client_ip = http_request.client.host if http_request.client else "unknown"

    allowed, retry_after = await limiter.check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Too many optimization requests from this address.",
            headers={"Retry-After": str(retry_after)},
        )

    acquired = await limiter.try_acquire_slot()
    if not acquired:
        try:
            await limiter.undo_rate_limit_entry(client_ip)
        except Exception:
            pass
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server at capacity. At most {limits.max_concurrent_runs} optimizations "
                "can run simultaneously. Please retry shortly."
            ),
            headers={"Retry-After": "5"},
        )

    return body


@router.post("/loss/validate", response_model=LossValidateResponse)
def validate_loss_source(body: LossValidateRequest) -> LossValidateResponse:
    from pydegas.optimize.dsl import extract_params, validate_loss

    errors = validate_loss(body.source)
    params = extract_params(body.source) if not errors else []
    return LossValidateResponse(
        errors=errors,
        params=[LossParamInfo(**p) for p in params],
    )


@router.get("/loss-functions")
def get_loss_functions() -> list[LossFunctionInfo]:
    return [
        LossFunctionInfo(
            name=name,
            params=[LossParamDef(**p) for p in LOSS_PARAM_SCHEMA.get(name, [])],
        )
        for name in LOSS_REGISTRY
    ]


@router.get("/optimizers")
def get_optimizers() -> list[str]:
    return list(OPTIMIZER_REGISTRY.keys())


@router.get("/pruning-strategies")
def get_pruning_strategies() -> list[str]:
    return list(get_args(OptimizationRequest.model_fields["pruning"].annotation))


@router.post(
    "/run",
    response_model=None,
)
async def run_optimization(
    http_request: Request,
    stream: bool = Query(
        default=False,
        description="If true, stream steps as Server-Sent Events (text/event-stream).",
    ),
    body: OptimizationRequest = Depends(check_optimization_limits),
) -> Union[OptimizationResponse, StreamingResponse]:
    """Setup (compile, CFG, loss) runs before streaming starts — invalid input returns 422.
    Mid-run failures yield ``event: error``."""
    limiter: OptimizationRateLimiter | None = getattr(
        http_request.app.state, "orl_limiter", None
    )
    limits = app_settings.optimization_limits

    try:
        program = body.program
        if body.program_language == "soga_highlevel":
            program = compile_to_soga_text(program, seed=body.compile_seed)

        cfg = from_text(program)
        smooth(
            cfg, smooth_eps=body.smooth_eps
        ) if body.smooth_eps is not None else smooth(cfg)

        if body.loss_source:
            from pydegas.optimize.dsl import compile_loss, extract_params

            params_info = extract_params(body.loss_source)
            coerced: dict[str, Any] = {}
            for p in params_info:
                name, type_ann = p["name"], p["type"]
                raw = body.loss_bindings.get(name)
                if raw is None:
                    continue
                if type_ann == "traj_set":
                    if not isinstance(raw, list):
                        raise ValueError(f"binding '{name}' must be a 2-D array.")
                    if len(raw) > limits.max_trajectory_rows:
                        raise ValueError(
                            f"binding '{name}': {len(raw)} rows exceeds maximum {limits.max_trajectory_rows}."
                        )
                    if raw and len(raw[0]) > limits.max_trajectory_cols:
                        raise ValueError(
                            f"binding '{name}': {len(raw[0])} columns exceeds maximum {limits.max_trajectory_cols}."
                        )
                    coerced[name] = torch.tensor(raw, dtype=torch.float64)
                elif type_ann == "index_list":
                    coerced[name] = [int(x) for x in raw]
                elif type_ann == "scalar":
                    coerced[name] = torch.tensor(float(raw), dtype=torch.float64)
                elif type_ann == "int":
                    coerced[name] = int(raw)
                else:
                    coerced[name] = raw
            loss_fn = compile_loss(body.loss_source, **coerced)
        else:
            loss_kwargs = dict(body.loss_kwargs)
            if "trajectories" in loss_kwargs and not isinstance(
                loss_kwargs["trajectories"], torch.Tensor
            ):
                raw = loss_kwargs["trajectories"]
                if not isinstance(raw, list):
                    raise ValueError("trajectories must be a list of rows.")
                if len(raw) > limits.max_trajectory_rows:
                    raise ValueError(
                        f"trajectories has {len(raw)} rows; maximum is {limits.max_trajectory_rows}."
                    )
                if raw and len(raw[0]) > limits.max_trajectory_cols:
                    raise ValueError(
                        f"trajectories has {len(raw[0])} columns; maximum is {limits.max_trajectory_cols}."
                    )
                loss_kwargs["trajectories"] = torch.tensor(raw, dtype=torch.float64)
            loss_fn = get_loss(body.loss_function, **loss_kwargs)

        optimizer_kwargs = dict(body.optimizer_kwargs) if body.optimizer_kwargs else {}
        if body.optimizer == "LBFGS":
            optimizer_kwargs["max_iter"] = min(optimizer_kwargs.get("max_iter", 20), 20)

        run = OptimizationRun(
            cfg=cfg,
            initial_params=body.initial_params,
            loss_fn=loss_fn,
            optimizer=body.optimizer,
            optimizer_kwargs=optimizer_kwargs or None,
            Kmax=body.Kmax,
            pruning=body.pruning,
            tolerance=body.tolerance,
            patience=body.patience,
        )
    except (ValueError, PyDeGASError) as e:
        if limiter is not None:
            await limiter.release_slot()
        raise HTTPException(status_code=422, detail=str(e)) from e

    max_seconds = limits.max_run_seconds

    if not stream:
        try:
            loop = asyncio.get_running_loop()
            out_steps: list[StepOut] = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: [
                        _step_to_out(s, include_dist=body.return_dist_summary)
                        for s in run.run_generator(body.n_steps)
                    ],
                ),
                timeout=max_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"Optimization timed out after {max_seconds}s.",
            )
        finally:
            if limiter is not None:
                await limiter.release_slot()

        final = out_steps[-1] if out_steps else None
        return OptimizationResponse(
            steps=out_steps,
            converged=run.has_converged(),
            final_params=final.params if final else dict(body.initial_params),
        )

    _limiter = limiter
    _request = http_request
    _loop = asyncio.get_running_loop()
    _sentinel = object()

    async def _iter_events() -> AsyncIterator[str]:
        # Starlette calls aclose() on disconnect, firing the finally block immediately.
        try:
            yield _sse_event("start", {"n_steps": body.n_steps})

            last_out: StepOut | None = None
            gen = run.run_generator(body.n_steps)
            deadline = time.monotonic() + max_seconds

            while True:
                if await _request.is_disconnected():
                    logger.debug("client disconnected — stopping optimization stream")
                    return

                if time.monotonic() > deadline:
                    logger.warning(
                        "optimization stream timed out after {}s", max_seconds
                    )
                    yield _sse_event(
                        "error",
                        {"detail": f"Optimization timed out after {max_seconds}s."},
                    )
                    return

                try:
                    step = await _loop.run_in_executor(None, next, gen, _sentinel)
                except Exception as e:
                    logger.exception("error during optimization stream")
                    yield _sse_event("error", {"detail": str(e)})
                    return

                if step is _sentinel:
                    break

                last_out = _step_to_out(step, include_dist=body.return_dist_summary)
                yield _sse_event("step", last_out.model_dump())

            yield _sse_event(
                "end",
                {
                    "converged": run.has_converged(),
                    "final_params": last_out.params
                    if last_out
                    else dict(body.initial_params),
                },
            )
        finally:
            if _limiter is not None:
                await _limiter.release_slot()

    return StreamingResponse(_iter_events(), media_type="text/event-stream")
