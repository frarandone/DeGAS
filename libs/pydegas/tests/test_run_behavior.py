from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def make_run(**kwargs):
    from pydegas.cfg.builder import from_text
    from pydegas.optimize import OptimizationRun

    options = {"optimizer": "SGD", "optimizer_kwargs": {"lr": 0.1}, "tolerance": None, **kwargs}
    return OptimizationRun(from_text("x = _mu;"), {"mu": 1.0}, lambda dist: dist.gm.mean().square().sum(), **options)


def test_step_records_pre_update_loss_and_post_update_parameters():
    run = make_run()
    first = run.step(0)
    assert first.loss == pytest.approx(1.0)
    assert first.params["mu"] == pytest.approx(0.8)
    assert first.dist.gm.mean().item() == pytest.approx(1.0)
    second = run.step(1)
    assert second.loss == pytest.approx(0.64)
    assert second.params["mu"] == pytest.approx(0.64)
    assert first.dist.gm.mean().item() == pytest.approx(1.0)
    assert first.params["mu"] == pytest.approx(0.8)
    assert not first.dist.gm.mu.requires_grad
    assert first.dist.gm.mu.data_ptr() != second.dist.gm.mu.data_ptr()


def test_generator_matches_eager_run():
    eager = make_run().run(5)
    streamed = list(make_run().run_generator(5))
    assert [r.step for r in streamed] == list(range(5))
    assert [r.loss for r in streamed] == pytest.approx([r.loss for r in eager])
    assert [r.params["mu"] for r in streamed] == pytest.approx([r.params["mu"] for r in eager])


def test_callback_stops_after_yielding_the_requested_step():
    seen = []

    def stop(result):
        seen.append(result.step)
        return result.step == 2

    results = make_run(on_step=stop).run(10)
    assert seen == [0, 1, 2]
    assert [r.step for r in results] == seen


def test_convergence_waits_for_full_patience_window():
    results = make_run(optimizer_kwargs={"lr": 0.0}, tolerance=1e-8, patience=2).run(10)
    assert len(results) == 3
    assert [r.converged for r in results] == [False, False, True]


def test_none_tolerance_disables_early_stopping():
    results = make_run(optimizer_kwargs={"lr": 0.0}, patience=2).run(10)
    assert len(results) == 10
    assert not any(r.converged for r in results)


def test_optimizer_class_matches_registered_name():
    by_name = make_run().run(3)
    by_class = make_run(optimizer=torch.optim.SGD).run(3)
    assert [r.params["mu"] for r in by_class] == pytest.approx([r.params["mu"] for r in by_name])


def test_optimizer_instance_updates_the_run_parameters():
    from pydegas.cfg.builder import from_text
    from pydegas.optimize import OptimizationRun

    parameter = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    run = OptimizationRun(from_text("x = _mu;"), {"mu": 1.0}, lambda d: d.gm.mean().square().sum(), optimizer=optimizer)
    result = run.step(0)
    assert result.params["mu"] == pytest.approx(0.8)
    assert run.optimizer is optimizer
    assert run.params["mu"] is parameter
    assert parameter.item() == pytest.approx(0.8)


def test_optimizer_instance_preserves_accumulated_momentum():
    from pydegas.cfg.builder import from_text
    from pydegas.optimize import OptimizationRun

    parameter = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    parameter.square().backward()
    optimizer.step()
    momentum = optimizer.state[parameter]["momentum_buffer"]
    run = OptimizationRun(
        from_text("x = _mu;"), {"mu": parameter.item()}, lambda d: d.gm.mean().square().sum(), optimizer
    )
    assert optimizer.state[parameter]["momentum_buffer"] is momentum
    result = run.step(0)
    assert result.params["mu"] == pytest.approx(0.46)
    assert momentum.item() == pytest.approx(3.4)


def test_optimizer_instance_preserves_groups_and_initializes_values():
    from pydegas.cfg.builder import from_text
    from pydegas.optimize import OptimizationRun

    first = torch.tensor(7.0, requires_grad=True)
    second = torch.tensor(8.0, requires_grad=True)
    optimizer = torch.optim.SGD([{"params": [first], "lr": 0.1}, {"params": [second], "lr": 0.2}])
    run = OptimizationRun(
        from_text("x = _first; y = _second;"),
        {"first": 1.0, "second": 2.0},
        lambda d: d.gm.mean().square().sum(),
        optimizer,
    )
    assert run.params["first"] is first and run.params["second"] is second
    assert first.item() == 1.0 and second.item() == 2.0
    assert run.step(0).params == pytest.approx({"first": 0.8, "second": 1.2})
    assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.2]


@pytest.mark.parametrize("kind", ["count", "vector", "frozen"])
def test_incompatible_optimizer_instance_is_rejected_before_mutation(kind):
    from pydegas.cfg.builder import from_text
    from pydegas.optimize import OptimizationRun

    parameter = torch.tensor([7.0] if kind == "vector" else 7.0, requires_grad=kind != "frozen")
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    initial = {"mu": 1.0, "extra": 2.0} if kind == "count" else {"mu": 1.0}
    with pytest.raises(ValueError, match="Optimizer"):
        OptimizationRun(from_text("x = _mu;"), initial, lambda d: d.gm.mean().sum(), optimizer)
    assert parameter.item() == 7.0


def test_resolver_rejects_optimizer_with_unrelated_tensors():
    from pydegas.optimize import resolve_optimizer

    owned = torch.tensor(1.0, requires_grad=True)
    unrelated = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.SGD([owned], lr=0.1)
    with pytest.raises(ValueError, match="must own"):
        resolve_optimizer(optimizer, [unrelated])


def test_lbfgs_instance_matches_registered_optimizer():
    parameter = torch.tensor(1.0, requires_grad=True)
    options = {"lr": 0.1, "max_iter": 3}
    optimizer = torch.optim.LBFGS([parameter], **options)
    expected = make_run(optimizer="LBFGS", optimizer_kwargs=options).run(2)
    actual = make_run(optimizer=optimizer, optimizer_kwargs={}).run(2)
    assert [result.loss for result in actual] == pytest.approx([result.loss for result in expected])
    assert [result.params["mu"] for result in actual] == pytest.approx([result.params["mu"] for result in expected])


def test_run_does_not_print_or_write_distribution_statistics(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    make_run().run(3)
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert list(tmp_path.iterdir()) == []


def test_cached_assignment_tree_does_not_reuse_parameter_bindings():
    from pydegas.cfg.builder import from_text
    from pydegas.semantics.engine import start_soga

    cfg = from_text("x = _mu;")
    first = start_soga(cfg, {"mu": torch.tensor(1.0)})
    second = start_soga(cfg, {"mu": torch.tensor(3.0)})
    assert first.gm.mean().item() == pytest.approx(1.0)
    assert second.gm.mean().item() == pytest.approx(3.0)
