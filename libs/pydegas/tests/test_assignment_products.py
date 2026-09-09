from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("coefficient, expected", [("2", 12.0), ("-2", -12.0), ("_scale", 12.0), ("a[0]", 12.0)])
@pytest.mark.parametrize("smoothed", [False, True])
def test_coefficient_products_in_programs(coefficient, expected, smoothed):
    from pydegas.cfg.builder import from_text
    from pydegas.cfg.smoother import smooth
    from pydegas.semantics.engine import start_soga

    cfg = from_text(f"data a = [2]; y = 2; z = 3; x = {coefficient}*y*z;")
    if smoothed:
        smooth(cfg)
    scale = torch.tensor(2.0, requires_grad=True)
    actual = start_soga(cfg, {"scale": scale})
    mean = actual.gm.mean()[actual.var_list.index("x")]
    assert mean.item() == pytest.approx(expected)
    if coefficient == "_scale":
        assert torch.autograd.grad(mean, scale)[0].item() == pytest.approx(6.0)


@pytest.mark.parametrize("coefficient", [2.0, -2.0])
def test_coefficient_scales_product_mean_cross_covariances_and_variance(make_dist, coefficient):
    from pydegas.semantics.update import update_rule

    dist = make_dist(["x", "y", "z"], [1.0], [[0.0, 2.0, 3.0]], [[[1.0, 0.0, 0.0], [0.0, 4.0, 1.0], [0.0, 1.0, 9.0]]])
    actual = update_rule(dist, f"x = {coefficient}*y*z", {}, {})
    torch.testing.assert_close(actual.gm.mean(), torch.tensor([coefficient * 7, 2.0, 3.0]))
    torch.testing.assert_close(
        actual.gm.cov(),
        torch.tensor(
            [
                [coefficient**2 * 121, coefficient * 14, coefficient * 21],
                [coefficient * 14, 4.0, 1.0],
                [coefficient * 21, 1.0, 9.0],
            ]
        ),
    )


def test_coefficient_product_with_gaussian_literal(make_dist):
    from pydegas.semantics.update import update_rule

    dist = make_dist(["x", "y"], [1.0], [[0.0, 2.0]], [[[1.0, 0.0], [0.0, 1.0]]])
    actual = update_rule(dist, "x = 2*y*gm([1.0],[3.0],[2.0])", {}, {})
    assert actual.gm.mean()[0].item() == pytest.approx(12.0)
    assert actual.gm.cov()[0, 0].item() == pytest.approx(116.0)


def test_signed_product_of_a_variable_with_itself(make_dist):
    from pydegas.semantics.update import update_rule

    dist = make_dist(["x", "y"], [1.0], [[0.0, 2.0]], [[[1.0, 0.0], [0.0, 1.0]]])
    actual = update_rule(dist, "x = -y*y", {}, {})
    torch.testing.assert_close(actual.gm.mean(), torch.tensor([-5.0, 2.0]))
    torch.testing.assert_close(actual.gm.cov(), torch.tensor([[18.0, -4.0], [-4.0, 1.0]]))


def test_assignment_does_not_ignore_trailing_product_tokens(make_dist):
    from pydegas.exceptions import SyntaxParseError
    from pydegas.semantics.update import update_rule

    dist = make_dist(["x", "y"], [1.0], [[0.0, 2.0]], [[[1.0, 0.0], [0.0, 1.0]]])
    with pytest.raises(SyntaxParseError):
        update_rule(dist, "x = 2*y*", {}, {})


def test_product_of_three_random_variables_is_rejected(make_dist):
    from pydegas.exceptions import SemanticError
    from pydegas.semantics.update import update_rule

    dist = make_dist(["x", "y"], [1.0], [[0.0, 2.0]], [[[1.0, 0.0], [0.0, 1.0]]])
    with pytest.raises(SemanticError, match="two random variables"):
        update_rule(dist, "x = y*y*y", {}, {})


def test_regenerated_parser_preserves_data_index_and_gm_parameter_gradients():
    from pydegas.cfg.builder import from_text
    from pydegas.semantics.engine import start_soga

    cfg = from_text("data a = [2,3]; array[2] X; for i in range(2) { X[i] = a[i]*gm([1.0],[_mu],[1.0]); } end for;")
    mu = torch.tensor(1.0, requires_grad=True)
    actual = start_soga(cfg, {"mu": mu})
    torch.testing.assert_close(actual.gm.mean(), torch.tensor([2.0, 3.0]))
    assert torch.autograd.grad(actual.gm.mean().sum(), mu)[0].item() == pytest.approx(5.0)
