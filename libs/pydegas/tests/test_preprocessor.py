from __future__ import annotations

import pytest


def test_compile_to_soga_text_rewrites_gauss_to_gm() -> None:
    from pydegas.parse.preprocessor import compile_to_soga_text

    compiled = compile_to_soga_text("x = gauss(0.0, 1.0);", seed=0)
    assert "gauss(" not in compiled
    assert "gm(" in compiled


def test_preprocess_error_is_raised_for_invalid_gauss_args() -> None:
    from pydegas.exceptions import PreprocessError
    from pydegas.parse.preprocessor import compile_to_soga_text

    with pytest.raises(PreprocessError, match=r"Invalid gauss\(0\.0\) arguments"):
        compile_to_soga_text("x = gauss(0.0);", seed=0)


def test_preprocess_error_is_raised_for_invalid_uniform_args() -> None:
    from pydegas.exceptions import PreprocessError
    from pydegas.parse.preprocessor import compile_to_soga_text

    with pytest.raises(PreprocessError, match=r"Invalid uniform"):
        compile_to_soga_text("x = uniform([0, 1]);", seed=0)
