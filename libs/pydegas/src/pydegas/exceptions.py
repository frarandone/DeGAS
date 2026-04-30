"""
This module defines the exception hierarchy for the pydegas library.
"""

from __future__ import annotations


class PyDeGASError(Exception):
    """Base class for all pydegas-raised errors."""


class ParseError(PyDeGASError):
    """Base class for parse failures."""


class SyntaxParseError(ParseError):
    """Syntax-level parse failure (unexpected token, malformed construct)."""


class PreprocessError(ParseError):
    """Preprocessing step failed (normalization, helper syntax rewriting, etc.)."""
