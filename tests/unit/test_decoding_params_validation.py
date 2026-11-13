"""Unit tests for decoding parameters validation (F5)."""

import pytest
from src.engine.types import DecodingParams


def test_valid_decoding_params():
    """Verify valid default decoding parameters initialize properly."""
    p = DecodingParams(min_new_tokens=60, max_new_tokens=2048)
    assert p.min_new_tokens == 60
    assert p.max_new_tokens == 2048


def test_max_new_tokens_less_than_min_raises_value_error():
    """F5: Verify max_new_tokens < min_new_tokens raises explicit ValueError."""
    with pytest.raises(ValueError, match="cannot be less than anti-truncation requirement"):
        DecodingParams(min_new_tokens=60, max_new_tokens=40)
