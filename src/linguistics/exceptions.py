"""Domain exceptions for Italian linguistic normalization pipeline (Milestone M2, WP0)."""


class LinguisticNormalizationError(Exception):
    """Base exception for all linguistic text normalization errors."""
    pass


class EntityShieldingCorruptionError(LinguisticNormalizationError):
    """Raised when protected entity placeholders cannot be faithfully restored."""
    pass
