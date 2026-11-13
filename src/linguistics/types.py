"""Types and dataclasses for linguistic Text Normalization (TN) contracts (R4, C7)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TNAnnotation:
    """Annotation representing an individual expanded token (e.g. number, currency, acronym)."""
    raw_span: str
    expanded_span: str
    category: str  # "cardinal", "ordinal", "currency", "date", "time", "acronym"
    char_start: int
    char_end: int


@dataclass(frozen=True)
class NormalizedText:
    """Result of text normalization contract."""
    raw_text: str
    normalized_text: str
    annotations: List[TNAnnotation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
