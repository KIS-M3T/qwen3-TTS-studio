"""Entity Shielding and Deshielding subsystem (Milestone M2, WP1).

Implements Stage 1 (Shielding) and Stage 8 (Deshielding) of the Italian
linguistic normalization pipeline (R1, R2, S3, P4).

Isolates structured technical entities (URLs, emails, file paths, UUIDs,
technical codes) before linguistic processing to prevent corruption of
numbers, dots, slashes, or hyphens inside them.
"""

from dataclasses import dataclass
import re
from typing import List, Optional, Tuple
import uuid

from src.linguistics.exceptions import EntityShieldingCorruptionError


@dataclass(frozen=True)
class ShieldedEntity:
    """Record of a shielded technical entity."""
    placeholder: str
    original_text: str
    category: str  # "url", "email", "path_windows", "path_posix", "uuid", "technical_code"
    char_start: int
    char_end: int


# Category priority for overlap resolution:
# URL > Email > Windows Path > POSIX Path > UUID > Technical Code
CATEGORY_PRIORITY = {
    "url": 100,
    "email": 90,
    "path_windows": 80,
    "path_posix": 70,
    "uuid": 60,
    "technical_code": 50,
}

# Compiled regex patterns for entity recognition
PATTERNS = [
    # URLs (http, https)
    (
        "url",
        re.compile(
            r"https?://(?:[a-zA-Z0-9$_.+!*'(),;?&=-]|%[0-9a-fA-F]{2})+"
            r"(?:/(?:[a-zA-Z0-9$_.+!*'(),;?&=-]|%[0-9a-fA-F]{2})*)*",
            re.IGNORECASE,
        ),
    ),
    # Email addresses
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,10}\b",
            re.IGNORECASE,
        ),
    ),
    # Windows absolute or UNC paths (e.g. C:\Windows\..., \\server\share\...)
    (
        "path_windows",
        re.compile(
            r"(?:[a-zA-Z]:\\|\\\\)(?:[^\\/:*?\"<>|\r\n\s]+\\)*[^\\/:*?\"<>|\r\n\s]+",
            re.IGNORECASE,
        ),
    ),
    # Common POSIX paths starting with root directories (e.g. /etc/hosts, /var/log/syslog)
    (
        "path_posix",
        re.compile(
            r"(?:/(?:bin|boot|dev|etc|home|lib|media|mnt|opt|proc|root|run|sbin|srv|sys|tmp|usr|var|Users)"
            r"(?:/[a-zA-Z0-9_.-]+)+)",
            re.IGNORECASE,
        ),
    ),
    # Standard UUID (8-4-4-4-12 hex digits)
    (
        "uuid",
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
            re.IGNORECASE,
        ),
    ),
    # Certified technical alphanumeric codes with hyphen (e.g. ISO-9001, F-16, COVID-19)
    (
        "technical_code",
        re.compile(
            r"\b(?:ISO-\d{3,5}|[A-Z]{1,5}-\d{1,5}|COVID-19)\b",
            re.IGNORECASE,
        ),
    ),
]

# Pattern matching any shielded placeholder produced by this subsystem
SHIELD_PLACEHOLDER_REGEX = re.compile(r"§§§SHIELD_[0-9a-fA-F]+_\d+§§§")


class EntityShieldManager:
    """Manages collision-free entity shielding and verified deshielding."""

    @staticmethod
    def _generate_nonce(text: str) -> str:
        """Generate a random hex nonce guaranteed not to collide with user text."""
        while True:
            candidate = uuid.uuid4().hex[:12]
            if candidate not in text:
                return candidate

    @classmethod
    def shield(cls, text: str) -> Tuple[str, List[ShieldedEntity]]:
        """Identify and replace technical entities with non-colliding private placeholders.

        Args:
            text: Raw input text.

        Returns:
            Tuple of (shielded_text, list_of_ShieldedEntity).
        """
        if not text:
            return text, []

        candidates = []
        for category, pattern in PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span()
                # Strip trailing punctuation often caught in URLs/paths (.,;:)
                matched_str = match.group(0)
                while matched_str and matched_str[-1] in ".,;:!?)":
                    end -= 1
                    matched_str = text[start:end]

                if start < end:
                    candidates.append((start, end, category, matched_str))

        if not candidates:
            return text, []

        # Resolve overlaps by priority, then longest match
        # Sort candidates: start ascending, priority descending, length descending
        candidates.sort(
            key=lambda c: (
                c[0],
                -CATEGORY_PRIORITY.get(c[2], 0),
                -(c[1] - c[0]),
            )
        )

        selected: List[Tuple[int, int, str, str]] = []
        last_end = -1
        for start, end, category, matched_str in candidates:
            if start >= last_end:
                selected.append((start, end, category, matched_str))
                last_end = end

        if not selected:
            return text, []

        nonce = cls._generate_nonce(text)
        shielded_entities: List[ShieldedEntity] = []

        # Build replacement string from left to right
        result_parts = []
        current_idx = 0

        for i, (start, end, category, matched_str) in enumerate(selected):
            result_parts.append(text[current_idx:start])
            placeholder = f"§§§SHIELD_{nonce}_{i}§§§"
            result_parts.append(placeholder)
            shielded_entities.append(
                ShieldedEntity(
                    placeholder=placeholder,
                    original_text=matched_str,
                    category=category,
                    char_start=start,
                    char_end=end,
                )
            )
            current_idx = end

        result_parts.append(text[current_idx:])
        shielded_text = "".join(result_parts)

        return shielded_text, shielded_entities

    @classmethod
    def deshield(cls, text: str, shielded_entities: List[ShieldedEntity]) -> str:
        """Faithfully restore shielded entities and verify shielding integrity.

        Args:
            text: Normalized text containing placeholders.
            shielded_entities: List of ShieldedEntity recorded during Stage 1.

        Returns:
            Deshielded text with original technical entities restored.

        Raises:
            EntityShieldingCorruptionError: If placeholders are corrupted, missing, or mismatched.
        """
        if not shielded_entities:
            return text

        result = text
        for entity in shielded_entities:
            if entity.placeholder not in result:
                raise EntityShieldingCorruptionError(
                    f"Integrity check failed: placeholder '{entity.placeholder}' for "
                    f"original entity '{entity.original_text}' was lost or corrupted during normalization."
                )
            # Replace exactly once per placeholder
            result = result.replace(entity.placeholder, entity.original_text, 1)

        # Check for any lingering unexpected shield placeholders
        remaining = SHIELD_PLACEHOLDER_REGEX.findall(result)
        if remaining:
            raise EntityShieldingCorruptionError(
                f"Integrity check failed: unexpected lingering shield placeholders detected: {remaining}"
            )

        return result
