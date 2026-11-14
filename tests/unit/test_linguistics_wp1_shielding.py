"""Unit tests for Milestone M2 Workpackage WP1 (Core Shielding, Offset Tracker & Shared Core Numbers)."""

import pytest

from src.linguistics import (
    EntityShieldManager,
    EntityShieldingCorruptionError,
    ItalianNormalizerConfig,
    ItalianTextNormalizer,
    OffsetTracker,
    ShieldedEntity,
    cardinal_to_italian,
    ordinal_to_italian,
)


class TestWP1EntityShielding:
    """Verifies Stage 1 (Shielding) and Stage 8 (Deshielding) entity preservation (R1, R2, S3, P4)."""

    def test_shield_and_deshield_url(self):
        """Verify URL with query parameters, slashes and dots is preserved intact."""
        text = "Visita il sito https://www.uniroma1.it/it/ateneo?v=1&k=2 per informazioni."
        shielded, entities = EntityShieldManager.shield(text)
        assert len(entities) == 1
        assert entities[0].category == "url"
        assert entities[0].original_text == "https://www.uniroma1.it/it/ateneo?v=1&k=2"
        assert entities[0].placeholder in shielded

        restored = EntityShieldManager.deshield(shielded, entities)
        assert restored == text

    def test_shield_and_deshield_email(self):
        """Verify email address is shielded and restored faithfully."""
        text = "Invia una mail a info.supporto@studio-legale.it entro domani."
        shielded, entities = EntityShieldManager.shield(text)
        assert len(entities) == 1
        assert entities[0].category == "email"
        assert entities[0].original_text == "info.supporto@studio-legale.it"

        restored = EntityShieldManager.deshield(shielded, entities)
        assert restored == text

    def test_shield_and_deshield_windows_path(self):
        """Verify Windows file path with drive letter and backslashes is shielded."""
        text = r"Il file si trova in C:\Windows\System32\drivers\etc\hosts sul server."
        shielded, entities = EntityShieldManager.shield(text)
        assert len(entities) == 1
        assert entities[0].category == "path_windows"
        assert entities[0].original_text == r"C:\Windows\System32\drivers\etc\hosts"

        restored = EntityShieldManager.deshield(shielded, entities)
        assert restored == text

    def test_shield_and_deshield_posix_path(self):
        """Verify POSIX path starting with system directory is shielded."""
        text = "Controlla la configurazione in /etc/nginx/nginx.conf prima del riavvio."
        shielded, entities = EntityShieldManager.shield(text)
        assert len(entities) == 1
        assert entities[0].category == "path_posix"
        assert entities[0].original_text == "/etc/nginx/nginx.conf"

        restored = EntityShieldManager.deshield(shielded, entities)
        assert restored == text

    def test_shield_and_deshield_uuid(self):
        """Verify standard 36-char UUID is shielded."""
        text = "La chiave di sessione è 550e8400-e29b-41d4-a716-446655440000 nel database."
        shielded, entities = EntityShieldManager.shield(text)
        assert len(entities) == 1
        assert entities[0].category == "uuid"
        assert entities[0].original_text == "550e8400-e29b-41d4-a716-446655440000"

        restored = EntityShieldManager.deshield(shielded, entities)
        assert restored == text

    def test_shield_technical_codes(self):
        """Verify technical hyphenated codes (ISO-9001, F-16, COVID-19) are shielded."""
        text = "Lo standard ISO-9001, il velivolo F-16 e il virus COVID-19."
        shielded, entities = EntityShieldManager.shield(text)
        assert len(entities) == 3
        codes = [e.original_text for e in entities]
        assert "ISO-9001" in codes
        assert "F-16" in codes
        assert "COVID-19" in codes

        restored = EntityShieldManager.deshield(shielded, entities)
        assert restored == text

    def test_overlap_priority_url_over_path(self):
        """Verify URL takes priority over internal path segments."""
        text = "Vedi https://server.com/etc/hosts per dettagli."
        shielded, entities = EntityShieldManager.shield(text)
        assert len(entities) == 1
        assert entities[0].category == "url"
        assert entities[0].original_text == "https://server.com/etc/hosts"

    def test_placeholder_non_collision_with_user_text(self):
        """Verify that user text containing literal placeholder strings does not collide."""
        text = "Il testo contiene §§§SHIELD_0§§§ e __PROTECTED_ENTITY_1__ come stringhe."
        shielded, entities = EntityShieldManager.shield(text)
        # The user strings should not be corrupted or misinterpreted as active placeholders
        restored = EntityShieldManager.deshield(shielded, entities)
        assert restored == text

    def test_composite_shielding_with_unicode(self):
        """Corresponds to M2-TC-081: URL + UUID + multi-byte quotes («, ») + prices."""
        text = (
            "L'URL è https://test.org/id?v=1&k=2, il prezzo è 15 € e l'UUID è "
            "550e8400-e29b-41d4-a716-446655440000 con caratteri «speciali»."
        )
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.raw_text == text
        # URL, UUID and typographic quotes must remain completely intact
        assert "https://test.org/id?v=1&k=2" in res.normalized_text
        assert "550e8400-e29b-41d4-a716-446655440000" in res.normalized_text
        assert "«speciali»" in res.normalized_text

    def test_deshield_raises_on_corrupted_placeholder(self):
        """Verify EntityShieldingCorruptionError is raised if placeholder is altered."""
        text = "Visita https://example.org per info."
        shielded, entities = EntityShieldManager.shield(text)
        # Corrupt the placeholder in intermediate text
        corrupted = shielded.replace("§§§SHIELD_", "§§§MUTATED_")
        with pytest.raises(EntityShieldingCorruptionError, match="lost or corrupted"):
            EntityShieldManager.deshield(corrupted, entities)


class TestWP1OffsetTracker:
    """Verifies coordinate mapping and delta compensation in OffsetTracker."""

    def test_single_replacement_offset_mapping(self):
        """Verify single replacement maps original coordinates exactly."""
        raw = "Ho speso 50 € per il pranzo."
        tracker = OffsetTracker(raw)
        # Replace "50 €" (indices 9..13 in raw, length 4) with "cinquanta euro" (length 14)
        ann = tracker.record_replacement(
            intermediate_start=9,
            intermediate_end=13,
            replacement="cinquanta euro",
            category="currency",
        )
        assert ann.char_start == 9
        assert ann.char_end == 13
        assert ann.raw_span == "50 €"
        assert ann.expanded_span == "cinquanta euro"
        assert tracker.current_text == "Ho speso cinquanta euro per il pranzo."

    def test_multiple_sequential_replacements_offset_mapping(self):
        """Verify multiple length-changing replacements maintain correct raw_text coordinates."""
        raw = "Da 10 a 20 persone."
        tracker = OffsetTracker(raw)

        # First replacement: "10" at [3, 5) -> "dieci" (length 2 -> 5, delta = +3)
        ann1 = tracker.record_replacement(3, 5, "dieci", "cardinal")
        assert ann1.char_start == 3
        assert ann1.char_end == 5
        assert ann1.raw_span == "10"

        # Intermediate text is now "Da dieci a 20 persone."
        # In intermediate text, "20" is at index 11 (was at 8 in raw, shifted by +3)
        idx_20 = tracker.current_text.index("20")
        assert idx_20 == 11

        # Second replacement: "20" at [11, 13) -> "venti"
        ann2 = tracker.record_replacement(idx_20, idx_20 + 2, "venti", "cardinal")
        # In original raw_text, "20" was at [8, 10)
        assert ann2.char_start == 8
        assert ann2.char_end == 10
        assert ann2.raw_span == "20"
        assert ann2.expanded_span == "venti"


class TestWP1SharedCoreNumbers:
    """Verifies cardinal_to_italian and ordinal_to_italian conforming to Italian orthography."""

    def test_cardinals_under_20(self):
        assert cardinal_to_italian(0) == "zero"
        assert cardinal_to_italian(1) == "uno"
        assert cardinal_to_italian(3) == "tre"
        assert cardinal_to_italian(11) == "undici"
        assert cardinal_to_italian(14) == "quattordici"
        assert cardinal_to_italian(19) == "diciannove"

    def test_tens_truncation_before_one_and_eight(self):
        """Verify phonotactic truncation: drop final vowel before uno (1) and otto (8)."""
        assert cardinal_to_italian(21) == "ventuno"
        assert cardinal_to_italian(28) == "ventotto"
        assert cardinal_to_italian(31) == "trentuno"
        assert cardinal_to_italian(38) == "trentotto"
        assert cardinal_to_italian(41) == "quarantuno"
        assert cardinal_to_italian(48) == "quarantotto"
        assert cardinal_to_italian(81) == "ottantuno"
        assert cardinal_to_italian(88) == "ottantotto"

    def test_compounds_of_three_acute_accent(self):
        """Verify Treccani rule: compounds ending in 3 require acute accent (ventitré)."""
        assert cardinal_to_italian(3) == "tre"
        assert cardinal_to_italian(23) == "ventitré"
        assert cardinal_to_italian(33) == "trentatré"
        assert cardinal_to_italian(43) == "quarantatré"
        assert cardinal_to_italian(93) == "novantatré"

    def test_hundreds_and_elision(self):
        assert cardinal_to_italian(100) == "cento"
        assert cardinal_to_italian(105) == "centocinque"
        assert cardinal_to_italian(180) == "centottanta"
        assert cardinal_to_italian(200) == "duecento"
        assert cardinal_to_italian(999) == "novecentonovantanove"

    def test_thousands_and_historical_years(self):
        assert cardinal_to_italian(1000) == "mille"
        assert cardinal_to_italian(2000) == "duemila"
        assert cardinal_to_italian(1945) == "millenovecentoquarantacinque"
        assert cardinal_to_italian(2026) == "duemilaventisei"

    def test_millions_and_billions(self):
        assert cardinal_to_italian(1_000_000) == "un milione"
        assert cardinal_to_italian(2_000_000) == "due milioni"
        assert cardinal_to_italian(1_250_000) == "un milione duecentocinquantamila"
        assert cardinal_to_italian(1_000_000_000) == "un miliardo"
        assert cardinal_to_italian(2_500_000_000) == "due miliardi cinquecento milioni"

    def test_negative_input_raises_value_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            cardinal_to_italian(-5)

    def test_ordinals(self):
        assert ordinal_to_italian(1, "m") == "primo"
        assert ordinal_to_italian(1, "f") == "prima"
        assert ordinal_to_italian(2, "m") == "secondo"
        assert ordinal_to_italian(10, "m") == "decimo"
        assert ordinal_to_italian(21, "m") == "ventunesimo"
        assert ordinal_to_italian(23, "m") == "ventitreesimo"

        with pytest.raises(ValueError, match="must be >= 1"):
            ordinal_to_italian(0)
