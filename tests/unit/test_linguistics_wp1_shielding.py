"""Unit tests for Milestone M2 Workpackage WP1 (Core Shielding, Offset Tracker & Shared Core Numbers).

Covers regression verifications for reviewer defects WP1-B1, WP1-B2, WP1-B3,
and numbers_core parameter validation (S1, S3, P4).
"""

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
    """Verifies Stage 1 (Shielding) and Stage 8 (Deshielding) entity preservation (WP1-B2, WP1-B3)."""

    def test_shield_and_deshield_url_with_port_query_fragment(self):
        """Verify WP1-B3: URL with port, query params and fragment is protected with exact span check."""
        text = "Visita il server http://localhost:8080/api/v1?id=10&k=2#sec per i dettagli."
        shielded, entities, nonce = EntityShieldManager.shield(text)

        assert len(entities) == 1
        ent = entities[0]
        assert ent.category == "url"
        expected_url = "http://localhost:8080/api/v1?id=10&k=2#sec"
        assert ent.original_text == expected_url

        # WP1-B3: Check exact protected interval on original text
        assert ent.char_start == 17
        assert ent.char_end == 17 + len(expected_url)
        assert text[ent.char_start:ent.char_end] == expected_url

        restored = EntityShieldManager.deshield(shielded, entities, session_nonce=nonce)
        assert restored == text

    def test_shield_and_deshield_www_address(self):
        """Verify WP1-B3: www. address without http:// protocol is shielded with exact coordinates."""
        text = "Consulta www.uniroma1.it/didattica/corsi?anno=2026#orari per l'orario."
        shielded, entities, nonce = EntityShieldManager.shield(text)

        assert len(entities) == 1
        ent = entities[0]
        assert ent.category == "url"
        expected_url = "www.uniroma1.it/didattica/corsi?anno=2026#orari"
        assert ent.original_text == expected_url

        # WP1-B3: Verify exact slice matching
        assert text[ent.char_start:ent.char_end] == expected_url
        assert ent.char_start == text.index("www.uniroma1.it")
        assert ent.char_end == ent.char_start + len(expected_url)

        restored = EntityShieldManager.deshield(shielded, entities, session_nonce=nonce)
        assert restored == text

    def test_shield_and_deshield_email(self):
        """Verify email address is shielded and restored faithfully."""
        text = "Invia una mail a info.supporto@studio-legale.it entro domani."
        shielded, entities, nonce = EntityShieldManager.shield(text)
        assert len(entities) == 1
        assert entities[0].category == "email"
        assert entities[0].original_text == "info.supporto@studio-legale.it"

        restored = EntityShieldManager.deshield(shielded, entities, session_nonce=nonce)
        assert restored == text

    def test_shield_and_deshield_windows_path(self):
        """Verify Windows file path with drive letter and backslashes is shielded."""
        text = r"Il file si trova in C:\Windows\System32\drivers\etc\hosts sul server."
        shielded, entities, nonce = EntityShieldManager.shield(text)
        assert len(entities) == 1
        assert entities[0].category == "path_windows"
        assert entities[0].original_text == r"C:\Windows\System32\drivers\etc\hosts"

        restored = EntityShieldManager.deshield(shielded, entities, session_nonce=nonce)
        assert restored == text

    def test_shield_and_deshield_posix_path(self):
        """Verify POSIX path starting with system directory is shielded."""
        text = "Controlla la configurazione in /etc/nginx/nginx.conf prima del riavvio."
        shielded, entities, nonce = EntityShieldManager.shield(text)
        assert len(entities) == 1
        assert entities[0].category == "path_posix"
        assert entities[0].original_text == "/etc/nginx/nginx.conf"

        restored = EntityShieldManager.deshield(shielded, entities, session_nonce=nonce)
        assert restored == text

    def test_shield_and_deshield_uuid(self):
        """Verify standard 36-char UUID is shielded."""
        text = "La chiave di sessione è 550e8400-e29b-41d4-a716-446655440000 nel database."
        shielded, entities, nonce = EntityShieldManager.shield(text)
        assert len(entities) == 1
        assert entities[0].category == "uuid"
        assert entities[0].original_text == "550e8400-e29b-41d4-a716-446655440000"

        restored = EntityShieldManager.deshield(shielded, entities, session_nonce=nonce)
        assert restored == text

    def test_shield_technical_codes(self):
        """Verify technical hyphenated codes (ISO-9001, F-16, COVID-19) are shielded."""
        text = "Lo standard ISO-9001, il velivolo F-16 e il virus COVID-19."
        shielded, entities, nonce = EntityShieldManager.shield(text)
        assert len(entities) == 3
        codes = [e.original_text for e in entities]
        assert "ISO-9001" in codes
        assert "F-16" in codes
        assert "COVID-19" in codes

        restored = EntityShieldManager.deshield(shielded, entities, session_nonce=nonce)
        assert restored == text

    def test_overlap_priority_url_over_path(self):
        """Verify URL takes priority over internal path segments."""
        text = "Vedi https://server.com/etc/hosts per dettagli."
        shielded, entities, nonce = EntityShieldManager.shield(text)
        assert len(entities) == 1
        assert entities[0].category == "url"
        assert entities[0].original_text == "https://server.com/etc/hosts"

    def test_wp1_b2_user_literal_placeholder_preservation(self):
        """Verify WP1-B2: user literal string containing §§§SHIELD_deadbeef1234_0§§§ passes cleanly."""
        text = "Visita https://example.org e scrivi §§§SHIELD_deadbeef1234_0§§§"
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        # Must pass without exceptions and leave the user's literal string intact
        assert res.normalized_text == text
        assert res.raw_text == text

    def test_deshield_raises_on_corrupted_placeholder(self):
        """Verify WP1-B2: EntityShieldingCorruptionError is raised if placeholder is altered."""
        text = "Visita https://example.org per info."
        shielded, entities, nonce = EntityShieldManager.shield(text)
        corrupted = shielded.replace("§§§SHIELD_", "§§§MUTATED_")
        with pytest.raises(EntityShieldingCorruptionError, match="lost or corrupted"):
            EntityShieldManager.deshield(corrupted, entities, session_nonce=nonce)

    def test_deshield_raises_on_duplicated_placeholder(self):
        """Verify WP1-B2: EntityShieldingCorruptionError is raised if placeholder is duplicated."""
        text = "Visita https://example.org per info."
        shielded, entities, nonce = EntityShieldManager.shield(text)
        # Duplicate the placeholder
        duplicated = shielded + " e ancora " + entities[0].placeholder
        with pytest.raises(EntityShieldingCorruptionError, match="duplicated"):
            EntityShieldManager.deshield(duplicated, entities, session_nonce=nonce)

    def test_composite_shielding_with_unicode(self):
        """Corresponds to M2-TC-081: URL + UUID + multi-byte quotes («, ») + prices."""
        text = (
            "L'URL è https://test.org/id?v=1&k=2, il prezzo è 15 € e l'UUID è "
            "550e8400-e29b-41d4-a716-446655440000 con caratteri «speciali»."
        )
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.raw_text == text
        assert "https://test.org/id?v=1&k=2" in res.normalized_text
        assert "550e8400-e29b-41d4-a716-446655440000" in res.normalized_text
        assert "«speciali»" in res.normalized_text


class TestWP1OffsetTracker:
    """Verifies coordinate mapping, delta compensation and WP1-B1 requirements."""

    def test_wp1_b1_at_least_three_sequential_replacements(self):
        """Verify WP1-B1: at least 3 length-changing replacements across the sentence."""
        raw = "Pagati 10 € per pane, 20 € per pasta e 30 € per frutta."
        tracker = OffsetTracker(raw)

        # 1st replacement: '10 €' at index 7..11 in raw (length 4 -> 14)
        tracker.record_replacement(7, 11, "dieci euro", "currency")
        assert raw[tracker.annotations[0].char_start:tracker.annotations[0].char_end] == "10 €"

        # 2nd replacement: find '20 €' in intermediate text
        idx2 = tracker.current_text.index("20 €")
        tracker.record_replacement(idx2, idx2 + 4, "venti euro", "currency")
        assert raw[tracker.annotations[1].char_start:tracker.annotations[1].char_end] == "20 €"

        # 3rd replacement: find '30 €' in intermediate text
        idx3 = tracker.current_text.index("30 €")
        tracker.record_replacement(idx3, idx3 + 4, "trenta euro", "currency")
        assert raw[tracker.annotations[2].char_start:tracker.annotations[2].char_end] == "30 €"

        assert len(tracker.annotations) == 3
        assert tracker.annotations[0].raw_span == "10 €"
        assert tracker.annotations[1].raw_span == "20 €"
        assert tracker.annotations[2].raw_span == "30 €"
        assert (
            tracker.current_text
            == "Pagati dieci euro per pane, venti euro per pasta e trenta euro per frutta."
        )

    def test_wp1_b1_multiple_successive_modifications_to_same_region(self):
        """Verify WP1-B1: successive modifications to the same region do not double-remap."""
        raw = "Il totale dovuto è 50 €."
        tracker = OffsetTracker(raw)

        idx = tracker.current_text.index("50 €")
        # 1st pass: 50 € -> cinquanta euro
        tracker.record_replacement(idx, idx + 4, "cinquanta euro", "currency")
        ann1 = tracker.annotations[-1]
        assert ann1.char_start == 19
        assert ann1.char_end == 23
        assert raw[ann1.char_start:ann1.char_end] == "50 €"

        # 2nd pass: refine 'cinquanta euro' -> 'cinquanta euro tondi' in the SAME region
        idx_refined = tracker.current_text.index("cinquanta euro")
        tracker.record_replacement(
            idx_refined, idx_refined + len("cinquanta euro"), "cinquanta euro tondi", "currency"
        )
        ann2 = tracker.annotations[-1]

        # Crucial WP1-B1 check: char_start and char_end must remain [19, 23) in original raw_text!
        assert ann2.char_start == 19
        assert ann2.char_end == 23
        assert raw[ann2.char_start:ann2.char_end] == "50 €"
        assert tracker.current_text == "Il totale dovuto è cinquanta euro tondi."

    def test_wp1_b1_shielded_url_before_numbers(self):
        """Verify WP1-B1: shielded URL before numbers aligns tracker with shielded text and maps correctly."""
        raw = "Vedi https://example.org/api:8080 per il prezzo di 50 €."
        # Step 1: Shield the URL
        shielded, entities, nonce = EntityShieldManager.shield(raw)
        tracker = OffsetTracker(raw)
        tracker.apply_shielding(entities)

        # Verify tracker.current_text is strictly aligned with shielded_text
        assert tracker.current_text == shielded

        # Step 2: Normalization in intermediate text (WP2-WP5 simulation)
        idx_50 = tracker.current_text.index("50 €")
        tracker.record_replacement(idx_50, idx_50 + 4, "cinquanta euro", "currency")

        ann = tracker.annotations[-1]
        # Verify the mapped coordinates on raw_text point exactly to "50 €"
        assert ann.raw_span == "50 €"
        assert raw[ann.char_start:ann.char_end] == "50 €"

        # Step 3: Deshield in tracker
        for ent in entities:
            tracker.deshield_entity(ent.placeholder, ent.original_text)

        assert (
            tracker.current_text
            == "Vedi https://example.org/api:8080 per il prezzo di cinquanta euro."
        )

    def test_wp1_b1a_exact_shielding_alignment_selective_occurrence(self):
        """Verify WP1-B1a: exact shielding alignment without find() on ambiguous substrings.

        Test string: 'Codice 123ISO-9001, certificazione ISO-9001.'
        Verifies that exclusively the second occurrence is shielded, tracker.current_text
        matches shielded_text identically, and deshielding restores original text.
        """
        text = "Codice 123ISO-9001, certificazione ISO-9001."
        shielded_text, entities, nonce = EntityShieldManager.shield(text)

        # Only the second occurrence (isolated code with word boundary) must be shielded
        assert len(entities) == 1
        assert entities[0].original_text == "ISO-9001"
        assert entities[0].char_start == 35
        assert entities[0].char_end == 43
        assert text[entities[0].char_start:entities[0].char_end] == "ISO-9001"

        tracker = OffsetTracker(text)
        tracker.apply_shielding(entities)

        # WP1-B1a: tracker.current_text must coincide EXACTLY with shielded_text
        assert tracker.current_text == shielded_text
        # First occurrence (123ISO-9001) must NOT be shielded in tracker
        assert "123ISO-9001" in tracker.current_text
        assert f"123{entities[0].placeholder}" not in tracker.current_text

        # Stage 8: Deshielding in tracker
        tracker.deshield_entity(entities[0].placeholder, entities[0].original_text)
        assert tracker.current_text == text

        # Verify end-to-end through normalizer pipeline
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == text
        assert res.raw_text == text

    def test_wp1_b1b_partial_modification_of_expanded_content_provenance(self):
        """Verify WP1-B1b: explicit provenance rule for partial modifications of expanded content.

        Expanded content represents an indivisible semantic unit originating from [raw_start, raw_end).
        Partial sub-modifications must NOT expose arbitrary linearly interpolated raw coordinates.
        They must inherit the enclosing raw source span without fractional character slicing.
        """
        raw = "Il prezzo è 15 € per articolo."
        tracker = OffsetTracker(raw)

        # 1st pass: 15 € (chars 12..16) -> "quindici euro"
        tracker.record_replacement(12, 16, "quindici euro", "currency")
        ann1 = tracker.annotations[-1]
        assert ann1.char_start == 12
        assert ann1.char_end == 16
        assert ann1.raw_span == "15 €"
        assert raw[ann1.char_start:ann1.char_end] == "15 €"

        # 2nd pass: partial modification within the expanded region!
        # Modify only the sub-segment 'euro' inside 'quindici euro' -> 'euro netti'
        idx_euro = tracker.current_text.index("euro")
        tracker.record_replacement(idx_euro, idx_euro + 4, "euro netti", "currency")
        ann2 = tracker.annotations[-1]

        # Explicit WP1-B1b provenance rule verification:
        # Rather than inventing an arbitrary fractional raw offset (e.g. 14..16 or slicing '5 €'),
        # the sub-modification inherits the full valid enclosing provenance [12, 16) = "15 €".
        assert ann2.char_start == 12
        assert ann2.char_end == 16
        assert ann2.raw_span == "15 €"
        assert raw[ann2.char_start:ann2.char_end] == "15 €"
        assert tracker.current_text == "Il prezzo è quindici euro netti per articolo."


class TestWP1SharedCoreNumbers:
    """Verifies cardinal_to_italian, ordinal_to_italian and secondary observation parameter validation."""

    def test_cardinals_under_20(self):
        assert cardinal_to_italian(0) == "zero"
        assert cardinal_to_italian(1) == "uno"
        assert cardinal_to_italian(3) == "tre"
        assert cardinal_to_italian(11) == "undici"
        assert cardinal_to_italian(14) == "quattordici"
        assert cardinal_to_italian(19) == "diciannove"

    def test_tens_truncation_before_one_and_eight(self):
        assert cardinal_to_italian(21) == "ventuno"
        assert cardinal_to_italian(28) == "ventotto"
        assert cardinal_to_italian(31) == "trentuno"
        assert cardinal_to_italian(38) == "trentotto"
        assert cardinal_to_italian(41) == "quarantuno"
        assert cardinal_to_italian(48) == "quarantotto"
        assert cardinal_to_italian(81) == "ottantuno"
        assert cardinal_to_italian(88) == "ottantotto"

    def test_compounds_of_three_acute_accent(self):
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

    def test_boundary_and_edge_case_cardinals(self):
        """Secondary observation: edge-cases up to upper limit."""
        assert cardinal_to_italian(100_000) == "centomila"
        assert cardinal_to_italian(1_000_001) == "un milione uno"
        assert cardinal_to_italian(1_001_000) == "un milione mille"
        assert cardinal_to_italian(1_000_000_001) == "un miliardo uno"
        max_val = 999_999_999_999
        res = cardinal_to_italian(max_val)
        assert res.startswith("novecentonovantanove miliardi")
        assert res.endswith("novecentonovantanovemilanovecentonovantanove")

    def test_parameter_validation_cardinals(self):
        """Secondary observation: strict argument validation."""
        # Type errors
        with pytest.raises(TypeError, match="requires an int"):
            cardinal_to_italian(3.14)  # type: ignore
        with pytest.raises(TypeError, match="requires an int"):
            cardinal_to_italian("100")  # type: ignore
        with pytest.raises(TypeError, match="requires an int"):
            cardinal_to_italian(True)  # type: ignore

        # Value errors
        with pytest.raises(ValueError, match="non-negative"):
            cardinal_to_italian(-1)
        with pytest.raises(ValueError, match="upper limit"):
            cardinal_to_italian(1_000_000_000_000)

    def test_ordinals(self):
        assert ordinal_to_italian(1, "m") == "primo"
        assert ordinal_to_italian(1, "f") == "prima"
        assert ordinal_to_italian(2, "m") == "secondo"
        assert ordinal_to_italian(10, "m") == "decimo"
        assert ordinal_to_italian(21, "m") == "ventunesimo"
        assert ordinal_to_italian(23, "m") == "ventitreesimo"

    def test_parameter_validation_ordinals(self):
        """Secondary observation: ordinal validation."""
        with pytest.raises(TypeError, match="requires an int"):
            ordinal_to_italian(False)  # type: ignore
        with pytest.raises(ValueError, match="must be >= 1"):
            ordinal_to_italian(0)
        with pytest.raises(ValueError, match="gender"):
            ordinal_to_italian(5, gender="x")  # type: ignore
        with pytest.raises(ValueError, match="upper limit"):
            ordinal_to_italian(1_000_000_000_000)
