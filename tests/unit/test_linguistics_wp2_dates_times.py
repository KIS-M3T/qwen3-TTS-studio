"""Unit tests for Milestone M2 Workpackage WP2 (Dates and Times).

Verifies:
- Full Gregorian dates (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY)
- Abbreviated numeric dates (DD/MM, DD-MM)
- Alphanumeric dates with named months ('25 aprile 1945')
- Mandatory rule of the first of the month ('1°', '01/', '1/' -> 'primo')
- Conservative No-Op on invalid calendar dates (31/02, 29/02 non-leap, 32/13)
- 24-hour clock times (HH:MM) with natural phonotactic contractions ('alle 12:00' -> 'a mezzogiorno')
- Clock times with seconds (HH:MM:SS) in confirmed clock contexts ('dodici, quindici minuti e trenta secondi')
- Chronometric durations in confirmed timing contexts ('un'ora, quindici minuti e trenta secondi')
- Conservative No-Op on ambiguous uncontextualized HH:MM:SS ('12:15:30')
- Conservative No-Op on impossible hours and minutes ('25:70', '12:99')
- Offset tracking, TNAnnotation category ('date' / 'time') and raw_text coordinate alignment
- Composite sentences with shielded URLs, email and technical codes
"""

import pytest

from src.linguistics.dates import DateNormalizer, is_valid_gregorian_date
from src.linguistics.normalizer import ItalianTextNormalizer
from src.linguistics.times import TimeNormalizer
from src.linguistics.tracker import OffsetTracker


class TestWP2Dates:
    """Verifies Gregorian date normalization, 1st-of-month rule, and invalid date No-Ops."""

    def test_gregorian_date_validation_helper(self):
        """Verify is_valid_gregorian_date detects leap years and out-of-range days/months."""
        assert is_valid_gregorian_date(15, 3, 2026) is True
        assert is_valid_gregorian_date(29, 2, 2024) is True  # Leap year
        assert is_valid_gregorian_date(29, 2, 2025) is False  # Non-leap year
        assert is_valid_gregorian_date(31, 2, 2026) is False  # 31 Feb impossible
        assert is_valid_gregorian_date(31, 4, 2026) is False  # 30 days in April
        assert is_valid_gregorian_date(32, 13, 2024) is False  # Month 13 out of range
        assert is_valid_gregorian_date(15, 8, None) is True
        assert is_valid_gregorian_date(31, 2, None) is False

    def test_standard_numeric_date_dd_mm_yyyy(self):
        """Corresponds to M2-TC-017: 15/03/2026 -> quindici marzo duemilaventisei."""
        text = "La conferenza si terrà il 15/03/2026."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "La conferenza si terrà il quindici marzo duemilaventisei."
        assert len(res.annotations) == 1
        ann = res.annotations[0]
        assert ann.category == "date"
        assert ann.raw_span == "15/03/2026"
        assert ann.expanded_span == "quindici marzo duemilaventisei"
        assert text[ann.char_start:ann.char_end] == "15/03/2026"

    def test_first_of_month_with_ordinal_symbol(self):
        """Corresponds to M2-TC-018: 1° maggio -> primo maggio."""
        text = "La festa dei lavoratori è il 1° maggio."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "La festa dei lavoratori è il primo maggio."
        assert len(res.annotations) == 1
        ann = res.annotations[0]
        assert ann.category == "date"
        assert ann.raw_span == "1° maggio"
        assert ann.expanded_span == "primo maggio"
        assert text[ann.char_start:ann.char_end] == "1° maggio"

    def test_first_of_month_numeric_slash(self):
        """Corresponds to M2-TC-019: 01/11/2024 -> primo novembre duemilaventiquattro."""
        text = "L'evento inizia il 01/11/2024."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "L'evento inizia il primo novembre duemilaventiquattro."
        assert len(res.annotations) == 1
        ann = res.annotations[0]
        assert ann.category == "date"
        assert ann.raw_span == "01/11/2024"
        assert ann.expanded_span == "primo novembre duemilaventiquattro"
        assert text[ann.char_start:ann.char_end] == "01/11/2024"

    def test_date_with_named_month_and_historical_year(self):
        """Corresponds to M2-TC-020: 25 aprile 1945 -> venticinque aprile millenovecentoquarantacinque."""
        text = "La liberazione avvenne il 25 aprile 1945."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert (
            res.normalized_text
            == "La liberazione avvenne il venticinque aprile millenovecentoquarantacinque."
        )
        assert len(res.annotations) == 1
        ann = res.annotations[0]
        assert ann.category == "date"
        assert ann.raw_span == "25 aprile 1945"
        assert text[ann.char_start:ann.char_end] == "25 aprile 1945"

    def test_date_without_year_dd_mm(self):
        """Corresponds to M2-AB-003: 15/08 -> quindici agosto."""
        text = "Ci vediamo il 15/08 per il pranzo di Ferragosto."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "Ci vediamo il quindici agosto per il pranzo di Ferragosto."
        assert len(res.annotations) == 1
        assert res.annotations[0].category == "date"
        assert res.annotations[0].raw_span == "15/08"

    def test_first_of_month_without_year(self):
        """Verify 01/03 without year expands to 'primo marzo'."""
        text = "Il termine scade il 01/03."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "Il termine scade il primo marzo."

    def test_date_format_delimiters_dash_and_dot(self):
        """Verify hyphens and dots as valid date delimiters."""
        normalizer = ItalianTextNormalizer()
        res1 = normalizer.normalize("Data: 15-03-2026.")
        assert res1.normalized_text == "Data: quindici marzo duemilaventisei."

        res2 = normalizer.normalize("Data: 15.03.2026.")
        assert res2.normalized_text == "Data: quindici marzo duemilaventisei."

    def test_impossible_date_31_february_noop(self):
        """Corresponds to M2-TC-066: 31/02/2026 preserved intact (No-Op)."""
        text = "La riunione è fissata per il 31/02/2026."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == text
        assert res.annotations == []

    def test_impossible_date_non_leap_year_29_february_noop(self):
        """Corresponds to M2-TC-067: 29/02/2025 preserved intact (No-Op)."""
        text = "Il contratto scade il 29/02/2025."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == text
        assert res.annotations == []

    def test_leap_year_29_february_accepted(self):
        """Verify 29/02/2024 is accepted and normalized on a leap year."""
        text = "Il giorno 29/02/2024 era bisestile."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "Il giorno ventinove febbraio duemilaventiquattro era bisestile."
        assert len(res.annotations) == 1

    def test_impossible_date_out_of_range_noop(self):
        """Corresponds to M2-TC-068: 32/13/2024 preserved intact (No-Op)."""
        text = "Il documento riporta la data 32/13/2024."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == text
        assert res.annotations == []


class TestWP2Times:
    """Verifies 24h times, durations, clock with seconds, and impossible time No-Ops."""

    def test_standard_24h_time_hh_mm(self):
        """Corresponds to M2-TC-021: 14:30 -> quattordici e trenta."""
        text = "Il treno parte alle 14:30 dal binario quattro."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "Il treno parte alle quattordici e trenta dal binario quattro."
        assert len(res.annotations) == 1
        ann = res.annotations[0]
        assert ann.category == "time"
        assert ann.raw_span == "alle 14:30"
        assert text[ann.char_start:ann.char_end] == "alle 14:30"

    def test_time_null_minutes_alle_08_00(self):
        """Corresponds to M2-TC-022: alle 08:00 -> alle otto."""
        text = "L'ufficio apre alle 08:00 precise."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "L'ufficio apre alle otto precise."
        assert len(res.annotations) == 1
        assert res.annotations[0].raw_span == "alle 08:00"
        assert text[res.annotations[0].char_start:res.annotations[0].char_end] == "alle 08:00"

    def test_time_noon_with_preposition_mezzogiorno(self):
        """Corresponds to M2-TC-024: alle 12:00 -> a mezzogiorno."""
        text = "La seduta è convocata alle 12:00."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "La seduta è convocata a mezzogiorno."
        assert len(res.annotations) == 1
        ann = res.annotations[0]
        assert ann.category == "time"
        assert ann.raw_span == "alle 12:00"
        assert ann.expanded_span == "a mezzogiorno"
        assert text[ann.char_start:ann.char_end] == "alle 12:00"

    def test_time_midnight_with_preposition_mezzanotte(self):
        """Verify alle 00:00 -> a mezzanotte."""
        text = "Il silenzio cala alle 00:00."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "Il silenzio cala a mezzanotte."
        assert len(res.annotations) == 1
        assert res.annotations[0].raw_span == "alle 00:00"

    def test_time_one_o_clock_all_una(self):
        """Verify alle 01:00 -> all'una and alle 01:30 -> all'una e trenta."""
        normalizer = ItalianTextNormalizer()
        res1 = normalizer.normalize("La pausa finisce alle 01:00.")
        assert res1.normalized_text == "La pausa finisce all'una."

        res2 = normalizer.normalize("Ci vediamo alle 01:30.")
        assert res2.normalized_text == "Ci vediamo all'una e trenta."

    def test_time_clock_with_seconds_context(self):
        """Corresponds to M2-TC-023: 12:15:30 in clock context -> dodici, quindici minuti e trenta secondi."""
        text = "L'orologio della torre segnava le 12:15:30."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert (
            res.normalized_text
            == "L'orologio della torre segnava le dodici, quindici minuti e trenta secondi."
        )
        assert len(res.annotations) == 1
        ann = res.annotations[0]
        assert ann.category == "time"
        assert ann.raw_span == "12:15:30"
        assert text[ann.char_start:ann.char_end] == "12:15:30"

    def test_time_chronometric_duration_context(self):
        """Corresponds to M2-TC-023-B: 01:15:30 in duration context -> un'ora, quindici minuti e trenta secondi."""
        text = "Il tempo registrato dal cronometro è di 01:15:30."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert (
            res.normalized_text
            == "Il tempo registrato dal cronometro è di un'ora, quindici minuti e trenta secondi."
        )
        assert len(res.annotations) == 1
        ann = res.annotations[0]
        assert ann.category == "time"
        assert ann.raw_span == "01:15:30"
        assert text[ann.char_start:ann.char_end] == "01:15:30"

    def test_time_ambiguous_noop_preservation(self):
        """Corresponds to M2-TC-023-C: 12:15:30 in ambiguous/neutral context is preserved intact (No-Op)."""
        text = "Il display di collaudo riporta il valore 12:15:30."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == text
        assert res.annotations == []

    def test_impossible_time_hour_out_of_range_noop(self):
        """Corresponds to M2-TC-069: alle 25:70 preserved intact (No-Op)."""
        text = "L'allarme è scattato alle 25:70 di notte."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == text
        assert res.annotations == []

    def test_impossible_time_minute_out_of_range_noop(self):
        """Corresponds to M2-TC-070: 12:99 preserved intact (No-Op)."""
        text = "Il log indica un orario di 12:99 nel sistema."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == text
        assert res.annotations == []

    def test_wp2_b1_partial_match_prevention_impossible_seconds(self):
        """Verify WP2-B1: HH:MM prefix inside invalid HH:MM:SS is NOT partially normalized."""
        normalizer = ItalianTextNormalizer()
        cases = [
            "12:15:60",
            "Il sensore riporta 12:15:60 alle porte.",
            "L'appuntamento era alle 12:15:99.",
            "alle 12:15:99",
            "L'orologio segnava le 12:15:60.",
        ]
        for text in cases:
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []

    def test_wp2_b1_partial_match_prevention_longer_colon_chains(self):
        """Verify WP2-B1: HH:MM or HH:MM:SS inside longer colon-chains is preserved as No-Op."""
        normalizer = ItalianTextNormalizer()
        cases = [
            "12:15:30:40",
            "Il timestamp tecnico è 12:15:30:40.",
            "L'orologio segnava le 12:15:30:40.",
            "Il cronometro è di 01:15:30:40.",
            "alle 12:15:30:40",
        ]
        for text in cases:
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []

    def test_wp2_b2_context_gating_false_clock_marker_in_window(self):
        """Verify WP2-B2: unrelated 'alle' in prior clause or non-governing phrase stays No-Op."""
        normalizer = ItalianTextNormalizer()
        text1 = "Siamo alle porte, il display mostra 12:15:30."
        res1 = normalizer.normalize(text1)
        assert res1.normalized_text == text1
        assert res1.annotations == []

        text2 = "Siamo alle porte e il display mostra 12:15:30."
        res2 = normalizer.normalize(text2)
        assert res2.normalized_text == text2
        assert res2.annotations == []

    def test_wp2_b2_context_gating_false_duration_marker_across_sentence(self):
        """Verify WP2-B2: 'cronometro' in prior sentence or non-governing phrase stays No-Op."""
        normalizer = ItalianTextNormalizer()
        text1 = "Il cronometro è stato riposto. Il display mostra 01:15:30."
        res1 = normalizer.normalize(text1)
        assert res1.normalized_text == text1
        assert res1.annotations == []

        text2 = "Il cronometro è fermo e il display mostra 01:15:30."
        res2 = normalizer.normalize(text2)
        assert res2.normalized_text == text2
        assert res2.annotations == []

    def test_night_time_23_45(self):
        """Corresponds to M2-AB-004: alle 23:45 -> alle ventitré e quarantacinque."""
        text = "Il treno parte alle 23:45 dal binario otto."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "Il treno parte alle ventitré e quarantacinque dal binario otto."
        assert len(res.annotations) == 1
        assert res.annotations[0].category == "time"


class TestWP2CompositeAndOffsetTracking:
    """Verifies interaction between dates, times, shielded entities, and coordinate mapping."""

    def test_composite_date_and_time_same_sentence(self):
        """Corresponds to M2-AB-001: 1° ottobre + alle 09:00 in the same sentence."""
        text = "La seduta è fissata per il 1° ottobre alle 09:00."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "La seduta è fissata per il primo ottobre alle nove."
        assert len(res.annotations) == 2

        # 1st annotation: date
        ann_date = res.annotations[0]
        assert ann_date.category == "date"
        assert ann_date.raw_span == "1° ottobre"
        assert text[ann_date.char_start:ann_date.char_end] == "1° ottobre"

        # 2nd annotation: time
        ann_time = res.annotations[1]
        assert ann_time.category == "time"
        assert ann_time.raw_span == "alle 09:00"
        assert text[ann_time.char_start:ann_time.char_end] == "alle 09:00"

    def test_date_and_time_with_shielded_url_and_code(self):
        """Verify date and time normalization alongside shielded URL with ports and technical code."""
        text = (
            "Collegati su https://server.com:8443/api in data 15/03/2026 alle 14:30 "
            "per la verifica del certificato ISO-9001."
        )
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)

        # URL and ISO code must remain shielded and restored intact
        assert "https://server.com:8443/api" in res.normalized_text
        assert "ISO-9001" in res.normalized_text

        # Date and time must be properly normalized
        assert "quindici marzo duemilaventisei" in res.normalized_text
        assert "alle quattordici e trenta" in res.normalized_text

        # Annotations check
        categories = [a.category for a in res.annotations]
        assert "date" in categories
        assert "time" in categories

        for ann in res.annotations:
            assert text[ann.char_start:ann.char_end] == ann.raw_span

    def test_temporal_idempotence(self):
        """Verify that normalizing an already normalized temporal string yields an identical result."""
        text = "Riunione il 15/03/2026 alle 14:30."
        normalizer = ItalianTextNormalizer()
        res1 = normalizer.normalize(text)
        res2 = normalizer.normalize(res1.normalized_text)
        assert res2.normalized_text == res1.normalized_text
        # Second run should produce zero new annotations because numbers are already spelled out in letters
        assert len(res2.annotations) == 0
