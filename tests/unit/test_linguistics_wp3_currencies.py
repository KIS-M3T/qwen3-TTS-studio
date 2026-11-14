"""Unit tests for Milestone M2 Workpackage WP3 (Currencies and Monetary Amounts).

Verifies formal monetary register (Decision D5.A):
- Euro suffix/prefix symbols (€, EUR) and strict plural invariance ('euro', never 'euri')
- Foreign currencies ($, USD -> dollaro/dollari; £, GBP -> sterlina/sterline)
- Singular agreement ('un euro', 'un centesimo', 'un dollaro', 'una sterlina')
- Fractional cents ('0,01 €' -> 'un centesimo', '0,50 €' -> 'cinquanta centesimi')
- Explicit zero amounts ('0 €', '0,00 €' -> 'zero euro')
- Negative amounts ('-15,00 €' -> 'meno quindici euro')
- Italian thousands separators ('€ 2.500', '1.250,50 €', '1.000.000 €')
- Safety guards: numeric ranges ('10-20 €', '€ 10-20'), malformed decimals ('12,500 €', '12.50 €'),
  dual symbols ('€ 50 €'), unshielded tokens, and shielded URLs/UUIDs (`M2-TC-081`)
- Offset tracking, TNAnnotation category ('currency'), and idempotence
"""

from src.linguistics.normalizer import ItalianTextNormalizer


class TestWP3CurrenciesEuro:
    """Verifies Euro normalization under Decision D5.A."""

    def test_euro_suffix_basic_and_invariance(self):
        """Corresponds to M2-TC-029 & M2-TC-030: 50 €, 150 € -> cinquanta/centocinquanta euro."""
        normalizer = ItalianTextNormalizer()

        res1 = normalizer.normalize("La quota di iscrizione è di 50 €.")
        assert res1.normalized_text == "La quota di iscrizione è di cinquanta euro."
        assert len(res1.annotations) == 1
        assert res1.annotations[0].category == "currency"
        assert res1.annotations[0].raw_span == "50 €"

        res2 = normalizer.normalize("Ho speso 150 € per l'albergo.")
        assert res2.normalized_text == "Ho speso centocinquanta euro per l'albergo."
        assert "euri" not in res2.normalized_text

    def test_euro_mixed_cents_plural(self):
        """Corresponds to M2-TC-031 & M2-AB-006: 12,50 € -> dodici euro e cinquanta centesimi."""
        text = "Il conto ammonta a 12,50 €."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "Il conto ammonta a dodici euro e cinquanta centesimi."
        ann = res.annotations[0]
        assert ann.category == "currency"
        assert text[ann.char_start:ann.char_end] == "12,50 €"

    def test_euro_mixed_singular_one_euro_one_cent(self):
        """Corresponds to M2-TC-032 & M2-AB-007: 1,01 € -> un euro e un centesimo."""
        text = "C'è una differenza di 1,01 €."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "C'è una differenza di un euro e un centesimo."

    def test_euro_prefix_with_thousands_separator(self):
        """Corresponds to M2-TC-033 & M2-AB-008: € 2.500 -> duemilacinquecento euro."""
        normalizer = ItalianTextNormalizer()

        text1 = "Il bonifico di € 2.500 è stato eseguito."
        res1 = normalizer.normalize(text1)
        assert res1.normalized_text == "Il bonifico di duemilacinquecento euro è stato eseguito."
        assert text1[res1.annotations[0].char_start:res1.annotations[0].char_end] == "€ 2.500"

        text2 = "Il rimborso ammonta a € 2.450 complessivi."
        res2 = normalizer.normalize(text2)
        assert res2.normalized_text == "Il rimborso ammonta a duemilaquattrocentocinquanta euro complessivi."

    def test_euro_zero_amount(self):
        """Corresponds to M2-TC-034: 0 € and 0,00 € -> zero euro."""
        normalizer = ItalianTextNormalizer()
        assert normalizer.normalize("Il costo totale è di 0 €.").normalized_text == "Il costo totale è di zero euro."
        assert normalizer.normalize("Saldo: 0,00 €.").normalized_text == "Saldo: zero euro."

    def test_euro_pure_fractional_cents(self):
        """Verify Spec § 2.3 fractional cents: 0,01 € -> un centesimo; 0,50 € -> cinquanta centesimi."""
        normalizer = ItalianTextNormalizer()
        assert normalizer.normalize("Resto di 0,01 €.").normalized_text == "Resto di un centesimo."
        assert normalizer.normalize("Costo di 0,50 €.").normalized_text == "Costo di cinquanta centesimi."
        assert normalizer.normalize("Valore 0,05 €.").normalized_text == "Valore cinque centesimi."

    def test_euro_negative_amount(self):
        """Corresponds to M2-TC-035: -15,00 € -> meno quindici euro."""
        text = "Il saldo contabile è di -15,00 €."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == "Il saldo contabile è di meno quindici euro."
        ann = res.annotations[0]
        assert text[ann.char_start:ann.char_end] == "-15,00 €"

        assert normalizer.normalize("Scarto: -0,50 €.").normalized_text == "Scarto: meno cinquanta centesimi."
        assert normalizer.normalize("Importo: -€ 15,00.").normalized_text == "Importo: meno quindici euro."

    def test_euro_iso_code_eur(self):
        """Corresponds to M2-TC-036: 200 EUR -> duecento euro."""
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize("Il versamento di 200 EUR è stato registrato.")
        assert res.normalized_text == "Il versamento di duecento euro è stato registrato."

        res_prefix = normalizer.normalize("Importo EUR 200 approvato.")
        assert res_prefix.normalized_text == "Importo duecento euro approvato."

    def test_euro_millions_preposition(self):
        """Verify large exact million amounts insert 'di' properly."""
        normalizer = ItalianTextNormalizer()
        assert normalizer.normalize("Fondo da 1.000.000 €.").normalized_text == "Fondo da un milione di euro."
        assert normalizer.normalize("Budget di € 2.000.000.").normalized_text == "Budget di due milioni di euro."


class TestWP3ForeignCurrencies:
    """Verifies Dollar ($ / USD) and Pound Sterling (£ / GBP) normalization."""

    def test_dollars_prefix_and_suffix(self):
        """Corresponds to M2-TC-037: $ 100 -> cento dollari, singular -> un dollaro."""
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize("Il prezzo negli Stati Uniti è di $ 100.")
        assert res.normalized_text == "Il prezzo negli Stati Uniti è di cento dollari."

        assert normalizer.normalize("Costa $ 1.").normalized_text == "Costa un dollaro."
        assert normalizer.normalize("Prezzo 25,50 USD.").normalized_text == "Prezzo venticinque dollari e cinquanta centesimi."

    def test_pounds_prefix_and_feminine_singular(self):
        """Corresponds to M2-TC-038: £ 50 -> cinquanta sterline, singular -> una sterlina."""
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize("Il biglietto per Londra costa £ 50.")
        assert res.normalized_text == "Il biglietto per Londra costa cinquanta sterline."

        assert normalizer.normalize("Costa £ 1.").normalized_text == "Costa una sterlina."
        assert normalizer.normalize("Prezzo 100 GBP.").normalized_text == "Prezzo cento sterline."


class TestWP3SafetyAndEdgeCases:
    """Verifies conservative guards, non-corruption of invalid inputs, shielding, and composability."""

    def test_numeric_ranges_are_not_corrupted(self):
        """Verify hyphenated ranges like '10-20 €' or '€ 10-20' are preserved without mangling."""
        normalizer = ItalianTextNormalizer()
        for text in ("Prezzo tra 10-20 €.", "Fascia € 10-20.", "Fascia € 10 - 20."):
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []

    def test_malformed_or_excess_decimals_are_noop(self):
        """Verify 3-decimal amounts or dot-decimal amounts are not partially mangled."""
        normalizer = ItalianTextNormalizer()
        for text in ("Valore 12,500 €.", "Importo 12.50 €."):
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []

    def test_dual_currency_sandwiches_and_suffix_letters_noop(self):
        """Verify malformed dual symbols or shorthand like $100k stay intact."""
        normalizer = ItalianTextNormalizer()
        for text in ("Errore € 50 €.", "Errore $ 100 USD.", "Valore $100k.", "Costa 50 EURO."):
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []

    def test_composite_shielding_unicode_m2_tc_081(self):
        """Corresponds to M2-TC-081: Shielded URL & UUID + Unicode guillemets + 15 €."""
        text = (
            "L'URL è https://test.org/id?v=1&k=2, il prezzo è 15 € e l'UUID è "
            "550e8400-e29b-41d4-a716-446655440000 con caratteri «speciali»."
        )
        expected = (
            "L'URL è https://test.org/id?v=1&k=2, il prezzo è quindici euro e l'UUID è "
            "550e8400-e29b-41d4-a716-446655440000 con caratteri «speciali»."
        )
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert res.normalized_text == expected
        assert len(res.annotations) == 1
        ann = res.annotations[0]
        assert ann.category == "currency"
        assert ann.raw_span == "15 €"
        assert text[ann.char_start:ann.char_end] == "15 €"

    def test_composite_dates_times_currencies_and_offset_alignment(self):
        """Verify combined Stage 1 (Shielding) + Stage 2 (Dates/Times) + Stage 3 (Currencies)."""
        text = (
            "Ricevuta https://pay.it/$99/tx del 01/05/2026 alle 12:00 per 1.250,50 €."
        )
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert (
            res.normalized_text
            == "Ricevuta https://pay.it/$99/tx del primo maggio duemilaventisei a mezzogiorno per milleduecentocinquanta euro e cinquanta centesimi."
        )
        assert [a.category for a in res.annotations] == ["date", "time", "currency"]
        for ann in res.annotations:
            assert text[ann.char_start:ann.char_end] == ann.raw_span

    def test_currency_idempotence(self):
        """Verify normalizing an already normalized monetary text yields an identical result."""
        text = "Il totale è di 1.250,50 € e $ 100."
        normalizer = ItalianTextNormalizer()
        res1 = normalizer.normalize(text)
        res2 = normalizer.normalize(res1.normalized_text)
        assert res2.normalized_text == res1.normalized_text
        assert res2.annotations == []
