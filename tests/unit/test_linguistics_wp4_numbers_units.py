"""Unit tests for Milestone M2 Workpackage WP4 (Units, Percentages & Numbers Pipeline).

Verifies Stages 4 & 5 of Spec Rev 2.1:
- Stage 4: Percentages ('25%', '0,5%') and metric/physical units ('10 km', '1 kg', '80 km/h', '85 m²')
- Stage 5: Cardinals ('42', '1.250.000'), gender agreement on '1' ('1 mela' -> 'una mela'),
  decimals with zero rules ('3,14', '3,05', '3,50'), negatives ('-5'), ordinals ('1°', '2ª'),
  Roman centuries ('XXI secolo'), abbreviated centuries ("'900" -> 'Novecento'),
  postal codes ('00187'), certified D2.A phones ('+39 347 1234567'), alphanumeric codes ('AB123CD')
- Structural preservation of shielded codes, impossible dates/times, rejected currency clusters,
  dot decimals, and >4300-digit (5000-digit) safety limits
- Composite Stage 1-5 offset tracking, idempotence, and p95 < 3.0 ms latency gate.
"""

import time

from src.linguistics.normalizer import ItalianTextNormalizer


class TestWP4UnitsAndPercentages:
    """Verifies Stage 4 percentages and metric/physical unit expansion."""

    def test_percentages_corpus_m2_tc_008_009_ab_012(self):
        normalizer = ItalianTextNormalizer()

        res1 = normalizer.normalize("Il tasso di interesse è salito al 25% annuo.")
        assert res1.normalized_text == "Il tasso di interesse è salito al venticinque per cento annuo."
        assert res1.annotations[0].category == "percentage"
        assert res1.annotations[0].raw_span == "25%"

        res2 = normalizer.normalize("La variazione è stata dello 0,5% su base mensile.")
        assert res2.normalized_text == "La variazione è stata dello zero virgola cinque per cento su base mensile."

        res3 = normalizer.normalize("L'indice di gradimento ha raggiunto il 78,4% degli intervistati.")
        assert (
            res3.normalized_text
            == "L'indice di gradimento ha raggiunto il settantotto virgola quattro per cento degli intervistati."
        )

        assert normalizer.normalize("Sconto 1%.").normalized_text == "Sconto uno per cento."
        assert normalizer.normalize("Calo -5%.").normalized_text == "Calo meno cinque per cento."

    def test_metric_units_corpus_m2_tc_010_to_013_ab_010(self):
        normalizer = ItalianTextNormalizer()

        res1 = normalizer.normalize("Il percorso misura 10 km.")
        assert res1.normalized_text == "Il percorso misura dieci chilometri."
        assert res1.annotations[0].category == "unit_measure"

        res2 = normalizer.normalize("Il pacco pesa esattamente 1 kg.")
        assert res2.normalized_text == "Il pacco pesa esattamente un chilogrammo."

        res3 = normalizer.normalize("Il limite di velocità è di 80 km/h.")
        assert res3.normalized_text == "Il limite di velocità è di ottanta chilometri orari."

        res4 = normalizer.normalize("L'appartamento ha una superficie di 85 m².")
        assert res4.normalized_text == "L'appartamento ha una superficie di ottantacinque metri quadrati."

        res5 = normalizer.normalize("La distanza da percorrere è di 120 km ad una velocità di 90 km/h.")
        assert (
            res5.normalized_text
            == "La distanza da percorrere è di centoventi chilometri ad una velocità di novanta chilometri orari."
        )

    def test_additional_units_decimals_and_case_safety(self):
        normalizer = ItalianTextNormalizer()
        assert normalizer.normalize("Peso 1,5 kg e 500 g.").normalized_text == "Peso uno virgola cinque chilogrammi e cinquecento grammi."
        assert normalizer.normalize("Bottiglia da 2 l e temperatura di 20 °C.").normalized_text == "Bottiglia da due litri e temperatura di venti gradi centigradi."
        assert normalizer.normalize("Area di 1 m².").normalized_text == "Area di un metro quadrato."
        # Case sensitivity: '5G' (telecom) must NOT be misread as 'cinque grammi'
        assert normalizer.normalize("Connessione 5G attiva.").normalized_text == "Connessione 5G attiva."


class TestWP4CardinalsDecimalsAndGender:
    """Verifies Stage 5 cardinals, thousands separators, gender agreement on 1, decimals, and negatives."""

    def test_cardinals_and_millions_m2_tc_001_002_025_028(self):
        normalizer = ItalianTextNormalizer()

        assert normalizer.normalize("La stanza ha 42 posti a sedere.").normalized_text == "La stanza ha quarantadue posti a sedere."
        assert normalizer.normalize("La città conta 1.250.000 abitanti.").normalized_text == "La città conta un milione duecentocinquantamila abitanti."
        assert (
            normalizer.normalize("La prima guerra mondiale si svolse dal 1915 al 1918.").normalized_text
            == "La prima guerra mondiale si svolse dal millenovecentoquindici al millenovecentodiciotto."
        )
        assert normalizer.normalize("La sala ospita da 10 a 15 persone.").normalized_text == "La sala ospita da dieci a quindici persone."

    def test_cardinal_one_gender_and_elision_agreement_m2_tc_003(self):
        normalizer = ItalianTextNormalizer()

        assert (
            normalizer.normalize("Abbiamo raccolto 1 mela e 1 pera nel frutteto.").normalized_text
            == "Abbiamo raccolto una mela e una pera nel frutteto."
        )
        assert normalizer.normalize("Aspetta 1 ora.").normalized_text == "Aspetta un'ora."
        assert normalizer.normalize("Resta 1 posto libero.").normalized_text == "Resta un posto libero."
        assert normalizer.normalize("Porta 1 zaino.").normalized_text == "Porta uno zaino."
        assert normalizer.normalize("Contiamo da 1 a 3.").normalized_text == "Contiamo da uno a tre."
        assert normalizer.normalize("Il binario 1 è chiuso.").normalized_text == "Il binario uno è chiuso."

    def test_decimals_and_zeros_m2_tc_004_to_006(self):
        normalizer = ItalianTextNormalizer()

        res1 = normalizer.normalize("Il valore di pi greco è circa 3,14.")
        assert res1.normalized_text == "Il valore di pi greco è circa tre virgola quattordici."
        assert res1.annotations[0].category == "decimal"

        assert normalizer.normalize("La temperatura misurata è di 3,05 gradi.").normalized_text == "La temperatura misurata è di tre virgola zero cinque gradi."
        assert normalizer.normalize("Il tempo sul giro è stato di 3,50 secondi.").normalized_text == "Il tempo sul giro è stato di tre virgola cinquanta secondi."
        assert normalizer.normalize("Valore -3,05.").normalized_text == "Valore meno tre virgola zero cinque."

    def test_negative_numbers_m2_tc_007_ab_009(self):
        normalizer = ItalianTextNormalizer()

        assert (
            normalizer.normalize("La temperatura minima registrata è stata di -5 gradi.").normalized_text
            == "La temperatura minima registrata è stata di meno cinque gradi."
        )
        assert (
            normalizer.normalize("La temperatura esterna è scesa a -3 gradi centigradi.").normalized_text
            == "La temperatura esterna è scesa a meno tre gradi centigradi."
        )


class TestWP4CenturiesOrdinalsPhonesPostalAndCodes:
    """Verifies Stage 5 centuries, ordinals, postal codes, certified D2.A phones, and codes."""

    def test_centuries_roman_and_apostrophe_m2_tc_026_027(self):
        normalizer = ItalianTextNormalizer()

        res1 = normalizer.normalize("Siamo nel XXI secolo.")
        assert res1.normalized_text == "Siamo nel ventunesimo secolo."
        assert res1.annotations[0].category == "century"

        res2 = normalizer.normalize("L'arte del '900 ha rivoluzionato i canoni classici.")
        assert res2.normalized_text == "L'arte del Novecento ha rivoluzionato i canoni classici."
        assert res2.annotations[0].category == "century"

        assert normalizer.normalize("Dal XIX al XX secolo.").normalized_text == "Dal diciannovesimo al ventesimo secolo."

    def test_numeric_ordinals(self):
        normalizer = ItalianTextNormalizer()
        assert normalizer.normalize("È arrivato 1° in gara.").normalized_text == "È arrivato primo in gara."
        assert normalizer.normalize("La 2ª classificata.").normalized_text == "La seconda classificata."
        assert normalizer.normalize("Nel 21° giorno.").normalized_text == "Nel ventunesimo giorno."

    def test_postal_code_m2_tc_014(self):
        normalizer = ItalianTextNormalizer()
        text = "La sede si trova in via Veneto, 00187 Roma."
        res = normalizer.normalize(text)
        assert res.normalized_text == "La sede si trova in via Veneto, zero zero uno otto sette Roma."
        assert res.annotations[0].category == "postal_code"
        assert text[res.annotations[0].char_start:res.annotations[0].char_end] == "00187"

        assert normalizer.normalize("CAP 20121 Milano.").normalized_text == "CAP due zero uno due uno Milano."

    def test_certified_phones_d2a_m2_tc_015_ab_011(self):
        normalizer = ItalianTextNormalizer()

        res1 = normalizer.normalize("Contattare il numero +39 347 1234567 per assistenza.")
        assert (
            res1.normalized_text
            == "Contattare il numero più tre nove tre quattro sette uno due tre quattro cinque sei sette per assistenza."
        )
        assert res1.annotations[0].category == "phone_number"

        res2 = normalizer.normalize("Per urgenze chiamare il numero +39 333 9876543 subito.")
        assert (
            res2.normalized_text
            == "Per urgenze chiamare il numero più tre nove tre tre tre nove otto sette sei cinque quattro tre subito."
        )

        # Domestic phone with explicit tel context
        res3 = normalizer.normalize("Telefono: 06 12345678.")
        assert res3.normalized_text == "Telefono: zero sei uno due tre quattro cinque sei sette otto."

    def test_alphanumeric_code_m2_tc_016_and_shielded_codes_m2_tc_071_to_073(self):
        normalizer = ItalianTextNormalizer()

        res1 = normalizer.normalize("Il codice di spedizione è AB123CD.")
        assert res1.normalized_text == "Il codice di spedizione è A B uno due tre C D."
        assert res1.annotations[0].category == "alphanumeric_code"

        # Shielded technical codes must stay 100% untouched
        for text in (
            "Il componente rispetta lo standard ISO-9001.",
            "Il velivolo da caccia F-16 è decollato dalla base.",
            "Durante la pandemia di COVID-19 sono state introdotte restrizioni.",
        ):
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []


class TestWP4SafetyCompositeLatencyAndIdempotence:
    """Verifies non-interference, 5000-digit safety, composite Stages 1-5, idempotence, and latency."""

    def test_structural_exclusions_and_prior_stage_noops_preserved(self):
        """Ensure Stage 4 & 5 never partially corrupt WP1-WP3 No-Op tokens or dot decimals."""
        normalizer = ItalianTextNormalizer()
        cases = (
            "La riunione è fissata per il 31/02/2026.",
            "Il display di collaudo riporta il valore 12:15:30.",
            "L'allarme è scattato alle 25:70 di notte.",
            "Codice 123ISO-9001, certificazione ISO-9001.",
            "Prezzo tra 10 - 20 €.",
            "Rapporto 12 / 50 €.",
            "Errore 50 € EUR.",
            "Importo € 12,,30.",
            "Versione 1.2.3 rilasciata.",
            "Valore 12.50.",
        )
        for text in cases:
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []

    def test_out_of_domain_5000_digit_safety_units_numbers(self):
        """Ensure >4300-digit numbers with units, percentages, or standalone stay safe No-Op."""
        normalizer = ItalianTextNormalizer()
        huge = "9" * 5000
        for text in (f"{huge}%", f"{huge} km", f"Numero {huge}.", f"Decimale 3,{huge}."):
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []

    def test_composite_pipeline_stages_1_to_5_and_offsets(self):
        text = "Ordine AB123CD del 15/03/2026 alle 14:30, peso 2,5 kg (+10% sconto) su https://shop.it/v1?p=9, totale 15 €."
        normalizer = ItalianTextNormalizer()
        res = normalizer.normalize(text)
        assert (
            res.normalized_text
            == "Ordine A B uno due tre C D del quindici marzo duemilaventisei alle quattordici e trenta, "
            "peso due virgola cinque chilogrammi (+dieci per cento sconto) su https://shop.it/v1?p=9, totale quindici euro."
        )
        for ann in res.annotations:
            assert text[ann.char_start:ann.char_end] == ann.raw_span

        # Idempotence
        res2 = normalizer.normalize(res.normalized_text)
        assert res2.normalized_text == res.normalized_text
        assert res2.annotations == []

    def test_wp4_pipeline_latency_gate_p95_under_3ms(self):
        """Verify Implementation Plan Rev 2 WP4 exit gate: p95 < 3.0 ms."""
        normalizer = ItalianTextNormalizer()
        sample = (
            "Il percorso di 10 km al 25% di pendenza costa 12,50 €; "
            "chiamare +39 347 1234567 o inviare a 00187 Roma (codice AB123CD)."
        )
        # Warmup
        for _ in range(20):
            normalizer.normalize(sample)

        times = []
        for _ in range(300):
            t0 = time.perf_counter()
            normalizer.normalize(sample)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times.sort()
        p95 = times[int(len(times) * 0.95)]
        assert p95 < 3.0, f"WP4 p95 latency {p95:.3f} ms exceeds 3.0 ms gate"
