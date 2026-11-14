"""Unit tests for Milestone M2 Workpackage WP5 (Titles, Abbreviations, Acronyms D4.B, Phonotactics)."""

import time

from src.linguistics import ItalianNormalizerConfig, ItalianTextNormalizer


class TestWP5TitlesAndAbbreviations:
    """Verifies Stage 6 professional titles (apocope rules) and Italian abbreviations."""

    def test_masculine_ore_titles_apocope_before_surname_corpus(self):
        normalizer = ItalianTextNormalizer()

        r1 = normalizer.normalize("Ha telefonato il dott. Rossi stamattina.")
        assert r1.normalized_text == "Ha telefonato il dottor Rossi stamattina."
        assert len(r1.annotations) == 1
        assert r1.annotations[0].category == "title"
        assert r1.annotations[0].raw_span == "dott."

        r2 = normalizer.normalize("La lezione del prof. Bianchi è rimandata.")
        assert r2.normalized_text == "La lezione del professor Bianchi è rimandata."

        r3 = normalizer.normalize("Il progetto è firmato dall'ing. Verdi.")
        assert r3.normalized_text == "Il progetto è firmato dall'ingegner Verdi."

        # Additional -ore apocope forms & surname compounds (AB-013, 014, 016 + variants)
        assert normalizer.normalize("Il dott. Rossi ha confermato l'orario della visita.").normalized_text == (
            "Il dottor Rossi ha confermato l'orario della visita."
        )
        assert normalizer.normalize("La conferenza del prof. Bianchi è stata molto applaudita.").normalized_text == (
            "La conferenza del professor Bianchi è stata molto applaudita."
        )
        assert normalizer.normalize("Il parere dell'ing. Verdi è stato fondamentale per la delibera.").normalized_text == (
            "Il parere dell'ingegner Verdi è stato fondamentale per la delibera."
        )
        assert normalizer.normalize("Il sig. De Luca e Dott. D'Angelo sono arrivati.").normalized_text == (
            "Il signor De Luca e Dottor D'Angelo sono arrivati."
        )

    def test_isolated_non_ore_and_feminine_titles_no_apocope_corpus(self):
        normalizer = ItalianTextNormalizer()

        # M2-TC-042: avv. does not end in -ore
        r1 = normalizer.normalize("L'atto è stato depositato dall'avv. Neri.")
        assert r1.normalized_text == "L'atto è stato depositato dall'avvocato Neri."

        # M2-TC-043: isolated 'dott.' without proper name uses full form
        r2 = normalizer.normalize("Ho chiesto al dott. se fosse disponibile.")
        assert r2.normalized_text == "Ho chiesto al dottore se fosse disponibile."

        # M2-TC-044 & M2-AB-015: feminine titles never apocopate
        r3 = normalizer.normalize("La dott.ssa Ferrari riceve per appuntamento.")
        assert r3.normalized_text == "La dottoressa Ferrari riceve per appuntamento."

        r4 = normalizer.normalize("La dott.ssa Ferrari presenterà la relazione annuale.")
        assert r4.normalized_text == "La dottoressa Ferrari presenterà la relazione annuale."

        assert normalizer.normalize("La prof.ssa Conti e l'arch. Neri firmano.").normalized_text == (
            "La professoressa Conti e l'architetto Neri firmano."
        )
        assert normalizer.normalize("il dott.").normalized_text == "il dottore"

    def test_abbreviations_and_cfr_syntax_corpus(self):
        normalizer = ItalianTextNormalizer()

        # M2-TC-045: contextual 'cfr.'
        r_cfr = normalizer.normalize("Per approfondire l'argomento, cfr. capitolo 2.")
        assert r_cfr.normalized_text == "Per approfondire l'argomento, confronta il capitolo due."

        assert normalizer.normalize("Vedi sezione sopra, cfr. sezione 4.").normalized_text == (
            "Vedi sezione sopra, confronta la sezione quattro."
        )
        assert normalizer.normalize("Per i dettagli, cfr. articolo 3.").normalized_text == (
            "Per i dettagli, confronta l'articolo tre."
        )
        assert normalizer.normalize("Approfondimenti: cfr. cap. 2.").normalized_text == (
            "Approfondimenti: confronta il capitolo due."
        )
        assert normalizer.normalize("Si rimanda alla nota, cfr. il testo.").normalized_text == (
            "Si rimanda alla nota, confronta il testo."
        )
        assert normalizer.normalize("Per ulteriori dettagli, si veda cfr. nota.").normalized_text == (
            "Per ulteriori dettagli, si veda cfr. nota."
        )

        # M2-TC-046..048: ecc., pag., p.za + additional safe forms
        r_ecc = normalizer.normalize("Portare penne, quaderni, matite, ecc.")
        assert r_ecc.normalized_text == "Portare penne, quaderni, matite, eccetera."

        r_pag = normalizer.normalize("Il testo prosegue a pag. 45 del manuale.")
        assert r_pag.normalized_text == "Il testo prosegue a pagina quarantacinque del manuale."

        r_pza = normalizer.normalize("Ci siamo incontrati in p.za Dante.")
        assert r_pza.normalized_text == "Ci siamo incontrati in piazza Dante."

        # Initial 'N. Rossi' must NOT expand to 'numero', whereas lowercase 'n. 5' expands cleanly
        assert normalizer.normalize("Firma: N. Rossi, pratica n. 5.").normalized_text == (
            "Firma: N. Rossi, pratica numero cinque."
        )
        # Ensure 'CAP 20121' is postal code (Stage 5) and never becomes 'capitolo'
        assert normalizer.normalize("Spedire a CAP 20121, v.le Monza.").normalized_text == (
            "Spedire a CAP due zero uno due uno, viale Monza."
        )


class TestWP5AcronymsClosedCatalogD4B:
    """Verifies Policy D4.B closed-dictionary acronym and initialism handling."""

    def test_lexicalized_acronyms_preserved_verbatim_noop(self):
        normalizer = ItalianTextNormalizer()
        cases = (
            "Il vertice della NATO si conclude oggi.",
            "I dati diffusi dall'ISTAT confermano la crescita.",
            "Il trattato istitutivo della NATO è stato ratificato a Washington.",
        )
        for text in cases:
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []

    def test_spelled_initialisms_expanded_for_bpe_corpus(self):
        normalizer = ItalianTextNormalizer()

        r1 = normalizer.normalize("Il comunicato è stato rilasciato dalla CGIL.")
        assert r1.normalized_text == "Il comunicato è stato rilasciato dalla C G I L."
        assert r1.annotations[0].category == "acronym"

        r2 = normalizer.normalize("È stato emesso un nuovo BTP decennale.")
        assert r2.normalized_text == "È stato emesso un nuovo B T P decennale."

        r3 = normalizer.normalize("La sede centrale dell'INPS si trova a Roma.")
        assert r3.normalized_text == "La sede centrale dell'I N P S si trova a Roma."

        r4 = normalizer.normalize("La delegazione della CGIL ha incontrato i rappresentanti del governo.")
        assert r4.normalized_text == "La delegazione della C G I L ha incontrato i rappresentanti del governo."

    def test_unclassified_uppercase_tokens_strictly_noop_d4b(self):
        normalizer = ItalianTextNormalizer()
        cases = (
            "Il marchio IKEA ha aperto un nuovo punto vendita.",
            "La start-up KWH-TECH sviluppa accumulatori.",
            "Il nuovo modello BMW e il server ACME usano PDF o HTML?",
        )
        # Check exact D4.B corpus strings (IKEA, KWH-TECH -> zero changes)
        for text in cases[:2]:
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []

        res3 = normalizer.normalize(cases[2])
        assert res3.normalized_text == "Il nuovo modello BMW e il server ACME usano P D F o H T M L?"


class TestWP5PhonotacticsElisionsAndD1B:
    """Verifies Stage 7 phonological apostrophes, accents, 'se'/'da', and D1.B homographs."""

    def test_apostrophes_and_qual_e_preservation_and_typo_fix(self):
        normalizer = ItalianTextNormalizer()

        # M2-TC-053 & M2-TC-054: native elisions & 'qual è' untouched
        for text in (
            "L'albero maestro della nave è imponente.",
            "Non si sa qual è la decisione corretta.",
        ):
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []

        # Erroneous apostrophe on 'qual'è' normalized cleanly
        r_fix = normalizer.normalize("Non so qual'è il problema, però L'albero resta.")
        assert r_fix.normalized_text == "Non so qual è il problema, però L'albero resta."

    def test_grave_to_acute_accents_and_se_da_disambiguation(self):
        normalizer = ItalianTextNormalizer()

        # M2-TC-055 & M2-AB-019
        assert normalizer.normalize("Non capisco perchè non sia venuto.").normalized_text == (
            "Non capisco perché non sia venuto."
        )
        assert normalizer.normalize("Non capisco perchè non sia ancora arrivato l'assegno.").normalized_text == (
            "Non capisco perché non sia ancora arrivato l'assegno."
        )
        assert normalizer.normalize("Perchè parli affinchè poi finchè puoi, benchè sia tardi?").normalized_text == (
            "Perché parli affinché poi finché puoi, benché sia tardi?"
        )

        # M2-TC-056: 'da se' -> 'da sé'
        r_se = normalizer.normalize("Chi fa da se fa per tre.")
        assert r_se.normalized_text == "Chi fa da sé fa per tre."

        # Additional safe 'se'/'da' disambiguation vs negative controls
        assert normalizer.normalize("Lotta tra se e se, ma non mi da fastidio.").normalized_text == (
            "Lotta tra sé e sé, ma non mi dà fastidio."
        )
        assert normalizer.normalize("Si da da fare e basta per se.").normalized_text == (
            "Si dà da fare e basta per sé."
        )
        for noop_text in (
            "Prendo un caffè e cioè un tè.",
            "Lui viene da Roma e dimmi se arriva.",
            "Lui da sempre aiuta se stesso.",
        ):
            res = normalizer.normalize(noop_text)
            assert res.normalized_text == noop_text
            assert res.annotations == []

    def test_heterophonic_homographs_noop_baseline_d1b(self):
        normalizer = ItalianTextNormalizer()
        assert normalizer.config.enable_homograph_disambiguation is False

        cases = (
            "La nave ha gettato l'ancora nel porto.",
            "Il treno non è ancora arrivato alla stazione.",
            "Devi venire subito qui.",
            "Il testimone ha subito una grave intimidazione.",
            "I principi del regno rispettano i principi etici.",
        )
        for text in cases:
            res = normalizer.normalize(text)
            assert res.normalized_text == text
            assert res.annotations == []


class TestWP5CompositeCorpusOffsetsIdempotenceAndLatency:
    """Verifies full Stages 1-8 composite corpus, shielding isolation, offsets, idempotence, and latency."""

    def test_composite_corpus_m2_tc_080_081_and_ab_020(self):
        normalizer = ItalianTextNormalizer()

        t80 = "Il dott. Rossi ha versato 1.250,50 € in data 01/05/2026."
        r80 = normalizer.normalize(t80)
        assert (
            r80.normalized_text
            == "Il dottor Rossi ha versato milleduecentocinquanta euro e cinquanta centesimi "
            "in data primo maggio duemilaventisei."
        )
        assert r80.metadata["pipeline_stage"] == "wp5_phonotactics"
        for ann in r80.annotations:
            assert t80[ann.char_start:ann.char_end] == ann.raw_span

        t81 = (
            "L'URL è https://test.org/id?v=1&k=2, il prezzo è 15 € e l'UUID è "
            "550e8400-e29b-41d4-a716-446655440000 con caratteri «speciali»."
        )
        r81 = normalizer.normalize(t81)
        assert (
            r81.normalized_text
            == "L'URL è https://test.org/id?v=1&k=2, il prezzo è quindici euro e l'UUID è "
            "550e8400-e29b-41d4-a716-446655440000 con caratteri «speciali»."
        )
        for ann in r81.annotations:
            assert t81[ann.char_start:ann.char_end] == ann.raw_span

        t_ab20 = "L'inaugurazione del nuovo teatro avverrà alle ore 20:30 del 1° giugno."
        r_ab20 = normalizer.normalize(t_ab20)
        assert (
            r_ab20.normalized_text
            == "L'inaugurazione del nuovo teatro avverrà alle ore venti e trenta del primo giugno."
        )

    def test_shielding_protection_against_wp5_rules_and_idempotence(self):
        normalizer = ItalianTextNormalizer()
        raw = (
            "Il prof. Verdi (CGIL) ha inviato https://ente.it/dott.html?pag=1&cfr=etc "
            "e il file C:\\dott.ssa\\perchè\\etc\\hosts, perchè §§§SHIELD_0§§§ resta intact."
        )
        res = normalizer.normalize(raw)
        assert (
            res.normalized_text
            == "Il professor Verdi (C G I L) ha inviato https://ente.it/dott.html?pag=1&cfr=etc "
            "e il file C:\\dott.ssa\\perchè\\etc\\hosts, perché §§§SHIELD_0§§§ resta intact."
        )
        for ann in res.annotations:
            assert raw[ann.char_start:ann.char_end] == ann.raw_span

        # Idempotence check
        res2 = normalizer.normalize(res.normalized_text)
        assert res2.normalized_text == res.normalized_text
        assert res2.annotations == []

    def test_config_flags_isolate_wp5_stages_cleanly(self):
        cfg = ItalianNormalizerConfig(
            enable_titles=False,
            enable_abbreviations=False,
            enable_acronyms=False,
            enable_phonotactics=False,
        )
        normalizer = ItalianTextNormalizer(cfg)
        raw = "Il dott. Rossi della CGIL chiede perchè pag. 5 costa 10 €."
        res = normalizer.normalize(raw)
        assert res.normalized_text == "Il dott. Rossi della CGIL chiede perchè pag. cinque costa dieci euro."

    def test_wp5_full_pipeline_latency_gate_p95_under_3ms(self):
        """Verify Stages 1-8 latency meets M2 p95 < 3.0 ms requirement."""
        normalizer = ItalianTextNormalizer()
        sample = (
            "Il dott. Rossi della CGIL ha inviato 1.250,50 € a p.za Dante, pag. 12, "
            "perchè la NATO resta valida e cfr. capitolo 2 su https://example.it/doc."
        )
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
        assert p95 < 3.0, f"WP5 p95 latency {p95:.3f} ms exceeds 3.0 ms gate"

    def test_wp5_b1_ecc_final_punctuation_and_internal_abbreviations(self):
        """WP5-B1: Preserve sentence-final period on lowercase/multi-sentence 'ecc.' without internal dots."""
        normalizer = ItalianTextNormalizer()

        # Lowercase start & no comma before sentence-final 'ecc.' -> preserve dot
        assert normalizer.normalize("serve carta e penne ecc.").normalized_text == (
            "serve carta e penne eccetera."
        )
        # Multi-sentence ending in 'ecc.' -> preserve dot
        assert normalizer.normalize("prima frase. seconda frase con penne ecc.").normalized_text == (
            "prima frase. seconda frase con penne eccetera."
        )
        # Multi-sentence where first sentence ends with 'ecc.' -> preserve dot
        assert normalizer.normalize("prendi penne ecc. Poi usciamo.").normalized_text == (
            "prendi penne eccetera. Poi usciamo."
        )

        # Negative checks: internal abbreviations must NOT get arbitrary periods; 'il dott.' stays intact
        assert normalizer.normalize("prendi penne, ecc. e matite poi esci.").normalized_text == (
            "prendi penne, eccetera e matite poi esci."
        )
        assert normalizer.normalize("vedi pag. 4 del manuale.").normalized_text == (
            "vedi pagina quattro del manuale."
        )
        assert normalizer.normalize("il dott.").normalized_text == "il dottore"

    def test_wp5_b2_proclitic_da_verb_with_temporal_adverbs_vs_prepositions(self):
        """WP5-B2: Proclitic pronoun + 'da' normalizes to 'dà' even before temporal adverbs; prepositions stay No-Op."""
        normalizer = ItalianTextNormalizer()

        # Positive verbal readings before temporal adverbs
        assert normalizer.normalize("Mi da oggi una risposta.").normalized_text == (
            "Mi dà oggi una risposta."
        )
        assert normalizer.normalize("Mi da sempre fastidio.").normalized_text == (
            "Mi dà sempre fastidio."
        )
        assert normalizer.normalize("Glielo da domani senza problemi.").normalized_text == (
            "Glielo dà domani senza problemi."
        )
        assert normalizer.normalize("Ci da ieri sera un segnale?").normalized_text == (
            "Ci dà ieri sera un segnale?"
        )
        assert normalizer.normalize("Se ne da pensiero?").normalized_text == (
            "Se ne dà pensiero?"
        )

        # Negative prepositional / ambiguous / Roman-numeral contexts stay strictly No-Op
        for noop_text in (
            "Lui da sempre aiuta gli amici.",
            "Lui viene da Roma oggi.",
            "Il pacco da consegnare domani è pronto.",
            "Da oggi cambia tutto.",
            "Paolo VI da giovane studiava molto.",
        ):
            res = normalizer.normalize(noop_text)
            assert res.normalized_text == noop_text
            assert res.annotations == []

    def test_wp5_b3_all_caps_accent_normalization_and_d4b_compliance(self):
        """WP5-B3: Accent normalization preserves ALL-CAPS casing and respects D4.B closed catalog."""
        normalizer = ItalianTextNormalizer()

        r1 = normalizer.normalize("PERCHÈ NON RISPONDI AFFINCHÈ CAPISCA?")
        assert r1.normalized_text == "PERCHÉ NON RISPONDI AFFINCHÉ CAPISCA?"
        assert r1.normalized_text.isupper()
        for ann in r1.annotations:
            assert r1.raw_text[ann.char_start:ann.char_end] == ann.raw_span

        r2 = normalizer.normalize("Perchè parli NÈ più nè meno, QUAL'È il punto?")
        assert r2.normalized_text == "Perché parli NÉ più né meno, QUAL È il punto?"

        # D4.B & conservative uppercase boundaries remain untouched
        for d4b_noop in (
            "IKEA E BMW DA SE",
            "MI DA SEMPRE",
            "NATO E ISTAT",
        ):
            res = normalizer.normalize(d4b_noop)
            assert res.normalized_text == d4b_noop
            assert res.annotations == []

        # Idempotence on all-caps normalized output
        assert normalizer.normalize(r1.normalized_text).normalized_text == r1.normalized_text
        assert normalizer.normalize(r1.normalized_text).annotations == []
