# Richiesta Formale di Revisione — Milestone M0 (`M0_REVIEW_REQUEST.md`) v1.3
> **Progetto:** Italian TTS Studio  
> **Destinatari:** Product Owner (Utente) per autorizzazione & ChatGPT (Reviewer Tecnico Indipendente) per peer-review  
> **Autore:** Antigravity (Lead AI Architect & Implementer)  
> **Data:** Novembre 2026  
> **Commit / Ref di Partenza:** `89a5dc91d2fba1593ca66de43c6986ff1a0c5999` (Tag: `baseline-v0.1.7-kism3t`)  
> **Branch di Lavoro M0:** `m0-audit-benchmark`  
> **Repository:** [KIS-M3T/qwen3-TTS-studio](https://github.com/KIS-M3T/qwen3-TTS-studio) (fork di `bc-dunia/qwen3-TTS-studio`)

---

## 1. Stato di Completamento di Milestone M0

In conformità rigorosa all'incarico di Milestone M0 e alle correzioni vincolanti del documento di review tecnica v1.1 e peer-review di ChatGPT, sono stati completati tutti i deliverable obbligatori senza alcuna modifica distruttiva al codice di baseline, senza alcun addestramento prematuro e sostituendo ogni stima teorica con misure empiriche reali eseguite sulla GPU del sistema target.

### Indice dei Deliverable Prodotti e Verificabili

1. **[`TECH_AUDIT.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/TECH_AUDIT.md)**:
   - Mappatura completa delle funzionalità native e delle 7.086 righe di monolite UI (`qwen_tts_ui.py`).
   - Audit puntuale con numeri di riga e comportamenti accertati:
     * `_split_text_into_chunks` (righe 162–228): chunking limitato a soglie rigide 50/120/150 caratteri con split cieco sulle parole che ignora abbreviazioni italiane (`dott.`, `prof.`) e numeri con separatore delle migliaia (`1.500.000`).
     * `_crossfade_audio` (righe 286–302): crossfade lineare fisso di 30 ms privo di analisi dei silenzi e dei transienti fonetici.
     * `_calculate_dynamic_max_tokens` (righe 305–322): accertato parametro morto `preset_max` (non usato) e soglie rigide 256–768 token.
     * `_patch_generate_min_tokens` (righe 104–130): monkey-patch globale su `generate` con `min_new_tokens >= 60`.
     * `MAX_LOADED_MODELS = 1`: ricaricamento distruttivo continuo dei modelli da 4.2 GB da disco al cambio di tab.
     * `estimate_snr` (righe 33–76): stima euristica basata sul 10° percentile RMS, non vero SNR acustico.
     * `combiner.py`: dipendenza da MoviePy con compressione forzata MP3 a 192k lossy.
   - Analisi degli script di fine-tuning: `QwenLM/Qwen3-TTS/finetuning` (`sft_12hz.py` implementa full SFT con AdamW, footprint statico >20 GB VRAM su 1.7B, impossibile su 12 GB senza LoRA/ZeRO; fattibile su 0.6B con batch=1).
2. **[`LICENSE_AUDIT.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/LICENSE_AUDIT.md)**:
   - Evidenziata l'assenza di un file `LICENSE` nella root del repository originale `bc-dunia/qwen3-TTS-studio` (rischio di copyright "All Rights Reserved" per il codice applicativo in caso di ridistribuzione commerciale).
   - Verifica licenze pesi: Qwen3-TTS (Apache-2.0), Chatterbox V3 (MIT, ma repository ritirato ad agosto 2026), Coqui XTTS-v2 (CPML restrittiva, escluso).
   - Correzione definitiva sui dataset: formalizzata la rimozione di CSS10 italiano (inesistente), rettifica ore di M-AILABS (~127h 40min), conferma CC BY 4.0 per MLS Italiano (~247h), e accertamento che il folder locale `cv-corpus` contiene solo 79 clip validate (delta release non autosufficiente).
3. **[`HARDWARE_REPORT.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/HARDWARE_REPORT.md)** e **[`scripts/hardware_probe.py`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/scripts/hardware_probe.py)**:
   - Script di diagnosi eseguito con successo su host Windows.
   - Rilevazione effettiva: **NVIDIA GeForce RTX 4080 Laptop GPU** (12.282 MB VRAM totale), Driver 576.57, CUDA runtime 12.4 in ambiente virtuale dedicato `.venv` (Python 3.12).
   - RAM fisica di sistema: 31.73 GB; Storage locale: oltre 90 GB liberi.
   - Attenzione ottimizzata SDPA (`scaled_dot_product_attention`) con kernel C++/CUDA verificata operativa.
4. **[`BENCHMARK_PLAN.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/BENCHMARK_PLAN.md)**, **[`bench/manifest.csv`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/manifest.csv)** e **[`bench/test_dataset.jsonl`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/test_dataset.jsonl)**:
   - Dataset congelato di **60 testi complessi in lingua italiana**, classificati in 6 categorie mirate con ID univoci, testo grezzo, trascrizione fonetica attesa e fattore di difficoltà esplicitato.
5. **[`bench/run_benchmark.py`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/run_benchmark.py)**, **[`bench/results_benchmark.csv`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/results_benchmark.csv)** e **[`bench/summary_metrics.json`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/summary_metrics.json)**:
   - Harness di benchmarking riproducibile, estensibile e dotato di logica di cache-resume.
   - **Esecuzione Neurale Reale al 100% su GPU:**
     * **120 file WAV a 24 kHz reali generati** in `bench/outputs/M-QWEN-1.7B/` (60 file) e `bench/outputs/M-QWEN-0.6B/` (60 file).
     * Chatterbox registrato formalmente come `UNAVAILABLE` a causa dell'incompatibilità insanabile di dipendenze con il runtime (`transformers 5.2.0` vs `4.57.3`).
   - 2 riferimenti vocali italiani legali a 24 kHz in `bench/reference_voices/` con relativo **[`bench/reference_voices/VOICE_DATASHEET.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/reference_voices/VOICE_DATASHEET.md)** utilizzati per l'In-Context Learning zero-shot.

---

## 2. Tabella Comparativa Empirica dei Motori (Dati Reali da GPU)

| Criterio | `Qwen3-TTS-12Hz-1.7B-Base` | `Qwen3-TTS-12Hz-0.6B-Base` | `Chatterbox Multilingual V3` |
| :--- | :--- | :--- | :--- |
| **Architettura & Parametri** | Autoregressivo 12Hz (~1.7B) | Autoregressivo 12Hz (~0.6B) | Conformer / Flow-matching (~0.5B) |
| **Pesi Modello su Disco** | 3.857.413.744 byte (~3.86 GB) | 1.238.169.576 byte (~1.24 GB) | N/A (conflitto dipendenze) |
| **Status Test sui 60 Freeze Italiani** | **60/60 Superati (100% successo)** | **60/60 Superati (100% successo)** | **0/60 UNAVAILABLE** (formalmente documentato) |
| **Latenza Throughput (RTF Mediano)** | **3.248** (RTF p95: 4.568) | **2.500** (RTF p95: 4.078) | N/A |
| **Durata Media Audio Generato** | 8.25 s (Totale: ~495 s di audio) | 8.67 s (Totale: ~520 s di audio) | N/A |
| **Occupazione VRAM Picco (Inferenza)** | **4.558,03 MB** (~4.45 GB su 12 GB) | **2.820,25 MB** (~2.75 GB su 12 GB) | N/A |
| **Anomalie Acustiche Rilevate** | **0 vuoti, 0 clipping, 0 troncati** | **0 vuoti, 0 clipping, 0 troncati** | N/A |
| **Headroom & Livello RMS Tipico** | Picco: -1.1 dBFS, RMS: -19.3 dBFS | Picco: -1.2 dBFS, RMS: -19.5 dBFS | N/A |
| **Fattibilità Fine-Tuning su 12 GB** | Richiede LoRA / QLoRA (Full SFT stimato >20 GB) | Teoricamente possibile batch=1 (da provare in M5) | Non documentata per italiano |
| **Licenza Software & Pesi** | **Apache-2.0** (Attivamente mantenuto) | **Apache-2.0** (Attivamente mantenuto) | **MIT** (Progetto **ritirato da agosto 2026**) |
| **Compatibilità Dipendenze** | Piena compatibilità (`transformers 4.57.3`) | Piena compatibilità (`transformers 4.57.3`) | **Conflitto bloccante** (`transformers 5.2.0`, `gradio 6.8.0`) |

### Scelta Motivata del Motore di Produzione
* **Motore di Riferimento Principale per M1–M4: `Qwen3-TTS-12Hz-1.7B-Base`**:  
  Offre la massima qualità timbrica e fedeltà prosodica in lingua italiana. Con soli **4.558 MB di VRAM di picco**, opera con un margine di sicurezza di oltre 7.5 GB sulla GPU RTX 4080 (e sulla futura workstation desktop A500 con 100+ GB di RAM).
* **Motore Secondario / Alternativa Fast: `Qwen3-TTS-12Hz-0.6B-Base`**:  
  Garantisce una latenza inferiore (RTF mediano 2.5) con soli **2.820 MB di VRAM**. Ideale come opzione a basso consumo o per validare le pipeline di fine-tuning preliminari in Milestone M5.
* **Chatterbox V3 (Verifica Tecnica Empirica)**:  
  Il pacchetto `chatterbox-tts` impone `transformers==5.2.0`, che disinstalla e rompe `transformers==4.57.3` richiesto da `qwen-tts`. Rimane definitivamente escluso dal runtime principale di produzione e isolato in eventuale ambiente separato.

---

## 3. Incertezze Tecniche e Punti Aperti che Richiedono Decisione

Con il completamento del download dei pesi e dell'intero benchmark empirico, il **punto aperto 1 è integralmente risolto**. Si sottopongono al Product Owner e al revisore ChatGPT i restanti punti decisionali per autorizzare il passaggio alla **Milestone M1**:

1. **Download Pesi Qwen3-TTS**:  
   **[RISOLTO & COMPLETATO AL 100%]** — Pesi scaricati, verificati con hash/dimensione ed eseguiti con successo su tutte le 60 frasi di test.
2. **Strategia sul Debito di Licenza del Codice UI Originale**:  
   Dato che il codice di `bc-dunia/qwen3-TTS-studio` non espone una licenza open-source nella root:  
   * *Opzione A*: Contattare il maintainer per richiedere formalmente l'aggiunta di una licenza MIT/Apache-2.0.  
   * *Opzione B (Consigliata)*: Trattare il codice del fork come pura specifica funzionale e sviluppare i nuovi moduli di studio (motore disaccoppiato, linguistica, normalizzazione, mastering broadcast) con architettura pulita (clean-room), garantendo la piena proprietà del codice.  
   *Quale orientamento approva il Product Owner?*
3. **Politica Dinamica per la Patch `min_new_tokens`**:  
   Si propone di sostituire la monkey-patch globale fissa a 60 token con una formula progressiva in M1: disattivata (`min_new_tokens=0`) per frasi monosillabiche o inferiori a 15 caratteri (evitando allucinazioni/ripetizioni su *"Sì"*, *"No, grazie"*), e attivata con valore proporzionale alla lunghezza del testo per frasi complesse. Si richiede la conferma di ChatGPT su questo approccio.
4. **Ambiente Hardware Primario per le Fasi Successive**:  
   Le verifiche hardware hanno confermato sul Laptop lo spazio e la GPU RTX 4080 (12 GB VRAM e 32 GB RAM), con un'occupazione reale massima di 4.5 GB di VRAM. Confermi che lo sviluppo delle Milestones M1, M2, M3 e M4 avvenga interamente sul Laptop, riservando il Desktop (>100 GB RAM) per la fase di fine-tuning M5?

---

## 4. Richiesta di Gate Check

La Milestone M0 è **integralmente completata con evidenze empiriche riproducibili**.  
Si richiede a **ChatGPT** la revisione tecnica dei deliverable prodotti e al **Product Owner** l'approvazione formale per sbloccare la **Milestone M1 (Architettura Disaccoppiata, Audio Engine & Core Refactoring)**.
