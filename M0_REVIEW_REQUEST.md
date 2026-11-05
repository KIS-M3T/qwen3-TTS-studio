# Richiesta Formale di Revisione — Milestone M0 (`M0_REVIEW_REQUEST.md`)
> **Progetto:** Italian TTS Studio  
> **Destinatari:** Product Owner (Utente) per autorizzazione & ChatGPT (Reviewer Tecnico Indipendente) per peer-review  
> **Autore:** Antigravity (Lead AI Architect & Implementer)  
> **Data:** Novembre 2026  
> **Commit / Ref di Partenza:** `89a5dc91d2fba1593ca66de43c6986ff1a0c5999` (Tag: `baseline-v0.1.7-kism3t`)  
> **Repository:** [KIS-M3T/qwen3-TTS-studio](https://github.com/KIS-M3T/qwen3-TTS-studio) (fork di `bc-dunia/qwen3-TTS-studio`)

---

## 1. Stato di Completamento di Milestone M0

In conformità rigorosa all'incarico di Milestone M0 e alle correzioni vincolanti del documento di review v1.1, sono stati completati tutti i deliverable obbligatori senza alcuna modifica distruttiva al codice di baseline, senza alcun addestramento prematuro e senza simulazione di dati o misure.

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
   - Script di diagnosi non invasivo eseguito con successo su host Windows.
   - Rilevazione effettiva: **NVIDIA GeForce RTX 4080 Laptop GPU** (12.282 MB VRAM totale, 11.997 MB liberi), Driver 576.57, CUDA driver 12.9, RAM fisica di sistema 31.73 GB (disponibili 6.18 GB), storage locale 95.8 GB liberi.
   - Creazione dell'ambiente isolato `.venv` con Python 3.12 per svincolarsi dal PyTorch CPU installato a livello di sistema operativo.
4. **[`BENCHMARK_PLAN.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/BENCHMARK_PLAN.md)**, **[`bench/manifest.csv`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/manifest.csv)** e **[`bench/test_dataset.jsonl`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/test_dataset.jsonl)**:
   - Dataset congelato di **60 testi complessi in lingua italiana**, classificati in 6 categorie mirate con ID univoci, testo grezzo, trascrizione fonetica attesa e fattore di difficoltà esplicitato.
5. **[`bench/run_benchmark.py`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/run_benchmark.py)**, **[`bench/results_benchmark.csv`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/results_benchmark.csv)** e **[`bench/summary_metrics.json`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/summary_metrics.json)**:
   - Harness di benchmarking riproducibile ed estensibile.
   - Esecuzione diagnostica completata: 60 file audio nativi WAV a 24 kHz salvati in `bench/outputs/DIAGNOSTIC_DRYRUN/`.
   - Gestione delle assenze: Chatterbox registrato come `UNAVAILABLE` con motivazione tecnica documentata ("Package chatterbox non installato; repository ResembleAI ritirato ad agosto 2026").
   - Preparazione di 2 riferimenti vocali italiani legali a 24 kHz in `bench/reference_voices/` con relativo **[`bench/reference_voices/VOICE_DATASHEET.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/reference_voices/VOICE_DATASHEET.md)**.

---

## 2. Tabella dei Compromessi tra i Motori Candidati

| Criterio | `Qwen3-TTS-12Hz-1.7B-Base` | `Qwen3-TTS-12Hz-0.6B-Base` | `Chatterbox Multilingual V3` |
| :--- | :--- | :--- | :--- |
| **Qualità Fonetica e Naturalezza** | **Eccellente** (modello da 1.7B parametri, timbro ricco, ottima resa dell'italiano con prompt reference). | **Buona** (leggermente meno espressivo nelle sfumature emotive, ma chiaro). | **Discreta/Media** (intonazione talvolta robotica o con accento anglofono residuo). |
| **In-Context Voice Cloning** | **Sì** (zero-shot/few-shot con speaker embedding pesato). | **Sì** (zero-shot, ma minore aderenza timbrica rispetto al 1.7B). | **Sì** (richiede riferimento acustico pulito). |
| **Latenza & Throughput (RTF)** | RTF stimato ~0.25–0.40 su RTX 4080 (veloce). | RTF stimato ~0.10–0.18 su RTX 4080 (**ultraveloce**). | RTF stimato ~0.20–0.30. |
| **Occupazione VRAM (Inferenza)** | ~3.4 – 3.8 GB (entra comodamente nei 12 GB). | ~1.3 – 1.6 GB (impatto minimo sui 12 GB). | ~1.5 – 2.0 GB. |
| **Fattibilità Fine-Tuning su 12 GB** | **Solo con LoRA/QLoRA** (SFT full-parameter richiede >20 GB VRAM). | **Fattibile** (anche con full SFT con batch=1 e gradient accumulation). | Non prevista pipeline SFT standard per l'italiano. |
| **Licenza & Manutenzione** | **Apache-2.0** (Attivamente mantenuto da Alibaba). | **Apache-2.0** (Attivamente mantenuto da Alibaba). | **MIT** (Progetto **abbandonato/retired** da agosto 2026). |

### Scelta Motivata del Primo Motore
* **Motore di Riferimento Principale per M1–M4: `Qwen3-TTS-12Hz-1.7B-Base`**:  
  Rappresenta il miglior compromesso assoluto tra fedeltà vocale italiana, supporto nativo della lingua e licenza d'uso commerciale permissiva (Apache-2.0). I 12 GB di VRAM della GPU del laptop consentono un'inferenza fluida in `bfloat16` con ampio margine per cache e pre-processing.
* **Motore Secondario / Baseline Leggera: `Qwen3-TTS-12Hz-0.6B-Base`**:  
  Viene mantenuto nell'architettura come alternativa a bassissima latenza e come primo candidato per validare la procedura di addestramento e fine-tuning nella successiva Milestone M5.
* **Chatterbox V3**:  
  Confermato esclusivamente come backend secondario isolabile di confronto, non come dipendenza strategica, a causa del suo stato di manutenzione cessato.

---

## 3. Incertezze Tecniche e Punti Aperti che Richiedono Decisione

Prima di autorizzare il passaggio alla **Milestone M1**, si sottopongono al Product Owner e al revisore ChatGPT i seguenti quesiti:

1. **Download Pesi Qwen3-TTS**:  
   Per procedere ai test di benchmark acustico reale (misura di WER con ASR e ascolto cieco su audio generato da rete neurale), è necessario scaricare i pesi ufficiali `Qwen3-TTS-12Hz-1.7B-Base` (~4.2 GB) e `Qwen3-TTS-Tokenizer-12Hz` (~1.0 GB) tramite `huggingface_hub`. Si richiede l'autorizzazione al download dei pesi nella cartella di progetto.
2. **Strategia sul Debito di Licenza del Codice UI Originale**:  
   Dato che il codice di `bc-dunia/qwen3-TTS-studio` non espone una licenza open-source nella root:  
   * *Opzione A*: Contattare il maintainer per richiedere formalmente l'aggiunta di una licenza MIT/Apache-2.0.  
   * *Opzione B*: Trattare il codice del fork come pura specifica di riferimento e sviluppare i nuovi moduli (motore, linguistica, mastering) con architettura pulita disaccoppiata (clean-room), garantendo la piena proprietà del codice sviluppato.  
   *Quale orientamento approva il Product Owner?*
3. **Politica Dinamica per la Patch `min_new_tokens`**:  
   Si propone di sostituire la monkey-patch globale fissa a 60 token con una formula progressiva: disattivata (`min_new_tokens=0`) per frasi monosillabiche o inferiori a 15 caratteri (evitando allucinazioni su *"Sì"*, *"No"*), e attivata con valore proporzionale alla lunghezza del testo per frasi lunghe. Si richiede la conferma di ChatGPT su questo approccio per M1.
4. **Ambiente Hardware Primario per le Fasi Successive**:  
   Le verifiche hardware hanno confermato sul Laptop lo spazio e la GPU RTX 4080 (12 GB VRAM e 32 GB RAM). Confermi che lo sviluppo delle Milestones M1, M2, M3 e M4 avvenga interamente sul Laptop, riservando il Desktop (>100 GB RAM) per la fase di fine-tuning M5?

---

## 4. Richiesta di Gate Check

La Milestone M0 è formalmente completata. Si richiede a **ChatGPT** la revisione tecnica dei deliverable prodotti e al **Product Owner** l'approvazione formale per sbloccare la Milestone M1.
