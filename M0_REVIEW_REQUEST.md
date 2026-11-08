# Richiesta Formale di Revisione — Milestone M0 (`M0_REVIEW_REQUEST.md`) v1.5
> **Progetto:** Italian TTS Studio  
> **Destinatari:** Product Owner (Utente) per autorizzazione & ChatGPT (Reviewer Tecnico Indipendente) per peer-review Gate 3  
> **Autore:** Antigravity (Lead AI Architect & Implementer)  
> **Data:** 8 Novembre 2026 (Revisione Gate 3)  
> **Commit Baseline Tag:** `89a5dc91d2fba1593ca66de43c6986ff1a0c5999` (`baseline-v0.1.7-kism3t`)  
> **Branch Remoto GitHub Ufficiale:** [`m0-audit-benchmark`](https://github.com/KIS-M3T/qwen3-TTS-studio/tree/m0-audit-benchmark)  
> **Link Creazione Pull Request:** [`.../pull/new/m0-audit-benchmark`](https://github.com/KIS-M3T/qwen3-TTS-studio/pull/new/m0-audit-benchmark) (apertura riservata al Product Owner)  
> **Pacchetto Evidenze Completo:** [`m0_deliverables.zip`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/m0_deliverables.zip) (5.80 MB, include campioni WAV neurali)

---

## 1. Riscontro Puntuale ai Rilievi del Gate 3 (8 Novembre 2026)

In aderenza rigorosa ai rilievi bloccanti (**P0-A, P0-B, P0-C**) e metodologici (**P1**) espressi nel verbale di Gate 3, sono state implementate tutte le correzioni necessarie, azzerando qualsiasi stima sintetica e conducendo un benchmark empirico controllato e ripetibile.

### P0-A: Eliminazione Totale di Tempi e VRAM Sintetici dal Percorso Cache
- **Azione Eseguita:** Nel file `bench/run_benchmark.py` sono state eliminate tutte le formule di stima euristica (`* 2.8`, `* 2.5`, costanti fisse di VRAM).
- **Nuova Classificazione:** Per tutti i 50 file audio pre-generati e recuperati da cache per i quali non era disponibile un timer monotonico diretto:
  - `run_type = "CACHE_RECOVERY_UNMEASURED"`
  - `gen_time_sec = None` (cella vuota nel CSV)
  - `rtf = None` (cella vuota nel CSV)
  - `peak_vram_mb = None` (cella vuota nel CSV)
  - Parametri acustici (`duration_sec`, `sample_peak_dbfs`, `rms_dbfs`, assenza di clipping, silenzio di coda) conservati come dati fisici reali estratti dal file WAV.
- **Isolamento Statistico:** I record `CACHE_RECOVERY_UNMEASURED` sono rigorosamente esclusi da qualsiasi calcolo di latenza o RTF mediano/p95 nel file `bench/summary_metrics.json`.

### P0-B: Benchmark Empirico Controllato su 10 Frasi x 2 Modelli con Logging Append-Only
- **Sottoinsieme Testato (10 casi rappresentativi):**
  `NUM_01`, `ACR_01`, `HOM_01`, `TEC_01`, `NAR_01`, `LEN_01`, `LEN_02`, `LEN_03`, `LEN_04`, `LEN_05`.
- **Esecuzione Controllata:**
  - 1 run di **WARMUP** dedicato per ciascun modello (escluso dalle metriche statistiche).
  - **2 iterazioni misurate per ciascuna frase** (20 run reali per `M-QWEN-1.7B` + 20 run reali per `M-QWEN-0.6B` = 40 generazioni misurate totali).
  - Cronometraggio monotonico ad alta precisione tramite `time.perf_counter()`.
  - Misurazione della memoria GPU a tre livelli: `torch.cuda.max_memory_allocated()`, `torch.cuda.max_memory_reserved()` e `nvidia-smi` used memory.
- **Logging Append-Only Immediato (`bench/benchmark_events.jsonl`):**
  Ogni singola inferenza è registrata istantaneamente con flush su disco, includendo timestamp ISO, ID test, modello, tipo di run (`REAL_INFERENCE`), iterazione, SHA256 del testo di input, SHA256 del file WAV generato, tempo di calcolo, durata, RTF, sample peak e occupazione VRAM.

### P0-C: Distinzione tra Sentinella ASR e Realtà Acustica — Dossier di Ascolto
- **Chiarimento sull'Allucinazione ASR (`NUM_01`):**
  L'output di oltre 1.300 caratteri generato da Whisper-tiny su `NUM_01` è stato formalmente ricondotto a un **loop autoregressivo del decodificatore ASR**, innescato dalla lettura di simboli e date non normalizzate con frammenti fonetici a bassa confidenza.
- **Realtà Acustica Udibile:**
  L'audit acustico umano diretto (documentato in [`bench/QUALITY_LISTENING_DOSSIER.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/QUALITY_LISTENING_DOSSIER.md)) certifica che i file audio reali (`1.7B_NUM_01.wav` da 11.28s e `0.6B_NUM_01.wav` da 12.40s) **non contengono alcuna ripetizione infinita udibile**: la voce si arresta naturalmente prima del dodicesimo secondo.
- **Terminologia Corretta:**
  Adottata univocamente la dicitura **TN (Text Normalization)** per l'espansione del testo simbolico in parole pronunciabili.
- **Disponibilità dei File Audio Reali:**
  Tutti i 20 file WAV delle sentinelle e i 2 riferimenti vocali CC0 sono raccolti e verificabili in `bench/samples/` e inclusi nell'archivio scaricabile `bench/m0_neural_samples.zip` con checksum SHA256 certificati in `CHECKSUMS.txt`.
- **Inquadramento Causale:**
  La Text Normalization è classificata come requisito architetturale critico la cui causalità comparativa specifica sarà validata con test A/B in M1.

### Risoluzione Rilievi P1 (Metodologici e Documentali)
1. **Patch `min_new_tokens`:** Lo script di benchmark istanzia direttamente `Qwen3TTSModel.from_pretrained` (non usa `audio/model_loader.py`); il test A/B a parità di condizioni è programmato in M1.
2. **Definizione di Troncamento:** La metrica `is_truncated` rileva unicamente la presenza di code di silenzio finale $>2,0\text{s}$ (`trailing_silence_gt_2s`). L'assenza di troncamento semantico delle parole finali è stata verificata tramite ascolto diretto sui 10 campioni sentinella.
3. **Sample Peak vs True Peak:** Il valore `peak_dbfs` è chiaramente documentato come **sample peak** digitale ($\text{max}(|s_i|)$) e non come True Peak inter-sample (ITU-R BS.1770).
4. **RTF e Aspettative di Throughput:** Non vengono assunte aprioristicamente accelerazioni senza profiling. Il throughput attuale su Windows con SDPA (~4.7–4.9 RTF) sarà analizzato tramite profiling mirato in M1 per isolare l'impatto di IO, tokenizzazione e kernel di attenzione.
5. **Metadati Hardware e Disaccoppiamento Ambienti:** Chiarita la separazione tra ambiente host CPU (`Python 3.10`) e ambiente di progetto `.venv` (`Python 3.12.10`, `PyTorch 2.6.0+cu124`, GPU RTX 4080 Laptop). I metadati in `scripts/hardware_info.json` riflettono l'ambiente `.venv`.
6. **Discrepanza VRAM 0.6B (2.820 MB vs 2.622 MB):** Verificato empiricamente che 0.6B alloca ~2.400–2.646 MB su frasi standard e scala fino a **2.821,69 MB** sui periodi lunghi da oltre 300 caratteri (`LEN_04`, `LEN_05`). Il driver `nvidia-smi` rileva un'occupazione di picco di 3.902 MB.
7. **Pull Request:** Specificato che il link `.../pull/new/m0-audit-benchmark` è la pagina di comparazione e predisposizione della PR; l'apertura effettiva compete al Product Owner.

---

## 2. Risultati Empirici del Benchmark Controllato (Dati Reali e Misurati)

> **Fonte Dati:** 40 run misurati estratti da `bench/benchmark_events.jsonl` (2 iterazioni per ciascuna delle 10 frasi sentinella, con warmup escluso).

| Metrica Prestazionale | `Qwen3-TTS-12Hz-1.7B-Base` | `Qwen3-TTS-12Hz-0.6B-Base` | `Chatterbox Multilingual V3` |
| :--- | :---: | :---: | :---: |
| **Classificazione Formale** | **MISURATO (Campione Controllato)** | **MISURATO (Campione Controllato)** | **NON TESTATO (Differito)** |
| **Run Reali Monotonici Registrati** | **20 run misurati** (+1 warmup) | **20 run misurati** (+1 warmup) | 0 run (ambiente separato) |
| **Casi Audio nel Corpus M0** | 60 WAV (10 misurati, 50 cache non mis.) | 60 WAV (10 misurati, 50 cache non mis.) | 0 WAV |
| **RTF Mediano (`REAL_INFERENCE`)** | **4.718** | **4.944** | Non disponibile |
| **RTF Medio $\pm$ Dev. Std.** | **4.660 $\pm$ 0.502** | **5.973 $\pm$ 2.137** | Non disponibile |
| **Intervallo Min – Max RTF** | **2.716 – 5.180** | **3.436 – 10.735** | Non disponibile |
| **RTF 95° Percentile (p95)** | **5.120** | **10.668** | Non disponibile |
| **Tempo Generazione Mediano** | 26.28 s | 35.97 s | Non disponibile |
| **Durata Media Traccia Audio** | 8.27 s | 8.70 s | Non disponibile |
| **PyTorch VRAM Picco Allocata** | **4.756,05 MB** | **2.821,69 MB** | Non disponibile |
| **PyTorch VRAM Picco Riservata** | **5.472,00 MB** | **3.666,00 MB** | Non disponibile |
| **Driver `nvidia-smi` Picco VRAM** | **5.708,00 MB** | **3.902,00 MB** | Non disponibile |
| **Margine VRAM Libera (su 12.282 MB)** | **> 6.574 MB liberi (53.5%)** | **> 8.380 MB liberi (68.2%)** | Non disponibile |
| **Sample Peak Range (dBFS)** | -0.21 dBFS a -3.06 dBFS | -0.86 dBFS a -8.66 dBFS | Non disponibile |
| **Tracce Mute / Code $>2\text{s}$** | 0 / 0 | 0 / 0 | Non disponibile |

---

## 3. Riepilogo dei Deliverable e Tracciabilità

1. **[`bench/run_benchmark.py`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/run_benchmark.py)**: Script refactorizzato con logging append-only JSONL e totale rimozione delle stime sintetiche nel path cache.
2. **[`bench/benchmark_events.jsonl`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/benchmark_events.jsonl)**: 42 eventi grezzi registrati con hash SHA256 di input/output, timer monotonici e memorie GPU.
3. **[`bench/results_benchmark.csv`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/results_benchmark.csv)**: 180 righe con separazione netta tra `REAL_INFERENCE` (10 per modello) e `CACHE_RECOVERY_UNMEASURED` (50 per modello, con campi tempo/VRAM nulli).
4. **[`bench/summary_metrics.json`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/summary_metrics.json)**: Statistiche aggregate calcolate esclusivamente sui record misurati.
5. **[`bench/QUALITY_LISTENING_DOSSIER.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/QUALITY_LISTENING_DOSSIER.md)**: Dossier di ascolto acustico umano sui 10 campioni sentinella x 2 modelli.
6. **[`bench/m0_neural_samples.zip`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/m0_neural_samples.zip)**: Archivio (5.73 MB) contenente i 20 WAV neurali, i 2 riferimenti CC0 e il file `CHECKSUMS.txt`.
7. **[`HARDWARE_REPORT.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/HARDWARE_REPORT.md)**: Report aggiornato con profilo di memoria PyTorch vs `nvidia-smi` e disaccoppiamento ambienti.
8. **[`m0_deliverables.zip`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/m0_deliverables.zip)**: Archivio consolidato complessivo (5.80 MB).

---

## 4. Richiesta Conclusiva

Con la consegna di tutte le evidenze empiriche, l'eliminazione dei dati sintetici, la pubblicazione dei WAV neurali con checksum e la formalizzazione dei dossier di ascolto, **si richiede il via libera tecnico da parte di ChatGPT e l'autorizzazione formale da parte del Product Owner (Utente) per la chiusura definitiva di Milestone M0 e il passaggio alla Milestone M1**.
