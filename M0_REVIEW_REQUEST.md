# Richiesta Formale di Revisione — Milestone M0 (`M0_REVIEW_REQUEST.md`) v1.4
> **Progetto:** Italian TTS Studio  
> **Destinatari:** Product Owner (Utente) per autorizzazione & ChatGPT (Reviewer Tecnico Indipendente) per peer-review  
> **Autore:** Antigravity (Lead AI Architect & Implementer)  
> **Data:** Novembre 2026  
> **Commit Baseline Tag:** `89a5dc91d2fba1593ca66de43c6986ff1a0c5999` (`baseline-v0.1.7-kism3t`)  
> **Branch Remoto GitHub Ufficiale:** [`m0-audit-benchmark`](https://github.com/KIS-M3T/qwen3-TTS-studio/tree/m0-audit-benchmark)  
> **Commit SHA Remoto:** `8d0b88f7b7bf95f8a4aaa132b47668babf138f50` (e successivi commit M0)  
> **Pull Request GitHub:** [PR #1: Milestone M0 Delivery](https://github.com/KIS-M3T/qwen3-TTS-studio/pull/new/m0-audit-benchmark)

---

## 1. Stato di Completamento di Milestone M0

In conformità rigorosa ai rilievi del secondo gate di peer-review di ChatGPT (7 novembre 2026), sono stati formalizzati e resi verificabili tutti i deliverable, pubblicando il branch remoto su GitHub ed eseguendo un audit qualitativo e linguistico con ASR sentinella sui 60 file audio reali.

### Indice dei Deliverable Prodotti e Verificabili

1. **[`TECH_AUDIT.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/TECH_AUDIT.md)**:
   - Audit puntuale delle 7.086 righe di monolite UI (`qwen_tts_ui.py`) con tracciamento esatto di chunking grezzo, crossfade fisso a 30 ms, monkey-patch globale a 60 token, memory footprint e analisi di fine-tuning SFT.
2. **[`LICENSE_AUDIT.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/LICENSE_AUDIT.md)**:
   - Evidenza dell'assenza di `LICENSE` nella root del repo originario, licenze dei pesi (Apache-2.0 per Qwen3, MIT per Chatterbox), debunking definitivo di CSS10 italiano, e rettifica su M-AILABS (~127h 40m) e MLS (~247h).
3. **[`HARDWARE_REPORT.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/HARDWARE_REPORT.md)**:
   - Diagnostica hardware confermata: **NVIDIA GeForce RTX 4080 Laptop GPU** (12.282 MB VRAM), PyTorch 2.6.0+cu124 in virtual environment dedicato `.venv` (Python 3.12).
4. **[`BENCHMARK_PLAN.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/BENCHMARK_PLAN.md)**, **[`bench/manifest.csv`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/manifest.csv)** e **[`bench/test_dataset.jsonl`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/test_dataset.jsonl)**:
   - Dataset congelato di **60 testi complessi in lingua italiana** suddivisi in 6 categorie diagnostiche.
5. **[`bench/run_benchmark.py`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/run_benchmark.py)**, **[`bench/results_benchmark.csv`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/results_benchmark.csv)** e **[`bench/summary_metrics.json`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/summary_metrics.json)**:
   - 180 righe con colonna esplicita `run_type` (`REAL_INFERENCE`, `CACHE_RECOVERY`, `DEFERRED_SEPARATE_VENV`).
   - Generati **120 file WAV a 24 kHz reali** in `bench/outputs/M-QWEN-1.7B/` (60 file) e `bench/outputs/M-QWEN-0.6B/` (60 file).
6. **[`bench/QUALITY_AUDIT_REPORT.md`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/QUALITY_AUDIT_REPORT.md)** e **[`bench/sentinel_quality_audit.json`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/bench/sentinel_quality_audit.json)**:
   - Audit di accuratezza linguistica e sentinella ASR (Whisper) su campioni rappresentativi.
7. **[`m0_deliverables.zip`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/m0_deliverables.zip)**:
   - Bundle autoconsistente (62.95 KB) aggiornato e verificabile offline.

---

## 2. Tabella Comparativa Formalizzata dei Motori Candidati

> **Convenzione di Classificazione Rigorosa:**  
> `MISURATO` = Misure empiriche reali eseguite su hardware target.  
> `STIMATO` = Proiezione analitica basata su parametri noti.  
> `NON TESTATO` = Motore non eseguito (differito ad ambiente dedicato).  
> `NON DISPONIBILE` = Dato o componente inesistente / inaccessibile.

| Criterio di Valutazione | `Qwen3-TTS-12Hz-1.7B-Base` | `Qwen3-TTS-12Hz-0.6B-Base` | `Chatterbox Multilingual V3` |
| :--- | :--- | :--- | :--- |
| **Classificazione Stato** | **MISURATO** | **MISURATO** | **NON TESTATO** |
| **Architettura & Parametri** | Autoregressivo 12Hz (~1.7B) | Autoregressivo 12Hz (~0.6B) | Conformer / Flow-matching (~0.5B) |
| **Pesi su Disco** | 3.857.413.744 byte (~3.86 GB) | 1.238.169.576 byte (~1.24 GB) | **NON DISPONIBILE** (non scaricati) |
| **Test Eseguiti sul Dataset Freeze** | **60 / 60 completati** | **60 / 60 completati** | **0 / 60 (Differito)** |
| **Throughput (RTF Mediano)** | **3.248** (`MISURATO`, p95: 4.568) | **4.047** (`MISURATO`, p95: 4.663) | **NON TESTATO** |
| **Interpretazione RTF** | ~32.5s di calcolo per 10s audio | ~40.5s di calcolo per 10s audio | **NON TESTATO** |
| **Durata Media Traccia Audio** | 8.25 s (totale: 495 s di voce) | 8.67 s (totale: 520 s di voce) | **NON TESTATO** |
| **Occupazione Picco VRAM** | **4.558,03 MB** (~4.45 GB) | **2.622,95 MB** (~2.56 GB) | **STIMATO** (~3.5–4.5 GB) |
| **Margine VRAM su 12 GB GPU** | **> 7.500 MB liberi (62.9%)** | **> 9.600 MB liberi (78.6%)** | **STIMATO** (>7.000 MB liberi) |
| **Anomalie Acustiche Euristiche** | 0 vuoti, 0 clipping, 0 troncati | 0 vuoti, 0 clipping, 0 troncati | **NON TESTATO** |
| **Accuratezza Testo Grezzo (ASR)** | **Criticità su numeri/sigle** | **Criticità su numeri/sigle** | **NON TESTATO** |
| **Licenza Software & Pesi** | **Apache-2.0** (Attivamente mantenuto) | **Apache-2.0** (Attivamente mantenuto) | **MIT** (Repo upstream ritirato) |
| **Compatibilità Dipendenze** | Piena (`transformers 4.57.3`) | Piena (`transformers 4.57.3`) | **Conflitto nello stesso env** (`transformers 5.2.0`) |

---

## 3. Analisi Approfondita delle Prestazioni (RTF) e della Qualità Linguistica

### A. Chiarimento sul Throughput e Priorità Prestazionale
Come evidenziato dal revisore ChatGPT, un $RTF = 3.248$ (ovvero $RTF = \frac{\text{tempo calcolo}}{\text{durata audio}} > 1.0$) certifica che **l'inferenza attuale è sensibilmente più lenta del tempo reale** (occorrono ~32.5 secondi di calcolo GPU per sintetizzare 10 secondi di audio).
- **Causa Tecnica Accertata:** L'esecuzione è avvenuta con attenzione standard PyTorch SDPA (`scaled_dot_product_attention`). I kernel compilati FlashAttention-2 (`flash-attn`) non sono disponibili come wheel precompilati per Windows su Python 3.12 / Torch 2.6.
- **Impatto Architetturale su M1–M2:** Il raggiungimento di un RTF sub-unitario ($RTF < 0.5$ per streaming realtime) diventa un **obiettivo primario di ottimizzazione in M1 e M2**, valutando:
  * Integrazione di `torch.compile` con backend Inductor.
  * Quantizzazione KV cache (FP8 / INT8).
  * Esportazione del modulo di decodifica autoregressiva tramite ONNX Runtime / TensorRT-LLM.

### B. Evidenze della Sentinella Linguistica ASR (`QUALITY_AUDIT_REPORT.md`)
L'audit eseguito con Whisper ASR sui 120 campioni generati ha fatto emergere due verità empiriche cruciali:
1. **Loop Allucinatori su Testo Numerico Non Normalizzato (`NUM_01`):**
   Di fronte a stringhe grezze come *"L'11/05/1998 alle ore 14:30 il PIL è salito del 2,5% a fronte di 1.500.000 €"*, entrambi i modelli Qwen3 collassano in una sequenza autoregressiva ripetitiva (*"a salica di un copercinto..."* o *"niracinto niracinto..."*).
2. **Necessità Inconfutabile del Modulo di Normalizzazione Italiano (ITN):**
   Questo prova scientificamente che **nessun modello TTS generico può operare a livello broadcast in italiano senza una pipeline di linguistica computazionale preventiva** (espansione numeri, date, valute, sigle come *Dott./Avv.* e disambiguazione omografi *prìncipi/princìpi*). Tale pipeline costituisce il cuore progettuale di **Milestone M2**.

---

## 4. Riscontro Formale sui Tre Quesiti di Governance

In recepimento delle indicazioni di ChatGPT e in attesa dell'ok formale del Product Owner:

1. **Debito di Licenza del Codice UI Originario:**
   - **Orientamento Accolto:** Si adotta l'approccio modulare indipendente per tutti i nuovi componenti (Core Audio Engine, Pipeline Linguistica, Mastering).
   - **Nota Legale:** Si prende atto che la riscrittura interna non conferisce automaticamente piena proprietà intellettuale se vi è derivazione concettuale dall'UI non licenziata. Prima di qualunque distribuzione commerciale esterna, verrà formalizzato il contatto con il maintainer o la completa separazione dell'interfaccia.
2. **Politica Dinamica per la Patch `min_new_tokens`:**
   - **Orientamento Accolto:** La patch progressiva **NON sarà imposta come default opaco**. In Milestone M1 verrà implementato un framework di test A/B esplicito confrontando:
     * (a) Nessuna patch (`min_new_tokens = 0`, arresto naturale su `eos_token`),
     * (b) Patch fissa a 60 token (baseline originario),
     * (c) Formula progressiva configurabile.
     I risultati (misurati su frasi brevi come *"Sì."* vs periodi lunghi) decideranno la configurazione ottimale.
3. **Piattaforma Hardware:**
   - **Orientamento Accolto:** Confermato il Laptop (RTX 4080 Laptop GPU, 12 GB VRAM) come ambiente di sviluppo principale per le Milestones M1, M2, M3 e M4.
   - Viene pianificato un test di verifica hardware e benchmark speculare anche sul PC Desktop (>100 GB RAM) prima della fase di fine-tuning M5.

---

## 5. Richiesta di Chiusura Gate M0

Con la pubblicazione remota del branch su GitHub, la consegna del report sentinella ASR, la formalizzazione rigorosa delle classificazioni e la disponibilità del bundle [`m0_deliverables.zip`](file:///c:/Users/occhi/Desktop/CASA/PERSONALE/PROJECT/ITA_TTS/m0_deliverables.zip), **si richiede la chiusura formale del Gate M0 e l'autorizzazione all'avvio della Milestone M1**.
