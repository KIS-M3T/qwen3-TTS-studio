# Piano Operativo di Benchmark per il TTS Italiano (`BENCHMARK_PLAN.md`)
> **Milestone M0 — Italian TTS Studio**  
> **Oggetto:** Protocollo di valutazione comparativa rigorosa per motori TTS su lingua italiana.  
> **Data:** Novembre 2026

---

## 1. Obiettivo del Benchmark

Il presente protocollo definisce un framework di valutazione empirica, riproducibile e non simulata, volto a:
1. Misurare le prestazioni effettive (tempo di risposta, fattore tempo reale RTF, occupazione VRAM e stabilità acustica) dei modelli TTS candidati su hardware con 12 GB di VRAM.
2. Quantificare l'impatto degli errori fonetici, prosodici e linguistici dell'italiano (numeri, valute, sigle, omografi, frasi ipotattiche) in assenza e in presenza di normalizzazione.
3. Stabilire un confronto trasparente tra **Qwen3-TTS 1.7B-Base**, **Qwen3-TTS 0.6B-Base** e **Chatterbox Multilingual V3** (in qualità di alternativa isolata e ritirata ad agosto 2026).

---

## 2. Modelli Candidati e Condizioni Sperimentali

| ID Motore | Nome Modello | Pesi / Checkpoint | Architettura | Licenza | Stato & Ruolo nel Progetto |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M-QWEN-1.7B** | `Qwen3-TTS-12Hz-1.7B-Base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Autoregressivo 12Hz (~1.7B) | Apache-2.0 | **Candidato Principale** per clonazione e qualità broadcast. |
| **M-QWEN-0.6B** | `Qwen3-TTS-12Hz-0.6B-Base` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Autoregressivo 12Hz (~0.6B) | Apache-2.0 | **Baseline Leggera** ad alta velocità e test di fattibilità. |
| **M-CHAT-0.5B** | `Chatterbox Multilingual V3` | `ResembleAI/chatterbox` | Conformer/Flow-matching (~0.5B) | MIT | **Alternativa Isolata** (Retired research da agosto 2026). |

### Condizioni di Esecuzione
* **Precisione:** `bfloat16` nativo su GPU NVIDIA (con fallback FP16 ove necessario).
* **Attenzione:** FlashAttention-2 abilitato ove supportato dalla libreria.
* **Audio Reference:** Stesso file audio di riferimento vocale in lingua italiana (clip mono da 5–8 secondi a 24 kHz, tratta da M-AILABS o Mozilla Data Collective, esente da riverbero e priva di rumore di fondo) per tutti i modelli che supportano voice cloning.
* **Campioni:** Per ogni test deterministico viene registrato l'output nativo in formato **WAV non compresso**. Su un sottoinsieme rappresentativo vengono eseguiti 3 seed distinti per verificare la stabilità timbrica e l'assenza di allucinazioni stocastiche.

---

## 3. Il Dataset Congelato di Test (60 Esempi)

Il dataset è formalizzato nei file:
- **`bench/manifest.csv`** (formato tabellare per elaborazione statistica)
- **`bench/test_dataset.jsonl`** (formato JSON Lines con metadati dettagliati)

### Ripartizione delle Categorie:
1. **`numbers_currencies_dates` (12 testi, `NUM_01` – `NUM_12`):**
   - Date estese e compatte (`11/05/1998`), orari (`14:30`, `08:05`), decimali con virgola (`2,5%`), valute formattate (`1.500.000 €`, `4,50 $`), unità di misura (`km/h`, `mmHg`, `m s.l.m.`), numeri romani (`XXI secolo`), numeri telefonici e codici legislativi (`D.Lgs. 50/2016`).
2. **`acronyms_abbreviations_names` (10 testi, `ACR_01` – `ACR_10`):**
   - Sigle istituzionali e tecniche (`INPS`, `TAR`, `CNR`, `BCE`, `Wi-Fi`, `IEEE 802.11ax`), titoli abbreviati (`Dott.`, `Avv.`, `Prof.`, `sig.`), indirizzi email e PEC.
3. **`homographs_accents_phonetics` (10 testi, `HOM_01` – `HOM_10`):**
   - Disambiguazione dell'accento tonico (*prìncipi* vs *princìpi*, *àncora* vs *ancòra*, *sùbito* vs *subìto*), e apertura delle vocali fonetiche (*pèsca* vs *pésca*, *bòtte* vs *bótte*).
4. **`narrative_dialogues` (10 testi, `NAR_01` – `NAR_10`):**
   - Discorsi diretti con caporali («...»), incisi narrative, intonazione interrogativa ed esclamativa, cadenze emotive e pause riflessive.
5. **`technical_code_switching` (8 testi, `TEC_01` – `TEC_08`):**
   - Frasi in italiano tecnico con prestiti inglesi comuni nel software (*machine learning*, *memory leak*, *thread pool*, *docker compose*, *pull request*).
6. **`length_boundary_edge_cases` (10 testi, `LEN_01` – `LEN_10`):**
   - Casi limite di lunghezza: monosillabi (*"Sì."*), frasi cortissime (*"No, grazie."*) per verificare l'impatto della patch `min_new_tokens=60`, elenchi puntati con punti e virgole, espressioni matematiche con parentesi, e periodi lunghi complessi oltre i 300 e 400 caratteri per testare i limiti del chunker e del context buffer.

---

## 4. Metriche Misurate e Protocollo di Rilevazione

Per ogni generazione vengono registrati i seguenti valori quantitativi e qualitativi:

### 4.1 Metriche Computazionali & Hardware
* **Cold Start Latency (s):** Tempo intercorso tra l'invocazione e il primo campione audio generato (inclusivo dell'inizializzazione del grafo).
* **Generation Time (s):** Tempo effettivo di calcolo per il completamento dell'inferenza.
* **Audio Duration (s):** Durata effettiva del file WAV generato.
* **Real-Time Factor (RTF):** Calcolato come:
  $$\text{RTF} = \frac{\text{Tempo di Generazione (s)}}{\text{Durata Audio Generato (s)}}$$
  Un valore $\text{RTF} < 1.0$ indica generazione più veloce del tempo reale. Verranno calcolate **mediana** e **95° percentile (p95)**.
* **Peak VRAM Allocated & Reserved (MB):** Picco massimo di memoria GPU registrato tramite `torch.cuda.max_memory_allocated()` e `torch.cuda.max_memory_reserved()`.
* **RAM di Processo (MB):** Memoria fisica residente (RSS) occupata dal processo Python.

### 4.2 Metriche di Qualità Acustica & Robustezza
* **Test Audio Vuoto / Glitch:** Verifica automatica di picco assoluto $< 0.003$ o RMS $< 0.001$.
* **True Peak & Clipping (dBFS):** Misura dell'ampiezza massima di picco per intercettare saturazioni digitali (> 0.0 dBFS).
* **Troncamento / Silenzio di Coda:** Verifica della presenza di audio tagliato prima del completamento del testo o code di silenzio anomale $> 1.5\text{ s}$.
* **WER / CER Sentinella:** Calcolo del Word Error Rate e Character Error Rate tramite trascrizione ASR con modello Whisper Italiano (come indicatore diagnostico di omissione o ripetizione, accompagnato da verifica manuale).
* **Valutazione Soggettiva A/B alla Cieca:** Ascolto comparativo a doppio cieco su 10 frasi campione per valutare naturalezza, correttezza dell'accento italiano, fluidità e assenza di metallicità.

---

## 5. Struttura dei Risultati e Deliverable

Tutti i risultati saranno archiviati in:
```
bench/
├── manifest.csv               # Specifiche dei 60 testi
├── test_dataset.jsonl         # Dataset completo in JSONL
├── run_benchmark.py           # Script di esecuzione automatica e logging
├── results_benchmark.csv      # Misure tabellari complete
├── summary_metrics.json       # Mediane, p95, picchi VRAM aggregati
└── outputs/                   # Directory contenente i file WAV nativi
    ├── M-QWEN-1.7B/
    ├── M-QWEN-0.6B/
    └── M-CHAT-0.5B/
```
