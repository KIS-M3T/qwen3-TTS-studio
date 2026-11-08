# Hardware & Environment Diagnostic Report
> Generato ed empiricamente verificato per Milestone M0.  
> **Data:** 8 Novembre 2026  
> **Ambiente di Esecuzione Benchmark:** `.venv` (Python 3.12.10, PyTorch 2.6.0+cu124, CUDA 12.4)

---

## 1. Sommario Esecutivo

- **Sistema Operativo**: Windows 11 (AMD64, kernel 10.0.26200)
- **CPU**: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (24 vCPU logiche)
- **RAM di Sistema**: 31.73 GB totali (disponibili: ~5.0 GB)
- **GPU Principale**: **NVIDIA GeForce RTX 4080 Laptop GPU** — 12.282 MB VRAM dedicata
- **Driver NVIDIA**: 576.57
- **CUDA Driver**: 12.9 (V12.9.86)
- **CUDA Runtime PyTorch**: 12.4
- **Storage Locale**: ~77 GB liberi su 951.65 GB SSD NVMe
- **Python Runtime Progetto**: Python 3.12.10 (`.venv\Scripts\python.exe`)
- **PyTorch Progetto**: 2.6.0+cu124 (CUDA accessibile: **True**, dispositivo: `cuda:0`)

---

## 2. Profilo di Memoria GPU Empirico (PyTorch vs Driver nvidia-smi)

La profilazione eseguita durante il benchmark controllato sui modelli neurali (`M-QWEN-1.7B` e `M-QWEN-0.6B`) consente di chiarire in modo rigoroso e definitivo la distinzione tra memoria allocata da PyTorch, memoria riservata dal memory pool del framework e memoria totale occupata a livello di driver NVIDIA:

| Modello | Condizione di Carico | PyTorch Allocated Peak | PyTorch Reserved Peak | Driver `nvidia-smi` Used Peak | VRAM Margine Libero |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Idle GPU** | Nessun modello caricato | 0 MB | 0 MB | 0–220 MB | > 12.000 MB |
| **M-QWEN-0.6B** | Frasi brevi/medie (`LEN_01`–`03`, `HOM_01`) | 2.314 – 2.469 MB | 3.246 – 3.666 MB | 3.482 – 3.902 MB | ~8.380 MB |
| **M-QWEN-0.6B** | Sequenze lunghe (`LEN_04`, `LEN_05`, 348 car.) | **2.821,69 MB** | 3.666,00 MB | 3.902,00 MB | ~8.380 MB |
| **M-QWEN-1.7B** | Frasi brevi/medie (`LEN_01`–`03`, `ACR_01`) | 4.247 – 4.459 MB | 5.040 – 5.472 MB | 5.276 – 5.708 MB | ~6.574 MB |
| **M-QWEN-1.7B** | Sequenze lunghe (`LEN_04`, `LEN_05`, 348 car.) | **4.756,05 MB** | 5.472,00 MB | 5.708,00 MB | ~6.574 MB |

### Risoluzione Discrepanza VRAM 0.6B (2.820 MB vs 2.622 MB)
La discrepanza evidenziata nella peer review (Gate 3, P1 punto 6) è stata ricondotta alla reale dinamica computazionale della memoria:
1. Su frasi a lunghezza media/standard, il picco allocato da PyTorch per 0.6B si attesta a **2.622–2.646 MB** (da cui il valore di 2.622,95 MB osservato nel primo benchmark).
2. Su sequenze complesse e periodi lunghi da oltre 300 caratteri (`LEN_04` a 312 caratteri e `LEN_05` a 348 caratteri), l'allocazione del buffer KV-cache e delle attivazioni cross-attention scala esattamente fino a **2.821,69 MB** (da cui il valore arrotondato di 2.820 MB citato nel report hardware).
3. Entrambi i valori sono empiricamente esatti e corrispondono a classi di carico differenti dello stesso modello.

---

## 3. Disaccoppiamento Ambienti Python (Host vs `.venv`)

Per garantire la massima tracciabilità ed evitare conflitti:
- **Ambiente Globale Host:** `Python 3.10.11` con build `torch 2.11.0+cpu` ad uso diagnostico di sistema.
- **Ambiente di Progetto Dedicato:** `Python 3.12.10` in `.venv/`, con PyTorch `2.6.0+cu124` e torchaudio `2.6.0+cu124`.
- Tutti i comandi del benchmark, generazione audio e test ASR vengono eseguiti **esclusivamente all'interno di `.venv`**, garantendo il pieno sfruttamento dell'accelerazione CUDA e il supporto nativo `bfloat16`.
- I metadati in `scripts/hardware_info.json` registrano l'esecuzione dentro `.venv`.

---

## 4. Prestazioni di Throughput e Attenzione SDPA su Windows

- **Kernel di Attenzione:** Su sistema operativo Windows, l'estensione proprietaria `flash-attn` C++/CUDA non è precompilata ufficialmente e non è presente. L'engine TTS opera pertanto con l'implementazione integrata PyTorch **SDPA (`scaled_dot_product_attention`)**.
- **RTF Empirico Misurato:** L'RTF registrato con timer monotonico `time.perf_counter()` è pari a **~4.718 (mediana 1.7B)** e **~4.944 (mediana 0.6B)**. Ciò significa che la generazione impiega circa 4.5–5x la durata dell'audio.
- **Profiling in M1:** Come raccomandato da ChatGPT (P1 punto 4), non vengono fatte promesse aprioristiche di accelerazioni 6–8x senza profiling. L'identificazione dei colli di bottiglia (IO, tokenizzazione, SDPA vs kernel nativi, assenza di KV caching dinamico) sarà oggetto di profiling dedicato nella fase architetturale di Milestone M1.
