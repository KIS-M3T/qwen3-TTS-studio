# Hardware & Environment Diagnostic Report
> Generato da `scripts/hardware_probe.py` e aggiornato con misure empiriche del benchmark neurale M0.

## 1. Sommario Esecutivo

- **Sistema Operativo**: Windows 10 (AMD64)
- **CPU**: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- **RAM di Sistema**: 31.73 GB totali (disponibili: 6.18 GB)
- **GPU Principale**: **NVIDIA GeForce RTX 4080 Laptop GPU** — 12.282 MB VRAM
- **Driver NVIDIA**: 576.57
- **CUDA Version (Driver)**: 12.9, V12.9.86
- **Storage Locale**: >90 GB liberi su 951.65 GB (`C:\Users\occhi\Desktop\CASA\PERSONALE\PROJECT\ITA_TTS`)
- **Python Runtime**: 3.12.9 (`.venv\Scripts\python.exe`)
- **PyTorch**: 2.6.0+cu124 (CUDA disponibile: **True**, Device: NVIDIA GeForce RTX 4080 Laptop GPU)

## 2. Dettaglio GPU (nvidia-smi & Benchmark Reale)

| ID | Modello | VRAM Totale | VRAM Libera Idle | VRAM Usata Picco (1.7B) | Driver | Temp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | NVIDIA GeForce RTX 4080 Laptop GPU | 12282 MB | 11997 MB | 4558.03 MB (37.1%) | 576.57 | 49.0–52.0 °C |

## 3. Ambiente PyTorch & Accellerazione

- **PyTorch Installato**: True (`torch 2.6.0+cu124` & `torchaudio 2.6.0+cu124` in `.venv`)
- **CUDA Runtime PyTorch**: 12.4
- **Dispositivi CUDA accessibili da PyTorch**: 1 (`cuda:0`)
- **Supporto nativo BF16**: True (`torch.cuda.is_bf16_supported() == True`)
- **Supporto nativo FP16**: True
- **Attenzione Ottimizzata**: SDPA (`scaled_dot_product_attention`) abilitata e verificata con kernel C++/CUDA.

## 4. Note Operative per Milestone M0

1. **Verifica VRAM Effettiva**: La scheda GPU dispone di **12.282 MB di VRAM** dedicata. I test reali hanno confermato un'occupazione di picco di **4.558 MB (1.7B)** e **2.820 MB (0.6B)**, garantendo un margine di sicurezza di oltre 7.5 GB di VRAM libera.
2. **PyTorch Environment**: Configurato l'ambiente virtuale dedicato `.venv` con Python 3.12 e PyTorch compilato per CUDA 12.4 (`cu124`), perfettamente interoperabile con i driver NVIDIA 576.57 (CUDA 12.9).
3. **Spazio Disco**: Verificato sufficiente per i pesi modello Qwen3-TTS 0.6B (~1.24 GB) e 1.7B (~3.86 GB), il tokenizer (682 MB) e le 120 tracce audio WAV generate in `bench/outputs/` (~43 MB totali).
