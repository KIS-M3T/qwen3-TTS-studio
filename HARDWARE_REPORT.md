# Hardware & Environment Diagnostic Report
> Generato automaticamente da `scripts/hardware_probe.py` per Milestone M0.

## 1. Sommario Esecutivo

- **Sistema Operativo**: Windows 10 (AMD64)
- **CPU**: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- **RAM di Sistema**: 31.73 GB totali (disponibili: 6.18 GB)
- **GPU Principale**: **NVIDIA GeForce RTX 4080 Laptop GPU** — 12282 MB VRAM (Libera: 11997 MB)
- **Driver NVIDIA**: 576.57
- **CUDA Version (Driver)**: 12.9, V12.9.86
- **Storage Locale**: 95.82 GB liberi su 951.65 GB (C:\Users\occhi\Desktop\CASA\PERSONALE\PROJECT\ITA_TTS)
- **Python Runtime**: 3.10.11 (`C:\Users\occhi\AppData\Local\Programs\Python\Python310\python.exe`)
- **PyTorch**: 2.11.0+cpu (CUDA disponibile: False)

## 2. Dettaglio GPU (nvidia-smi)

| ID | Modello | VRAM Totale | VRAM Libera | VRAM Usata | Driver | Temp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | NVIDIA GeForce RTX 4080 Laptop GPU | 12282 MB | 11997 MB | 0 MB | 576.57 | 49.0 °C |

## 3. Ambiente PyTorch & Accellerazione

- **PyTorch Installato**: True
- **Versione PyTorch**: 2.11.0+cpu
- **CUDA Runtime PyTorch**: N/A
- **Dispositivi CUDA accessibili da PyTorch**: 0
- **Supporto nativo BF16**: False
- **Supporto nativo FP16**: False

## 4. Note Operative per Milestone M0

1. **Verifica VRAM Effettiva**: La scheda GPU rilevata dispone di **~12.288 MB di VRAM** dedicata.
2. **PyTorch Environment**: Se l'interprete Python globale ha installata una build PyTorch CPU (`+cpu`), per il benchmark con GPU CUDA è necessario operare all'interno di un ambiente virtuale (`.venv`) dedicato con PyTorch compilato per CUDA 12.x (`cu124` o `cu121`).
3. **Spazio Disco**: Verificato sufficiente per i pesi modello Qwen3-TTS 0.6B (~2.3 GB) e 1.7B (~4.2 GB) e l'audio generato.