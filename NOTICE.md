# NOTICE & CODICE DI PROVENIENZA DEI COMPONENTI
## Italian TTS Studio (`ITA_TTS`) — Milestone M1

> **Documento:** `NOTICE.md`  
> **Repository:** `KIS-M3T/qwen3-TTS-studio`  
> **Branch di Sviluppo:** `phase1/m1-modular-core-engine`  
> **Data:** Novembre 2026  
> **Stato:** Milestone M1 — WP0 (Audit Preliminare di Provenienza e Diritti)

---

## 1. Quadro Generale di Provenienza del Codice

Il progetto **Italian TTS Studio** nasce come fork e rifattorizzazione architetturale del repository `bc-dunia/qwen3-TTS-studio`. Al fine di garantire la massima trasparenza tecnica e conformità legale, si adotta una tracciabilità rigorosa tra codice originario ereditato, codice modificato e nuovi moduli sviluppati *ex-novo*.

### 1.1 Matrice di Provenienza Modulare

| Componente / Directory | Origine Primaria | Stato di Modifica | Licenza / Termini d'Uso |
| :--- | :--- | :--- | :--- |
| `podcast/` | `bc-dunia/qwen3-TTS-studio` | Invariato (client applicativo) | Copyright © 2026 bc-dunia (senza licenza esplicita standard) |
| `ui/` | `bc-dunia/qwen3-TTS-studio` | Invariato (interfaccia Gradio) | Copyright © 2026 bc-dunia (senza licenza esplicita standard) |
| `storage/` | `bc-dunia/qwen3-TTS-studio` | Invariato (modelli persona legacy) | Copyright © 2026 bc-dunia (senza licenza esplicita standard) |
| `audio/model_loader.py` | `bc-dunia/qwen3-TTS-studio` | **Modificato** (fix tokenizer, patch anti-troncamento) | Codice upstream modificato |
| `audio/generator.py` | `bc-dunia/qwen3-TTS-studio` | **Modificato** (supporto bfloat16, diagnostica troncamento) | Codice upstream modificato |
| `qwen_tts_ui.py` | `bc-dunia/qwen3-TTS-studio` | **Modificato** (adattato come client headless-ready) | Codice upstream modificato |
| `src/engine/` | **Italian TTS Studio** | **Nuovo Modulo Originale** | Copyright © 2026 Italian TTS Studio (`KIS-M3T`) |
| `src/linguistics/` | **Italian TTS Studio** | **Nuovo Modulo Originale** | Copyright © 2026 Italian TTS Studio (`KIS-M3T`) |
| `src/audio/` | **Italian TTS Studio** | **Nuovo Modulo Originale** | Copyright © 2026 Italian TTS Studio (`KIS-M3T`) |
| `src/instrumentation/` | **Italian TTS Studio** | **Nuovo Modulo Originale** | Copyright © 2026 Italian TTS Studio (`KIS-M3T`) |
| `tests/` | **Italian TTS Studio** | **Nuovo Modulo Originale** | Copyright © 2026 Italian TTS Studio (`KIS-M3T`) |
| `bench/` | **Italian TTS Studio** | **Nuovo Modulo Originale** (script di benchmark ed evidenze M0/M1) | Copyright © 2026 Italian TTS Studio (`KIS-M3T`) |

### 1.2 Politica sul Diritto di Redistribuzione del Fork
1. Poiché il repository upstream `bc-dunia/qwen3-TTS-studio` non include un file di licenza permissiva standard (es. MIT, Apache-2.0), il progetto Italian TTS Studio **non appone unilateralmente un file `LICENSE` conflittuale** all'intera base di codice.
2. I nuovi moduli implementati nella directory `src/` e nella suite `tests/` costituiscono opere indipendenti e proprietarie di Italian TTS Studio (`KIS-M3T`), distribuibili e modularmente isolabili rispetto alla logica legacy.

---

## 2. Modelli e Pesi Neurali

| Modello / Componente | Fornitore / Autore | Licenza | Utilizzo Ammesso |
| :--- | :--- | :--- | :--- |
| **Qwen3-TTS-12Hz-1.7B-Base** | Alibaba Cloud | Apache-2.0 | Libero uso personale, di ricerca e commerciale conformemente ai termini Apache-2.0 |
| **Qwen3-TTS-12Hz-0.6B-Base** | Alibaba Cloud | Apache-2.0 | Libero uso personale, di ricerca e commerciale conformemente ai termini Apache-2.0 |
| **Qwen3-TTS-12Hz-1.7B-CustomVoice** | Alibaba Cloud | Apache-2.0 | Libero uso conformemente ai termini Apache-2.0 |
| **Qwen3-TTS-Tokenizer-12Hz** | Alibaba Cloud | Apache-2.0 | Speech Tokenizer a 12 Hz distribuito sotto Apache-2.0 |

---

## 3. Tassonomia dei Diritti sui Dati Audio e Identità Vocale

In conformità al Regolamento UE 2024/1689 (AI Act) e all'ordinamento italiano (art. 10 Codice Civile sui diritti della personalità), l'uso di dati audio per la sintesi e clonazione vocale adotta una tassonomia formale a 5 livelli:

```
[Tassonomia Diritti Audio & Voce]
   ├── Livello 1: Diritto d'Autore sul Testo (Opera scritta originaria: PD o licenza compatibile)
   ├── Livello 2: Diritti sul Fonogramma (Master della registrazione: titolarità del file WAV)
   ├── Livello 3: Licenza del Dataset (Condizioni contrattuali: CC0, CC-BY, restrizioni d'uso)
   ├── Livello 4: Diritti Biometrici e della Personalità (Consenso esplicito dello speaker alla clonazione)
   └── Livello 5: Diritti sul Modello e Output (Titolarità dei pesi neurali e dell'audio generato)
```

### 3.1 Classificazione dei Campioni Vocali di Riferimento M0
- **File di Riferimento in Repository**: `bench/reference_voices/ref_voice_it_01.wav` e `bench/reference_voices/ref_voice_it_02.wav`.
- **Origine dei Dati**: Estratti dal subset italiano di *Mozilla Common Voice* (rilascio delta CC0/CC-BY).
- **Stato Giuridico e Consenso Biometrico**: La licenza open del file audio copre la fruizione della registrazione, ma **non attesta il consenso dello speaker originario alla creazione di cloni vocali sintetici per finalità commerciali o di impersonificazione**.
- **Vincolo Operativo Vincolante**: I campioni di riferimento attuali sono formalmente classificati come **«Riferimenti Tecnici di Benchmark per Uso Interno di Ricerca e Verifica Non Commerciale»**.
- **Requisito per Voci di Produzione (M3–M4)**: Nessuna voce o profilo LoRA sarà promosso a livello commerciale senza una liberatoria firmata (*Voice Talent Consent & Release Form*) che autorizzi specificamente la clonazione biometria dell'identità vocale.

---

## 4. Gestione della Sicurezza e Formato dei Prompt Vocali

1. **Divieto di Deserializzazione Arbitraria**:
   Le chiamate legacy a `pickle.load()` identificate in `audio/generator.py` (L548) e `qwen_tts_ui.py` (L2258) sono confinate e disabilitate per default nel nuovo runtime di inferenza (`allow_legacy_pickle=False`).
2. **Formato Ufficiale Obbligatorio**:
   Tutti i nuovi profili vocali salvati adotteranno il formato `safetensors` per i tensori di condizionamento vocale e `metadata.json` per i metadati testuali, eliminando completamente l'esecuzione di codice arbitrario all'atto del caricamento.
