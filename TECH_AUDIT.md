# Audit Tecnico del Codice e dell'Architettura (`TECH_AUDIT.md`)
> **Milestone M0 — Italian TTS Studio**  
> **Repository di riferimento:** [KIS-M3T/qwen3-TTS-studio](https://github.com/KIS-M3T/qwen3-TTS-studio)  
> **Upstream:** [bc-dunia/qwen3-TTS-studio](https://github.com/bc-dunia/qwen3-TTS-studio)  
> **Commit / Ref di partenza:** `89a5dc91d2fba1593ca66de43c6986ff1a0c5999` (Tag: `baseline-v0.1.7-kism3t`)  
> **Data:** Novembre 2026

---

## 1. Mappa del Repository e Responsabilità dei Moduli

Il repository è strutturato attorno a un'applicazione basata su Gradio per la famiglia di modelli Qwen3-TTS (0.6B e 1.7B):

```
qwen3-TTS-studio/
├── qwen_tts_ui.py              # Monolite UI Gradio (7.086 righe)
├── config.py                   # Gestione API keys e parametri ambiente
├── Dockerfile & .dockerignore  # Build container e test smoke
├── requirements.txt            # Dipendenze con vincoli (es. gradio>=4.44,<5.0)
│
├── audio/                      # Sottosistema Audio & TTS
│   ├── generator.py            # Inferenza TTS, chunking, crossfade, timeout
│   ├── model_loader.py         # Caricamento dinamico, patch token, gestione VRAM
│   ├── embedding_utils.py      # Estrazione embedding vocali e proxy SNR euristico
│   ├── combiner.py             # Concatenazione clip podcast tramite MoviePy
│   └── batch.py                # Pipeline di batch processing
│
├── podcast/                    # Orchestrazione Podcast & Dialoghi Multi-Speaker
│   ├── orchestrator.py         # Pipeline di generazione scalettata
│   ├── models.py               # Modelli Pydantic (Dialogue, Transcript, Speaker)
│   ├── outline.py / transcript.py # Generazione testo tramite LLM esterno
│   ├── script_parser.py        # Parsing di copioni testuali custom
│   ├── prompts.py              # Prompt di sistema per LLM
│   └── llm_client.py           # Client unificato (OpenAI, Ollama, Claude, OpenRouter)
│
├── storage/                    # Persistenza JSON / File
│   ├── history.py              # Storico generazioni e riproduzioni
│   ├── persona.py              # Profili e personalità speaker
│   ├── persona_models.py       # Pydantic models per speaker e ruoli
│   └── voice.py                # Salvataggio voci clonate (ref_spk_embedding e ref_code)
│
└── ui/                         # Componenti UI ausiliari
    ├── content_input.py        # Sezione input argomento
    ├── draft_editor.py         # Editor scaletta
    ├── draft_preview.py        # Preview HTML del dialogo
    ├── persona.py              # Gestione delle personas
    ├── progress.py             # Indicatori di progresso e CSS
    └── voice_cards.py          # Card selezione voci e ruoli
```

---

## 2. Analisi Dettagliata delle Funzioni Critiche

### 2.1 Chunking del Testo: `audio/generator.py:_split_text_into_chunks()` (righe 162–228)
* **Codice attuale**:
  ```python
  CHUNK_TARGET = 120
  CHUNK_MAX = 150
  CHUNK_MIN = 50

  def _split_text_into_chunks(text: str) -> list[str]:
      sentences = re.split(r"(?<=[.!?。！？])\s*", text)
      ...
      for sentence in sentences:
          if len(sentence) > CHUNK_MAX:
              words = sentence.split()
              ...
  ```
* **Comportamento verificato**:
  1. Esegue uno split preliminare basato su punteggiatura forte (`.!?。！？`). Non è quindi uno split puramente casuale.
  2. **Criticità per la lingua italiana**:
     - Ignora le abbreviazioni italiane con punto (`dott.`, `prof.`, `ing.`, `art.`, `pag.`, `sig.`, `ecc.`): una frase come *"Il dott. Rossi visita alle 10"* viene spezzata a metà con salto prosodico.
     - Ignora i punti usati come separatori delle migliaia nei numeri italiani (*1.500.000 €* viene spezzato in *"1"*, *"500"*, *"000 €"*).
     - Quando una frase supera `CHUNK_MAX=150`, la divide sulle parole (`words = sentence.split()`) accumulando fino a 120 caratteri, tagliando in modo cieco proposizioni subordinate o complementi, distruggendo la curva intonativa.

### 2.2 Crossfade Audio: `audio/generator.py:_crossfade_audio()` (righe 286–302)
* **Codice attuale**:
  ```python
  def _crossfade_audio(audio1: np.ndarray, audio2: np.ndarray, sr: int, fade_ms: int = 30) -> np.ndarray:
      fade_samples = int(sr * fade_ms / 1000)
      fade_out = np.linspace(1.0, 0.0, fade_samples)
      fade_in = np.linspace(0.0, 1.0, fade_samples)
      audio1_end = audio1[-fade_samples:] * fade_out
      audio2_start = audio2[:fade_samples] * fade_in
      crossfaded = audio1_end + audio2_start
      return np.concatenate([audio1[:-fade_samples], crossfaded, audio2[fade_samples:]])
  ```
* **Comportamento verificato**:
  - Applica una transizione lineare fissa di 30 ms (`0.03s`), senza verificare la presenza di silenzio naturale o confini fonetici.
  - Se il chunk precedente termina con una vocale aperta e il successivo inizia con un attacco duro, il crossfade di 30 ms sovrappone le due forme d'onda, provocando interferenze di fase o artefatti metallici.
  - **Miglioramento necessario**: Sostituire con concatenazione pulita con micro-fade ai transienti di silenzio (<5 ms) o inserimento di pause calibrate basate sulla sintassi.

### 2.3 Calcolo Dinamico dei Token: `audio/generator.py:_calculate_dynamic_max_tokens()` (righe 305–322)
* **Codice attuale**:
  ```python
  def _calculate_dynamic_max_tokens(text: str, preset_max: int) -> int:
      MIN_TOKENS = 256
      MAX_TOKENS = 768
      char_count = len(text)
      estimated = ceil(char_count * 2.5)
      dynamic_max = ceil(estimated * 1.3)
      max_new = max(MIN_TOKENS, min(dynamic_max, MAX_TOKENS))
      return max_new
  ```
* **Comportamento verificato**:
  - **Bug/Parametro morto accertato**: Il parametro `preset_max` è presente nella firma della funzione ma **completamente ignorato** nel corpo.
  - Per frasi brevissime (es. 2-5 caratteri, *"Sì"*, *"No"*), impone un `MIN_TOKENS=256`. Se combinato con la patch globale di generazione, impedisce al modello di chiudere rapidamente la sequenza.
  - Per frasi lunghe fissa un tetto rigido a 768 token, che rischia di troncare testi complessi prima che raggiungano la fine naturale.

### 2.4 Patch Anti-Truncation: `audio/model_loader.py:_patch_generate_min_tokens()` (righe 104–130)
* **Codice attuale**:
  ```python
  MIN_NEW_TOKENS_DEFAULT = int(os.environ.get("QWEN_TTS_MIN_NEW_TOKENS", "60"))

  def _patch_generate_min_tokens(model, min_new_tokens: int = MIN_NEW_TOKENS_DEFAULT):
      original_generate = model.model.generate
      @functools.wraps(original_generate)
      def patched_generate(*args, **kwargs):
          if "min_new_tokens" not in kwargs or kwargs.get("min_new_tokens", 0) < min_new_tokens:
              kwargs["min_new_tokens"] = min_new_tokens
          return original_generate(*args, **kwargs)
      model.model.generate = patched_generate
      return model
  ```
* **Comportamento verificato**:
  - La patch sovrascrive globalmente il metodo `generate` imponendo `min_new_tokens >= 60`.
  - Questa euristica riduce i troncamenti dovuti a premature emissioni del token `<eos>` da parte di Qwen3-TTS, ma per frasi corte (meno di 10-15 sillabe) forza il decoder a continuare la generazione per almeno 60 token audio (~5 secondi di parlato), inducendo allucinazioni fonetiche, suoni strascicati o ripetizioni.

### 2.5 Gestione VRAM e Modelli: `audio/model_loader.py` (righe 26, 152–185)
* **Codice attuale**:
  ```python
  MAX_LOADED_MODELS = 1
  loaded_models: OrderedDict = OrderedDict()
  ```
* **Comportamento verificato**:
  - Poiché `MAX_LOADED_MODELS=1`, il loader dealloca ed elimina forzatamente dalla memoria GPU qualsiasi modello precedentemente caricato appena si richiede una variante differente.
  - I tre checkpoint (1.7B-Base per clonazione, 1.7B-CustomVoice per preset, 1.7B-VoiceDesign per prompt testuale) sono file di pesi completamente distinti (~4.2 GB ciascuno).
  - Il passaggio tra tab costringe a scaricare e ricaricare da disco 4.2 GB di safetensors ogni volta, introducendo una latenza di 10–25 secondi.

### 2.6 Euristica SNR per la Clonazione Vocale: `audio/embedding_utils.py:estimate_snr()` (righe 33–76)
* **Codice attuale**:
  ```python
  rms_total = np.sqrt(np.mean(audio**2))
  frame_rms = [np.sqrt(np.mean(frame**2)) for frame in frames]
  noise_floor = np.percentile(frame_rms, 10)
  snr = 20 * np.log10(rms_total / noise_floor)
  ```
* **Comportamento verificato**:
  - Non calcola un vero SNR acustico (come WADA-SNR o NIST STNR basati su separazione spettrale).
  - È una semplice stima energetica basata sul 10° percentile dei frame RMS.
  - In registrazioni parlate continue senza pause di silenzio, il 10° percentile corrisponde a fonemi deboli (es. fricative o nasali) e non al rumore di fondo della stanza, sottostimando l'SNR. Non può essere usato come filtro insindacabile per i dataset.

### 2.7 Assemblaggio Audio: `audio/combiner.py`
* **Codice attuale**:
  - Utilizza `moviepy` (`AudioFileClip`, `concatenate_audioclips`) per leggere WAV ed esportare forzatamente in MP3 compresso (`libmp3lame`, 192 kbps).
  - Non esporta WAV o FLAC lossless. Se un file audio è corrotto o rimane aperto, può provocare memory leak o lock di file su Windows.

### 2.8 Monolite UI: `qwen_tts_ui.py`
* Misura esattamente **7.086 righe** in un singolo file.
* Contiene logica di presentazione, parsing, polling thread e costrutti `gr.State` intrecciati.
* **Direttiva M0**: È tassativo non ampliare ulteriormente questo monolite, isolando ogni nuovo modulo (linguistica, benchmarking, DSP) in package indipendenti e testabili unitariamente.

---

## 3. Ricognizione degli Strumenti di Fine-Tuning Esistenti

Abbiamo esaminato lo stato dei progetti esistenti per il fine-tuning di Qwen3-TTS:

### 3.1 `QwenLM/Qwen3-TTS/finetuning` (Script Ufficiale Alibaba)
* File principale: `sft_12hz.py`.
* Caratteristiche:
  - Implementa un Supervised Fine-Tuning (SFT) full-parameter per singolo speaker.
  - Ottimizzatore: standard `AdamW` in float32.
  - Non integra supporto nativo per PEFT / LoRA / QLoRA.
* **Impatto sui 12 GB di VRAM**:
  - Per il modello **1.7B**, i pesi in bf16 occupano ~3.4 GB, i gradienti ~3.4 GB, e gli stati dell'ottimizzatore AdamW (2 per parametro in fp32) ~13.6 GB. Il solo footprint statico supera i **20 GB di VRAM**.
  - **Conclusione**: Il fine-tuning full-parameter del modello 1.7B è **impossibile su 12 GB di VRAM** senza ZeRO-Offload.
  - Per il modello **0.6B**, il footprint statico è di circa 7.2 GB, rendendo teoricamente fattibile un esperimento single-speaker su 12 GB con batch=1 e gradient accumulation.

### 3.2 `bc-dunia/qwen3-TTS-finetune-studio`
* Repository gemella dell'autore di `qwen3-TTS-studio`.
* Fornisce una WebUI Gradio e una CLI che avvolgono la procedura ufficiale Alibaba (`prepare_data.py` e `sft_12hz.py`), con aggiunta di un'arena di comparazione dei checkpoint.
* **Riutilizzabilità**:
  - Il codice di pre-processing (estrazione dei codici audio con `prepare_data.py`) è solido e direttamente riutilizzabile.
  - La logica di training è tuttavia legata a `sft_12hz.py` (full SFT). Non risolve nativamente il vincolo dei 12 GB per il modello 1.7B.

### 3.3 Adattamento LoRA / PEFT
* Per addestrare il modello 1.7B entro 12 GB di VRAM, è necessario iniettare adattatori LoRA (rank 16/32) sui moduli lineari dell'attenzione (es. `q_proj`, `v_proj`) congelando il resto della rete.
* Questa componente **non è presente nei repository upstream** e dovrà essere implementata e misurata sperimentalmente solo a partire dalla Milestone M5, qualora giustificata dai benchmark.

---

## 4. Sintesi e Decisioni Tecniche per le Fasi Successive

1. **Baseline Preservation**: Il commit `89a5dc91d2fba1593ca66de43c6986ff1a0c5999` rimane il riferimento intatto; qualsiasi modifica al codice originale sarà introdotta tramite interfacce isolate.
2. **Nessun cambio globale non testato**: Le patch su `min_new_tokens` e `_calculate_dynamic_max_tokens` verranno rese configurabili e testate contro frasi brevi e lunghe.
3. **Pipeline Lossless**: Sostituire l'uso di MoviePy con un combiner basato su `soundfile`/`numpy` che preservi il formato WAV nativo a 24 kHz.
