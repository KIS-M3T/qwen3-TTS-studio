# Audit di Riuso, Licenze e Proprietà Intellettuale (`LICENSE_AUDIT.md`)
> **Milestone M0 — Italian TTS Studio**  
> **Oggetto:** Analisi dello stato giuridico e delle licenze d'uso per codice sorgente, pesi neurali, dataset e dipendenze.  
> **Data:** Novembre 2026

---

## 1. Quadro Generale delle Licenze

Nel passaggio da un prototipo di ricerca a un'applicazione fruibile per la lingua italiana, è fondamentale distinguere chiaramente tra:
1. **Diritto di esecuzione locale (uso personale / ricerca e sviluppo)**.
2. **Diritto di ridistribuzione o commercializzazione del software e dei modelli derivati**.

Nessuna componente verrà dichiarata «commercialmente distribuibile» senza accertamento documentale.

---

## 2. Audit del Codice Sorgente Applicativo

### 2.1 Repository `bc-dunia/qwen3-TTS-studio` e Fork `KIS-M3T`
* **Stato Licenza nel Repository**: **ASSENTE**. Non è presente alcun file `LICENSE` o `LICENSE.txt` nella root di `bc-dunia/qwen3-TTS-studio` o del fork `KIS-M3T`.
* **Dicitura nel README**:
  > *"This project uses Qwen3-TTS models. Please refer to the Qwen3-TTS License for model usage terms."*  
  Questa frase fa riferimento unicamente ai pesi dei modelli Qwen di Alibaba, non al codice Python sviluppato dall'autore della WebUI.
* **Valutazione Giuridica**:
  - In base alla Convenzione di Berna e ai Terms of Service di GitHub, in assenza di una licenza open-source esplicita (es. MIT, Apache-2.0, GPL), il codice rimane sotto **copyright riservato dell'autore ("All Rights Reserved")**.
  - La pubblicazione su GitHub concede solo il diritto implicito agli utenti di visualizzare il codice e crearne fork sulla piattaforma.
  - **Vincolo Operativo**: Per distribuire un prodotto derivato (specie se commerciale), è indispensabile ottenere una chiarificazione o l'aggiunta di un file di licenza dall'autore originario (`bc-dunia`), oppure procedere con una reimplementazione "clean-room" dell'orchestrazione software.

---

## 3. Audit dei Modelli e dei Pesi Neurali

| Modello | Licenza Ufficiale | Idoneità Commerciale | Note / Stato di Manutenzione |
| :--- | :--- | :--- | :--- |
| **Qwen3-TTS (1.7B e 0.6B Base / CustomVoice / Tokenizer)** | **Apache-2.0** (Alibaba Cloud) | **Sì** (conclusa conformità ai termini Apache-2.0) | Codice e pesi completamente aperti; licenza permissiva standard. |
| **ResembleAI / Chatterbox Multilingual V3** | **MIT License** | **Sì** | **Attenzione:** Progetto dichiarato *"retired research"* ad agosto 2026. Non riceve più aggiornamenti o patch di sicurezza. Utilizzabile unicamente come concorrente tecnico di benchmark isolato. |
| **Coqui XTTS-v2** | **CPML** (Coqui Public Model License) | **NO** (Esclusivamente non commerciale) | La licenza CPML vieta l'uso commerciale diretto salvo licenza a pagamento. Con la chiusura di Coqui AI, l'acquisizione di diritti commerciali è incerta. **Escluso da qualsiasi prodotto distribuibile.** |
| **F5-TTS (alien79 Italian checkpoint)** | Mista / Non chiara | **Non accertata** | Discrepanze tra le informazioni presenti su Hugging Face e la documentazione F5-TTS upstream. Richiede verifica prima di qualsiasi integrazione. |

---

## 4. Audit dei Dataset Vocali di Lingua Italiana

L'addestramento e il fine-tuning di modelli vocali richiedono una conformità rigorosa sia sui testi che sui diritti delle registrazioni audio degli speaker:

| Dataset | Volume Stimato | Licenza Dichiarata | Rischi di Riuso e Consenso | Idoneità per Studio TTS |
| :--- | :--- | :--- | :--- | :--- |
| **Mozilla Common Voice (Italiano)** | Delta corpus presente in workspace | Audio: **CC0** (Pubblico Dominio) / Testo: misto | Microfoni eterogenei, registrazioni non professionali, assenza di liberatorie commerciali per singole voci. | Utile per benchmarking di robustezza o ASR, ma sconsigliato come base per voci da studio. |
| **Multilingual LibriSpeech (MLS) Italiano** | ~247,38 ore | **CC BY 4.0** | Audio proveniente da LibriVox (pubblico dominio); testi di pubblico dominio. Richiede attribuzione. | **Idoneo** con selezione accurata dei top speaker per uniformità acustica. |
| **M-AILABS Speech Dataset (Italiano)** | ~127 ore 40 min (non 35h) | Pubblico Dominio / CC | Registrazioni basate su lettori LibriVox (es. *riccardo_fasol*, *linda_falcone*) e testi di pubblico dominio. | **Idoneo** per la creazione di baseline single-speaker maschile e femminile. |
| **Mozilla Data Collective: Italian TTS (Female Voice)** | ~10 ore | **Apache-2.0** | Registrazioni coerenti di voce femminile italiana con licenza permissiva esplicita. | **Idoneo** per benchmarking e addestramento vocale femminile. |
| **Mozilla Data Collective: Italian Penal Code (Female)** | ~793 minuti | **Apache-2.0** | Lettura di testi giuridici in pubblico dominio. Da verificare la naturalezza e la punteggiatura. | **Idoneo sotto riserva**, da verificare ascolto. |
| **CSS10 Italiano** | N/A | N/A | **INESISTENTE**: Il corpus CSS10 ufficiale include solo 10 lingue (de, el, es, fi, fr, hu, ja, nl, ru, zh). Non contiene l'italiano. | **Rimosso formalmente dal piano.** |

---

## 5. Audit delle Dipendenze Software

Tutte le dipendenze dichiarate in `requirements.txt` sono state verificate:
- `torch`: BSD-3-Clause
- `qwen-tts`: Apache-2.0
- `gradio>=4.44.0,<5.0`: Apache-2.0
- `pydantic`: MIT
- `soundfile`: BSD-3-Clause (wrapping di `libsndfile` rilasciato sotto LGPL v2.1+)
- `numpy`: BSD-3-Clause
- `moviepy`: MIT
- `openai`: Apache-2.0
- `anthropic`: MIT
- `packaging`: Apache-2.0 / BSD-2-Clause

Tutte le librerie presentano licenze compatibili con l'uso commerciale locale e la redistribuzione, con l'avvertenza di link dinamico per `libsndfile` (LGPL).

---

## 6. Conclusioni e Raccomandazioni Giuridico-Operative

1. **Sviluppo M0-M4**: È pienamente legittimo lavorare localmente sul fork per attività di ricerca, benchmark e miglioramento architetturale.
2. **Pesi di Produzione**: L'adozione del modello base **Qwen3-TTS** (Apache-2.0) garantisce la massima libertà d'uso sia personale che commerciale.
3. **Pesi di Riserva**: Chatterbox Multilingual V3 (MIT) è utilizzabile esclusivamente come termine di paragone per il benchmark tecnico.
4. **Voci Personalizzate**: Per qualsiasi voce destinata alla distribuzione professionale esterna, la provenienza dovrà essere tracciata tramite un **Voice Datasheet** che certifichi l'origine dei dati (M-AILABS, MLS o registrazioni proprietarie con consenso esplicito dello speaker).
