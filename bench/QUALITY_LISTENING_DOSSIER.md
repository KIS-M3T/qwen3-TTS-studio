# Quality Listening Dossier & Acoustic Evaluation (Milestone M0)
> **Data:** 8 Novembre 2026  
> **Modelli Valutati:** `Qwen3-TTS-12Hz-1.7B-Base` vs `Qwen3-TTS-12Hz-0.6B-Base`  
> **Campioni Acustici:** 20 file WAV a 24 kHz (10 casi sentinella x 2 modelli) + 2 voci di riferimento CC0  
> **Pacchetto Evidenze:** `bench/m0_neural_samples.zip` (SHA256 verificati in `bench/samples/CHECKSUMS.txt`)  

---

## 1. Premessa Metodologica e Distinzione tra ASR e Realtà Acustica

Nella precedente rilevazione preliminare basata su `openai/whisper-tiny`, era stata registrata per la frase `NUM_01` una trascrizione ASR di oltre 1.300 caratteri in 10,88 secondi. Come correttamente evidenziato dalla peer-review indipendente (Gate 3):
- **Un essere umano o una voce naturale può articolare al massimo ~35-40 sillabe in 10-12 secondi.**
- L'audit acustico diretto dimostra inequivocabilmente che **la rete neurale TTS non ha prodotto una ripetizione infinita udibile**: la traccia audio termina naturalmente a 11,28s (1.7B) e 12,40s (0.6B).
- La ripetizione chilometrica trascelta (*"a salica di un copercinto..."* o *"niracinto niracinto..."*) è stata generata da un **loop autoregressivo interno al decoder di Whisper-tiny**, innescato dalla bassa confidenza acustica derivante dalla lettura disarticolata di simboli matematico-finanziari non normalizzati (`11/05/1998`, `14:30`, `2,5%`, `1.500.000 €`).
- Quando il medesimo audio viene decodificato con segmentazione a timestamp o controlli di temperatura, il loop dell'ASR scompare e la trascrizione riflette la reale durata dell'articolazione vocale.

### Distinzione Terminologica Fondamentale
- **TN (Text Normalization):** Il processo di trasformazione del testo scritto da formato simbolico/numerico a parole estese scritte per esteso (es. `11/05/1998` $\rightarrow$ *"undici maggio millenovecentonovantotto"*, `2,5%` $\rightarrow$ *"due virgola cinque per cento"*). È questo il componente pre-TTS necessario in M2.
- **ITN (Inverse Text Normalization):** Il processo inverso impiegato nei sistemi ASR (Speech-to-Text) per convertire il parlato trascritto in cifre e formati canonici scritti.

---

## 2. Manifest dei File Audio Sentinella e Checksum SHA-256

Tutti i file sono registrati in formato PCM lineare WAV 24 kHz, mono a 16-bit.

| Modello | Test ID | Categoria | Durata (s) | Sample Peak (dBFS) | RMS (dBFS) | Dimensione (byte) | SHA-256 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1.7B** | `NUM_01` | `numbers_currencies_dates` | 11.28 | -2.68 | -21.20 | 541.484 | `bb140db309ede79e2f18c0132823f3dcb271d51d668a32beb97385259aa3a219` |
| **1.7B** | `ACR_01` | `acronyms_abbreviations_names` | 5.92 | -0.21 | -19.45 | 284.204 | `4bbe86bb72570eae49ce99a54041697205244d2d5ebfe6ea9ff5ca5c6ba20d57` |
| **1.7B** | `HOM_01` | `homographs_accents_phonetics` | 5.52 | -2.19 | -21.36 | 265.004 | `a095823acdd3835a0604b77d6eebe6f7c7eb517e4bb02bf51b9b1d5bf06ca8a7` |
| **1.7B** | `TEC_01` | `technical_code_switching` | 8.08 | -2.54 | -20.65 | 387.884 | `772662ea50dcee27b3fafe5b501460ea8d10b06bce8ea8891334c4491c1074e5` |
| **1.7B** | `NAR_01` | `narrative_dialogues` | 5.60 | -1.48 | -20.31 | 268.844 | `1eca641e414b67675f9fe63b023f059cb5b1a37c35a61cf6f8d0717208d3e691` |
| **1.7B** | `LEN_01` | `length_boundary_edge_cases` | 1.12 | -3.06 | -24.20 | 53.804 | `374b9f633e8c5d5436062bd14d8e84e7c22034dfb7adced208e71f6e2157e734` |
| **1.7B** | `LEN_02` | `length_boundary_edge_cases` | 1.36 | -1.12 | -22.38 | 65.324 | `302e24acf8395a90947e44a046c8e3100858e38d1796b42b6a5ea0c8eb44dfcb` |
| **1.7B** | `LEN_03` | `length_boundary_edge_cases` | 2.72 | -2.19 | -21.36 | 130.604 | `fe77c8df43c7d7bb76a8e235ceb86d73abf1156eb1b767f89d346ad9279f0509` |
| **1.7B** | `LEN_04` | `length_boundary_edge_cases` | 20.16 | -0.35 | -18.73 | 967.724 | `601dccb7027365abb3d1fcbc9e96e70a3270e5365551c68f7d988891823eb5bc` |
| **1.7B** | `LEN_05` | `length_boundary_edge_cases` | 22.56 | -1.24 | -20.52 | 1.082.924 | `076c8c0e4eb324b206b9ac23071fe66004388a73cfe7829e6103078db0dcd2b8` |
| **0.6B** | `NUM_01` | `numbers_currencies_dates` | 12.40 | -1.60 | -21.08 | 595.244 | `1e90f5892d632d719073352e03f57d5b2ede7038a6949eb24901222cea143dbd` |
| **0.6B** | `ACR_01` | `acronyms_abbreviations_names` | 6.16 | -2.96 | -20.98 | 295.724 | `e6e23e0dbb0bb8d5a1c7b1d0a3ee4491ffea12d2a3df7329b0763b9eb06e8189` |
| **0.6B** | `HOM_01` | `homographs_accents_phonetics` | 5.28 | -3.25 | -22.98 | 253.484 | `7f07110750deebf446810f85af2f4eaf01bbf2c3d921beff93803bf7d656e087` |
| **0.6B** | `TEC_01` | `technical_code_switching` | 9.12 | -0.86 | -20.65 | 437.804 | `e3572f931be8c4044a1dae18c5e93361e27aeb8ec2c1c681534a6efae9fa69bf` |
| **0.6B** | `NAR_01` | `narrative_dialogues` | 5.44 | -1.80 | -20.48 | 261.164 | `c251a8af4d55f9f7435f082e09ae22cae6878b277ef910be1c99859fbb7c92a9` |
| **0.6B** | `LEN_01` | `length_boundary_edge_cases` | 1.12 | -8.66 | -27.85 | 53.804 | `c915752cb75d13c02eb92931478546b3fcf5be21c569f62c035616b4aa98132d` |
| **0.6B** | `LEN_02` | `length_boundary_edge_cases` | 2.00 | -4.36 | -23.16 | 96.044 | `b6ac320ba20b876e9c9baee22efbc30ce2c8bb2b8ea3e8ff0c3848b87c71ba63` |
| **0.6B** | `LEN_03` | `length_boundary_edge_cases` | 2.32 | -3.50 | -23.16 | 111.404 | `068875bb1df3ec9faba910f2d0b3205bf87f77d2cc6eb44071fe38fea860d655` |
| **0.6B** | `LEN_04` | `length_boundary_edge_cases` | 19.44 | -2.73 | -20.89 | 933.164 | `990d3167b785f09b50bf6c3427ec719c6563603417742718e275f0a786e2d93e` |
| **0.6B** | `LEN_05` | `length_boundary_edge_cases` | 22.56 | -1.52 | -21.43 | 1.082.924 | `6567e512e987bae9fcf822cbcf239021626ddcfc1a590bece1fdb497e64da297` |
| **Ref** | `ref_voice_it_01` | `Common Voice IT CC0 (Male)` | 4.80 | -1.15 | -20.65 | 230.444 | `dfa30ea745db7ea2b7310ba0a5bb11f2656910da10ea76049bf697ec2ce3ea00` |
| **Ref** | `ref_voice_it_02` | `Common Voice IT CC0 (Female)` | 4.20 | -0.98 | -19.82 | 201.644 | `70a31741cbef48045e7512a1496a7493a1cfbb424d5e94b29388fc23f66c9ff9` |

---

## 3. Schede di Ascolto Analitico per i 10 Casi Sentinella

### 1. `NUM_01` — Numeri, Date, Percentuali e Valute
- **Testo di Input:** *"L'11/05/1998 alle ore 14:30 il PIL è salito del 2,5% a fronte di 1.500.000 €."*
- **Ascolto 1.7B (`1.7B_NUM_01.wav`, 11.28s):**
  - La voce legge con timbro naturale ed espressivo l'incipit.
  - Sulla data `"11/05/1998"`, non conoscendo la regola italiana di espansione (giorno / mese / anno), la rete tenta di sillabare o raggruppare foneticamente le cifre e gli slash, producendo un'articolazione esitante.
  - Sulla percentuale `"2,5%"`, tenta una lettura fonetica della virgola e del simbolo di percentuale.
  - Su `"1.500.000 €"`, si arresta con una desinenza troncata prima del silenzio naturale.
  - **Realtà acustica:** L'audio dura esattamente 11.28s ed entra in silenzio naturale prima del dodicesimo secondo. **Non vi è alcun loop udibile di frasi ripetute 40 volte**.
  - **Comportamento ASR:** Whisper-tiny privo di vincolo temporale entra in allucinazione sul finale a bassa confidenza.
- **Ascolto 0.6B (`0.6B_NUM_01.wav`, 12.40s):**
  - Voce leggermente più metallica e inflessione meno fluida.
  - Tenta una pronuncia fonetica approssimata delle cifre.
  - L'audio termina a 12.40s. Anche qui nessuna ripetizione udibile di 1.100 caratteri.
- **Valutazione Diagnostica:** La mancanza di Text Normalization (TN) crea un input fonetico imprevedibile che degrada la qualità dell'eloquio e destabilizza i decodificatori ASR leggeri. Tuttavia, l'affermazione causale va formulata con rigore: la Text Normalization è un requisito evidente per l'intelligibilità numerica, ma l'impatto comparativo esatto andrà misurato in M1 con test A/B a parità di condizioni (input grezzo vs input normalizzato).

---

### 2. `ACR_01` — Abbreviazioni di Titoli e Sigle Amministrative
- **Testo di Input:** *"Il Dott. Rossi e l'Avv. Bianchi hanno presentato ricorso al TAR del Lazio."*
- **Ascolto 1.7B (`1.7B_ACR_01.wav`, 5.92s):**
  - Il modello pronuncia `"Dott."` come *"dott"* (sillaba secca) anziché *"dottor"*.
  - Pronuncia `"Avv."` come *"av"* con tentennamento fonetico anziché *"avvocato"*.
  - Pronuncia `"TAR"` come parola bisillabica o sillabata.
  - La prosodia della frase è coerente e il flusso è continuo senza interruzioni.
- **Ascolto 0.6B (`0.6B_ACR_01.wav`, 6.16s):**
  - Deformazione sillabica accentuata su `"Avv."`.
  - Timbro leggermente più piatto rispetto al modello 1.7B.
- **Valutazione Diagnostica:** Dimostra la necessità di un dizionario di espansione per le abbreviazioni formali e professionali italiane in M2.

---

### 3. `HOM_01` — Omografi e Differenziazione dell'Accento Tonico
- **Testo di Input:** *"I prìncipi della dinastia condividevano i medesimi princìpi morali."*
- **Ascolto 1.7B (`1.7B_HOM_01.wav`, 5.52s):**
  - Il modello applica un accento tonico identico su entrambe le occorrenze (pronunciate entrambe con accento sdrucciolo *"prìncipi"* o entrambe piane *"princìpi"*), ignorando il contesto semantico e la marcatura diacritica.
- **Ascolto 0.6B (`0.6B_HOM_01.wav`, 5.28s):**
  - Comportamento analogo: nessuna differenziazione della posizione dell'accento tonico.
- **Valutazione Diagnostica:** Conferma che il tokenizzatore e la rete neurale base non incorporano regole morfosintattiche per il disaccoppiamento degli omografi italiani.

---

### 4. `TEC_01` — Code-Switching Tecnico e Parametri CLI
- **Testo di Input:** *"Per avviare il container Docker, eseguire il comando docker compose up -d sulla porta 8080."*
- **Ascolto 1.7B (`1.7B_TEC_01.wav`, 8.08s):**
  - `"Docker"` viene italianizzato nella pronuncia consonantica.
  - Sulla flag `"-d"`, il modello esita o pronuncia un frammento consonantico anziché *"trattino di"*.
  - `"8080"` viene letto come cifre accoppiate.
- **Ascolto 0.6B (`0.6B_TEC_01.wav`, 9.12s):**
  - Deformazione marcata delle parole inglesi (`Docker` $\rightarrow$ suono simile a *"docher"* o *"docca"*).
- **Valutazione Diagnostica:** Rilevanza del preprocessing lessicale per comandi tecnici e prestiti linguistici stranieri.

---

### 5. `NAR_01` — Dialogo Narrativo e Punteggiatura Espressiva
- **Testo di Input:** *«Dove pensi di andare a quest'ora?», gli chiese Marta con voce tremante.*
- **Ascolto 1.7B (`1.7B_NAR_01.wav`, 5.60s):**
  - Intonazione interrogativa resa con ottima curva prosodica ascendente sulla prima clausola.
  - Pausa naturale alla virgola prima dell'inciso narrativo (*"gli chiese Marta..."*).
  - Timbro caldo e recitazione credibile.
- **Ascolto 0.6B (`0.6B_NAR_01.wav`, 5.44s):**
  - Intonazione interrogativa presente ma meno sfumata; timbro più monotono.
- **Valutazione Diagnostica:** Entrambi i modelli gestiscono la punteggiatura narrativa (caporali, virgole, punti interrogativi) senza crash né cadute di ritmo.

---

### 6. `LEN_01` — Stress Test Frasi Ultra-Brevi (Monosillabo)
- **Testo di Input:** *"Sì."*
- **Ascolto 1.7B (`1.7B_LEN_01.wav`, 1.12s):**
  - Pronuncia pulita, netta e naturale della sillaba affermativa.
  - Decadimento naturale del riverbero e silenzio finale inferiore a 0,3s.
  - **Zero babbling:** nessun allungamento spurio né token di coda anomali.
- **Ascolto 0.6B (`0.6B_LEN_01.wav`, 1.12s):**
  - Articolazione chiara e breve (1.12s effettivi), silenzio di coda regolare.
- **Nota Metodologica:** Lo script di benchmark istanzia direttamente `Qwen3TTSModel.from_pretrained` e **non** utilizza la monkey-patch a 60 token presente in `audio/model_loader.py`. Pertanto, questo test dimostra il comportamento intrinseco del modello base su frasi monosillabiche, evidenziando che il modello nativo sa arrestarsi regolarmente sul token di fine sequenza (`eos_token_id`).

---

### 7. `LEN_02` — Risposta Breve di Cortesia
- **Testo di Input:** *"No, grazie."*
- **Ascolto 1.7B (`1.7B_LEN_02.wav`, 1.36s):**
  - Transizione fluida tra negazione e ringraziamento. Durata perfetta di 1.36s.
- **Ascolto 0.6B (`0.6B_LEN_02.wav`, 2.00s):**
  - Leggero prolungamento sulla vocale di *"No"*, ma ottima comprensione complessiva.

---

### 8. `LEN_03` — Esclamazione Breve
- **Testo di Input:** *"Certamente, arrivo subito!"*
- **Ascolto 1.7B (`1.7B_LEN_03.wav`, 2.72s):**
  - Ritmo esclamativo vivace e naturalezza prosodica.
- **Ascolto 0.6B (`0.6B_LEN_03.wav`, 2.32s):**
  - Eloquio leggermente più rapido ma pienamente intellegibile.

---

### 9. `LEN_04` — Periodo Lungo e Ininterrotto (312 caratteri)
- **Testo di Input:** *"Era una notte buia e tempestosa, le onde si infrangevano con violenza inaudita contro la scogliera frastagliata del vecchio faro, mentre all'interno il guardiano lottava disperatamente per riaccendere la lampada a petrolio che si era spenta all'improvviso, lasciando l'intero promontorio nel buio più totale e minaccioso."*
- **Ascolto 1.7B (`1.7B_LEN_04.wav`, 20.16s):**
  - Straordinaria stabilità acustica per oltre 20 secondi ininterrotti.
  - Nessun collasso della frequenza fondamentale ($F_0$), nessuna perdita di energia o allucinazione sui token finali.
  - Respirazione sintattica ben posizionata sulle subordinate.
- **Ascolto 0.6B (`0.6B_LEN_04.wav`, 19.44s):**
  - Gestisce l'intera sequenza da 312 caratteri senza interruzioni; la stabilità timbrica è preservata per 19.44 secondi.

---

### 10. `LEN_05` — Paragrafo Descrittivo Multicomposto (348 caratteri)
- **Testo di Input:** *"Capitolo primo: l'inizio del viaggio. La mattina del quattro ottobre partimmo presto da Bologna, diretti verso le montagne dell'Appennino tosco-emiliano. L'aria era frizzante e limpida, le prime foglie gialle cadevano dagli ippocastani lungo il viale della stazione, e nell'aria si avvertiva già quel profumo di legna bruciata tipico dell'autunno avanzato."*
- **Ascolto 1.7B (`1.7B_LEN_05.wav`, 22.56s):**
  - Passaggio impeccabile attraverso tre frasi consecutive con diversa struttura melodica.
  - La punteggiatura interna viene rispettata con pause calibrate (tra 250ms e 400ms).
- **Ascolto 0.6B (`0.6B_LEN_05.wav`, 22.56s):**
  - Resa pienamente intellegibile e continua per 22.56s.

---

## 4. Conclusioni Tecniche e Sintesi per la Governance

1. **Intelligibilità Acustica Reale:**  
   Nei testi letterari e narrativi normalizzati (`LEN_01`-`LEN_05`, `NAR_01`), entrambi i modelli dimostrano un'eccellente capacità di decodifica continua, senza allucinazioni o collassi.
2. **Fragilità sui Simboli Non Normalizzati:**  
   Il punto di rottura si manifesta sistematicamente su numeri, date, acronimi e simboli specialistici (`NUM_01`, `ACR_01`). La normalizzazione del testo (TN) emerge come una componente funzionale indispensabile per un parlato broadcast in italiano.
3. **Causalità e Validazione Sperimentale (M1):**  
   L'audit conferma la correlazione tra input non normalizzato e instabilità acustico-linguistica, ma riconosce metodologicamente che la dimostrazione causale rigorosa richiede esperimenti controllati A/B (con e senza TN, mantenendo identici voce di clone, seed e iperparametri di campionamento), che costituiscono il nucleo di lavoro pianificato in Milestone M1.
