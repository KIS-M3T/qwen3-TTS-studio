# Quality Audit & Sentinel ASR Report (Milestone M0)
> **Data:** 8 Novembre 2026 (Revisione Gate 3)  
> **Modelli Valutati:** `Qwen3-TTS-12Hz-1.7B-Base` vs `Qwen3-TTS-12Hz-0.6B-Base`  
> **Motore ASR Sentinella:** `openai/whisper-tiny` (inferenza su NVIDIA RTX 4080 CUDA)  
> **Scopo:** Analisi esplorativa dell'accuratezza linguistica e cross-check acustico sui campioni sentinella del benchmark.  
> **Dossier di Ascolto Completo:** Consultare [`bench/QUALITY_LISTENING_DOSSIER.md`](QUALITY_LISTENING_DOSSIER.md) per l'analisi acustica umana esaustiva.

---

## 1. Sommario Esecutivo

Come rilevato nella peer-review indipendente (Gate 3), **l'assenza di anomalie acustiche euristiche (zero clipping di sample peak, zero tracce mute, assenza di code di silenzio anomale) non costituisce prova di fedeltà linguistica**. 

L'audit acustico e linguistico eseguito combinando la sentinella ASR e l'ascolto umano diretto sui file WAV reali a 24 kHz ha consentito di chiarire la dinamica delle anomalie osservate:

1. **Sentinella ASR vs Realtà Acustica (`NUM_01`):**
   Whisper-tiny ha generato una trascrizione di oltre 1.100-1.300 caratteri ripetuti per un audio di ~11-12 secondi. L'ascolto acustico diretto dimostra che **il modello TTS non ha affatto pronunciato una sequenza infinita di parole**: l'eloquio dura effettivamente 11.28s (1.7B) e 12.40s (0.6B). L'anomalia chilometrica è il risultato di un **loop autoregressivo interno al decoder di Whisper-tiny**, innescato dalla bassa confidenza su una sequenza fonetica disarticolata causata dalla lettura di simboli matematici non normalizzati (`11/05/1998`, `14:30`, `2,5%`, `1.500.000 €`).
2. **Ruolo della Text Normalization (TN):**
   L'espansione del testo simbolico in parole scritte per esteso (**TN - Text Normalization**, non ITN che riguarda l'ASR inverso) è un requisito architetturale prioritario. La presenza di numeri, date e valute non espanse crea un input fuori distribuzione per il vocabolario del modello base. L'isolamento causale rigoroso (input grezzo vs input normalizzato a parità di voce, seed e parametri) è pianificato in Milestone M1.
3. **Pronuncia di Sigle e Abbreviazioni (`ACR_01`):**
   Titoli abbreviati come `Dott.` e `Avv.` vengono letti foneticamente come scritti (*"dott"*, *"av"*) anziché espansi nelle forme piene (*"dottor"*, *"avvocato"*).
4. **Disambiguazione degli Omografi (`HOM_01`):**
   In frasi come *"I prìncipi della dinastia condividevano i medesimi princìpi morali"*, il modello applica un'accentazione uniforme non differenziata dal contesto sintattico.
5. **Comportamento su Sequenze Brevi e Lunghe (`LEN_01` - `LEN_05`):**
   Su parole singole monosillabiche (*"Sì."*) e frasi di cortesia (*"No, grazie."*), il modello base si arresta correttamente sul token `eos` senza babbling o code anomale. Su periodi lunghi e complessi da 312 a 348 caratteri (`LEN_04`, `LEN_05`), la stabilità timbrica e la continuità vocale sono mantenute stabilmente per oltre 20-22 secondi.

---

## 2. Tabella di Confronto Sentinella ASR & Valutazione Acustica

| ID Test & Categoria | Testo Originale (Ground Truth) | Durata WAV (1.7B / 0.6B) | Trascrizione Whisper-tiny (1.7B / 0.6B) | Riscontro Acustico Reale & Diagnosi |
| :--- | :--- | :---: | :--- | :--- |
| **`NUM_01`**<br>(Numeri, Date, Valute) | *"L'11/05/1998 alle ore 14:30 il PIL è salito del 2,5% a fronte di 1.500.000 €."* | 11.28s / 12.40s | **1.7B:** *"In questo tempo... a salica di un copercinto [loop ASR]*"<br>**0.6B:** *"E' non di scienze rosti... niracinto niracinto [loop ASR]"* | **Anomalia ASR su parlato non normalizzato:** L'audio dura ~11-12s e termina regolarmente. La lettura disarticolata di date e cifre crea frammenti fonetici a bassa confidenza che mandano in loop il decoder ASR di Whisper-tiny. Necessaria **Text Normalization (TN)**. |
| **`ACR_01`**<br>(Abbreviazioni e Sigle) | *"Il Dott. Rossi e l'Avv. Bianchi hanno presentato ricorso al TAR del Lazio."* | 5.92s / 6.16s | **1.7B:** *"il metto resci e lagno... oltarbe le mezzo"*<br>**0.6B:** *"il decresci e la visi... cos'antar de l'azio"* | **Mancata espansione:** L'audio è continuo e pulito, ma `Dott.` e `Avv.` vengono pronunciati come sillabe secche (*"dott"*, *"av"*). Richiede modulo di espansione abbreviazioni. |
| **`HOM_01`**<br>(Omografi e Accenti) | *"I prìncipi della dinastia condividevano i medesimi princìpi morali."* | 5.52s / 5.28s | **1.7B:** *"E preuncipi... medesso e preuncipi morali"*<br>**0.6B:** *"e principe... mette insieme e principe in modale"* | **Accento non differenziato:** Entrambe le occorrenze ricevono identico accento tonico, senza distinzione tra sostantivo nobiliare e principio etico. |
| **`TEC_01`**<br>(Termini Tecnici) | *"Per avviare il container Docker, eseguire il comando docker compose up -d sulla porta 8080."* | 8.08s / 9.12s | **1.7B:** *"per avviare il contenere doppere..."*<br>**0.6B:** *"per agli contenere bocca..."* | **Italianizzazione fonetica:** Parole inglesi e opzioni CLI pronunciate in modo approssimativo. Richiede dizionario tecnico e regole G2P per prestiti. |
| **`NAR_01`**<br>(Dialogo Narrativo) | *«Dove pensi di andare a quest'ora?», gli chiese Marta con voce tremante.* | 5.60s / 5.44s | **1.7B:** *"Dove pensi che andare a costora..."*<br>**0.6B:** *"dove pensi di andare a costora..."* | **Prosodia accurata:** Curva intonativa interrogativa naturale e pausa sulla virgola narrativa. |
| **`LEN_01`**<br>(Ultra-breve) | *"Sì."* | 1.12s / 1.12s | **1.7B:** *"Sì."*<br>**0.6B:** *"Sì!"* | **Accurato:** Monosillabo pronunciato chiaramente senza allucinazioni o code prolungate. |
| **`LEN_02`**<br>(Breve cortesia) | *"No, grazie."* | 1.36s / 2.00s | **1.7B:** *"Noo, grazie!"*<br>**0.6B:** *"No, grazie."* | **Accurato:** Ottima transizione e naturalezza. |
| **`LEN_03`**<br>(Esclamazione) | *"Certamente, arrivo subito!"* | 2.72s / 2.32s | **1.7B:** *"Certo a mente a riposciuto."*<br>**0.6B:** *"Certamente, arrivo subito."* | **Buono:** Eloquio vivace ed esclamativo. |
| **`LEN_04`**<br>(Periodo 312 car.) | *"Era una notte buia e tempestosa..."* | 20.16s / 19.44s | Transcrizione continua verificata | **Alta stabilità:** Oltre 19-20s di generazione continua senza interruzioni né degrado timbrico. |
| **`LEN_05`**<br>(Paragrafo 348 car.) | *"Capitolo primo: l'inizio del viaggio..."* | 22.56s / 22.56s | Transcrizione multi-frase verificata | **Ottima transizione:** Pausa sintattica corretta tra i tre periodi consecutivi. |

---

## 3. Note Metodologiche su Troncamento e Clipping

1. **Definizione di Troncamento Acustico:**  
   Nei report preliminari, la colonna `is_truncated` identificava la presenza di **code di silenzio finale superiori a 2,0 secondi** (`trailing_silence_sec > 2.0`). Questa euristica rileva l'assenza di silenzi spuri prolungati alla fine dell'audio, ma **non dimostra il mancato troncamento linguistico** delle ultime parole del testo per tutti i 120 casi. L'ascolto diretto dei 10 campioni sentinella (`LEN_01`–`LEN_05`, `NAR_01`, `NUM_01`, `ACR_01`, `HOM_01`, `TEC_01`) ha confermato che il testo viene pronunciato nella sua interezza, senza troncamenti di parole sul finale.
2. **Definizione di Clipping:**  
   La metrica `peak_dbfs` corrisponde al **sample peak** digitale ($\text{max}(|s_i|)$) rispetto a 0 dBFS. Non rappresenta il True Peak inter-sample misurato tramite oversampling a 4x (normativa ITU-R BS.1770 / EBU R128). I valori registrati (tutti compresi tra -0,21 dBFS e -8,66 dBFS) certificano l'assenza di clipping digitale a livello di campioni interi.

---

## 4. Indicazioni per le Milestone Successive

- **In M1 (Architettura Modulare & Benchmark Controllato):**
  - Implementare il confronto A/B sistematico su input grezzo vs input con normalizzazione preliminare.
  - Testare l'effetto della patch dinamica `min_new_tokens` solo in modalità A/B isolata prima di considerarla come default.
- **In M2 (Linguistica Italiana):**
  - Integrare un modulo formale di **Text Normalization (TN)** per date, ore, numeri cardinali/ordinali, valute e unità di misura.
  - Sviluppare un dizionario di espansione per sigle e abbreviazioni italiane.
  - Introdurre un modulo G2P con annotazione prosodica e disambiguazione degli omografi tonici.
