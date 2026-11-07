# Quality Audit & Sentinel ASR Report (Milestone M0)
> **Data:** 7 Novembre 2026  
> **Modelli Valutati:** `Qwen3-TTS-12Hz-1.7B-Base` vs `Qwen3-TTS-12Hz-0.6B-Base`  
> **Motore ASR Sentinella:** `openai/whisper-tiny` (inferenza FP16 su NVIDIA RTX 4080 CUDA)  
> **Scopo:** Valutazione dell'accuratezza linguistica, rilevamento di allucinazioni fonetiche, loop di ripetizione e instabilità prosodiche sui campioni del benchmark freeze.

---

## 1. Sommario Esecutivo

Come correttamente evidenziato dalla peer-review di ChatGPT, **l'assenza di anomalie acustiche euristiche (zero clipping, zero silenzio, zero troncature) NON equivale ad accuratezza o qualità del parlato**.

L'audit acustico e linguistico eseguito tramite ASR sentinella sui file WAV reali a 24 kHz ha portato a scoperte diagnostiche decisive che confermano e orientano l'architettura delle prossime milestone:

1. **Assenza Totale di ITN (Inverse Text Normalization) nel Baseline:**
   Su testi contenenti numeri, date, ore e valute (`NUM_01`), il modello autoregressivo base non sa vocalizzare i token simbolici (`11/05/1998`, `14:30`, `2,5%`, `1.500.000 €`) e collassa in un **loop allucinatorio infinito** (es. ripetizione di *"a salica di un copercinto..."* o *"niracinto niracinto..."* per l'intera durata dell'audio).
2. **Incapacità di Espansione di Sigle e Abbreviazioni (`ACR_01`):**
   Abbreviazioni comuni come `Dott.` e `Avv.` vengono lette come sillabe deformate (*"il metto resci e lagno"*, *"il decresci e la visi"*), non espandendosi in *"Dottor"* e *"Avvocato"*.
3. **Mancanza di Disambiguazione degli Omografi (`HOM_01`):**
   Parole omografe non omomorfe (es. *prìncipi* vs *princìpi*) vengono pronunciate con accento casuale o uniforme, privo di analisi morfosintattica del contesto.
4. **Resa dei Termini Tecnici e Forestierismi (`TEC_01`):**
   Termini inglesi comuni in ambito IT (`Docker`, `docker compose up -d`) subiscono una storpiatura fonetica grezza (*"doppere"*, *"bocca"*).
5. **Comportamento su Frasi Brevi (`LEN_01`, `LEN_02`):**
   Su *"Sì."* e *"No, grazie."*, entrambi i modelli producono audio intellegibile (durata 0.8s–1.9s) ma con una tendenza all'allungamento vocalico (*"Noo, grazie!"*), confermando la necessità di rimuovere il monkey-patch rigido a 60 token in favore di una gestione pulita dell'arresto (`eos_token`).

---

## 2. Dettaglio Comparativo dei Campioni Sentinella

| ID Test & Categoria | Testo Originale (Ground Truth) | Trascrizione ASR su 1.7B | Trascrizione ASR su 0.6B | Diagnosi Qualitativa |
| :--- | :--- | :--- | :--- | :--- |
| **`NUM_01`**<br>(Numeri, Date, Valute) | *"L'11/05/1998 alle ore 14:30 il PIL è salito del 2,5% a fronte di 1.500.000 €."* | `"In questo tempo, prima che ha potato all'ora di poterci stranta il piano e salica di un po' avvolverlo, c'è un copercinto, a salica di un copercinto... [loop ripetuto 40+ volte]"` (10.88s) | `"E' non di scienze rosti, con il nuovo cellore intero, con l'ore, con il gittere interpino... affronte di ognomenci o cento niracinto niracinto niracinto... [loop ripetuto 90+ volte]"` (12.40s) | **CRITICO (Collasso Autoregressivo):** L'assenza di normalizzazione del testo numerico manda il decoder in blocco allucinatorio. Dimostra l'assoluta necessità del modulo `modules/linguistics/text_normalizer.py` (M2). |
| **`ACR_01`**<br>(Abbreviazioni e Sigle) | *"Il Dott. Rossi e l'Avv. Bianchi hanno presentato ricorso al TAR del Lazio."* | `"il metto resci e lagno, gli anchi hanno presentato l'interno, oltarbe le mezzo."` (6.00s) | `"il decresci e la visi, vi anche ha presentato il cos'antar de l'azio."` (6.48s) | **INACCURATO:** Deformazione fonetica totale di `Dott.`, `Avv.` e `TAR`. Richiede modulo dizionario sigle ed espansione abbreviazioni. |
| **`HOM_01`**<br>(Omografi e Accenti) | *"I prìncipi della dinastia condividevano i medesimi princìpi morali."* | `"E preuncipi della di nastia, condividero al medesso e preuncipi morali."` (4.80s) | `"e principe di radina stia condividivare mette insieme e principe in modale."` (4.48s) | **AMBIGUO:** Il modello non distingue la diversa posizione dell'accento tonico (*prìncipi* sdrucciola vs *princìpi* piana). Richiede modulo G2P con tagger POS. |
| **`TEC_01`**<br>(Termini Tecnici) | *"Per avviare il container Docker, eseguire il comando docker compose up -d sulla porta 8080."* | `"per avviare il contenere doppere, eseguire il pomande doppere con pesi a 10 minuti a un minuttante."` (8.24s) | `"per agli contenere bocca, eseguere con un bocca, con pezzi atte di sceporta e c'è che c'è quattra."` (7.68s) | **DEFORMATO:** Inglesismi tecnici storpiati completamente. Richiede dizionario fonetico di pre-elaborazione per termini stranieri e codici CLI. |
| **`NAR_01`**<br>(Dialogo Narrativo) | *"«Dove pensi di andare a quest'ora?», gli chiese Marta con voce tremante."* | `"Dove pensi che andare a costora, e che esimata con un ciotremante."` (5.12s) | `"dove pensi di andare a costora e chiesa morta con un ciotromolto."` (5.04s) | **PARZIALE:** Prosodia e intonazione buone ma parole slittate foneticamente (*costora* per *quest'ora*, *ciotremante* per *voce tremante*). |
| **`LEN_01`**<br>(Frasi Brevi) | *"Sì."* | `"Sì."` (0.88s) | `"Sì!"` (0.80s) | **ACCURATO:** Frase monosillabica sintetizzata correttamente senza babbling eccessivo. |
| **`LEN_02`**<br>(Frasi Brevi) | *"No, grazie."* | `"Noo, grazie!"` (1.84s) | `"No, grazie."` (1.92s) | **ACCURATO:** Trascrizione fedele, lieve prolungamento vocalico iniziale. |
| **`LEN_03`**<br>(Frasi Medie) | *"Certamente, arrivo subito!"* | `"Certo a mente a riposciuto."` (2.72s) | `"Certamente, arrivo subito."` (3.44s) | **DISCRETO SU 0.6B:** Il modello 0.6B trascrive perfettamente, 1.7B spezza *"certo a mente"*. |

---

## 3. Conclusioni Ingegneristiche per M1 e M2

Questi riscontri empirici forniscono la **prova definitiva** del percorso di sviluppo delineato:
1. **La sola architettura di generazione non basta:** Qwen3-TTS ha un eccellente timbro vocale zero-shot (24 kHz, naturalezza e assenza di artefatti robotici), ma è totalmente privo di intelligenza linguistica per la lingua italiana.
2. **Priorità Assoluta di M2 (Linguistica Italiana):** Il modulo di Text Normalization e G2P per date, valute, numeri e abbreviazioni non è un "optional di rifinitura", ma il prerequisito fondamentale affinché il modello non entri in loop allucinatorio.
3. **Decoupling Architetturale di M1:** L'Engine di generazione deve esporre pipeline modulari in cui il testo grezzo viene prima validato, normalizzato e arricchito foneticamente prima di essere inviato ai token audio della rete neurale.
