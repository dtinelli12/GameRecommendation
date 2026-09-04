# KBS Ibrido: Sistema Basato su Conoscenza per il Supporto alle Decisioni

Progetto per l'esame di **Ingegneria della Conoscenza (ICon)**  
**Dipartimento di Informatica — Università degli Studi di Bari Aldo Moro**  
* **Studente:** Domenico Tinelli (Matr. 777022)  
* **Anno Accademico:** 2025-2026  
* **Documentazione Completa:** consultare [`documentazione.md`](documentazione.md) per i dettagli teorici, le scelte di progetto e le valutazioni sperimentali.

---

## Descrizione del Progetto

Il sistema è un **Knowledge-Based System (KBS) ibrido** per l'analisi decisionale e la raccomandazione di software videoludico all'interno del catalogo Steam. L'architettura integra tre paradigmi computazionali cooperativi:
1. **Ragionamento Deduttivo (Prolog):** filtraggio deterministico su vincoli rigidi (budget, piattaforme, preferenze) e risoluzione ricorsiva della continuità narrativa delle saghe tramite *Negation as Failure (NAF)* sotto *Closed-World Assumption (CWA)*.
2. **Ragionamento Probabilistico (Reti Bayesiane):** modellazione dell'incertezza su Grafo Aciclico Orientato (DAG) con stima delle CPT tramite prior *BDeu* ed inferenza esatta con *Variable Elimination*.
3. **Apprendimento Supervisionato (Machine Learning):** stima predittiva del gradimento tramite *Random Forest Classifier*, calibrata e validata mediante *Stratified 10-Fold Cross-Validation* ($\mu \pm \sigma$).
4. **Graduatoria Finale:** combinazione convessa dei moduli pesata con lo *Shrinkage Bayesiano (Empirical Bayes Rating)* del consenso della community.

---

## Requisiti di Sistema e Installazione

### 1. Prerequisiti Software
* **Python:** versione `3.10` o superiore.
* **SWI-Prolog:** versione `9.x` installata sul sistema operativo e accessibile dalle variabili d'ambiente (`PATH`).  
  *(Nota: il modulo `pyswip` richiede i binari di SWI-Prolog installati sulla macchina host).*

### 2. Installazione delle Dipendenze Python
Dalla cartella principale del progetto, eseguire:

```bash
pip install -r requirements.txt
```

*(Librerie principali: `pyswip`, `pgmpy`, `scikit-learn`, `pandas`, `numpy`, `requests`).*

---

## Struttura dei File

* `game_recommendation.py`: script principale della pipeline. Coordina il filtro deduttivo Prolog, interroga la Rete Bayesiana e la Random Forest, e salva la graduatoria ordinata finale.
* `data/steam_games_raw.csv`: dataset grezzo con metadati e recensioni estratto da Kaggle.
* `data/dataset_games.csv`: catalogo processato con feature categoriche discretizzate e rating bayesiano.
* `logic/regole.pl`: assiomi deduttivi in clausole di Horn, modellazione del grafo aciclico delle saghe narrative e chiusura transitiva dei prequel tramite NAF.
* `logic/profilo_utente.pl`: base di fatti con la libreria dell'utente, i generi graditi e le fasce di budget ammissibili.
* `logic/conoscenza_giochi.pl`: base di conoscenza estensionale generata programmaticamente dal catalogo.
* `scripts/preprocess.py`: pipeline di pulizia dati, discretizzazione delle feature e generazione dei fatti Prolog.
* `scripts/bayesian_network.py`: definizione del DAG bayesiano, parameter learning (BDeu) e test di inferenza con Variable Elimination.
* `scripts/machine_learning.py`: benchmark comparativo (Decision Tree vs Random Forest) con Stratified 10-Fold Cross-Validation e feature importance.
* `scripts/fetch_steam_profile.py`: script di utilità per sincronizzare il profilo e la libreria giochi tramite Steam Web API.
* `raccomandazioni_finali.csv`: output generato contenente l'elenco ordinato delle raccomandazioni e i rispettivi punteggi parziali/globali.

---

## Esecuzione Rapida

I file `data/dataset_games.csv` e `logic/conoscenza_giochi.pl` sono già pre-elaborati e inclusi nel repository per consentire l'avvio immediato:

```bash
python game_recommendation.py
```

Al termine dell'esecuzione, la graduatoria verrà stampata a terminale e salvata nel file `raccomandazioni_finali.csv`.

---

## Collegamento di un Profilo Steam Personalizzato

Il sistema include un profilo di default in `logic/profilo_utente.pl`. Per testare il KBS con un account Steam differente, è possibile sincronizzare la propria libreria tramite lo script `scripts/fetch_steam_profile.py`:

1. **Requisiti dell'account Steam**:
   * Impostare il profilo e i dettagli dei giochi su **Pubblico** (*Modifica profilo > Impostazioni sulla privacy* nel client Steam).
   * Generare una chiave API dalla pagina ufficiale [Steam Web API](https://steamcommunity.com/dev/apikey).
   * Reperire il proprio identificativo numerico **SteamID64** a 17 cifre.

2. **Configurazione dello script**:
   * Aprire `scripts/fetch_steam_profile.py` e impostare le credenziali:
     ```python
     STEAM_API_KEY = "LA_TUA_CHIAVE_API"
     STEAM_ID = "IL_TUO_STEAM_ID_64"
     ```

3. **Estrazione della conoscenza**:
   * Eseguire lo script dalla radice del progetto:
     ```bash
     python scripts/fetch_steam_profile.py
     ```
   * Lo script interroga l'endpoint `GetOwnedGames`, calcola le ore giocate per genere e aggiorna automaticamente `logic/profilo_utente.pl` con i nuovi fatti assiomatici (`gia_giocato/1`, `genere_gradito/1`, `fascia_prezzo_accettabile/1`).

4. **Calcolo delle nuove raccomandazioni**:
   * Riavviare la pipeline principale per applicare i modelli sul nuovo profilo:
     ```bash
     python game_recommendation.py
     ```