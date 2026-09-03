# Sistema di Raccomandazione Videogiochi

Sistema ibrido per la raccomandazione personalizzata di titoli Steam basato sull'integrazione di programmazione logica, reti probabilistiche e machine learning.

### Struttura dei File

* `game_recommendation.py`: script principale del progetto. Esegue il filtraggio logico con Prolog, calcola i punteggi tramite Rete Bayesiana e Random Forest, e genera la graduatoria finale pesata con il Bayesian Rating della community.
* `data/steam_games_raw.csv`: dataset grezzo scaricato da Kaggle con i metadati originali e le recensioni dei giochi Steam.
* `data/dataset_games.csv`: dataset esportato dopo le operazioni di pulizia, categorizzazione delle feature e calcolo del rating bayesiano.
* `logic/regole.pl`: contiene le regole logiche in clausole di Horn per i vincoli di ammissibilità, l'albero narrativo delle saghe e la ricorsione per la chiusura transitiva dei prequel tramite Negation as Failure.
* `logic/profilo_utente.pl`: base di fatti con la libreria dell'utente, i generi graditi e le preferenze di budget.
* `logic/conoscenza_giochi.pl`: base di conoscenza generata automaticamente dal catalogo giochi per la risoluzione SLD in Prolog.
* `scripts/preprocess.py`: modulo per la pulizia del dataset grezzo, l'estrazione dei generi dai tag della community e la generazione dei fatti Prolog.
* `scripts/bayesian_network.py`: definizione del DAG, apprendimento delle tabelle di probabilità condizionata (CPT) e inferenza tramite Variable Elimination.
* `scripts/machine_learning.py`: addestramento e valutazione comparativa tra Decision Tree e Random Forest con Cross-Validation stratificata.
* `scripts/fetch_steam_profile.py`: script opzionale per sincronizzare i titoli giocati e i generi preferiti tramite le Steam Web API.
* `raccomandazioni_finali.csv`: report generato automaticamente contenente la lista ordinata dei giochi raccomandati e i rispettivi punteggi.

### Esecuzione

Dalla cartella principale del progetto, eseguire:

```bash
python game_recommendation.py
```

### Script Ausiliari e Riproducibilità

I file `data/dataset_games.csv` e `logic/conoscenza_giochi.pl` sono già inclusi nella repository per consentire l'esecuzione immediata della pipeline senza passaggi preliminari. Gli script contenuti nella cartella `scripts/` sono stati impiegati durante lo sviluppo per compiti specifici: `preprocess.py` ha ripulito il dataset grezzo e generato i fatti logici; `fetch_steam_profile.py` ha interrogato le API di Steam per creare il profilo utente di partenza; `bayesian_network.py` e `machine_learning.py` sono serviti a calibrare, validare e confrontare singolarmente i rispettivi modelli prima della loro integrazione nel modulo principale.

### Collegamento di un Profilo Steam Personalizzato

Il progetto include già un profilo di default in `logic/profilo_utente.pl`. Se si desidera testare il sistema con una libreria Steam diversa, è possibile sincronizzare un nuovo account tramite lo script `scripts/fetch_steam_profile.py`:

1. **Requisiti dell'account Steam**:
   * Impostare lo stato del profilo e i dettagli dei giochi su **Pubblico** (nel client Steam: *Modifica profilo > Impostazioni sulla privacy*).
   * Ottenere una chiave API gratuita dalla pagina [Steam Web API](https://steamcommunity.com/dev/apikey).
   * Recuperare il proprio identificativo numerico **SteamID64** a 17 cifre (visibile dall'URL del profilo o tramite strumenti come steamid.io).

2. **Configurazione dello script**:
   * Aprire il file `scripts/fetch_steam_profile.py`.
   * Assegnare i valori recuperati alle rispettive variabili di configurazione:
     ```python
     STEAM_API_KEY = "LA_TUA_CHIAVE_API"
     STEAM_ID = "IL_TUO_STEAM_ID_64"
     ```

3. **Estrazione e generazione della conoscenza**:
   * Eseguire lo script dalla radice del repository:
     ```bash
     python scripts/fetch_steam_profile.py
     ```
   * Lo script interrogherà le API di Steam (`GetOwnedGames`), analizzerà la cronologia d'uso per ricavare i generi con maggior tempo di gioco e sovrascriverà automaticamente `logic/profilo_utente.pl` con i nuovi fatti (`gia_giocato/1`, `genere_gradito/1`, `budget_utente/1`).

4. **Avvio della pipeline**:
   * Lanciare nuovamente lo script principale per calcolare le raccomandazioni sul nuovo profilo:
     ```bash
     python game_recommendation.py
     ```
