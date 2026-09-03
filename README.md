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