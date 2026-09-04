# KBS Ibrido per il Supporto alle Decisioni nel Catalogo Software
**Integrazione di Ragionamento Deduttivo, Modelli Grafici Probabilistici e Apprendimento Supervisionato**

### Gruppo di lavoro
* **Domenico Tinelli**, matricola 777022, d.tinelli12@studenti.uniba.it

**URL Repository:** https://github.com/dtinelli12/GameRecommendation  
**Anno Accademico:** 2025-2026

---

## Indice dei Contenuti
1. [Introduzione](#introduzione)[cite: 3]
2. [Sommario e Architettura di Sistema](#sommario-e-architettura-di-sistema)[cite: 3]
3. [Elenco degli Argomenti di Interesse](#elenco-degli-argomenti-di-interesse)[cite: 3]
4. [Sezione Argomento 1: Rappresentazione della Conoscenza e Ragionamento Deduttivo](#sezione-argomento-1-rappresentazione-della-conoscenza-e-ragionamento-deduttivo)[cite: 3]
   * [Sommario](#sommario-1)[cite: 3]
   * [Strumenti utilizzati](#strumenti-utilizzati)[cite: 3]
   * [Decisioni di Progetto](#decisioni-di-progetto)[cite: 3]
   * [Valutazione](#valutazione)[cite: 3]
5. [Sezione Argomento 2: Ragionamento in Condizioni di Incertezza (Rete Bayesiana)](#sezione-argomento-2-ragionamento-in-condizioni-di-incertezza-rete-bayesiana)[cite: 3]
   * [Sommario](#sommario-2)[cite: 3]
   * [Strumenti utilizzati](#strumenti-utilizzati-1)[cite: 3]
   * [Decisioni di Progetto](#decisioni-di-progetto-1)[cite: 3]
   * [Valutazione](#valutazione-1)[cite: 3]
6. [Sezione Argomento 3: Apprendimento Supervisionato e Modelli Predittivi](#sezione-argomento-3-apprendimento-supervisionato-e-modelli-predittivi)[cite: 3]
   * [Sommario](#sommario-3)[cite: 3]
   * [Strumenti utilizzati](#strumenti-utilizzati-2)[cite: 3]
   * [Decisioni di Progetto](#decisioni-di-progetto-2)[cite: 3]
   * [Valutazione e Risultati Sperimentali](#valutazione-e-risultati-sperimentali)[cite: 3]
7. [Integrazione Globale: Scoring e Formula di Rango](#integrazione-globale-scoring-e-formula-di-rango)
8. [Conclusioni e Sviluppi Futuri](#conclusioni-e-sviluppi-futuri)[cite: 3]
9. [Riferimenti Bibliografici](#riferimenti-bibliografici)[cite: 3]

---

<a id="introduzione"></a>
## Introduzione

Il dominio applicativo riguarda la selezione e l'analisi decisionale all'interno del catalogo di videogiochi distribuito sulla piattaforma Steam. Tale contesto è caratterizzato da elevata eterogeneità dei dati, vincoli tassonomici rigidi, dipendenze narrative sequenziali (saghe e prequel), incertezza legata alla reputazione della community e una marcata dimensionalità delle feature descrittive (generi, fasce di prezzo, tempi medi di fruizione). Invece di ricorrere a tradizionali euristiche di filtraggio collaborativo, il problema viene affrontato formalizzando il dominio mediante un sistema a base di conoscenza capace di gestire congiuntamente vincoli logici stringenti, stime probabilistiche e predizioni statistiche.

---

<a id="sommario-e-architettura-di-sistema"></a>
## Sommario e Architettura di Sistema

Il progetto implementa un **Knowledge-Based System (KBS) ibrido** a tre livelli, progettato per dimostrare la cooperazione sistematica tra paradigmi computazionali eterogenei:

* **Filtro Logico-Deduttivo:** Riduce lo spazio di ricerca escludendo a priori le entità non conformi ai vincoli deterministici dell'utente e alle dipendenze narrative del catalogo.
* **Inferenza Probabilistica Causal-Bayesiana:** Pesa l'ammissibilità dei candidati sotto condizioni di incertezza e informazione parziale.
* **Apprendimento Induttivo Supervisionato:** Estrae regolarità statistiche per stimare la propensione al gradimento globale, integrando il Bayesian Rating della community.

```text
           [ Catalogo Raw Steam (Kaggle) ]
                          │
               (scripts/preprocess.py)
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
[ data/dataset_games.csv ]     [ logic/conoscenza_giochi.pl ]
         │                     [ logic/profilo_utente.pl    ]
         │                     [ logic/regole.pl            ]
         │                                 │
         │                                 ▼
         │                 ┌───────────────────────────────┐
         │                 │  1. MODULO DEDUTTIVO (Prolog) │
         │                 │  - Risoluzione SLD            │
         │                 │  - Chiusura transitiva saghe  │
         │                 │  - Negation as Failure (NAF)  │
         │                 └───────────────┬───────────────┘
         │                                 │ Candidati ammissibili
         │                                 ▼
         │                 ┌───────────────────────────────┐
         ├────────────────>│  2. MODULO PROBABILISTICO     │
         │                 │  - Rete Bayesiana (DAG)       │
         │                 │  - Inferenza esatta (VE)      │
         │                 └───────────────┬───────────────┘
         │                                 │ P(Recommended | e)
         │                                 ▼
         │                 ┌───────────────────────────────┐
         ├────────────────>│  3. MODULO INDUTTIVO (ML)     │
         │                 │  - Random Forest Classifier   │
         │                 │  - Stratified 10-Fold CV      │
         │                 └───────────────┬───────────────┘
         │                                 │ Score predittivo
         │                                 ▼
         │                 ┌───────────────────────────────┐
         └────────────────>│  4. AGGREGAZIONE E RANKING    │
                           │  - Bayesian Shrinkage Rating  │
                           │  - Graduatoria pesata         │
                           └───────────────┬───────────────┘
                                           │
                                           ▼
                              [ raccomandazioni_finali.csv ]
```

---

## Elenco degli Argomenti di Interesse

| Argomento | Sezione Programma | Tecniche e Modelli Adottati | Ruolo Operativo nel KBS |
| :--- | :--- | :--- | :--- |
| **1. Rappresentazione e Deduzione** | **Parte II** *(Logica)* | Clausole di Horn, Chiusura transitiva, NAF (CWA), Risoluzione SLD | Filtro vincolare deterministico e validazione saghe |
| **2. Incertezza e Probabilità** | **Parte III** *(Reti Bayesiane)* | Modelli Grafici (DAG), Stima CPT con prior BDeu, Variable Elimination | Punteggio causale sotto informazione incompleta |
| **3. Apprendimento Supervisionato** | **Parte IV** *(Machine Learning)* | Decision Tree vs Random Forest, Stratified 10-Fold CV ($\mu \pm \sigma$) | Predizione non lineare del gradimento globale |

---

### Dettaglio degli Argomenti Trattati

* **Argomento 1: Rappresentazione della Conoscenza e Ragionamento Simbolico** *(Parte II del Programma)*
  * **Formalizzazione:** Clausole di Horn definite suddivise tra fatti ground estesi (`conoscenza_giochi.pl`), fatti dinamici di profilo (`profilo_utente.pl`) e regole deduttive (`regole.pl`).
  * **Meccanismo Inferenziale:** Risoluzione SLD (Selective Linear Definite clause resolution) con strategia depth-first e backtracking automatico.
  * **Complessità Assiomatica:** Modellazione della continuità narrativa su grafo aciclico orientato (DAG) tramite **chiusura transitiva ricorsiva** (`da_giocare_prima/2`) e **Negation as Failure (NAF)** sotto Closed-World Assumption (`saga_rispettata/1`), superando la natura di semplice pattern-matching tabellare.

* **Argomento 2: Ragionamento in Condizioni di Incertezza** *(Parte III del Programma)*
  * **Formalizzazione:** Modello Grafico Probabilistico (Rete Bayesiana a nodi discreti) con assunzioni esplicite di indipendenza condizionata tra feature strutturali (prezzo, durata, piattaforma) e target.
  * **Apprendimento Parametri:** Stima bayesiana delle CPT con prior uniforme equivalente di Dirichlet (**BDeu**, $s = 10$) per prevenire lo *zero-frequency problem* derivante da combinazioni di feature non campionate.
  * **Inferenza Esatta:** Algoritmo di **Variable Elimination (VE)** per calcolare la distribuzione a posteriori $P(\text{Recommended} = \text{yes} \mid \mathbf{e})$ marginalizzando le variabili latenti.

* **Argomento 3: Apprendimento Automatico Supervisionato** *(Parte IV del Programma)*
  * **Modelli a Confronto:** Classificatore singolo interpretativo (**Decision Tree**, indice di impurità di Gini) contro architettura ensemble (**Random Forest**, 150 alberi con bootstrap aggregating).
  * **Scelta degli Iperparametri:** Ottimizzazione preliminare via `GridSearchCV` su 5 fold (`max_depth = 8`, `min_samples_leaf = 10`) mirata a regolarizzare l'albero e tagliare i rami dominati da generi rari per prevenire l'overfitting.
  * **Protocollo di Valutazione:** **Stratified 10-Fold Cross-Validation** con preservazione del bilanciamento delle classi e aggregazione rigorosa di media empirica e deviazione standard ($\mu \pm \sigma$) su tutte le metriche.

---

<a id="sezione-argomento-1-rappresentazione-della-conoscenza-e-ragionamento-deduttivo"></a>
## Sezione Argomento 1: Rappresentazione della Conoscenza e Ragionamento Deduttivo

<a id="sommario-1"></a>
### Sommario
Il modulo implementa un motore logico-deduttivo fondato sul formalismo delle **Clausole di Horn definite** per modellare la conoscenza del dominio ed eseguire il filtraggio deterministico dei candidati. L'architettura della Base di Conoscenza (KB) è strutturata in tre componenti modulari:

* **Base di Conoscenza Estensionale (`conoscenza_giochi.pl`):** Generata programmaticamente a partire dal catalogo pre-processato, modella le istanze del dominio mediante fatti ottuari:
  ```prolog
  gioco(Id, Sviluppatore, Genere1, Genere2, FasciaPrezzo, CategoriaPlaytime, Piattaforma, RecCommunity).
  ```
* **Base di Fatti Dinamica (`profilo_utente.pl`):** Codifica le preferenze e lo storico di gioco dell'utente (interrogabili anche via Steam Web API):
  * `gia_giocato/1`: titoli già posseduti o completati.
  * `genere_gradito/1`: categorie tassonomiche di interesse.
  * `fascia_prezzo_accettabile/1`: vincoli di spesa ammissibili (`free`, `budget`, `mid_price`, `premium`).
* **Base di Conoscenza Intensionale (`regole.pl`):** Definisce gli assiomi relazionali, le regole di compatibilità e i vincoli sequenziali di fruizione.

---

<a id="strumenti-utilizzati"></a>
### Strumenti utilizzati
* **SWI-Prolog (v9.x):** Interprete logico basato sul principio di Risoluzione SLD (*Selective Linear Definite clause resolution*) con strategia Depth-First e meccanismo di backtracking cronologico [1].
* **PySwip:** Libreria Foreign Function Interface (FFI) basata su CFFI/ctypes per l'orchestrazione bidirezionale tra il runtime Python e il motore Prolog [1].

---

<a id="decisioni-di-progetto"></a>
### Decisioni di Progetto

In accordo con i vincoli didattici (che vietano ontologie ridotte a semplici tabelle di fatti interrogate con predicati di pattern matching banali), la KB implementa una logica relazionale complessa per la gestione della **continuità narrativa delle saghe videoludiche**.

#### 1. Modellazione del Grafo delle Saghe
Le dipendenze narrative dirette tra capitoli sono formalizzate come un grafo aciclico orientato (DAG) $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, dove ogni arco $(x, y) \in \mathcal{E}$ rappresenta la relazione ground `prequel_diretto(x, y)`:
```prolog
prequel_diretto('the_witcher_enhanced_edition', 'the_witcher_2_assassins_of_kings_enhanced_edition').
prequel_diretto('the_witcher_2_assassins_of_kings_enhanced_edition', 'the_witcher_3_wild_hunt').
```

#### 2. Chiusura Transitiva Ricorsiva
Per verificare le dipendenze narrative a profondità arbitraria, è stata definita la chiusura transitiva $\mathcal{E}^+$ mediante induzione strutturale:
```prolog
% Caso base: precedenza diretta
da_giocare_prima(X, Y) :- 
    prequel_diretto(X, Y).

% Passo induttivo: precedenza transitiva
da_giocare_prima(X, Y) :- 
    prequel_diretto(X, Z), 
    da_giocare_prima(Z, Y).
```

#### 3. Quantificazione Universale tramite Negation as Failure (NAF)
Un titolo $T$ appartenente a una saga non può essere raccomandato se l'utente non ha completato tutti i capitoli precedenti. Formalmente:

$$\forall P \, (\text{da\_giocare\_prima}(P, T) \implies \text{gia\_giocato}(P))$$

Sotto la **Closed-World Assumption (CWA)**, tale formula è convertita nella forma equivalente a quantificatore esistenziale negato:

$$\neg \exists P \, (\text{da\_giocare\_prima}(P, T) \land \neg \text{gia\_giocato}(P))$$

In Prolog, la regola è implementata sfruttando l'operatore NAF (`\+`):
```prolog
saga_rispettata(Titolo) :-
    \+ (da_giocare_prima(Prequel, Titolo), \+ gia_giocato(Prequel)).
```

#### 4. Regola Conclusiva di Ammissibilità Logica
L'obiettivo decisionale `consigliato(Titolo)` unifica e seleziona i titoli ammissibili imponendo congiuntamente tutti i filtri tassonomici e relazionali:
```prolog
consigliato(Titolo) :-
    gioco(Titolo, _Dev, G1, G2, Prezzo, _Playtime, _Platform, 'yes'),
    genere_compatibile(G1, G2),
    fascia_prezzo_accettabile(Prezzo),
    \+ gia_giocato(Titolo),
    saga_rispettata(Titolo).
```

---

<a id="valutazione"></a>
### Valutazione

#### Complessità Computazionale
Dato che il grafo delle saghe narrative $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ è rigorosamente privo di cicli (DAG), la risoluzione SLD per la chiusura transitiva ha una complessità temporale nel caso peggiore limitata superiormente da $O(|\mathcal{V}| + |\mathcal{E}|)$. Ciò assicura la terminazione finita dell'albero di derivazione senza rischio di ricorsione infinita.

#### Riduzione dello Spazio degli Stati
Il modulo Prolog opera come filtro vincolare deterministico (hard constraint). La sua efficacia nella riduzione dello spazio di ricerca per i modelli successivi (Rete Bayesiana e Machine Learning) è sintetizzata nella seguente tabella:

| Fase della Pipeline | Cardinalità Istanze | Descrizione Operativa |
| :--- | :---: | :--- |
| **Spazio Totale Catalogo** | $1.200$ | Catalogo software completo estratto e discretizzato da Steam |
| **Filtro Titoli Posseduti** | $1.154$ | Rimozione delle istanze già presenti nel profilo utente |
| **Filtro Tassonomico & Budget** | $312$ | Selezione su genere gradito (`G1` o `G2`) e fascia di prezzo compatibile |
| **Filtro Chiusura Saghe (NAF)** | **$148$** | **Spazio finale candidati ammissibili passati al modulo probabilistico** |

L'applicazione congiunta della risoluzione SLD e delle regole assiomatiche produce una **riduzione dell'87.6% dello spazio di ricerca iniziale**, permettendo ai successivi moduli probabilistici e predittivi di elaborare esclusivamente un insieme ristretto di alternative ammissibili.

<a id="sezione-argomento-2-ragionamento-in-condizioni-di-incertezza-rete-bayesiana"></a>
## Sezione Argomento 2: Ragionamento in Condizioni di Incertezza (Rete Bayesiana)

<a id="sommario-2"></a>
### Sommario
L'ammissibilità dedotta dal modulo logico costituisce una condizione necessaria ma non sufficiente per una decisione ottimale: essa opera su logica binaria e non quantifica il grado di incertezza intrinseco alla qualità del software e alla variabilità delle preferenze. 

Il secondo livello del KBS modella l'incertezza attraverso un **Modello Grafico Probabilistico (Directed Acyclic Graph)**. La rete bayesiana quantifica la probabilità di gradimento di ciascun titolo ammissibile condizionata alle sue caratteristiche oggettive, gestendo correlazioni condizionali ed evidenze parziali.

---

<a id="strumenti-utilizzati-1"></a>
### Strumenti utilizzati
* **pgmpy (v0.1.25):** Libreria Python per la strutturazione formale del DAG, la stima bayesiana delle CPT e l'inferenza probabilistica esatta [1, 3].
* **NetworkX:** Supporto alla validazione topologica dell'aciclicità del grafo e all'ordinamento topologico delle variabili condizionate.

---

<a id="decisioni-di-progetto-1"></a>
### Decisioni di Progetto

#### 1. Topologia della Rete e Assunzioni di Indipendenza Condizionata
Lo spazio degli stati del modello è definito dall'insieme di variabili discrete:

$$\mathcal{V}_{BN} = \{ \text{Genre}, \text{PriceCategory}, \text{PlaytimeCategory}, \text{PlatformSupport}, \text{Recommended} \}$$

La struttura delle dipendenze orientate (DAG) è stata definita imponendo assunzioni di indipendenza condizionata (*I-map*) motivate dalla fenomenologia del dominio:
* `Genre` agisce come nodo radice e genitore di `PlatformSupport`: generi altamente simulativi o gestionali complessi presentano dipendenza di distribuzione verso piattaforme singole (PC/Linux), a differenza di generi d'azione ad ampia compatibilità.
* `Recommended` (nodo foglia target, binario `yes`/`no`) è condizionato congiuntamente da genere, fascia di prezzo, impegno temporale richiesto e compatibilità di piattaforma.

```text
  [ Genre ]           [ PriceCategory ]       [ PlaytimeCategory ]
   │      \                   │                        │
   │       \                  │                        │
   ▼        \                 ▼                        │
[ PlatformSupport ] ───> [ Recommended (Target) ] <────┘
```

In base alla topologia adottata, la fattorizzazione della probabilità congiunta globale risulta:

$$P(G, Pr, Pl, Pt, R) = P(G) \cdot P(Pr) \cdot P(Pt) \cdot P(Pl \mid G) \cdot P(R \mid G, Pr, Pl, Pt)$$

#### 2. Giustificazione Ingegneristica della Discretizzazione
L'inclusione di variabili continue non discretizzate (es. prezzo in dollari float, monte ore medio esatto in minuti) genera due colli di bottiglia critici:
1. **Esplosione della dimensionalità delle CPT:** Con feature continue e campionamenti continui, la rappresentazione esatta delle distribuzioni congiunte richiede stime parametriche miste (Gaussian CPT) o partizioni a cardinalità infinita, saturando rapidamente la memoria di sistema durante l'inferenza congiunta.
2. **Fragilità dell'evidenza (OutOfDistribution / KeyError):** In fase di query, l'osservazione di un valore continuo non campionato nel training set produce il fallimento della marginalizzazione per mancanza di densità locale.

Tutte le feature sono state discretizzate a monte in `scripts/preprocess.py` in bin finiti e mutuamente esclusivi:
* **Fasce di Prezzo:** `free` ($0.00\$), `budget` ($0.01\$-14.99\$), `mid_price` ($15.00\$-29.99\$), `premium` ($\ge 30.00\$$).
* **Impegno Temporale (Playtime):** `short` ($< 5\text{ h}$), `medium` ($5-25\text{ h}$), `long` ($> 25\text{ h}$).
* **Piattaforme:** `single_platform` (solo Windows), `multiplatform` (supporto esteso Linux/macOS).

#### 3. Parameter Learning: Bayesian Estimator con Prior BDeu
La stima delle Tabelle di Probabilità Condizionata (CPT) tramite il classico stimatore di Massima Verosimiglianza (*Maximum Likelihood Estimator - MLE*) è inadeguata: per configurazioni di feature rare o non osservate nel dataset, MLE assegna probabilità nulla ($P = 0$), azzerando l'intero prodotto di probabilità congiunta durante l'inferenza (*zero-frequency problem*).

Si è adottato il **Bayesian Estimator con prior BDeu (Bayesian Dirichlet equivalent uniform)**:

$$P(X_i = k \mid \text{Pa}(X_i) = j) = \frac{N_{ijk} + \frac{s}{r_i \cdot q_i}}{N_{ij} + \frac{s}{q_i}}$$

dove $r_i$ è il numero di stati della variabile $X_i$, $q_i$ è il numero di configurazioni dei suoi nodi genitori $\text{Pa}(X_i)$, $N_{ijk}$ è il conteggio empirico osservato, e $s$ è la dimensione campionaria equivalente (*equivalent sample size*).

Il parametro di regolarizzazione è stato fissato a **`equivalent_sample_size = 10`**. Tale scelta assegna un pseudo-conteggio sufficiente a distribuire massa di probabilità sulle celle CPT prive di osservazioni, senza appiattire la varianza empirica del catalogo (come avverrebbe con valori di prior eccessivamente alti, es. $s \ge 50$).

---

<a id="valutazione-1"></a>
### Valutazione

#### Inferenza Esatta tramite Variable Elimination
Il punteggio probabilistico non richiede approssimazioni stocastiche (es. Gibbs Sampling) data la compattezza del grafo. L'inferenza è calcolata esattamente con l'algoritmo di **Variable Elimination (VE)**, marginalizzando per somme sui fattori intermedi relativi alle variabili non osservate [1, 3].

Dato un titolo ammesso dal filtro logico, le sue feature strutturali vengono caricate come evidenza $\mathbf{e}$:

$$\mathbf{e} = \{ \text{Genre} = g, \text{PriceCategory} = p, \text{PlaytimeCategory} = t, \text{PlatformSupport} = s \}$$

Il sistema interroga la distribuzione a posteriori del target di gradimento:

$$P_{BN}(\text{Recommended} = \text{'yes'} \mid \mathbf{e})$$

#### Analisi Comparativa delle Probabilità a Posteriori
La seguente tabella illustra il comportamento della rete su profili strutturali significativi del catalogo, evidenziando la capacità del modello di modulare la confidenza in base al contesto:

| Profilo Evidenza Software ($\mathbf{e}$) | Genere | Fascia Prezzo | Playtime | Piattaforme | $P(\text{Recommended} = \text{'yes'} \mid \mathbf{e})$ | Interpretazione Decisionale |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Profilo 1 (Indie Hit)** | `indie` | `budget` | `medium` | `multiplatform` | **$0.864$** | Forte confidenza: combinazione storicamente premiata su Steam. |
| **Profilo 2 (Action Mainstream)** | `action` | `premium` | `long` | `multiplatform` | **$0.781$** | Confidenza elevata con lieve penalizzazione su costo/lunghezza. |
| **Profilo 3 (Casual / Free)** | `casual` | `free` | `short` | `single_platform` | **$0.512$** | Zona di incertezza: alto tasso di gradimento discontinuo. |
| **Profilo 4 (Overpriced Short)** | `adventure`| `premium` | `short` | `single_platform` | **$0.327$** | Bassa confidenza: rapporto prezzo/longevità penalizzante. |

Il punteggio probabilistico calcolato funge da peso intermedio nel ranking finale: un titolo logicamente compatibile ma appartenente a una combinazione strutturalmente debole (es. Profilo 4) viene opportunamente scalzato da titoli con evidenze a maggiore supporto statistico.

<a id="sezione-argomento-3-apprendimento-supervisionato-e-modelli-predittivi"></a>
## Sezione Argomento 3: Apprendimento Supervisionato e Modelli Predittivi

<a id="sommario-3"></a>
### Sommario
Il terzo livello del sistema integra un modello induttivo supervisionato per stimare la funzione di classificazione $f: \mathcal{X} \rightarrow \{0, 1\}$, corrispondente alla probabilità che un'istanza software riceva accoglienza critica positiva da parte della community di Steam. 

Mentre il modello bayesiano impone una struttura condizionale causale fissa a monte, il modulo di Machine Learning opera direttamente sullo spazio vettoriale delle feature, estraendo pattern empirici complessi, correlazioni non lineari e interazioni incrociate tra attributi eterogenei. Lo spazio $\mathcal{X}$ include le caratteristiche categoriche binarizzate tramite One-Hot Encoding (genere primario, genere secondario, fascia di prezzo, impegno temporale, supporto multipiattaforma), garantendo compatibilità con i classificatori basati su iperpiani e partizionamento dello spazio.

---

<a id="strumenti-utilizzati-2"></a>
### Strumenti utilizzati
* **Scikit-Learn (v1.4+):** Pipeline di pre-elaborazione (One-Hot Encoding, imputazione), ottimizzazione su griglia degli iperparametri e implementazione degli estimatori [1, 4].
* **Pandas & NumPy:** Vettorizzazione matriciale e aggregazione numerica dei risultati statistici.

---

<a id="decisioni-di-progetto-2"></a>
### Decisioni di Progetto

In conformità con le direttive metodologiche del corso (che rigettano valutazioni limitate a singoli train/test split o semplici matrici di confusione non aggregate), la progettazione del modulo ha seguito criteri rigorosi di benchmark e validazione.

#### 1. Modelli a Confronto
Per determinare l'architettura predittiva ottimale, sono stati posti a confronto due paradigmi ad albero:
* **Decision Tree Classifier (Baseline interpretativo):** Modello singolo con criterio di split basato sull'impurità di Gini. Permette di tracciare le regole di decisione lineari ma è suscettibile a elevata varianza.
* **Random Forest Classifier (Ensemble a comitato):** Modello ensemble costituito da 150 stimatori ad albero con bootstrap aggregating (bagging) e feature subsampling randomico [4]. L'obiettivo è abbattere la varianza del singolo estimatore senza incrementare il bias.

#### 2. Ottimizzazione Iperparametri tramite GridSearchCV
I parametri di regolarizzazione non sono stati fissati in modo euristico o arbitrario, ma determinati tramite una ricerca su griglia sistematica con **GridSearchCV a 5 fold** su una porzione dedicata del training set. Lo spazio di ricerca esplorato ha riguardato:
* `max_depth`: `[4, 6, 8, 12, None]`
* `min_samples_leaf`: `[1, 5, 10, 20]`
* `min_samples_split`: `[2, 10, 20]`
* `criterion`: `['gini', 'entropy']`

La combinazione ottimale emersa per la Random Forest è:
```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=10,
    criterion='gini',
    random_state=42
)
```
La scelta di vincolare la profondità massima a `max_depth = 8` e la foglia minima a `min_samples_leaf = 10` risponde all'esigenza progettuale di potare i rami alimentati da combinazioni di generi rari a bassissima frequenza nel catalogo, neutralizzando il rischio di memorizzazione del rumore statistico (overfitting).

#### 3. Protocollo di Validazione: Stratified 10-Fold Cross-Validation
Per garantire affidabilità statistica ed eliminare il bias di campionamento:
* È stato adottato uno schema di **Stratified 10-Fold Cross-Validation**.
* La stratificazione assicura che in ciascun fold la percentuale di istanze appartenenti alla classe positiva (`Recommended = yes`) e negativa (`Recommended = no`) sia identica alla distribuzione reale del dataset ($65\% - 35\%$).
* Le prestazioni sono aggregate calcolando **media empirica ($\mu$) e deviazione standard ($\sigma$)** su tutti i 10 fold disgiunti.

---

<a id="valutazione-e-risultati-sperimentali"></a>
### Valutazione e Risultati Sperimentali

#### Tabella Comparativa Prestazionale (Stratified 10-Fold CV)
I risultati ottenuti sui 10 fold indipendenti sono riassunti nella seguente tabella comparativa:

| Modello | Accuracy ($\mu \pm \sigma$) | Precision ($\mu \pm \sigma$) | Recall ($\mu \pm \sigma$) | F1-Score ($\mu \pm \sigma$) |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | $0.7821 \pm 0.0142$ | $0.7645 \pm 0.0181$ | $0.7930 \pm 0.0125$ | $0.7784 \pm 0.0150$ |
| **Random Forest** | $\mathbf{0.8412 \pm 0.0088}$ | $\mathbf{0.8260 \pm 0.0112}$ | $\mathbf{0.8575 \pm 0.0079}$ | $\mathbf{0.8414 \pm 0.0091}$ |

#### Discussione dei Risultati
* **Incremento Prestazionale:** La Random Forest supera sistematicamente l'albero singolo su tutti i parametri, con un incremento di **$+5.91\%$ in Accuracy** e **$+6.30\%$ in F1-Score pesato**.
* **Stabilità e Riduzione della Varianza:** L'aspetto più rilevante è la contrazione della deviazione standard: la Random Forest dimezza quasi la dispersione dei risultati tra i fold ($\sigma_{F1} = 0.0091$ contro $0.0150$ del Decision Tree), confermando che l'aggregazione di più alberi bootstrap mitiga la sensibilità alle oscillazioni locali del dataset.
* **Trade-off Precision/Recall:** Il modello ensemble raggiunge un elevato valore di Recall ($0.8575$), minimizzando i falsi negativi (titoli validi scartati erroneamente), elemento cruciale in un sistema di supporto alle decisioni.

#### Analisi della Feature Importance (Gini Impurity Reduction)
L'ispezione della Mean Decrease in Impurity (MDI) calcolata dalla Random Forest evidenzia i descrittori a maggior potere informativo:

| Rango | Feature Estratta | Importanza Relativa (MDI) | Valutazione Ingegneristica |
| :---: | :--- | :---: | :--- |
| **1** | `playtime_category_long` | **$0.214$** | Titoli con longevità $>25\text{ h}$ mostrano forte correlazione con recensioni positive durature. |
| **2** | `genre_1_action` | **$0.182$** | Genere primario a maggiore volume e stabilità di consenso nel catalogo Steam. |
| **3** | `price_category_budget` | **$0.145$** | Rapporto qualità/prezzo favorevole (fascia $0.01\$-14.99\$$) riduce il bias negativo delle recensioni. |
| **4** | `platform_support_multiplatform` | **$0.118$** | La compatibilità con sistemi Linux/macOS incrementa l'indice di gradimento dell'ecosistema. |
| **5** | Altre feature / Generi secondari | **$0.341$** | Distribuzione diffusa sui generi specifici (`rpg`, `strategy`, `indie`). |

La probabilità calibrata $P_{RF}(\text{Recommended} = \text{'yes'} \mid \mathbf{x})$ prodotta dalla Random Forest viene quindi acquisita come terzo pilastro quantitativo per la graduatoria finale.

<a id="integrazione-globale-scoring-e-formula-di-rango"></a>
## Integrazione Globale: Scoring e Formula di Rango

### Architettura della Pipeline di Fusione
La selezione e l'ordinamento finale delle raccomandazioni non dipendono dall'output isolato di un singolo paradigma, ma scaturiscono da una pipeline a cascata che sintetizza deduzione simbolica, inferenza causale, predizione statistica e consenso empirico della community:

1. **Filtro Vincolare Determinante (Prolog):** Definisce lo spazio dei candidati ammissibili $\mathcal{C} = \{ T \mid \text{consigliato}(T) \}$. Qualsiasi titolo che violi i vincoli di budget, le preferenze di genere o la continuità delle saghe narrative riceve un'assegnazione binaria nulla ($0$) e viene escluso a monte dalla fase di ranking, abbattendo la complessità per i moduli successivi.
2. **Estrazione delle Evidenze e Scoring Congiunto:** Per ciascun candidato $T \in \mathcal{C}$, il sistema interroga in parallelo il modulo probabilistico (pgmpy) e il modello supervisionato (Scikit-Learn).
3. **Calcolo della Graduatoria Finale:** I punteggi quantitativi vengono aggregati secondo una combinazione lineare convessa, producendo il dataset ordinato `raccomandazioni_finali.csv`.

---

### Formulazione Matematica della Funzione di Rango
Per ogni titolo ammissibile $T \in \mathcal{C}$, il punteggio globale $\text{FinalScore}(T) \in [0, 1]$ è formalizzato come:

$$\text{FinalScore}(T) = w_1 \cdot P_{BN}(\text{Recommended} = \text{'yes'} \mid \mathbf{e}_T) + w_2 \cdot P_{RF}(\text{Recommended} = 1 \mid \mathbf{x}_T) + w_3 \cdot \text{BayesianScore}(T)$$

con il vincolo di normalizzazione convessa:

$$\sum_{i=1}^3 w_i = 1, \quad w_i > 0 \quad \forall i \in \{1, 2, 3\}$$

---

### Componente di Consenso: Empirical Bayes Rating (Shrinkage)
L'indice di gradimento grezzo delle recensioni della community su Steam (percentuale di recensioni positive $R = \frac{\text{positive}}{\text{positive} + \text{negative}}$) è fortemente distorto per titoli con un numero esiguo di recensioni: un gioco con 2 recensioni positive su 2 ($100\%$) risulterebbe ingiustamente superiore a un capolavoro acclamato con $45.000$ recensioni positive su $48.000$ ($93.7\%$).

Per correggere questa distorsione statistica senza introdurre euristiche arbitrarie, il termine $\text{BayesianScore}(T)$ implementa uno **Shrinkage Bayesiano (Empirical Bayes Estimator)**:

$$\text{BayesianScore}(T) = \frac{v}{v + m} \cdot R + \frac{m}{v + m} \cdot C$$

I parametri della formulazione sono definiti e calibrati come segue:
* **$v$ (Volume di evidenza locale):** Numero totale di recensioni rilasciate dagli utenti per il titolo $T$ ($v = \text{positive} + \text{negative}$).
* **$R$ (Media campionaria locale):** Rapporto di approvazione empirico del titolo $T$ ($R \in [0, 1]$).
* **$C$ (Prior globale di catalogo):** Valore medio di approvazione osservato sull'intero catalogo Steam discretizzato ($C \approx 0.718$). Rappresenta l'aspettativa a priori in assenza di evidenze locali.
* **$m$ (Soglia di confidenza dello shrinkage):** Fissata a **$m = 50$** recensioni. Quando il volume $v \ll m$, il punteggio viene fortemente regolarizzato verso la media globale $C$; al crescere delle recensioni ($v \gg m$), il peso si sposta deterministicamente verso il tasso reale $R$.

---

### Giustificazione Ingegneristica dei Pesi ($w_1, w_2, w_3$)
La configurazione dei pesi è stata calibrata sui valori **$w_1 = 0.35$**, **$w_2 = 0.35$**, **$w_3 = 0.30$**:

* **$w_1 = 0.35$ (Modulo Probabilistico - Rete Bayesiana):** Pesa la coerenza strutturale causale. Valuta quanto il profilo del gioco (combinazione di prezzo, longevità e compatibilità di piattaforma dato il genere) sia intrinsecamente solido, mitigando l'impatto di titoli sostenuti esclusivamente da campagne pubblicitarie ma deboli nel bilanciamento funzionale.
* **$w_2 = 0.35$ (Modulo Induttivo - Random Forest):** Cattura le complesse interazioni non lineari e multivariate tra generi primari, secondari e requisiti operativi, conferendo robustezza predittiva validata sui 10 fold.
* **$w_3 = 0.30$ (Consenso Community regolarizzato):** Riserva una quota determinante al gradimento reale espresso da decine di migliaia di giocatori, senza tuttavia consentire che la sola popolarità commerciale schiacci i vincoli strutturali e causali calcolati dai modelli di Intelligenza Artificiale.

---

### Esempio Pratico di Ordinamento e Risoluzione dei Conflitti
La seguente tabella illustra il meccanismo di ranking applicato a quattro titoli ammessi dal filtro Prolog, evidenziando come la combinazione dei tre punteggi premi l'eccellenza strutturale e corregga i casi limite:

| Titolo Software Ammissibile | $P_{BN}(\mathbf{e})$ (Causale) | $P_{RF}(\mathbf{x})$ (Predittivo) | Recensioni ($v$) | $R$ grezzo | $\text{BayesianScore}$ (Regolarizzato) | $\text{FinalScore}$ Globale | Decisione di Rango |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **The Witcher 3: Wild Hunt** | $0.884$ | $0.912$ | $52.300$ | $0.958$ | $0.958$ | **$0.916$** | **1° Classificato** (Consenso massivo + profilo eccellente) |
| **Hollow Knight** | $0.864$ | $0.875$ | $18.400$ | $0.962$ | $0.961$ | **$0.897$** | **2° Classificato** (Alta affinità indie-budget) |
| **Indie Sperimentale Recente** | $0.742$ | $0.710$ | $12$ | $1.000$ | $0.772$ | **$0.740$** | **3° Classificato** (Corretto dal prior: non sovrasta i colossi) |
| **Titolo AAA Fuori Prezzo** | $0.415$ | $0.560$ | $3.100$ | $0.620$ | $0.622$ | **$0.528$** | **4° Classificato** (Penalizzato da BN e RF nonostante il brand) |

Come evidenziato dall'esempio del titolo *Indie Sperimentale Recente*, lo Shrinkage Bayesiano riduce l'indice di gradimento dal fuorviante $100\%$ nominale al più realistico $0.772$, impedendo al gioco di scalzare produzioni di comprovata qualità pur mantenendolo in posizione favorevole nella parte medio-alta della graduatoria.

<a id="conclusioni-e-sviluppi-futuri"></a>
## Conclusioni e Sviluppi Futuri

### Sintesi delle Valutazioni
Il sistema a base di conoscenza ibrido sviluppato dimostra l'efficacia della cooperazione tra paradigmi computazionali eterogenei nel risolvere un problema decisionale complesso, superando i limiti intrinseci che ciascun modello manifesterebbe se impiegato isolatamente:

* **Efficienza del Filtro Deduttivo (Prolog):** L'inferenza simbolica mediante Risoluzione SLD ha ridotto lo spazio degli stati da esplorare dell'87.6%. La formalizzazione ricorsiva della chiusura transitiva e della Negation as Failure (NAF) sotto CWA ha garantito la risoluzione deterministica di vincoli di sequenzialità narrativa (saghe) e preferenze personali a costo computazionale trascurabile ($O(|\mathcal{V}| + |\mathcal{E}|)$), senza richiedere massicce basi di dati di addestramento.
* **Calibrazione dell'Incertezza (Rete Bayesiana):** L'adozione del Directed Acyclic Graph (DAG) con stima BDeu ($s = 10$) ha risolto con successo il problema delle combinazioni non osservate nel dataset (zero-frequency problem), consentendo l'inferenza esatta (Variable Elimination) di probabilità a posteriori condizionate alla struttura intrinseca del software.
* **Robustezza Predittiva (Random Forest):** La validazione condotta tramite Stratified 10-Fold Cross-Validation ha sancito la superiorità del modello ensemble rispetto al singolo albero decisionale ($F_1 = 0.8414 \pm 0.0091$ contro $0.7784 \pm 0.0150$), dimezzando la varianza tra fold e dimostrando elevata capacità di generalizzazione sui pattern non lineari del catalogo.
* **Equità nel Ranking (Shrinkage Bayesiano):** L'integrazione convessa finale, combinata con lo stimatore Empirical Bayes ($m = 50$), ha eliminato le distorsioni causate da titoli di nicchia con volumetria di recensioni insufficiente, producendo una graduatoria bilanciata tra solidità tecnica, approvazione predittiva e reputazione reale della community.

---

### Problematiche Affrontate e Compromessi Ingegneristici
Durante lo sviluppo sono emersi vincoli operativi che hanno richiesto compromessi tecnici specifici:
* **Discretizzazione delle Feature Continue:** Come emerso nei test preliminari con modelli bayesiani continui, l'inclusione di prezzi o ore di gioco float causava saturazione della memoria e fallimenti di inferenza (*KeyError* su valori continui mai osservati). La partizione a intervalli discreti ha risolto il problema di scalabilità al prezzo di una lieve perdita di granularità numerica.
* **Estrazione della Conoscenza delle Saghe:** La costruzione del grafo aciclico orientato per la KB estensionale ha richiesto la normalizzazione manuale e semi-automatica dei prequel/sequel dei titoli principali, stante l'assenza di un campo semantico esplicito per le serie narrative all'interno del catalogo grezzo di Steam.

---

### Sviluppi Futuri
In vista di future estensioni da parte di altri gruppi di ricerca o per una messa in produzione industriale, si delineano le seguenti linee di evoluzione architetturale:

* **Transizione a Ontologie OWL 2 DL e Web Semantico:** Sostituzione parziale o totale dei fatti Prolog con un'ontologia formale descritta in OWL 2 DL gestita tramite librerie come `Owlready2`. Tale estensione consentirebbe di sfruttare reasoner standard (es. Pellet, HermiT) per classificare gerarchie complesse di generi e interrogare la base di conoscenza tramite endpoint SPARQL.
* **Natural Language Processing (NLP) sulle Recensioni Utente:** Integrazione di un modello transformer (es. RoBERTa finetunato su testo videoludico) per elaborare le recensioni non strutturate ed estrarre indici di sentiment, bug frequenti o criticità di ottimizzazione hardware, iniettando tale informazione come ulteriore nodo di evidenza nel DAG bayesiano.
* **Interfaccia Utente e Deployment Dinamico:** Sviluppo di un'interfaccia grafica interattiva (es. Streamlit o framework web FastAPI) con autenticazione OAuth diretta su Steam API, permettendo all'utente di caricare automaticamente la propria libreria in tempo reale e ricevere raccomandazioni personalizzate immediate.

<a id="riferimenti-bibliografici"></a>
## Riferimenti Bibliografici

* **Ragionamento logico:** D. Poole, A. Mackworth: *Artificial Intelligence: Foundations of Computational Agents*. 3/e, Cambridge University Press [Ch.5][cite: 2].
* **Prolog:** D. Poole, A. Mackworth: *Artificial Intelligence: Foundations of Computational Agents*. 3/e, Cambridge University Press [Ch.15][cite: 2].
* **Ragionamento probabilistico e reti bayesiane:** D. Poole, A. Mackworth: *Artificial Intelligence: Foundations of Computational Agents*. 3/e, Cambridge University Press [Ch.9][cite: 2].
* **Apprendimento supervisionato:** D. Poole, A. Mackworth: *Artificial Intelligence: Foundations of Computational Agents*. 3/e, Cambridge University Press [Ch.7][cite: 2].
* **Documentazione Steam Web API e metriche catalogo:** https://partner.steamgames.com/doc/webapi
* **Specifiche Kaggle Steam Games Dataset:** https://www.kaggle.com/datasets/fronkongames/steam-games-dataset