# KBS Ibrido per il Supporto alle Decisioni nel Catalogo Software
**Integrazione di Ragionamento Deduttivo, Modelli Grafici Probabilistici e Apprendimento Supervisionato**

### Gruppo di lavoro
* **Domenico Tinelli**, matricola 777022, d.tinelli12@studenti.uniba.it

**URL Repository:** https://github.com/dtinelli12/GameRecommendation  
**Anno Accademico:** 2025-2026

---

## Indice dei Contenuti
1. [Introduzione](#introduzione)
2. [Sommario e Architettura di Sistema](#sommario-e-architettura-di-sistema)
3. [Elenco degli Argomenti di Interesse](#elenco-degli-argomenti-di-interesse)
4. [Sezione Argomento 1: Rappresentazione della Conoscenza e Ragionamento Deduttivo](#sezione-argomento-1-rappresentazione-della-conoscenza-e-ragionamento-deduttivo)
   * [Sommario](#sommario-1)
   * [Strumenti utilizzati](#strumenti-utilizzati)
   * [Decisioni di Progetto](#decisioni-di-progetto)
   * [Valutazione](#valutazione)
5. [Sezione Argomento 2: Ragionamento in Condizioni di Incertezza (Rete Bayesiana)](#sezione-argomento-2-ragionamento-in-condizioni-di-incertezza-rete-bayesiana)
   * [Sommario](#sommario-2)
   * [Strumenti utilizzati](#strumenti-utilizzati-1)
   * [Decisioni di Progetto](#decisioni-di-progetto-1)
   * [Valutazione](#valutazione-1)
6. [Sezione Argomento 3: Apprendimento Supervisionato e Modelli Predittivi](#sezione-argomento-3-apprendimento-supervisionato-e-modelli-predittivi)
   * [Sommario](#sommario-3)
   * [Strumenti utilizzati](#strumenti-utilizzati-2)
   * [Decisioni di Progetto](#decisioni-di-progetto-2)
   * [Valutazione e Risultati Sperimentali](#valutazione-e-risultati-sperimentali)
7. [Integrazione Globale: Scoring e Formula di Rango](#integrazione-globale-scoring-e-formula-di-rango)
8. [Conclusioni e Sviluppi Futuri](#conclusioni-e-sviluppi-futuri)
9. [Riferimenti Bibliografici](#riferimenti-bibliografici)

---

<a id="introduzione"></a>
## Introduzione

Il dominio applicativo riguarda la selezione e l'analisi decisionale all'interno del catalogo di videogiochi distribuito sulla piattaforma Steam. Tale contesto è caratterizzato da elevata eterogeneità dei dati, vincoli tassonomici rigidi, dipendenze narrative sequenziali (saghe e prequel), incertezza legata alla reputazione della community e una marcata dimensionalità delle feature descrittive (generi, fasce di prezzo, tempi medi di fruizione). Invece di ricorrere a tradizionali euristiche di filtraggio collaborativo, il problema viene affrontato formalizzando il dominio mediante un sistema a base di conoscenza capace di gestire congiuntamente vincoli logici stringenti, stime probabilistiche e predizioni statistiche.

---

<a id="sommario-e-architettura-di-sistema"></a>
## Sommario e Architettura di Sistema

Il progetto implementa un **Knowledge-Based System (KBS) ibrido a tre paradigmi computazionali**, orchestrati su quattro stadi operativi per dimostrare la cooperazione sistematica tra modelli eterogenei:

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
          │                    [ logic/profilo_utente.pl    ]
          │                    [ logic/regole.pl            ]
          │                                 │
          │                                 ▼
          │                ┌───────────────────────────────┐
          │                │  1. MODULO DEDUTTIVO (Prolog) │
          │                │  - Risoluzione SLD            │
          │                │  - Chiusura transitiva saghe  │
          │                │  - Negation as Failure (NAF)  │
          │                └───────────────┬───────────────┘
          │                                │ Candidati ammissibili
          │                                ▼
          │                ┌───────────────────────────────┐
          ├───────────────>│  2. MODULO PROBABILISTICO     │
          │                │  - Rete Bayesiana (DAG)       │
          │                │  - Inferenza esatta (VE)      │
          │                └───────────────┬───────────────┘
          │                                │ P(Recommended | e)
          │                                ▼
          │                ┌───────────────────────────────┐
          ├───────────────>│  3. MODULO INDUTTIVO (ML)     │
          │                │  - Random Forest Classifier   │
          │                │  - Stratified 10-Fold CV      │
          │                └───────────────┬───────────────┘
          │                                │ Score predittivo
          │                                ▼
          │                ┌───────────────────────────────┐
          └───────────────>│  4. AGGREGAZIONE E RANKING    │
                           │  - Bayesian Shrinkage Rating  │
                           │  - Graduatoria pesata         │
                           └───────────────┬───────────────┘
                                           │
                                           ▼
                              [ raccomandazioni_finali.csv ]
```

---

<a id="elenco-degli-argomenti-di-interesse"></a>
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
  gioco(Titolo, Sviluppatore, Genere1, Genere2, FasciaPrezzo, CategoriaPlaytime, Piattaforma, RecCommunity).
  ```
* **Base di Fatti Dinamica (`profilo_utente.pl`):** Codifica le preferenze e lo storico di gioco dell'utente (estratte tramite Steam Web API):
  * `gia_giocato/1`: titoli posseduti o completati, utilizzati come base per la Negation as Failure.
  * `genere_gradito/1`: categorie tassonomiche di gradimento dell'utente (`racing`, `soccer`, `action`, `horror`, `fps`).
  * `fascia_prezzo_accettabile/1`: vincoli di budget ammissibili definiti dall'utente (`free`, `budget`, `mid_price`), escludendo tassativamente la fascia a prezzo pieno (`aaa_full`).
* **Base di Conoscenza Intensionale (`regole.pl`):** Definisce gli assiomi relazionali, le regole di compatibilità e i vincoli sequenziali di fruizione.

---

<a id="strumenti-utilizzati"></a>
### Strumenti utilizzati
* **SWI-Prolog (v9.x):** Interprete logico basato sul principio di Risoluzione SLD (*Selective Linear Definite clause resolution*) con strategia Depth-First e meccanismo di backtracking cronologico [Ch.15].
* **PySwip:** Libreria Foreign Function Interface (FFI) basata su CFFI/ctypes per l'orchestrazione bidirezionale tra il runtime Python e il motore Prolog.

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
Dato che il grafo delle saghe narrative $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ è privo di cicli (DAG), la risoluzione SLD per la chiusura transitiva ha una complessità temporale nel caso peggiore limitata superiormente da $O(|\mathcal{V}| + |\mathcal{E}|)$, assicurando la terminazione finita dell'albero di derivazione.

#### Riduzione dello Spazio degli Stati
Il modulo Prolog opera come filtro vincolare deterministico (hard constraint). La sua efficacia nella riduzione dello spazio di ricerca per i modelli successivi è sintetizzata nella seguente tabella:

| Fase della Pipeline | Cardinalità Istanze | Descrizione Operativa |
| :--- | :---: | :--- |
| **Spazio Totale Catalogo** | $11.811$ | Catalogo software filtrato per significatività statistica ($\ge 50$ voti) |
| **Filtro Titoli Posseduti** | $11.602$ | Rimozione delle istanze già presenti nella libreria dell'utente (`gia_giocato`) |
| **Filtro Tassonomico & Budget** | $2.189$ | Selezione su genere gradito (`G1` o `G2`) e fasce di prezzo accettabili |
| **Filtro Chiusura Saghe (NAF)** | **$2.155$** | **Spazio finale candidati ammissibili passati alla pipeline di ranking** |

L'applicazione congiunta della risoluzione SLD e delle regole assiomatiche produce una **riduzione dell'81.8% dello spazio di ricerca iniziale**, consentendo ai moduli probabilistici e predittivi di concentrare il calcolo esclusivamente su alternative ammissibili.

---

<a id="sezione-argomento-2-ragionamento-in-condizioni-di-incertezza-rete-bayesiana"></a>
## Sezione Argomento 2: Ragionamento in Condizioni di Incertezza (Rete Bayesiana)

<a id="sommario-2"></a>
### Sommario
Il secondo modulo dell'architettura modella le dipendenze probabilistiche e l'incertezza intrinseca che caratterizzano il successo critico di un videogioco. A differenza del paradigma deduttivo (che opera per vincoli rigidi booleani), il modello probabilistico stima la probabilità a posteriori che un titolo sia raccomandabile condizionatamente alle sue caratteristiche strutturali e commerciali.

La distribuzione congiunta è formalizzata mediante una Rete Bayesiana discreta (Directed Acyclic Graph, DAG). I parametri condizionati (CPT) sono appresi statisticamente dal catalogo Steam e l'inferenza probabilistica viene risolta in modo esatto tramite l'algoritmo di Eliminazione di Variabili (Variable Elimination).

---

<a id="strumenti-utilizzati-1"></a>
### Strumenti utilizzati
* **pgmpy (v0.1.25+):** Libreria per la definizione della topologia del grafo orientato, l'apprendimento dei parametri e l'inferenza probabilistica su fattori discreti.
* **Pandas & NumPy:** Manipolazione matriciale, raggruppamento delle frequenze e gestione degli stati categorici.

---

<a id="decisioni-di-progetto-1"></a>
### Decisioni di Progetto

#### 1. Topologia del Grafo Aciclico Orientato (DAG)
La struttura causale della rete comprende 5 nodi discreti ed è definita per riflettere le reali dinamiche di produzione e percezione del mercato videoludico:
* **`genre_1` (Genere primario):** Nodo radice che descrive l'archetipo ludico principale (13 stati: le 12 categorie a maggior frequenza nel catalogo, la classe specializzata `horror` e il valore residuale `other` per evitare l'esplosione combinatoria delle CPT).
* **`platform_support` (Supporto piattaforme):** Nodo radice indipendente (2 stati: `windows_only`, `multiplatform`).
* **`price_category` (Fascia di prezzo):** Nodo dipendente da `genre_1` (4 stati: `free`, `budget`, `mid_price`, `aaa_full`). Il genere influenza direttamente la politica di monetizzazione (es. le simulazioni o i titoli indie tendono a fasce budget/free, mentre produzioni action complesse si collocano spesso a prezzo pieno).
* **`playtime_category` (Longevità media):** Nodo dipendente da `genre_1` (3 stati: `short`, `medium`, `long`). La tipologia di gameplay determina fisiologicamente la durata media richiesta all'utente.
* **`recommended` (Variabile Target):** Nodo foglia binario (`yes`, `no`), condizionato congiuntamente da genere, prezzo, longevità e compatibilità di piattaforma.

```
       [ genre_1 ]              [ platform_support ]
        /   |   \                        |
       /    |    \                       |
      v     |     v                      |
[ price ]   |   [ playtime ]             |
      \     |     /                      |
       \    |    /                       |
        v   v   v                        |
     [ recommended ] <-------------------+
```

#### 2. Fattorizzazione della Distribuzione di Probabilità Congiunta
In virtù delle proprietà di indipendenza condizionale codificate dalla struttura ad anelli (I-map), la probabilità congiunta globale si fattorizza secondo la regola di scomposizione a catena bayesiana:

$$P(G, Pl, Pr, Pt, R) = P(G) \cdot P(Pl) \cdot P(Pr \mid G) \cdot P(Pt \mid G) \cdot P(R \mid G, Pr, Pt, Pl)$$

dove $G = \text{genre\_1}$, $Pl = \text{platform\_support}$, $Pr = \text{price\_category}$, $Pt = \text{playtime\_category}$ e $R = \text{recommended}$.

#### 3. Apprendimento dei Parametri (BDeu Prior)
Per stimare le tabelle di probabilità condizionata (CPT) evitando probabilità nulle per configurazioni rare o non osservate nel dataset (zero-frequency problem), è stato adottato il metodo **BayesianEstimator** con prior coniugato uniforme di Dirichlet (**BDeu**, Bayesian Dirichlet equivalent uniform) e dimensione campionaria equivalente fissata a:

$$s = 10$$

La CPT del nodo target `recommended` gestisce 4 variabili condizionanti con una cardinalità combinatoria complessiva di:

$$13 \, (\text{generi}) \times 2 \, (\text{piattaforme}) \times 3 \, (\text{longevità}) \times 4 \, (\text{prezzi}) = 312 \text{ configurazioni genitoriali}$$

generando una tabella condizionale a $624$ parametri probabilistici coerenti e validati assiomaticamente.

---

<a id="valutazione-1"></a>
### Valutazione

L'inferenza esatta è eseguita interrogando la distribuzione a posteriori su $11.811$ istanze mediante **Variable Elimination**. Di seguito si riportano i risultati quantitativi verificati empiricamente sul catalogo:

#### Query A — Titolo Horror in fascia Budget con durata Media
* **Evidenza assegnata:** $\mathbf{e}_A = \{\text{genre\_1} = \text{'horror'}, \, \text{price\_category} = \text{'budget'}, \, \text{playtime\_category} = \text{'medium'}\}$
* **Probabilità inferita:**
  $$P(\text{recommended} = \text{'yes'} \mid \mathbf{e}_A) = \mathbf{0.9112} \quad (91.12\%)$$
* **Interpretazione:** Un costo di accesso contenuto abbinato a un'esperienza horror di durata bilanciata produce una probabilità di gradimento molto elevata, coerente con le tendenze riscontrate nella community.

#### Query B — Titolo Action ad Alto Budget ma Breve Longevità
* **Evidenza assegnata:** $\mathbf{e}_B = \{\text{genre\_1} = \text{'action'}, \, \text{price\_category} = \text{'aaa\_full'}, \, \text{playtime\_category} = \text{'short'}\}$
* **Probabilità inferita:**
  $$P(\text{recommended} = \text{'yes'} \mid \mathbf{e}_B) = \mathbf{0.6471} \quad (64.71\%)$$
* **Interpretazione:** Il modello penalizza sensibilmente la combinazione di prezzo pieno (`aaa_full`) e longevità ridotta (`short`), abbattendo la fiducia statistica di oltre $26$ punti percentuali rispetto alla Query A.

#### Query C — Impatto Marginale del Supporto Multipiattaforma su Titoli Indie
* **Evidenza Windows-only:** `{'genre_1': 'indie', 'platform_support': 'windows_only'}` $\rightarrow P = \mathbf{50.60\%}$
* **Evidenza Multiplatform:** `{'genre_1': 'indie', 'platform_support': 'multiplatform'}` $\rightarrow P = \mathbf{67.17\%}$
* **Interpretazione:** Il supporto esteso agli ambienti Linux e macOS garantisce un delta positivo netto di **$+16.57\%$** sulla probabilità di raccomandazione positiva, evidenziando il valore strategico della portabilità per le produzioni indipendenti.

---

<a id="sezione-argomento-3-apprendimento-supervisionato-e-modelli-predittivi"></a>
## Sezione Argomento 3: Apprendimento Supervisionato e Modelli Predittivi

<a id="sommario-3"></a>
### Sommario
Il terzo modulo integra un classificatore supervisionato per catturare correlazioni non lineari e interazioni multivariate tra i metadati del catalogo, stimando la probabilità empirica di gradimento di ciascun titolo.

Il problema è formalizzato come un task di classificazione binaria: dato il vettore di attributi discretizzati di un videogioco $\mathbf{x}_i$, il modello deve stimare la probabilità a posteriori $P(Y = 1 \mid \mathbf{x}_i)$, dove $Y \in \{0, 1\}$ rappresenta la classe target `recommended` (derivata dalla soglia empirica del $75\%$ di gradimento positivo). Per garantire affidabilità statistica ed evitare bias di stima, la validazione è stata condotta tramite **Stratified 10-Fold Cross-Validation**, mettendo a confronto un singolo Albero di Decisione (Decision Tree) e un'architettura Ensemble ad aggregazione bootstrap (Random Forest).

---

<a id="strumenti-utilizzati-2"></a>
### Strumenti utilizzati
* **Scikit-Learn (v1.3+):** Moduli `tree.DecisionTreeClassifier`, `ensemble.RandomForestClassifier`, `model_selection.StratifiedKFold` e `model_selection.cross_validate`.
* **Pandas & NumPy:** Pipeline di codifica One-Hot (vettorizzazione degli stati categorici), binarizzazione del target e calcolo delle metriche di dispersione statistica ($\mu \pm \sigma$).

---

<a id="decisioni-di-progetto-2"></a>
### Decisioni di Progetto

#### 1. Feature Engineering e Spazio di Rappresentazione
Il dataset impiegato comprende $11.811$ titoli filtrati per significatività statistica ($\ge 50$ recensioni totali). La distribuzione del target riflette la composizione naturale del catalogo:
* **Classe 0 (Non Raccomandato / Negativo):** $5.089$ campioni ($43.09\%$)
* **Classe 1 (Raccomandato / Positivo):** $6.722$ campioni ($56.91\%$)

Per evitare la dispersione delle dimensioni dovuta a categorie rare, i generi (`genre_1` e `genre_2`) sono stati limitati ai $15$ più frequenti, collassando le classi marginali nella categoria residuale `other`. Il vettore delle feature categoriche:

$$\mathbf{x} = [\text{genre\_1}, \, \text{genre\_2}, \, \text{price\_category}, \, \text{playtime\_category}, \, \text{platform\_support}]$$

è stato trasformato tramite **One-Hot Encoding** con rimozione della prima colonna dummy (`drop_first=True`) per prevenire la multicollinearità, ottenendo una matrice di input sparsa a $36$ feature binarie.

#### 2. Configurazione e Regolarizzazione dei Modelli
Per bilanciare la capacità espressiva e prevenire l'overfitting, sono stati confrontati due modelli con vincoli strutturali omogenei:
* **Decision Tree (Baseline):** Modello singolo addestrato con partizionamento ricorsivo basato sull'indice di impurità di Gini, profondità massima vincolata a `max_depth = 8` e foglia minima `min_samples_leaf = 10`.
* **Random Forest (Ensemble Bagging):** Foresta di $150$ stimatori (`n_estimators = 150`), con `max_depth = 8`, `min_samples_leaf = 10` e campionamento casuale delle feature per ogni split ($\sqrt{36} = 6$ feature candidate per nodo).

#### 3. Protocollo Sperimentale (Stratified 10-Fold CV)
La valutazione è stata condotta dividendo il dataset in $K = 10$ partizioni bilanciate. La stratificazione preserva rigorosamente in ciascun fold la proporzione originaria tra classi ($43.09\%$ classe 0 vs $56.91\%$ classe 1). Su ciascuna iterazione sono state calcolate quattro metriche prestazionali, riportando la media empirica ($\mu$) e la deviazione standard ($\sigma$).

---

<a id="valutazione-e-risultati-sperimentali"></a>
### Valutazione e Risultati Sperimentali

#### Tabella Comparativa Prestazionale
La seguente tabella riassume i risultati ottenuti dalla Stratified 10-Fold Cross-Validation sulle $11.811$ istanze:

| Modello | Accuracy ($\mu \pm \sigma$) | Precision ($\mu \pm \sigma$) | Recall ($\mu \pm \sigma$) | F1-Score ($\mu \pm \sigma$) |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | $0.6146 \pm 0.0139$ | $\mathbf{0.6659 \pm 0.0149}$ | $0.6485 \pm 0.0141$ | $0.6570 \pm 0.0115$ |
| **Random Forest** | $\mathbf{0.6193 \pm 0.0145}$ | $0.6447 \pm 0.0124$ | $\mathbf{0.7380 \pm 0.0170}$ | $\mathbf{0.6881 \pm 0.0121}$ |

#### Analisi Comparativa delle Prestazioni
1. **Sensibilità Predittiva (Recall):** Il vantaggio architetturale della Random Forest emerge nella Recall, che passa da $0.6485$ a **$0.7380$** (un incremento netto di **$+8.95\%$** a favore dell'ensemble). Nel contesto di un sistema di raccomandazione, massimizzare la Recall è priorità primaria: minimizzare i falsi negativi garantisce che titoli di valore compatibili con il profilo utente non vengano erroneamente scartati a monte.
2. **Bilanciamento Armonico (F1-Score):** L'F1-Score medio sale a **$0.6881$** per la Random Forest contro lo $0.6570$ dell'albero singolo, confermando che l'aggregazione di più alberi indipendenti riduce la varianza di stima dell'errore e stabilizza la predizione.
3. **Generalizzazione:** La deviazione standard contenuta in tutti i fold ($\sigma \approx 0.012$) attesta l'assenza di overfitting e l'eccellente capacità di generalizzazione del modello ensemble su segmenti non visti del catalogo.

#### Feature Importance (Mean Decrease in Impurity)
L'estrazione dell'importanza delle feature calcolata tramite la riduzione media dell'impurità di Gini evidenzia i fattori maggiormente discriminanti nella decisione della Random Forest:

| Rango | Feature Codificata | Importanza (MDI) | Valenza nel Dominio |
| :---: | :--- | :---: | :--- |
| **1** | `platform_support_windows_only` | **$0.2547$** | Esclusività di piattaforma (forte penalità per assenza di supporto Linux/Mac) |
| **2** | `price_category_mid_price` | **$0.1147$** | Fascia di prezzo intermedia (€10–€30), equilibrio ottimale per la community |
| **3** | `genre_1_other` | **$0.1080$** | Generi non convenzionali e produzioni composite |
| **4** | `genre_1_simulation` | **$0.0612$** | Genere con forte polarizzazione di recensioni tra appassionati |
| **5** | `price_category_free` | **$0.0418$** | Modello Free-to-Play, frequentemente soggetto a review bombing |
| **6** | `playtime_category_short` | **$0.0384$** | Titoli brevi (< 5 ore), critici sul rapporto longevità/prezzo |
| **7** | `genre_1_anime` | **$0.0361$** | Nicchia tematica ad altissimo coinvolgimento |
| **8** | `price_category_budget` | **$0.0297$** | Fascia economica (< €10), associata a ridotto rischio d'acquisto |

---

<a id="integrazione-globale-scoring-e-formula-di-rango"></a>
## Integrazione Globale: Scoring e Formula di Rango

### Architettura di Cooperazione e Pipeline a Due Fasi
L'architettura del sistema implementa un modello a due stadi che coniuga la rigidità delle garanzie logiche con la granularità dei modelli probabilistici e statistici:

```
[ Catalogo Steam: 11.811 Titoli ]
               |
               v
  [ Modulo Deduttivo (Prolog) ]  ---> Filtro vincoli rigidi (Budget, Generi, NAF Saghe)
               |
               v  (Riduzione: 2.155 Titoli Ammissibili, -81.8%)
  [ Pipeline di Ranking Ibrido ]
         /           |           \
        v            v            v
  [ Rete Bayes ]  [ Random ]   [ Empirical Bayes ]
     (DAG)        [ Forest ]   [    Shrinkage    ]
    w1 = 0.35     w2 = 0.35        w3 = 0.30
        \            |            /
         ----->  [ Score Finale ]  <-----
                     |
                     v
   [ Graduatoria Ordinata (Top 10) ]
```

1. **Stadio 1 — Filtro Simbolico Deterministico (Prolog):** Elimina categoricamente tutti i titoli che violano vincoli assiomatici dell'utente. Su un catalogo iniziale di **$11.811$ titoli**, la Risoluzione SLD con Negation as Failure e chiusura transitiva ne ha ammessi **$2.155$**, abbattendo l'**$81.8\%$** dello spazio di ricerca ed eliminando giochi già posseduti, fuori budget o con prequel non giocati.
2. **Stadio 2 — Ranking Multi-Criterio Integrato:** Ciascuno dei $2.155$ titoli ammissibili viene valutato concorrentemente dai tre modelli per assegnare un punteggio normalizzato continuo $S(g) \in [0, 1]$.

---

### Formalizzazione Matematica dello Score Globale

Il punteggio finale di raccomandazione per ogni titolo ammissibile $g$ è definito come combinazione lineare convessa pesata:

$$S(g) = w_1 \cdot P_{BN}(g) + w_2 \cdot P_{RF}(g) + w_3 \cdot \mathcal{B}(g)$$

con vincolo di partizione dell'unità $\sum_{i=1}^3 w_i = 1.0$, configurato con i seguenti pesi operativi:
* **$w_1 = 0.35$ — Ragionamento Causale (Rete Bayesiana, $P_{BN}$):** quantifica la coerenza strutturale del titolo ($P(\text{recommended} = \text{'yes'} \mid \text{genere}, \text{prezzo}, \text{durata}, \text{piattaforma})$), premiando le configurazioni con elevata probabilità a priori di gradimento intrinseco.
* **$w_2 = 0.35$ — Predizione Multivariata (Random Forest, $P_{RF}$):** stima empirica $P(\hat{Y} = 1 \mid \mathbf{x})$ basata sull'aggregazione ensemble di alberi di decisione, catturando complesse interazioni non lineari tra generi primari, generi secondari e supporto multipiattaforma.
* **$w_3 = 0.30$ — Shrinkage Bayesiano della Community ($\mathcal{B}(g)$):** stima regolarizzata del consenso reale degli utenti Steam calcolata tramite stimatore Empirical Bayes:

$$\mathcal{B}(g) = \left( \frac{v_g}{v_g + m} \right) \cdot R_g + \left( \frac{m}{v_g + m} \right) \cdot C$$

dove:
* $v_g$ è il volume totale di recensioni ricevute dal gioco ($v_g = \text{positive\_ratings} + \text{negative\_ratings}$).
* $R_g$ è il rapporto grezzo di recensioni positive ($R_g = \text{positive\_ratings} / v_g$).
* $C$ è il gradimento medio complessivo del catalogo ($C \approx 0.748$).
* $m = 300$ è la soglia di inerzia a priori che attira i titoli con pochi voti verso la media globale $C$, impedendo a titoli con campionamento ridotto (es. 50 voti tutti positivi, $100\%$) di scavalcare ingiustamente capolavori consolidati con decine di migliaia di recensioni.

---

### Risultati Sperimentali e Graduatoria Top 10

Applicando la pipeline integrata sul profilo reale estratto tramite Steam Web API, il sistema ha prodotto la seguente graduatoria per le prime 10 posizioni:

| # | Titolo | Genere Primario (`genre_1`) | Fascia Prezzo | Voti Totali | % Positiva | $P_{BN}$ | $P_{RF}$ | Bayes Rating | Score Finale |
| :-: | :--- | :---: | :---: | :-: | :-: | :-: | :-: | :-: | :---: |
| **1** | **Fran Bow** | `horror` | `mid_price` | $5.088$ | $96.2\%$ | $0.997$ | $0.780$ | $0.950$ | **$0.9071$** |
| **2** | **Broforce** | `america` *(G2: action)* | `mid_price` | $32.092$ | $96.7\%$ | $0.925$ | $0.828$ | $0.965$ | **$0.9030$** |
| **3** | **Alien: Isolation** | `horror` | `mid_price` | $26.492$ | $92.6\%$ | $0.997$ | $0.780$ | $0.924$ | **$0.8993$** |
| **4** | **Ultimate Chicken Horse** | `local_multiplayer` *(G2: action)* | `mid_price` | $8.359$ | $94.6\%$ | $0.925$ | $0.828$ | $0.939$ | **$0.8953$** |
| **5** | **N++ (NPLUSPLUS)** | `platformer` *(G2: action)* | `mid_price` | $1.683$ | $95.1\%$ | $0.925$ | $0.828$ | $0.920$ | **$0.8894$** |
| **6** | **Batman: Arkham City** | `action` | `mid_price` | $27.250$ | $95.4\%$ | $0.995$ | $0.729$ | $0.952$ | **$0.8887$** |
| **7** | **Assault Android Cactus** | `twin_stick_shooter` *(G2: action)* | `mid_price` | $1.353$ | $94.8\%$ | $0.925$ | $0.828$ | $0.911$ | **$0.8868$** |
| **8** | **Mad Max** | `open_world` *(G2: action)* | `mid_price` | $40.033$ | $90.5\%$ | $0.925$ | $0.828$ | $0.904$ | **$0.8846$** |
| **9** | **Hitman: Absolution™** | `stealth` *(G2: action)* | `mid_price` | $25.755$ | $90.4\%$ | $0.925$ | $0.828$ | $0.902$ | **$0.8841$** |
| **10**| **Monstrum** | `horror` | `mid_price` | $1.618$ | $89.7\%$ | $0.997$ | $0.776$ | $0.873$ | **$0.8826$** |

*Nota di coerenza logico-tassonomica e convergenza dei modelli:* 
1. I titoli la cui feature primaria `genre_1` non compare tra i generi graditi diretti dell'utente (es. `america`, `local_multiplayer`, `open_world`, `stealth`, `twin_stick_shooter`) sono stati ammessi dal motore Prolog poiché soddisfano l'assioma relazionale `genere_compatibile(G1, G2)` attraverso il genere secondario `G2 = action`.
2. La convergenza dei punteggi ($P_{BN} = 0.925$ e $P_{RF} = 0.828$) per tali titoli deriva dal fatto che i loro generi primari rari vengono collassati nella classe residuale `other`, condividendo lo stesso profilo categorico modale (`mid_price`, `windows_only`). La discriminazione fine del rango viene conseguentemente risolta dallo stimatore Empirical Bayes $\mathcal{B}(g)$, che premia la solida volumetria statistica e il consenso reale della community.

#### Analisi Critica del Risultato
* **Efficacia del vincolo narrativo:** *Batman: Arkham City* si colloca al 6° posto solo perché l'utente ha già giocato ad *Arkham Asylum* (come documentato in `profilo_utente.pl`). Se *Arkham Asylum* non fosse stato registrato come completato, la NAF di Prolog lo avrebbe scartato a monte, a prescindere dal suo elevato score ($0.8887$).
* **Equilibrio Multi-Paradigma:** *Fran Bow* si aggiudica la prima posizione grazie alla massima confidenza probabilistica nel genere horror ($P_{BN} = 0.997$), a una forte predizione di gradimento ($P_{RF} = 0.780$) e a un consenso reale solido ($96.2\%$ su oltre $5.000$ voti, stabilizzato a Bayes Rating $0.950$).
* **Mitigazione dei Bias di Scala:** Giochi di nicchia con poche recensioni ma rating grezzo molto alto non monopolizzano la vetta: la presenza del termine di Shrinkage ($m = 300$) valorizza titoli consolidati con oltre $25.000$–$40.000$ recensioni (*Alien: Isolation*, *Mad Max*), garantendo raccomandazioni affidabili e commercialmente concrete.

---

<a id="conclusioni-e-sviluppi-futuri"></a>
## Conclusioni e Sviluppi Futuri

### Sintesi delle Valutazioni
Il sistema a base di conoscenza ibrido sviluppato dimostra l'efficacia della cooperazione tra paradigmi computazionali eterogenei nel risolvere un problema decisionale complesso, superando i limiti intrinseci che ciascun modello manifesterebbe se impiegato isolatamente:

* **Efficienza del Filtro Deduttivo (Prolog):** L'inferenza simbolica mediante Risoluzione SLD ha ridotto lo spazio degli stati da esplorare dell'**$81.8\%$** (da $11.811$ a $2.155$ candidati). La formalizzazione ricorsiva della chiusura transitiva e della Negation as Failure (NAF) sotto CWA ha garantito la risoluzione deterministica di vincoli di sequenzialità narrativa (saghe) e preferenze personali a costo computazionale lineare ($O(|\mathcal{V}| + |\mathcal{E}|)$), senza richiedere massicce basi di dati di addestramento.
* **Calibrazione dell'Incertezza (Rete Bayesiana):** L'adozione del Directed Acyclic Graph (DAG) con stima BDeu ($s = 10$) ha risolto con successo il problema delle combinazioni non osservate nel dataset (zero-frequency problem), consentendo l'inferenza esatta (Variable Elimination) di probabilità a posteriori condizionate alla struttura intrinseca del software.
* **Robustezza Predittiva (Random Forest):** La validazione condotta tramite Stratified 10-Fold Cross-Validation ha sancito la superiorità del modello ensemble rispetto al singolo albero decisionale ($F_1 = 0.6881 \pm 0.0121$ contro $0.6570 \pm 0.0115$, con Recall a favore dell'ensemble pari a $0.7380$ contro $0.6485$), minimizzando i falsi negativi e dimostrando un'eccellente capacità di generalizzazione sui pattern multivariati del catalogo.
* **Equità nel Ranking (Shrinkage Bayesiano):** L'integrazione convessa finale ($w_1 = 0.35, w_2 = 0.35, w_3 = 0.30$), combinata con lo stimatore Empirical Bayes ($m = 300$), ha eliminato le distorsioni causate da titoli con volumetria di recensioni ridotta, producendo una graduatoria bilanciata tra solidità causale, approvazione predittiva e reputazione reale della community.

---

### Problematiche Affrontate e Compromessi Ingegneristici
Durante lo sviluppo sono emersi vincoli operativi che hanno richiesto compromessi tecnici specifici:
* **Discretizzazione delle Feature Continue:** Come emerso nei test preliminari con modelli bayesiani continui, l'inclusione di prezzi o ore di gioco float causava saturazione della memoria e fallimenti di inferenza (*KeyError* su valori continui mai osservati). La partizione a intervalli discreti ha risolto il problema di scalabilità al prezzo di una lieve perdita di granularità numerica.
* **Estrazione della Conoscenza delle Saghe:** La costruzione del grafo aciclico orientato per la KB estensionale ha richiesto la normalizzazione e l'allineamento dei prequel/sequel dei titoli principali, stante l'assenza di un campo semantico esplicito per le serie narrative all'interno del catalogo grezzo di Steam.

---

### Sviluppi Futuri
In vista di future estensioni da parte di altri gruppi di ricerca o per una messa in produzione industriale, si delineano le seguenti linee di evoluzione architetturale:

* **Transizione a Ontologie OWL 2 DL e Web Semantico:** Sostituzione parziale o totale dei fatti Prolog con un'ontologia formale descritta in OWL 2 DL gestita tramite librerie come `Owlready2`. Tale estensione consentirebbe di sfruttare reasoner standard (es. Pellet, HermiT) per classificare gerarchie complesse di generi e interrogare la base di conoscenza tramite endpoint SPARQL.
* **Natural Language Processing (NLP) sulle Recensioni Utente:** Integrazione di un modello transformer (es. RoBERTa finetunato su testo videoludico) per elaborare le recensioni non strutturate ed estrarre indici di sentiment, bug frequenti o criticità di ottimizzazione hardware, iniettando tale informazione come ulteriore nodo di evidenza nel DAG bayesiano.
* **Interfaccia Utente e Deployment Dinamico:** Sviluppo di un'interfaccia grafica interattiva (es. Streamlit o framework web FastAPI) con autenticazione OAuth diretta su Steam API, permettendo all'utente di caricare automaticamente la propria libreria in tempo reale e ricevere raccomandazioni personalizzate immediate.

---

<a id="riferimenti-bibliografici"></a>
## Riferimenti Bibliografici

* **Ragionamento logico:** D. Poole, A. Mackworth: *Artificial Intelligence: Foundations of Computational Agents*. 3/e, Cambridge University Press [Ch.5].
* **Prolog:** D. Poole, A. Mackworth: *Artificial Intelligence: Foundations of Computational Agents*. 3/e, Cambridge University Press [Ch.15].
* **Ragionamento probabilistico e reti bayesiane:** D. Poole, A. Mackworth: *Artificial Intelligence: Foundations of Computational Agents*. 3/e, Cambridge University Press [Ch.9].
* **Apprendimento supervisionato:** D. Poole, A. Mackworth: *Artificial Intelligence: Foundations of Computational Agents*. 3/e, Cambridge University Press [Ch.7].
* **Documentazione Steam Web API e metriche catalogo:** https://partner.steamgames.com/doc/webapi
* **Specifiche Kaggle Steam Games Dataset:** https://www.kaggle.com/datasets/fronkongames/steam-games-dataset