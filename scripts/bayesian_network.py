import os
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

print("=== FASE 3: Costruzione e Addestramento Rete Bayesiana ===")

# 1. Caricamento del dataset con gestione dinamica del percorso
if os.path.exists("data/dataset_games.csv"):
    csv_path = "data/dataset_games.csv"
elif os.path.exists("../data/dataset_games.csv"):
    csv_path = "../data/dataset_games.csv"
else:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_games.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Impossibile trovare il dataset in: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Dataset caricato da '{csv_path}' con {len(df)} istanze.")

# 2. Selezione e pretrattamento delle variabili
# Limitiamo il nodo genre_1 ai generi più frequenti + horror per evitare l'esplosione delle CPT
top_genres = df['genre_1'].value_counts().nlargest(12).index.tolist()
if 'horror' not in top_genres:
    top_genres.append('horror')

df_bn = df[['genre_1', 'price_category', 'playtime_category', 'platform_support', 'recommended']].copy()
df_bn['genre_1'] = df_bn['genre_1'].apply(lambda g: g if g in top_genres else 'other')
df_bn = df_bn.dropna().reset_index(drop=True)

print(f"Righe utilizzate: {len(df_bn)}")
print(f"Stati del nodo 'genre_1' ({len(df_bn['genre_1'].unique())}): {df_bn['genre_1'].unique().tolist()}")

# 3. Definizione del DAG (Directed Acyclic Graph)
edges = [
    ('genre_1', 'price_category'),        # Il genere influenza le fasce di prezzo
    ('genre_1', 'playtime_category'),     # Il genere influenza la longevità
    ('genre_1', 'recommended'),           # Gradimento intrinseco del genere
    ('price_category', 'recommended'),    # Impatto del prezzo sul giudizio
    ('playtime_category', 'recommended'), # Rapporto durata/prezzo
    ('platform_support', 'recommended')   # Disponibilità di piattaforme
]

model = DiscreteBayesianNetwork(edges)
print("\nStruttura del DAG definita:")
for parent, child in model.edges():
    print(f"  {parent} -> {child}")

# 4. Apprendimento dei Parametri (CPT) tramite BayesianEstimator
print("\nApprendimento delle CPT tramite Bayesian Estimator (prior BDeu)...")
estimator = BayesianEstimator(model, df_bn)
cpds = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=10)
model.add_cpds(*cpds)

# Validazione assiomatica del modello probabilistico
assert model.check_model(), "Errore: il modello bayesiano contiene distribuzioni non valide!"
print("Modello validato con successo: tutte le CPT rispettano gli assiomi della probabilità.")

# 5. Ispezione della CPT del nodo target 'recommended'
cpt_target = model.get_cpds('recommended')
print("\n--- CPT del nodo 'recommended' ---")
print(f"Variabile: {cpt_target.variable}")
print(f"Variabili condizionanti (genitori): {cpt_target.variables[1:]}")
print(f"Forma della tabella (valori): {cpt_target.values.shape}")

# 6. Inferenza Probabilistica Esatta (Variable Elimination)
infer = VariableElimination(model)

def get_prob_yes(factor):
    """Estrae la probabilità a posteriori per recommended='yes'."""
    try:
        return float(factor.get_value(recommended='yes'))
    except Exception:
        idx = list(factor.state_names['recommended']).index('yes')
        return float(factor.values[idx])

print("\n=======================================================")
print(" SIMULAZIONE QUERY DI INFERENZA PROBABILISTICA ")
print("=======================================================\n")

# Caso A: Gioco Horror, fascia Budget, durata Medium
ev_horror = {'genre_1': 'horror', 'price_category': 'budget', 'playtime_category': 'medium'}
q_horror = infer.query(variables=['recommended'], evidence=ev_horror, show_progress=False)
p_horror = get_prob_yes(q_horror)
print(f"Query A - Evidenza: {ev_horror}")
print(f"-> P(recommended = 'yes' | evidenza) = {p_horror:.4f} ({p_horror*100:.2f}%)\n")

# Caso B: Gioco Action, fascia AAA Full price, durata Short
ev_action = {'genre_1': 'action', 'price_category': 'aaa_full', 'playtime_category': 'short'}
q_action = infer.query(variables=['recommended'], evidence=ev_action, show_progress=False)
p_action = get_prob_yes(q_action)
print(f"Query B - Evidenza: {ev_action}")
print(f"-> P(recommended = 'yes' | evidenza) = {p_action:.4f} ({p_action*100:.2f}%)\n")

# Caso C: Impatto del solo supporto Multipiattaforma su un Indie
ev_win = {'genre_1': 'indie', 'platform_support': 'windows_only'}
ev_multi = {'genre_1': 'indie', 'platform_support': 'multiplatform'}
q_win = infer.query(variables=['recommended'], evidence=ev_win, show_progress=False)
q_multi = infer.query(variables=['recommended'], evidence=ev_multi, show_progress=False)
print("Query C - Effetto marginale di Windows-only vs Multiplatform su Indie:")
print(f"-> Windows only:  {get_prob_yes(q_win)*100:.2f}%")
print(f"-> Multiplatform: {get_prob_yes(q_multi)*100:.2f}%\n")

# 7. Funzione per la pipeline finale di raccomandazione
def get_recommendation_probability(genre, price, playtime, platform):
    genre_clean = genre if genre in top_genres else 'other'
    evidence = {
        'genre_1': genre_clean,
        'price_category': price,
        'playtime_category': playtime,
        'platform_support': platform
    }
    # Filtra solo gli stati validi riconosciuti dalla rete
    valid_ev = {k: v for k, v in evidence.items() if v in model.get_cpds(k).state_names[k]}
    res = infer.query(variables=['recommended'], evidence=valid_ev, show_progress=False)
    return get_prob_yes(res)

if __name__ == "__main__":
    score = get_recommendation_probability('horror', 'budget', 'medium', 'multiplatform')
    print(f"Punteggio di raccomandazione test (Horror, Budget, Medium, Multi): {score:.4f}")