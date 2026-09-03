import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pyswip import Prolog

# Import dei moduli pgmpy e scikit-learn
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.ensemble import RandomForestClassifier

print("===================================================================")
print(" SISTEMA IBRIDO: LOGICA + RETE BAYESIANA + ML + COMMUNITY RATING ")
print("===================================================================\n")

# Controllo preliminare di consistenza
dataset_path = "data/dataset_games.csv" if os.path.exists("data") else "dataset_games.csv"
conoscenza_path = "logic/conoscenza_giochi.pl" if os.path.exists("logic") else "conoscenza_giochi.pl"

if not os.path.exists(dataset_path) or not os.path.exists(conoscenza_path):
    print("[INIT] File di conoscenza non trovati. Avvio preprocessing automatico...")
    script_prep = "scripts/preprocess.py" if os.path.exists("scripts/preprocess.py") else "preprocess.py"
    subprocess.run([sys.executable, script_prep], check=True)

# 1. Caricamento Dataset e Lookup
print("[1/5] Caricamento del dataset unificato...")
df = pd.read_csv(dataset_path)

def format_prolog_atom(text):
    if pd.isna(text): return "unknown"
    raw = str(text).split('/')[0].lower()
    for noise in [' hd remaster', ' hd', ' remaster', ' remastered']:
        raw = raw.replace(noise, '')
    clean = "".join(c for c in raw.replace(" ", "_").replace("-", "_").replace(":", "") if c.isalnum() or c == '_')
    while "__" in clean: clean = clean.replace("__", "_")
    clean = clean.strip("_")
    if not clean or clean[0].isdigit(): clean = "g_" + clean
    return clean

df['atom_name'] = df['name'].apply(format_prolog_atom)

catalog = {}
for _, row in df.iterrows():
    catalog[row['atom_name']] = {
        'name': row['name'],
        'genre_1': row['genre_1'],
        'genre_2': row['genre_2'],
        'price': row['price_category'],
        'playtime': row['playtime_category'],
        'platform': row['platform_support'],
        'total_ratings': int(row['total_ratings']),
        'positive_ratio': float(row['positive_ratio']),
        'bayesian_rating': float(row['bayesian_rating']),
        'review_tier': row['review_tier']
    }

# 2. Modulo Simbolico (Prolog NAF + Saghe)
print("[2/5] Consultazione Base di Conoscenza Prolog ed estrazione candidati ammissibili...")
prolog = Prolog()
prolog.consult(conoscenza_path)
prolog.consult("logic/regole.pl" if os.path.exists("logic/regole.pl") else "regole.pl")

simbolico_candidati = list(prolog.query("consigliato(Titolo)"))
titoli_ammessi = list(set(str(res['Titolo']) for res in simbolico_candidati))
print(f"-> Giochi ammessi dopo il filtro logico (Prolog): {len(titoli_ammessi)}")

# 3. Modulo Probabilistico (Rete Bayesiana)
print("\n[3/5] Addestramento Rete Bayesiana (DAG)...")
top_genres_bn = df['genre_1'].value_counts().nlargest(12).index.tolist()
if 'horror' not in top_genres_bn: top_genres_bn.append('horror')

df_bn = df[['genre_1', 'price_category', 'playtime_category', 'platform_support', 'recommended']].copy()
df_bn['genre_1'] = df_bn['genre_1'].apply(lambda g: g if g in top_genres_bn else 'other')
df_bn = df_bn.dropna()

edges = [
    ('genre_1', 'price_category'),
    ('genre_1', 'playtime_category'),
    ('genre_1', 'recommended'),
    ('price_category', 'recommended'),
    ('playtime_category', 'recommended'),
    ('platform_support', 'recommended')
]
bn_model = DiscreteBayesianNetwork(edges)
estimator = BayesianEstimator(bn_model, df_bn)
cpds = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=10)
bn_model.add_cpds(*cpds)
bn_infer = VariableElimination(bn_model)

def get_bayes_prob(g1, price, playtime, platform):
    genre_clean = g1 if g1 in top_genres_bn else 'other'
    ev = {'genre_1': genre_clean, 'price_category': price, 'playtime_category': playtime, 'platform_support': platform}
    valid_ev = {k: v for k, v in ev.items() if v in bn_model.get_cpds(k).state_names[k]}
    res = bn_infer.query(variables=['recommended'], evidence=valid_ev, show_progress=False)
    try:
        return float(res.get_value(recommended='yes'))
    except Exception:
        idx = list(res.state_names['recommended']).index('yes')
        return float(res.values[idx])

# 4. Modulo Machine Learning (Random Forest)
print("[4/5] Addestramento Random Forest...")
features_ml = ['genre_1', 'genre_2', 'price_category', 'playtime_category', 'platform_support']
X_raw = df[features_ml].copy()
y_ml = (df['recommended'] == 'yes').astype(int)

top_genres_ml = X_raw['genre_1'].value_counts().nlargest(15).index.tolist()
X_raw['genre_1'] = X_raw['genre_1'].apply(lambda g: g if g in top_genres_ml else 'other')
X_raw['genre_2'] = X_raw['genre_2'].apply(lambda g: g if g in top_genres_ml else 'other')

X_encoded = pd.get_dummies(X_raw, drop_first=True)
ml_feature_names = X_encoded.columns.tolist()

rf_model = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
rf_model.fit(X_encoded, y_ml)

def get_ml_prob(g1, g2, price, playtime, platform):
    g1_c = g1 if g1 in top_genres_ml else 'other'
    g2_c = g2 if g2 in top_genres_ml else 'other'
    sample = {col: 0 for col in ml_feature_names}
    for k, v in [('genre_1', g1_c), ('genre_2', g2_c), ('price_category', price), 
                 ('playtime_category', playtime), ('platform_support', platform)]:
        col = f"{k}_{v}"
        if col in sample: sample[col] = 1
    return float(rf_model.predict_proba(pd.DataFrame([sample]))[0][1])

# 5. Pipeline Integrata Multi-Criterio
print("\n[5/5] Calcolo Ranking Integrato...")

candidati_valutati = []
for atom in titoli_ammessi:
    if atom not in catalog: continue
    meta = catalog[atom]
    
    p_bayes = get_bayes_prob(meta['genre_1'], meta['price'], meta['playtime'], meta['platform'])
    p_ml = get_ml_prob(meta['genre_1'], meta['genre_2'], meta['price'], meta['playtime'], meta['platform'])
    score_modelli = (0.50 * p_bayes) + (0.50 * p_ml)
    
    b_rating = meta['bayesian_rating']
    score_finale = (0.50 * score_modelli) + (0.50 * b_rating)
    
    candidati_valutati.append({
        'name': meta['name'],
        'genre_1': meta['genre_1'],
        'price': meta['price'],
        'tier': meta['review_tier'],
        'total_votes': meta['total_ratings'],
        'pos_ratio': round(meta['positive_ratio'] * 100, 1),
        'p_bayes': round(p_bayes, 3),
        'p_ml': round(p_ml, 3),
        'b_rating': round(b_rating, 3),
        'score_finale': round(score_finale, 4)
    })

df_ranking = pd.DataFrame(candidati_valutati).sort_values(by='score_finale', ascending=False).reset_index(drop=True)

# Visualizzazione Top 15
print("\n" + "="*105)
print(" TOP 15 RACCOMANDAZIONI FINALI PERSONALIZZATE ")
print("="*105)
print(f"{'#':<3} | {'TITOLO':<30} | {'GENERE':<10} | {'PREZZO':<9} | {'VOTI':<8} | {'% POS':<6} | {'BAYES':<6} | {'ML':<6} | {'COMMUNITY':<9} | {'FINALE'}")
print("-" * 105)

for i in range(min(15, len(df_ranking))):
    r = df_ranking.iloc[i]
    t = r['name'][:28]
    g = r['genre_1'][:9]
    p = r['price'][:8]
    print(f"{i+1:<3} | {t:<30} | {g:<10} | {p:<9} | {r['total_votes']:<8} | {r['pos_ratio']:<6.1f} | {r['p_bayes']:<6.3f} | {r['p_ml']:<6.3f} | {r['b_rating']:<9.3f} | {r['score_finale']:.4f}")

df_ranking.to_csv("raccomandazioni_finali.csv", index=False)
print("\nReport esportato con successo in 'raccomandazioni_finali.csv'!")