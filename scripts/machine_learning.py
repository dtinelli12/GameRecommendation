import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

print("===================================================================")
print(" MODULO MACHINE LEARNING: COMPARAZIONE E STRATIFIED K-FOLD CV ")
print("===================================================================\n")

# 1. Caricamento del dataset elaborato con percorso flessibile
if os.path.exists("data/dataset_games.csv"):
    dataset_path = "data/dataset_games.csv"
elif os.path.exists("../data/dataset_games.csv"):
    dataset_path = "../data/dataset_games.csv"
else:
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_games.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Impossibile trovare il dataset in '{dataset_path}'. Eseguire prima preprocess.py.")

df = pd.read_csv(dataset_path)
print(f"Dataset caricato correttamente: {len(df)} istanze.")

# 2. Definizione Features e Target
features = ['genre_1', 'genre_2', 'price_category', 'playtime_category', 'platform_support']
X_raw = df[features].copy()
y = (df['recommended'] == 'yes').astype(int)

# Raggruppamento generi rari in 'other' per evitare sparsità
top_genres = X_raw['genre_1'].value_counts().nlargest(15).index.tolist()
X_raw['genre_1'] = X_raw['genre_1'].apply(lambda g: g if g in top_genres else 'other')
X_raw['genre_2'] = X_raw['genre_2'].apply(lambda g: g if g in top_genres else 'other')

# One-Hot Encoding
X = pd.get_dummies(X_raw, drop_first=True)

print(f"Feature matrix X: {X.shape[0]} righe, {X.shape[1]} colonne codificate.")
print(f"Distribuzione classi target: {np.bincount(y)} (0: Negativo, 1: Positivo)\n")

# 3. Definizione Modelli da confrontare
modelli = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=10,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
}

# 4. Configurazione Stratified K-Fold CV (10 fold)
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

risultati = []

print(f"Avvio validazione ({n_splits}-Fold Stratified Cross Validation)...")

for nome, modello in modelli.items():
    scores = cross_validate(modello, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    
    acc_mean, acc_std = np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])
    prec_mean, prec_std = np.mean(scores['test_precision']), np.std(scores['test_precision'])
    rec_mean, rec_std = np.mean(scores['test_recall']), np.std(scores['test_recall'])
    f1_mean, f1_std = np.mean(scores['test_f1']), np.std(scores['test_f1'])
    
    risultati.append({
        "Modello": nome,
        "Accuracy (mean ± std)": f"{acc_mean:.4f} ± {acc_std:.4f}",
        "Precision (mean ± std)": f"{prec_mean:.4f} ± {prec_std:.4f}",
        "Recall (mean ± std)": f"{rec_mean:.4f} ± {rec_std:.4f}",
        "F1-Score (mean ± std)": f"{f1_mean:.4f} ± {f1_std:.4f}",
        "_acc_num": acc_mean,
        "_f1_num": f1_mean
    })

df_risultati = pd.DataFrame(risultati)

# 5. Stampa Tabella Riassuntiva Richiesta dalle Linee Guida
print("\n" + "="*90)
print(" TABELLA RIASSUNTIVA COMPARATIVA (Stratified 10-Fold CV) ")
print("="*90)
colonne_stampa = ["Modello", "Accuracy (mean ± std)", "Precision (mean ± std)", "Recall (mean ± std)", "F1-Score (mean ± std)"]
print(df_risultati[colonne_stampa].to_string(index=False))
print("="*90)

# 6. Feature Importance (Random Forest)
rf = modelli["Random Forest"]
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nTop 10 Feature più determinanti per la classificazione (Random Forest):")
for feat, imp in importances.head(10).items():
    print(f"- {feat:<30}: {imp:.4f}")