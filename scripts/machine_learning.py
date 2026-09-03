import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("=== FASE 4: Apprendimento Supervisionato (Decision Tree & Random Forest) ===")

# 1. Caricamento del dataset
df = pd.read_csv("dataset_games.csv")

# 2. Selezione feature e encoding
features = ['genre_1', 'genre_2', 'price_category', 'playtime_category', 'platform_support']
X = df[features].copy()
y = (df['recommended'] == 'yes').astype(int) # Target binario: 1 = yes, 0 = no

# Raggruppamento generi rari per limitare la dimensionalità della matrice sparsa
top_genres = X['genre_1'].value_counts().nlargest(15).index.tolist()
X['genre_1'] = X['genre_1'].apply(lambda g: g if g in top_genres else 'other')
X['genre_2'] = X['genre_2'].apply(lambda g: g if g in top_genres else 'other')

# One-Hot Encoding per le variabili categoriche
X_encoded = pd.get_dummies(X, drop_first=True)
feature_names = X_encoded.columns.tolist()

print(f"Campioni: {X_encoded.shape[0]} | Feature codificate: {X_encoded.shape[1]}")
print(f"Bilanciamento classi target: {dict(y.value_counts(normalize=True).round(3))}")

# 3. Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Modello 1: Decision Tree (con regolarizzazione della profondità)
dt_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=42)
dt_model.fit(X_train, y_train)

# 5. Modello 2: Random Forest (Ensemble di 150 stimatori)
rf_model = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 6. Validazione Incrociata (Stratified 5-Fold Cross Validation)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_dt = cross_val_score(dt_model, X_train, y_train, cv=cv, scoring='f1')
cv_rf = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1')

print("\n--- Risultati Cross-Validation (F1-Score 5-Fold) ---")
print(f"Decision Tree F1: {cv_dt.mean():.4f} (+/- {cv_dt.std():.4f})")
print(f"Random Forest F1: {cv_rf.mean():.4f} (+/- {cv_rf.std():.4f})")

# 7. Valutazione sul Test Set Indipendente
y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n=======================================================")
print(" REPORT DI CLASSIFICAZIONE: RANDOM FOREST ")
print("=======================================================")
print(classification_report(y_test, y_pred_rf, target_names=['No', 'Yes']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_rf):.4f}")

# 8. Feature Importance (Interpretazione delle decisioni)
importances = pd.Series(rf_model.feature_importances_, index=feature_names)
print("\nTop 7 Feature più Determinanti secondo Random Forest:")
for feat, imp in importances.sort_values(ascending=False).head(7).items():
    print(f"  - {feat:<35}: {imp:.4f} ({imp*100:.1f}%)")

# 9. Funzione Predittiva per la Pipeline Finale
def predict_ml_score(genre_1, genre_2, price, playtime, platform):
    """Calcola la probabilità di raccomandazione predetta dal Random Forest."""
    g1 = genre_1 if genre_1 in top_genres else 'other'
    g2 = genre_2 if genre_2 in top_genres else 'other'
    
    sample_dict = {col: 0 for col in feature_names}
    
    # Assegnazione valori Dummy
    for k, v in [('genre_1', g1), ('genre_2', g2), ('price_category', price), 
                 ('playtime_category', playtime), ('platform_support', platform)]:
        col_name = f"{k}_{v}"
        if col_name in sample_dict:
            sample_dict[col_name] = 1
            
    sample_df = pd.DataFrame([sample_dict])
    prob_yes = rf_model.predict_proba(sample_df)[0][1]
    return float(prob_yes)

if __name__ == "__main__":
    test_p = predict_ml_score('horror', 'action', 'budget', 'medium', 'multiplatform')
    print(f"\nPredizione test Random Forest (Horror, Budget, Medium): {test_p:.4f}")