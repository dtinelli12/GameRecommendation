import os
import pandas as pd
import numpy as np

print("--- FASE 1: Preprocessing e Generazione Conoscenza ---")

# Creazione cartelle di output se non esistono
os.makedirs("data", exist_ok=True)
os.makedirs("logic", exist_ok=True)

# 1. Caricamento dataset grezzo
csv_in = "data/steam_games_raw.csv" if os.path.exists("data/steam_games_raw.csv") else "steam_games_raw.csv"
df = pd.read_csv(csv_in)
print(f"Righe caricate da '{csv_in}': {len(df)}")

# 2. Filtro significatività statistica (>= 50 voti)
df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
df = df[df['total_ratings'] >= 50].copy()

# 3. Rimozione duplicati
df['name_clean'] = df['name'].astype(str).str.lower().str.strip()
df = df.drop_duplicates(subset=['name_clean'], keep='first').reset_index(drop=True)

# 4. Bayesian Rating ponderato e target binario
df['positive_ratio'] = df['positive_ratings'] / df['total_ratings']
C = df['positive_ratio'].mean()
m = 300
df['bayesian_rating'] = (df['total_ratings'] / (df['total_ratings'] + m)) * df['positive_ratio'] + (m / (df['total_ratings'] + m)) * C
df['recommended'] = np.where(df['positive_ratio'] >= 0.75, 'yes', 'no')

def get_review_tier(row):
    r, v = row['positive_ratio'], row['total_ratings']
    if r >= 0.95 and v >= 500: return 'overwhelmingly_positive'
    if r >= 0.85: return 'very_positive'
    if r >= 0.70: return 'mostly_positive'
    if r >= 0.40: return 'mixed'
    return 'negative'

df['review_tier'] = df.apply(get_review_tier, axis=1)

# 5. Discretizzazione Feature
df['price_category'] = pd.cut(df['price'], bins=[-1, 0, 9.99, 29.99, 1000], labels=['free', 'budget', 'mid_price', 'aaa_full']).astype(str)
df['playtime_category'] = pd.cut(df['average_playtime'] / 60, bins=[-1, 5, 20, 100000], labels=['short', 'medium', 'long']).astype(str)
df['platform_support'] = df['platforms'].apply(lambda p: 'multiplatform' if any(x in str(p).lower() for x in ['linux', 'mac']) else 'windows_only')

# 6. Formattazione atomi Prolog e gestione tag
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

TAGS_DA_IGNORARE = {
    'singleplayer', 'multiplayer', 'co_op', 'online_co_op', 'local_co_op',
    'atmospheric', 'great_soundtrack', 'story_rich', 'difficult', 'funny',
    'female_protagonist', 'first_person', 'third_person', '2d', '3d',
    'vr', 'controller', 'early_access', 'casual', 'short', 'masterpiece'
}

def extract_smart_genres(row):
    all_tokens = [t.strip().lower() for t in str(row.get('steamspy_tags', '')).split(';') if t.strip()]
    for g in str(row.get('genres', '')).split(';'):
        gc = g.strip().lower()
        if gc and gc not in all_tokens: all_tokens.append(gc)
    filtered = []
    for t in all_tokens:
        atom = format_prolog_atom(t)
        if atom in TAGS_DA_IGNORARE or len(atom) < 2: continue
        if 'horror' in atom: atom = 'horror'
        if atom not in filtered: filtered.append(atom)
    g1 = filtered[0] if len(filtered) > 0 else 'unknown'
    g2 = filtered[1] if len(filtered) > 1 else 'unknown'
    return pd.Series([g1, g2], index=['genre_1', 'genre_2'])

df[['genre_1', 'genre_2']] = df.apply(extract_smart_genres, axis=1)
df['developer'] = df['developer'].apply(format_prolog_atom)
df['atom_title'] = df['name'].apply(format_prolog_atom)

# 7. Esportazione CSV
colonne_finali = [
    'name', 'developer', 'genre_1', 'genre_2',
    'price_category', 'playtime_category', 'platform_support', 'recommended',
    'total_ratings', 'positive_ratio', 'bayesian_rating', 'review_tier'
]
df[colonne_finali].to_csv("data/dataset_games.csv", index=False)
print("Salvato: 'data/dataset_games.csv'")

# 8. Generazione automatica dei fatti Prolog
with open("logic/conoscenza_giochi.pl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(
            f"gioco('{row['atom_title']}', '{row['developer']}', '{row['genre_1']}', "
            f"'{row['genre_2']}', '{row['price_category']}', '{row['playtime_category']}', "
            f"'{row['platform_support']}', '{row['recommended']}').\n"
        )
print("Generato: 'logic/conoscenza_giochi.pl'")