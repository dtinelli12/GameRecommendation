import os
import sys
import requests
import pandas as pd
import numpy as np

# Configurazione credenziali Steam Web API
API_KEY = "LA_TUA_CHIAVE_API"  # Inserisci la tua API Key da https://steamcommunity.com/dev/apikey
STEAM_ID = "76561198439482035" # SteamID64 di Domy_nator

# 1. Risoluzione dinamica dei percorsi di input e output
if os.path.exists("data/dataset_games.csv"):
    dataset_path = "data/dataset_games.csv"
    output_dir = "logic"
elif os.path.exists("../data/dataset_games.csv"):
    dataset_path = "../data/dataset_games.csv"
    output_dir = "../logic"
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "data", "dataset_games.csv")
    output_dir = os.path.join(base_dir, "..", "logic")

os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "profilo_utente.pl")

# 2. Funzione di atomizzazione identica a preprocess.py e game_recommendation.py
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

print("--- 1. Download libreria da Steam Web API ---")
if API_KEY == "LA_TUA_CHIAVE_API":
    print("[ATTENZIONE] Inserisci una Steam API Key valida nella variabile API_KEY!")
    sys.exit(1)

url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
params = {
    "key": API_KEY,
    "steamid": STEAM_ID,
    "include_appinfo": True,
    "include_played_free_games": True,
    "format": "json"
}

try:
    response = requests.get(url, params=params, timeout=10)
except Exception as e:
    print(f"Errore di connessione a Steam API: {e}")
    sys.exit(1)

if response.status_code != 200:
    print(f"Errore API (Status {response.status_code}): verifica che l'API key sia corretta.")
    sys.exit(1)

data = response.json()
games = data.get("response", {}).get("games", [])

if not games:
    print("Nessun gioco trovato. Assicurati che nelle impostazioni privacy di Steam 'Dettagli dei giochi' sia su Pubblico.")
    sys.exit(1)

print(f"Trovati {len(games)} giochi nell'account Steam ({STEAM_ID}).")

# Creazione DataFrame dei giochi posseduti
df_user = pd.DataFrame(games)
df_user['atom_name'] = df_user['name'].apply(format_prolog_atom)
df_user['playtime_hours'] = df_user['playtime_forever'] / 60

# --- 2. Inferenza automatica dei generi preferiti dal gameplay reale ---
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Impossibile trovare il dataset in '{dataset_path}'.")

df_games = pd.read_csv(dataset_path)
df_games['atom_name'] = df_games['name'].apply(format_prolog_atom)

merged = pd.merge(df_user, df_games, on='atom_name', how='inner')

genre_hours = {}
for _, row in merged.iterrows():
    hours = row['playtime_hours']
    for g in [row['genre_1'], row['genre_2']]:
        if pd.notna(g) and g != 'unknown':
            genre_hours[g] = genre_hours.get(g, 0.0) + hours

top_genres = sorted(genre_hours.items(), key=lambda x: x[1], reverse=True)
preferred_genres = [g[0] for g in top_genres[:5]] if top_genres else ['racing', 'soccer', 'action', 'horror', 'fps']

print("\nGeneri preferiti dedotti automaticamente dalle ore di gioco:")
for g, h in top_genres[:5]:
    print(f"- {g}: {h:.1f} ore registrate")

# Lista atomi posseduti
giochi_posseduti = set(df_user['atom_name'].dropna().tolist())

# Normalizzazione alias per coerenza con il grafo saghe di regole.pl
if any("batman_arkham_asylum" in g for g in giochi_posseduti):
    giochi_posseduti.add("batman_arkham_asylum")

# --- 3. Scrittura del file Prolog logic/profilo_utente.pl ---
with open(output_file, "w", encoding="utf-8") as f:
    f.write("% ========================================================\n")
    f.write("% PROFILO_UTENTE.PL - Fatti estratti da Steam Web API\n")
    f.write(f"% Utente: Domy_nator (SteamID64: {STEAM_ID})\n")
    f.write("% ========================================================\n\n")
    
    f.write("% Generi preferiti (dedotti dallo storico di gioco)\n")
    for pg in preferred_genres:
        f.write(f"genere_gradito('{pg}').\n")
    
    f.write("\n% Fasce di prezzo considerate accettabili\n")
    f.write("fascia_prezzo_accettabile('free').\n")
    f.write("fascia_prezzo_accettabile('budget').\n")
    f.write("fascia_prezzo_accettabile('mid_price').\n\n")
    
    f.write("% Giochi posseduti / gia' affrontati (per Negation as Failure)\n")
    for titolo in sorted(giochi_posseduti):
        f.write(f"gia_giocato('{titolo}').\n")

print(f"\nFile '{output_file}' generato con successo e perfettamente allineato!")