import requests
import pandas as pd
import numpy as np

# Inserisci qui la chiave a 32 caratteri ottenuta da https://steamcommunity.com/dev/apikey
API_KEY = "LA_TUA_CHIAVE_API"
STEAM_ID = "IL_TUO_STEAM_ID_64"

def format_prolog_atom(text):
    if pd.isna(text): return "unknown"
    formatted = str(text).lower().replace(" ", "_").replace("-", "_").replace(":", "")
    clean = "".join(c for c in formatted if c.isalnum() or c == '_')
    while "__" in clean: clean = clean.replace("__", "_")
    clean = clean.strip("_")
    if not clean or clean[0].isdigit(): clean = "g_" + clean
    return clean

print("--- 1. Download libreria da Steam Web API ---")
url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
params = {
    "key": API_KEY,
    "steamid": STEAM_ID,
    "include_appinfo": True,
    "include_played_free_games": True,
    "format": "json"
}

response = requests.get(url, params=params)

if response.status_code != 200:
    print(f"Errore API (Status {response.status_code}): verifica che l'API key sia corretta.")
    exit()

data = response.json()
games = data.get("response", {}).get("games", [])

if not games:
    print("Nessun gioco trovato. Assicurati che nelle impostazioni privacy di Steam 'Dettagli dei giochi' sia su Pubblico.")
    exit()

print(f"Trovati {len(games)} giochi nell'account di Domy_nator.")

# Creazione DataFrame dei giochi posseduti
df_user = pd.DataFrame(games)
df_user['atom_name'] = df_user['name'].apply(format_prolog_atom)
df_user['playtime_hours'] = df_user['playtime_forever'] / 60

# --- 2. Inferenza automatica dei generi preferiti dal gameplay reale ---
# Carichiamo il dataset per trovare i generi associati ai giochi che hai giocato
df_games = pd.read_csv("dataset_games.csv")
df_games['atom_name'] = df_games['name'].apply(format_prolog_atom)

merged = pd.merge(df_user, df_games, on='atom_name', how='inner')

# Raccogliamo le ore spese per genere
genre_hours = {}
for _, row in merged.iterrows():
    hours = row['playtime_hours']
    for g in [row['genre_1'], row['genre_2']]:
        if pd.notna(g) and g != 'unknown':
            genre_hours[g] = genre_hours.get(g, 0.0) + hours

# Ordiniamo i generi per tempo di gioco e prendiamo i principali
top_genres = sorted(genre_hours.items(), key=lambda x: x[1], reverse=True)
preferred_genres = [g[0] for g in top_genres[:5]] if top_genres else ['action', 'adventure', 'rpg']

print("\nGeneri preferiti dedotti automaticamente dalle ore di gioco:")
for g, h in top_genres[:5]:
    print(f"- {g}: {h:.1f} ore registrate")

# --- 3. Scrittura del file Prolog profilo_utente.pl ---
with open("profilo_utente.pl", "w", encoding="utf-8") as f:
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
    for _, row in df_user.iterrows():
        titolo = row['atom_name']
        f.write(f"gia_giocato('{titolo}').\n")

print("\nFile 'profilo_utente.pl' generato con successo!")
