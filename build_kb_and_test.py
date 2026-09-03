import pandas as pd
from pyswip import Prolog

print("--- 1. Generazione di conoscenza_giochi.pl dal Dataset ---")
df = pd.read_csv("dataset_games.csv")

def format_prolog_atom(text):
    if pd.isna(text):
        return "unknown"
    clean = "".join(c for c in str(text).lower() if c.isalnum() or c == '_')
    if not clean or clean[0].isdigit():
        clean = "g_" + clean
    return clean

with open("conoscenza_giochi.pl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        titolo = format_prolog_atom(row['name'])
        dev = format_prolog_atom(row['developer'])
        g1 = format_prolog_atom(row['genre_1'])
        g2 = format_prolog_atom(row['genre_2'])
        prezzo = format_prolog_atom(row['price_category'])
        playtime = format_prolog_atom(row['playtime_category'])
        platform = format_prolog_atom(row['platform_support'])
        target = format_prolog_atom(row['recommended'])

        fatto = f"gioco('{titolo}', '{dev}', '{g1}', '{g2}', '{prezzo}', '{playtime}', '{platform}', '{target}').\n"
        f.write(fatto)

print("File 'conoscenza_giochi.pl' generato con successo!")

print("\n--- 2. Consultazione Base di Conoscenza e Inferenza ---")
prolog = Prolog()
# Carichiamo i fatti del catalogo e poi le regole (che importano profilo_utente.pl)
prolog.consult("conoscenza_giochi.pl")
prolog.consult("regole.pl")

# Esecuzione della query: trova i titoli raccomandati
query_str = "consigliato(Titolo)"
risultati = list(prolog.query(query_str))

print(f"\nTotale giochi consigliati da Prolog per il tuo profilo: {len(risultati)}")
print("Primi 10 titoli raccomandati:")
for res in risultati[:10]:
    print(f"- {res['Titolo']}")